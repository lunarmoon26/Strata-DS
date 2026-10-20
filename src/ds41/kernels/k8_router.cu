// K8: one weight read for the token tile, followed by a GPU-only stable top six.
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/config.hpp"
#include "k8/math.hpp"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::ds41::kernels {
namespace {

constexpr int kMaxTokens = 8;  // Fixed interface, not a benchmark-derived limit.
constexpr int kWarp = 32;
constexpr int kSelectThreads = kWarp;  // One independent warp CTA per token.
static_assert(kDim == 5120 && kExperts == 384 && kTopK == 6);
static_assert(384 % 8 == 0);

void check(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "ds41 router: %s: %s\n", where, cudaGetErrorString(error));
        std::abort();
    }
}

struct Workspace {
    int device;
    double* scores;
};

double* workspace() {
    // Exactly one K8-owned, maximum-size allocation per device. The engine
    // warms up before capture and never overlaps two K8 calls on one device.
    // Other task kernels have independent storage and may run concurrently.
    static std::mutex mutex;
    static std::vector<Workspace> buffers;
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    std::lock_guard<std::mutex> lock(mutex);
    for (const Workspace& buffer : buffers) {
        if (buffer.device == device) return buffer.scores;
    }
    double* scores = nullptr;
    check(cudaMalloc(&scores, kMaxTokens * kExperts * sizeof(double)), "initialize scratch");
    buffers.push_back({device, scores});
    return scores;  // Kept for process lifetime, including captured graph replay.
}

__device__ __forceinline__ float warp_sum(float value) {
    // Identical lane ownership, FMA order, and reduction tree to bf16_gemv_k
    // in ops.cu. In particular, do not split K or reassociate partial sums.
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

// One warp per expert, kDecodeWarps experts per CTA. The warps' logits meet in shared memory and one lane per expert
// computes the double score: a warp's lane 0 doing it alone issued kDecodeWarps times as many FP64 instructions on
// the SM's few FP64 units (RTX 5090 Laptop: 12.3 -> 10.5 us; the score bits are unchanged).
constexpr int kDecodeWarps = 8;
__global__ void decode_scores(const __nv_bfloat16* __restrict__ x,
                              const __nv_bfloat16* __restrict__ w,
                              double* __restrict__ scores) {
    __shared__ float logits[kDecodeWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int expert = blockIdx.x * kDecodeWarps + warp;
    const __nv_bfloat16* row = w + expert * kDim;
    float acc = 0.0f;
#pragma unroll 8
    for (int d = lane; d < kDim; d += kWarp) {
        acc = __fmaf_rn(__bfloat162float(x[d]), __bfloat162float(row[d]), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[warp] = acc;
    __syncthreads();
    if (threadIdx.x < kDecodeWarps)
        scores[blockIdx.x * kDecodeWarps + threadIdx.x] = k8_detail::score(logits[threadIdx.x]);
}

template <int Tokens>
__global__ void tile_scores(const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ w,
                            double* __restrict__ scores) {
    // Each CTA owns one expert. Cooperatively load and convert its entire
    // BF16 row once, then reuse it across all token warps. Compared with one
    // warp holding eight accumulators, this exposes independent token warps
    // without changing any token's FP32 accumulation sequence.
    __shared__ float row[kDim];
    const int expert = blockIdx.x;
    for (int d = threadIdx.x; d < kDim; d += blockDim.x) {
        row[d] = __bfloat162float(w[expert * kDim + d]);
    }
    __syncthreads();
    const int token = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (token < Tokens) {
        float acc = 0.0f;
#pragma unroll 8
        for (int d = lane; d < kDim; d += kWarp) {
            acc = __fmaf_rn(__bfloat162float(x[token * kDim + d]), row[d], acc);
        }
        acc = warp_sum(acc);
        // Scores are computed where logits are produced, spreading the small
        // nonlinear step across the GEMV grid instead of bottlenecking one CTA.
        if (lane == 0) scores[token * kExperts + expert] = k8_detail::score(acc);
    }
}

// The biased score as an integer with the same order: a > b in double exactly when key(a) > key(b), and equal
// doubles give equal keys (-0 is folded to +0). The selector compares keys, because FP64 compares are slow on GPUs
// with few FP64 units (RTX 5090 Laptop: select_top6 9.7 -> 3.4 us). A NaN gets the lowest key, below the -inf of
// the empty candidate: as with the double compare, it never wins a comparison.
__device__ __forceinline__ long long order_key(double value) {
    const long long bits = __double_as_longlong(value);
    const long long magnitude = bits & 0x7fffffffffffffffll;
    if (magnitude > 0x7ff0000000000000ll) return LLONG_MIN;
    if (magnitude == 0) return 0;
    return bits < 0 ? (bits ^ 0x7fffffffffffffffll) : bits;
}

__device__ __forceinline__ bool better_key(long long a, int ai, long long b, int bi) {
    return a > b || (a == b && ai < bi);
}

__device__ __forceinline__ void warp_best(long long& key, int& id) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const long long other = __shfl_down_sync(0xffffffffu, key, offset);
        const int other_id = __shfl_down_sync(0xffffffffu, id, offset);
        if (better_key(other, other_id, key, id)) {
            key = other;
            id = other_id;
        }
    }
}

__global__ void select_top6(const double* __restrict__ scores,
                            const float* __restrict__ bias,
                            int32_t* __restrict__ ids,
                            float* __restrict__ weights) {
    constexpr int kPerThread = kExperts / kWarp;
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    // Twelve register-resident candidates per lane, with the original expert
    // IDs retained for ties. Each token is an independent, full-warp CTA.
    // The comparisons use order_key(score + bias): the same order as k8_detail::better.
    double raw[kPerThread];
    long long keys[kPerThread];
    int expert_ids[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int id = lane + j * kWarp;
        raw[j] = scores[token * kExperts + id];
        keys[j] = order_key(raw[j] + double(bias[id]));
        expert_ids[j] = id;
    }
    const long long none = order_key(-INFINITY);
    double selected = 0.0;
    double sum = 0.0;
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        long long key = none;
        int id = kExperts;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            if (better_key(keys[j], expert_ids[j], key, id)) {
                key = keys[j];
                id = expert_ids[j];
            }
        }
        warp_best(key, id);
        const int chosen = __shfl_sync(0xffffffffu, id, 0);
        double unbiased = 0.0;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            if (expert_ids[j] == chosen) {
                unbiased = raw[j];
                keys[j] = none;
                expert_ids[j] = kExperts;
            }
        }
        // Broadcast the owner's original score, including zero. Subtracting
        // the bias from a rounded comparison value would change normalization.
        unbiased = __shfl_sync(0xffffffffu, unbiased, chosen & (kWarp - 1));
        if (lane == 0) {
            ids[token * kTopK + i] = chosen;
            // Exactly the reference's left-to-right selected-score sum.
            sum += unbiased;
        }
        if (lane == i) selected = unbiased;
    }
    sum = __shfl_sync(0xffffffffu, sum, 0);
    if (lane < kTopK) {
        weights[token * kTopK + lane] = float(selected / (sum + 1e-20) * double(kRouteScale));
    }
}

template <int Tokens>
void launch_tile(const __nv_bfloat16* x, const __nv_bfloat16* w, double* scores, cudaStream_t stream) {
    constexpr int threads = Tokens <= 4 ? 128 : 256;
    tile_scores<Tokens><<<kExperts, threads, 0, stream>>>(x, w, scores);
}

}  // namespace

void router_init() { (void) workspace(); }

void router_topk(const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias,
                 int32_t* ids, float* weights, cudaStream_t stream) {
    if (m < 1 || m > kMaxTokens) return;
    double* scores = workspace();
    switch (m) {
        case 1: decode_scores<<<kExperts / kDecodeWarps, kDecodeWarps * kWarp, 0, stream>>>(x, w, scores); break;
        case 2: launch_tile<2>(x, w, scores, stream); break;
        case 3: launch_tile<3>(x, w, scores, stream); break;
        case 4: launch_tile<4>(x, w, scores, stream); break;
        case 5: launch_tile<5>(x, w, scores, stream); break;
        case 6: launch_tile<6>(x, w, scores, stream); break;
        case 7: launch_tile<7>(x, w, scores, stream); break;
        case 8: launch_tile<8>(x, w, scores, stream); break;
    }
    select_top6<<<m, kSelectThreads, 0, stream>>>(scores, bias, ids, weights);
    check(cudaGetLastError(), "launch");
}

}  // namespace strata::ds41::kernels
