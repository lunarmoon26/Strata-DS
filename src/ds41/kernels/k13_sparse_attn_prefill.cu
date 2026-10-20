// K13-02: bounded, staged BF16 tensor-core attention for prefill.
// QK, softmax and PV are separate batched kernels. No CTA retains the full
// score matrix; small operand stages leave shared memory available for residency.
#include "strata/ds41/kernels/k13_sparse_attn_prefill.hpp"

#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace strata::ds41::kernels {
namespace {

using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;
constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kMaxIndices = 1024;
constexpr int kQueryChunk = 256;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kQKRows = 64;
constexpr int kQKDepth = 64;
constexpr int kQKStride = kQKDepth + 8;
constexpr int kPVDim = 64;
constexpr int kPVDepth = 32;
constexpr int kPStride = kPVDepth + 8;
constexpr int kVStride = kPVDim + 8;
constexpr unsigned kFullWarp = 0xffffffffu;

struct __align__(32) QKStorage {
    bf16 q[kHeads * kQKStride];
    bf16 k[kQKRows * kQKStride];
    int indices[kQKRows];
};
struct __align__(32) PVOperands {
    bf16 p[kHeads * kPStride];
    bf16 v[kPVDepth * kVStride];
};
union __align__(32) PVStorage {
    PVOperands operands;
    float output[kWarps][16 * 16];
};
static_assert(sizeof(QKStorage) == 18688, "QK shared layout changed");
static_assert(sizeof(PVStorage) == 9728, "PV shared layout changed");
static_assert(sizeof(QKStorage) <= 99 * 1024 && sizeof(PVStorage) <= 99 * 1024,
              "K13 shared-memory limit exceeded");

// One CTA computes all 64 heads by 64 listed KV rows for one query. The
// KV tile is shared by every head, with arbitrary order, duplicates and -1
// padding permitted. Both tensor-core inputs are BF16; accumulators are FP32.
__global__ __launch_bounds__(kThreads, 4) void qk_stage(
        const bf16* __restrict__ q, const bf16* __restrict__ kv,
        const int32_t* __restrict__ idx, int n_idx, float scale,
        float* __restrict__ scores) {
    __shared__ QKStorage shared;
    const int query = blockIdx.y;
    const int first = blockIdx.x * kQKRows;
    const int warp = threadIdx.x / 32;
    const int head = (warp / 4) * 32;
    const int row = (warp % 4) * 16;
    if (threadIdx.x < kQKRows) {
        const int t = first + threadIdx.x;
        shared.indices[threadIdx.x] = t < n_idx ? idx[static_cast<size_t>(query) * n_idx + t] : -1;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[2];
    wmma::fill_fragment(accum[0], 0.0f);
    wmma::fill_fragment(accum[1], 0.0f);
    __syncthreads();
    for (int dim = 0; dim < kDim; dim += kQKDepth) {
#pragma unroll
        for (int i = threadIdx.x; i < kHeads * kQKDepth / 8; i += kThreads) {
            const int h = i / (kQKDepth / 8);
            const int d = (i % (kQKDepth / 8)) * 8;
            *reinterpret_cast<uint4*>(shared.q + h * kQKStride + d) =
                *reinterpret_cast<const uint4*>(q + (static_cast<size_t>(query) * kHeads + h) * kDim + dim + d);
        }
#pragma unroll
        for (int i = threadIdx.x; i < kQKRows * kQKDepth / 8; i += kThreads) {
            const int r = i / (kQKDepth / 8);
            const int d = (i % (kQKDepth / 8)) * 8;
            const int j = shared.indices[r];
            uint4 value = make_uint4(0, 0, 0, 0);
            if (j >= 0)
                value = *reinterpret_cast<const uint4*>(kv + static_cast<size_t>(j) * kDim + dim + d);
            *reinterpret_cast<uint4*>(shared.k + r * kQKStride + d) = value;
        }
        __syncthreads();
#pragma unroll
        for (int d = 0; d < kQKDepth; d += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
            wmma::load_matrix_sync(b, shared.k + row * kQKStride + d, kQKStride);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::load_matrix_sync(a, shared.q + (head + h * 16) * kQKStride + d, kQKStride);
                wmma::mma_sync(accum[h], a, b, accum[h]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int i = 0; i < accum[h].num_elements; ++i) accum[h].x[i] *= scale;
        // The fixed 1024-column scratch pitch safely contains a final partial
        // 64-row tile. Softmax masks padding using the actual index list.
        float* dst = scores + (static_cast<size_t>(query) * kHeads + head + h * 16) * kMaxIndices + first + row;
        wmma::store_matrix_sync(dst, accum[h], kMaxIndices, wmma::mem_row_major);
    }
}

// One warp per head, eight independent heads per CTA. Keep the unrounded
// exponential sum in FP32, round only the PV multiplicands, and do not include
// the sink in the maximum or numerator: this is exactly ops::sparse_attn's math.
__global__ __launch_bounds__(kThreads) void softmax_stage(
        const float* __restrict__ scores, const int32_t* __restrict__ idx,
        int n_idx, const float* __restrict__ sink, bf16* __restrict__ probabilities,
        float* __restrict__ denominators) {
    const int query = blockIdx.y;
    const int head = blockIdx.x * kWarps + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const size_t row = static_cast<size_t>(query) * kHeads + head;
    float values[kMaxIndices / 32];
    float maximum = -1.0e30f;
#pragma unroll
    for (int i = 0; i < kMaxIndices / 32; ++i) {
        const int t = lane + i * 32;
        const bool valid = t < n_idx && idx[static_cast<size_t>(query) * n_idx + t] >= 0;
        values[i] = valid ? scores[row * kMaxIndices + t] : -INFINITY;
        maximum = fmaxf(maximum, values[i]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(kFullWarp, maximum, offset));
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxIndices / 32; ++i) {
        const float p = values[i] == -INFINITY ? 0.0f : expf(values[i] - maximum);
        sum += p;
        probabilities[row * kMaxIndices + lane + i * 32] = __float2bfloat16_rn(p);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(kFullWarp, sum, offset);
    if (lane == 0) denominators[row] = sum + expf(sink[head] - maximum);
}

// One CTA computes 64 heads by 64 output dimensions. A 32-row operand stage
// is reused for the whole reduction. Output staging is warp-private, so only
// an 8 KiB overlay is needed rather than a full FP32 64x64 shared matrix.
__global__ __launch_bounds__(kThreads, 3) void pv_stage(
        const bf16* __restrict__ probabilities, const float* __restrict__ denominators,
        const bf16* __restrict__ kv, const int32_t* __restrict__ idx, int n_idx,
        bf16* __restrict__ output) {
    __shared__ PVStorage shared;
    const int query = blockIdx.y;
    const int first_dim = blockIdx.x * kPVDim;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int head = (warp / 4) * 32;
    const int dim = (warp % 4) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) wmma::fill_fragment(accum[h], 0.0f);
    for (int first = 0; first < n_idx; first += kPVDepth) {
#pragma unroll
        for (int i = threadIdx.x; i < kHeads * kPVDepth / 8; i += kThreads) {
            const int h = i / (kPVDepth / 8);
            const int t = (i % (kPVDepth / 8)) * 8;
            *reinterpret_cast<uint4*>(shared.operands.p + h * kPStride + t) =
                *reinterpret_cast<const uint4*>(probabilities + (static_cast<size_t>(query) * kHeads + h) * kMaxIndices + first + t);
        }
#pragma unroll
        for (int i = threadIdx.x; i < kPVDepth * kPVDim / 8; i += kThreads) {
            const int t = i / (kPVDim / 8);
            const int d = (i % (kPVDim / 8)) * 8;
            const int j = first + t < n_idx ? idx[static_cast<size_t>(query) * n_idx + first + t] : -1;
            uint4 value = make_uint4(0, 0, 0, 0);
            if (j >= 0)
                value = *reinterpret_cast<const uint4*>(kv + static_cast<size_t>(j) * kDim + first_dim + d);
            *reinterpret_cast<uint4*>(shared.operands.v + t * kVStride + d) = value;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < kPVDepth; t += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
            wmma::load_matrix_sync(b, shared.operands.v + t * kVStride + dim, kVStride);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::load_matrix_sync(a, shared.operands.p + (head + h * 16) * kPStride + t, kPStride);
                wmma::mma_sync(accum[h], a, b, accum[h]);
            }
        }
        __syncthreads();
    }
    // The last CTA barrier above retires all operand readers before the
    // output overlay is used. With n_idx=0 there were no operand readers.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        wmma::store_matrix_sync(shared.output[warp], accum[h], 16, wmma::mem_row_major);
        __syncwarp(kFullWarp);
#pragma unroll
        for (int i = lane; i < 16 * 16; i += 32) {
            const int out_head = head + h * 16 + i / 16;
            const int out_dim = first_dim + dim + i % 16;
            const size_t out_row = static_cast<size_t>(query) * kHeads + out_head;
            output[out_row * kDim + out_dim] =
                __float2bfloat16_rn(shared.output[warp][i] / denominators[out_row]);
        }
        __syncwarp(kFullWarp);
    }
}

struct Scratch {
    float* scores;
    bf16* probabilities;
    float* denominators;
};

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "K13-02 %s: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

Scratch scratch_for_device() {
    // One fixed-size allocation per device for this task, retained for the
    // process lifetime. Chunking covers the full m<=16384 interface without
    // allocating for a specific test shape or growing inside a captured call.
    static std::mutex mutex;
    static std::map<int, Scratch> scratch;
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = scratch.find(device);
    if (found != scratch.end()) return found->second;
    constexpr size_t elements = static_cast<size_t>(kQueryChunk) * kHeads * kMaxIndices;
    constexpr size_t score_bytes = elements * sizeof(float);
    constexpr size_t probability_bytes = elements * sizeof(bf16);
    constexpr size_t denominator_bytes = static_cast<size_t>(kQueryChunk) * kHeads * sizeof(float);
    void* allocation = nullptr;
    cuda_check(cudaMalloc(&allocation, score_bytes + probability_bytes + denominator_bytes), "first-call scratch");
    auto* bytes = static_cast<unsigned char*>(allocation);
    Scratch result{reinterpret_cast<float*>(bytes), reinterpret_cast<bf16*>(bytes + score_bytes),
                   reinterpret_cast<float*>(bytes + score_bytes + probability_bytes)};
    scratch.emplace(device, result);
    return result;
}

}  // namespace

void sparse_attn_prefill(const bf16* q, const bf16* kv, const int32_t* idx, int m, int n_idx,
                         const float* sink, float scale, bf16* o, cudaStream_t stream) {
    if (m <= 0) return;
    const Scratch scratch = scratch_for_device();
    for (int first = 0; first < m; first += kQueryChunk) {
        const int count = m - first < kQueryChunk ? m - first : kQueryChunk;
        const size_t query_offset = static_cast<size_t>(first) * kHeads * kDim;
        const int32_t* indices = idx + static_cast<size_t>(first) * n_idx;
        if (n_idx > 0)
            qk_stage<<<dim3((n_idx + kQKRows - 1) / kQKRows, count), kThreads, 0, stream>>>(
                q + query_offset, kv, indices, n_idx, scale, scratch.scores);
        softmax_stage<<<dim3(kHeads / kWarps, count), kThreads, 0, stream>>>(
            scratch.scores, indices, n_idx, sink, scratch.probabilities, scratch.denominators);
        pv_stage<<<dim3(kDim / kPVDim, count), kThreads, 0, stream>>>(
            scratch.probabilities, scratch.denominators, kv, indices, n_idx, o + query_offset);
    }
}

}  // namespace strata::ds41::kernels
