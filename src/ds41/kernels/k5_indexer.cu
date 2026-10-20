// K5-14: K5-11 tensor scores and caller-storage radix with warp-private histograms.
// The same pipeline is used eagerly and in CUDA graphs; no internal global scratch exists.
#include "strata/ds41/kernels/k5_indexer.hpp"

#include <mma.h>
#include <math_constants.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
constexpr int kThreads = 256;
constexpr int kRadixItems = 4096;
constexpr int kMetadataHeader = 8;
constexpr int kSelectItems = 1024;
constexpr unsigned kFullWarp = 0xffffffffu;
using Count = unsigned long long;

void check(cudaError_t e, const char* operation) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "K5 %s: %s\n", operation, cudaGetErrorString(e));
        std::abort();
    }
}
__device__ __forceinline__ float rounded(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ int64_t live_length(const int* pos, int ratio, int64_t cap) {
    const int64_t t = (int64_t(*pos) + 1) / ratio;
    assert(t >= 0 && t <= cap);
    return t;
}

// Graph selection: the single-CTA radix below this many keys, the partitioned radix from here.
// RTX 5090 Laptop, scores included: 4096 keys 26 us single vs 32 us partitioned; 262144 keys 890 vs 165 us.
constexpr int64_t kDeviceParallelFrom = 8192;

// Length of a partitioned selection. pos == nullptr: the host length n, always run. Otherwise n is the capacity;
// the live length replaces it, and the selection runs only where the single-CTA kernel yields to it.
struct Live {
    const int* pos = nullptr;
    int ratio = 1;
    int k = 0;
};
__device__ __forceinline__ bool partitioned(Live live, int64_t& n) {
    if (!live.pos) return true;
    n = live_length(live.pos, live.ratio, n);
    return n >= kDeviceParallelFrom && live.k > 0 && live.k < n;
}
__host__ __device__ __forceinline__ int64_t partitions_of(int64_t n) { return (n - 1) / kRadixItems + 1; }
__host__ __device__ __forceinline__ int histogram_ctas(int64_t n) {
    return int(partitions_of(n) < 256 ? partitions_of(n) : 256);
}

__global__ void small_scores(const bf16* q, const bf16* keys, int64_t n, const bf16* w,
                             const uint8_t* cand, float* scores, const int* pos = nullptr, int ratio = 1) {
    if (pos) {
        n = live_length(pos, ratio, n);
        if (n > 512) return;
    }
    const int64_t j = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (j >= n) return;
    if (cand && !cand[j]) {
        if (lane == 0) scores[j] = -CUDART_INF_F;
        return;
    }
    float total = 0.0f;
    for (int h = 0; h < 32; ++h) {
        float dot = 0.0f;
        for (int d = lane; d < 128; d += 32)
            dot += __bfloat162float(q[h * 128 + d]) * __bfloat162float(keys[j * 128 + d]);
        for (int delta = 16; delta; delta >>= 1)
            dot += __shfl_xor_sync(kFullWarp, dot, delta);
        total += rounded(fmaxf(rounded(dot), 0.0f) * __bfloat162float(w[h]));
    }
    if (lane == 0) scores[j] = rounded(total);
}

// Float-flip maps finite values and infinities to unsigned ascending order.
// Signed zero compares equal under the public contract, so canonicalize it.
// BF16 scores need only two radix bytes. Candidate block inputs retain all four.
template <bool Bf16Scores>
__device__ __forceinline__ uint32_t key_of(float value) {
    uint32_t bits = value == 0.0f ? 0u : __float_as_uint(value);
    uint32_t key = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    if constexpr (Bf16Scores) key >>= 16;
    return key;
}

struct Selection {
    uint32_t prefix;
    Count remaining;  // Number of ties still needed within this prefix.
};

// Each warp computes 32 heads x 16 positions. The matrix intermediates are
// FP32; the three BF16 conversions remain separate from tensor accumulation.
__global__ void tensor_scores(const bf16* q, const bf16* keys, int64_t n, const bf16* w,
                              const uint8_t* cand, float* scores, const int* pos = nullptr, int ratio = 1) {
    if (pos) {
        n = live_length(pos, ratio, n);
        if (n <= 512 || int64_t(blockIdx.x) * 64 >= n) return;
    }
    __shared__ __align__(32) bf16 sq[32 * 128];
    __shared__ __align__(32) bf16 sk[64 * 128];
    __shared__ __align__(32) float dots[32 * 64];
    __shared__ float weights[32];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int64_t base = int64_t(blockIdx.x) * 64;
    for (int i = tid; i < 32 * 128; i += 128) sq[i] = q[i];
    for (int i = tid; i < 64 * 128; i += 128) {
        const int64_t j = base + i / 128;
        sk[i] = j < n ? keys[j * 128 + i % 128] : __float2bfloat16_rn(0.0f);
    }
    if (tid < 32) weights[tid] = __bfloat162float(w[tid]);
    __syncthreads();

    namespace wm = nvcuda::wmma;
    wm::fragment<wm::matrix_a, 16, 16, 16, bf16, wm::row_major> a0, a1;
    wm::fragment<wm::matrix_b, 16, 16, 16, bf16, wm::col_major> b;
    wm::fragment<wm::accumulator, 16, 16, 16, float> c0, c1;
    wm::fill_fragment(c0, 0.0f);
    wm::fill_fragment(c1, 0.0f);
#pragma unroll
    for (int d = 0; d < 128; d += 16) {
        wm::load_matrix_sync(a0, sq + d, 128);
        wm::load_matrix_sync(a1, sq + 16 * 128 + d, 128);
        wm::load_matrix_sync(b, sk + warp * 16 * 128 + d, 128);
        wm::mma_sync(c0, a0, b, c0);
        wm::mma_sync(c1, a1, b, c1);
    }
    wm::store_matrix_sync(dots + warp * 16, c0, 64, wm::mem_row_major);
    wm::store_matrix_sync(dots + 16 * 64 + warp * 16, c1, 64, wm::mem_row_major);
    __syncthreads();
    if (tid < 64 && base + tid < n) {
        if (cand && !cand[base + tid]) {
            scores[base + tid] = -CUDART_INF_F;
        } else {
            float total = 0.0f;
#pragma unroll
            for (int h = 0; h < 32; ++h)
                total += rounded(fmaxf(rounded(dots[h * 64 + tid]), 0.0f) * weights[h]);
            scores[base + tid] = rounded(total);
        }
    }
}

// Candidate block maxima are computed on demand. This removes an unbounded
// temporary allocation, and preserves every FP32 bit in candidate selection.
template <bool Candidate>
struct Values {
    const float* scores;
    int64_t positions;
    int block_size;
    __device__ __forceinline__ float get(int64_t j) const {
        if constexpr (!Candidate) {
            return scores[j];
        } else {
            if (j == (positions - 1) / block_size) return CUDART_INF_F;
            float value = -CUDART_INF_F;
            const int64_t begin = j * block_size;
            // the decode block (8): two 16-byte loads instead of eight dependent ones; only the last block is partial
            if (block_size == 8 && (reinterpret_cast<uintptr_t>(scores) & 15) == 0) {
                const float4 a = *reinterpret_cast<const float4*>(scores + begin);
                const float4 b = *reinterpret_cast<const float4*>(scores + begin + 4);
                return fmaxf(value, fmaxf(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)), fmaxf(fmaxf(b.x, b.y), fmaxf(b.z, b.w))));
            }
            for (int d = 0; d < block_size && begin + d < positions; ++d)
                value = fmaxf(value, scores[begin + d]);
            return value;
        }
    }
};

__global__ void all_positions(int64_t n, int32_t offset, int32_t* out) {
    const int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < n) out[j] = int32_t(j) + offset;
}

__global__ void all_candidates(const float* scores, int64_t n, int block_size, uint8_t* cand) {
    const int64_t blocks = (n - 1) / block_size + 1;
    for (int64_t b = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; b < blocks;
         b += int64_t(gridDim.x) * blockDim.x) {
        float best = -CUDART_INF_F;
        for (int d = 0; d < block_size && b * block_size + d < n; ++d)
            best = fmaxf(best, scores[b * block_size + d]);
        const uint8_t keep = best != -CUDART_INF_F || b == (n - 1) / block_size;
        for (int d = 0; d < block_size && b * block_size + d < n; ++d) cand[b * block_size + d] = keep;
    }
}

// A single CTA owns the complete cutoff and ordered emission. Histograms and
// scan state live only in shared memory, so every invocation and every graph
// replay has independent storage. There are no host-side lifetime assumptions.
template <bool Candidate>
__global__ void select_without_scratch(Values<Candidate> values, int64_t n,
                                       int k, int32_t offset, int32_t* out,
                                       uint8_t* cand, const int* pos = nullptr, int ratio = 1,
                                       bool yield = false) {
    if (pos) {
        values.positions = live_length(pos, ratio, values.positions);
        if (values.positions == 0) return;
        n = Candidate ? (values.positions - 1) / values.block_size + 1 : values.positions;
        if (yield && n >= kDeviceParallelFrom && k > 0 && k < n) return;   // the partitioned selection runs
        k = int(n < k ? n : k);
        if (k <= 0) {
            if constexpr (Candidate) {
                for (int64_t j = threadIdx.x; j < values.positions; j += blockDim.x) cand[j] = 0;
            }
            return;
        }
    }
    __shared__ Count bins[256];
    __shared__ Selection state;
    __shared__ uint32_t greater_prefix[32], equal_prefix[32];
    __shared__ uint32_t batch_greater, batch_equal;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t lane_mask = (uint32_t(1) << lane) - 1;
    uint32_t prefix = 0, prefix_mask = 0;
    constexpr int first_shift = Candidate ? 24 : 8;
    for (int shift = first_shift; shift >= 0; shift -= 8) {
        bins[tid] = 0;
        __syncthreads();
        for (int64_t base = 0; base < n; base += kThreads) {
            const int64_t j = base + tid;
            const uint32_t key = j < n ? key_of<!Candidate>(values.get(j)) : 0;
            const bool valid = j < n && (key & prefix_mask) == prefix;
            const unsigned active = __ballot_sync(kFullWarp, valid);
            if (valid) {
                const uint32_t bin = (key >> shift) & 255u;
                const unsigned peers = __match_any_sync(active, bin);
                if (lane == __ffs(peers) - 1) atomicAdd(bins + bin, Count(__popc(peers)));
            }
        }
        __syncthreads();
        if (tid == 0) {
            Count remaining = shift == first_shift ? Count(k) : state.remaining;
            for (int bin = 255; bin >= 0; --bin) {
                if (bins[bin] >= remaining) {
                    state.prefix = prefix | (uint32_t(bin) << shift);
                    state.remaining = remaining;
                    break;
                }
                remaining -= bins[bin];
            }
        }
        __syncthreads();
        prefix = state.prefix;
        prefix_mask |= 255u << shift;
    }
    const Count quota = state.remaining;
    Count greater_before = 0, equal_before = 0;
    // Four consecutive 256-position parts reduce barrier overhead. The first
    // warp scans all 32 (part, warp) totals in positional order. Its exclusive
    // prefixes then give both the tie quota and the ascending output address.
    for (int64_t base = 0; base < n; base += kSelectItems) {
        unsigned greater_lanes[4], equal_lanes[4];
        uint32_t keys[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int64_t j = base + part * kThreads + tid;
            const uint32_t key = j < n ? key_of<!Candidate>(values.get(j)) : 0;
            keys[part] = key;
            greater_lanes[part] = __ballot_sync(kFullWarp, j < n && key > prefix);
            equal_lanes[part] = __ballot_sync(kFullWarp, j < n && key == prefix);
            if (lane == 0) {
                greater_prefix[part * 8 + warp] = __popc(greater_lanes[part]);
                equal_prefix[part * 8 + warp] = __popc(equal_lanes[part]);
            }
        }
        __syncthreads();
        if (warp == 0) {
            const uint32_t own_greater = greater_prefix[lane];
            const uint32_t own_equal = equal_prefix[lane];
            uint32_t greater = own_greater, equal = own_equal;
#pragma unroll
            for (int delta = 1; delta < 32; delta <<= 1) {
                const uint32_t previous_greater = __shfl_up_sync(kFullWarp, greater, delta);
                const uint32_t previous_equal = __shfl_up_sync(kFullWarp, equal, delta);
                if (lane >= delta) { greater += previous_greater; equal += previous_equal; }
            }
            greater_prefix[lane] = greater - own_greater;
            equal_prefix[lane] = equal - own_equal;
            if (lane == 31) { batch_greater = greater; batch_equal = equal; }
        }
        __syncthreads();
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int64_t j = base + part * kThreads + tid;
            const Count preceding_greater = greater_before + greater_prefix[part * 8 + warp]
                + __popc(greater_lanes[part] & lane_mask);
            const Count preceding_equal = equal_before + equal_prefix[part * 8 + warp]
                + __popc(equal_lanes[part] & lane_mask);
            const bool selected = j < n && (keys[part] > prefix ||
                (keys[part] == prefix && preceding_equal < quota));
            if constexpr (Candidate) {
                if (j < n) {
                    const uint8_t keep = selected && keys[part] != key_of<false>(-CUDART_INF_F);
                    const int64_t begin = j * values.block_size;
                    for (int d = 0; d < values.block_size && begin + d < values.positions; ++d)
                        cand[begin + d] = keep;
                }
            } else if (selected) {
                out[preceding_greater + (preceding_equal < quota ? preceding_equal : quota)] = int32_t(j) + offset;
            }
        }
        greater_before += batch_greater;
        equal_before += batch_equal;
        __syncthreads();
    }
}

// Every produced score is an exactly representable BF16 float, hence its low
// halfword is zero. Between score production and restoration ONLY these PTX
// halfword accesses touch scores: high halves are read-only values, low halves
// hold temporary metadata. No overlapping 32-bit access occurs in this phase.
// Separate same-stream kernel launches order metadata reuse and restoration.
__device__ __forceinline__ uint16_t load_word(const float* scores, int64_t word) {
    uint16_t value;
    const char* address = reinterpret_cast<const char*>(scores) + word * 2;
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(value) : "l"(address) : "memory");
    return value;
}
__device__ __forceinline__ void store_low(float* scores, int64_t index, uint16_t value) {
    char* address = reinterpret_cast<char*>(scores) + index * 4;
    asm volatile("st.global.u16 [%0], %1;" :: "l"(address), "h"(value) : "memory");
}
__device__ __forceinline__ uint32_t high_key(const float* scores, int64_t j) {
    uint32_t bits = load_word(scores, 2 * j + 1);
    if ((bits & 0x7fffu) == 0) bits = 0;
    return bits ^ ((bits & 0x8000u) ? 0xffffu : 0x8000u);
}
__device__ __forceinline__ Count load_count(const float* scores, int64_t index) {
    Count result = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r)
        result |= Count(load_word(scores, 2 * (index + r))) << (16 * r);
    return result;
}
__device__ __forceinline__ void store_count(float* scores, int64_t index, Count value) {
#pragma unroll
    for (int r = 0; r < 4; ++r) store_low(scores, index + r, uint16_t(value >> (16 * r)));
}

// Four low halfwords hold each 64-bit count. With P<=ceil(n/4096),
// 8+1024*P<=n for every n>=4096, including a partial final partition.
// Each bounded round visits at most 16 batches of 256 positions per CTA.
// A warp therefore contributes at most 16*32=512 to any uint32_t local bin,
// regardless of n or the 256-CTA launch cap. Each bin's owner converts all
// eight local counts to Count before accumulating into its 64-bit total.
// The scorer, two radix bytes, global count layout, and launch count are unchanged.
template<bool First>
__global__ void parallel_histogram(float* scores, int64_t n, Live live) {
    if (!partitioned(live, n)) return;
    // each CTA owns metadata slots [8+1024*b, 8+1024*(b+1)): exactly histogram_ctas(n) CTAs take part
    const int ctas = live.pos ? histogram_ctas(n) : int(gridDim.x);
    if (int(blockIdx.x) >= ctas) return;
    __shared__ uint32_t warp_bins[kThreads / 32][256];
    constexpr int batches_per_round = kRadixItems / kThreads;
    static_assert(batches_per_round * 32 == 512, "warp-local count bound");
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t prefix = First ? 0u : load_word(scores, 0);
    const int64_t stride = int64_t(ctas) * kThreads;
    Count total = 0;
    for (int64_t begin = int64_t(blockIdx.x) * kThreads; begin < n;
         begin += stride * batches_per_round) {
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) warp_bins[w][tid] = 0;
        __syncthreads();
        for (int batch = 0; batch < batches_per_round; ++batch) {
            const int64_t base = begin + int64_t(batch) * stride;
            if (base >= n) break;  // Uniform for the complete CTA.
            const int64_t j = base + tid;
            const uint32_t key = j < n ? high_key(scores, j) : 0;
            const bool valid = j < n && (First || (key & 0xff00u) == prefix);
            const unsigned active = __ballot_sync(kFullWarp, valid);
            if (valid) {
                const uint32_t bin = First ? key >> 8 : key & 255u;
                const unsigned peers = __match_any_sync(active, bin);
                if (lane == __ffs(peers) - 1)
                    atomicAdd(&warp_bins[warp][bin], uint32_t(__popc(peers)));
            }
        }
        __syncthreads();
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) total += Count(warp_bins[w][tid]);
        // All eight readers finish before any thread resets the next round.
        // No extra barrier is needed after the final round's read-only reduction.
        if (begin + stride * batches_per_round < n) __syncthreads();
    }
    store_count(scores, kMetadataHeader + (int64_t(blockIdx.x) * 256 + tid) * 4, total);
}

template<bool First>
__global__ void choose_parallel_byte(float* scores, int parts, int k, int64_t n, Live live) {
    if (!partitioned(live, n)) return;
    if (live.pos) parts = histogram_ctas(n);
    __shared__ Count bins[256];
    Count sum = 0;
    for (int p = 0; p < parts; ++p)
        sum += load_count(scores, kMetadataHeader + (int64_t(p) * 256 + threadIdx.x) * 4);
    bins[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        Count remaining = First ? Count(k) : load_count(scores, 1);
        uint32_t prefix = First ? 0u : load_word(scores, 0);
        for (int b = 255; b >= 0; --b) {
            if (bins[b] >= remaining) {
                prefix |= uint32_t(b) << (First ? 8 : 0);
                store_low(scores, 0, uint16_t(prefix));
                store_count(scores, 1, remaining);
                break;
            }
            remaining -= bins[b];
        }
    }
}

struct Pair { Count greater, equal; };
__device__ __forceinline__ Pair scan_pair(Pair own, Pair* warps, Pair& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    Pair running = own;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const Count g = __shfl_up_sync(kFullWarp, running.greater, d);
        const Count e = __shfl_up_sync(kFullWarp, running.equal, d);
        if (lane >= d) { running.greater += g; running.equal += e; }
    }
    if (lane == 31) warps[warp] = running;
    __syncthreads();
    if (warp == 0) {
        Pair p = lane < 8 ? warps[lane] : Pair{0, 0};
#pragma unroll
        for (int d = 1; d < 8; d <<= 1) {
            const Count g = __shfl_up_sync(kFullWarp, p.greater, d);
            const Count e = __shfl_up_sync(kFullWarp, p.equal, d);
            if (lane >= d) { p.greater += g; p.equal += e; }
        }
        if (lane < 8) warps[lane] = p;
    }
    __syncthreads();
    const Pair before = warp ? warps[warp - 1] : Pair{0, 0};
    total = warps[7];
    const Pair prefix{before.greater + running.greater - own.greater,
                      before.equal + running.equal - own.equal};
    __syncthreads();
    return prefix;
}

__global__ void count_partitions(float* scores, int64_t n, Live live) {
    if (!partitioned(live, n)) return;
    __shared__ Pair warps[8];
    const uint32_t cutoff = load_word(scores, 0);
    const int64_t partitions = (n - 1) / kRadixItems + 1;
    for (int64_t partition = blockIdx.x; partition < partitions; partition += gridDim.x) {
        Pair count{0, 0};
#pragma unroll
        for (int part = 0; part < kRadixItems / kThreads; ++part) {
            const int64_t j = partition * kRadixItems + part * kThreads + threadIdx.x;
            if (j < n) {
                const uint32_t key = high_key(scores, j);
                count.greater += key > cutoff;
                count.equal += key == cutoff;
            }
        }
        Pair total;
        scan_pair(count, warps, total);
        if (threadIdx.x == 0) {
            const int64_t metadata = kMetadataHeader + partition * 8;
            store_count(scores, metadata, total.greater);
            store_count(scores, metadata + 4, total.equal);
        }
    }
}

__global__ void prefix_partitions(float* scores, int64_t parts, int64_t n, Live live) {
    if (!partitioned(live, n)) return;
    if (live.pos) parts = partitions_of(n);
    __shared__ Pair warps[8];
    Pair carry{0, 0};
    for (int64_t base = 0; base < parts; base += kThreads) {
        const int64_t part = base + threadIdx.x;
        const int64_t metadata = kMetadataHeader + part * 8;
        const Pair own = part < parts ? Pair{load_count(scores, metadata), load_count(scores, metadata + 4)}
                                      : Pair{0, 0};
        Pair total;
        const Pair prefix = scan_pair(own, warps, total);
        if (part < parts) {
            store_count(scores, metadata, carry.greater + prefix.greater);
            store_count(scores, metadata + 4, carry.equal + prefix.equal);
        }
        carry.greater += total.greater;
        carry.equal += total.equal;
        __syncthreads();
    }
}

__global__ void emit_partitions(const float* scores, int64_t n, int32_t offset, int32_t* out, Live live) {
    if (!partitioned(live, n)) return;
    __shared__ uint32_t prefix_g[32], prefix_e[32], batch_g, batch_e;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t lane_mask = (uint32_t(1) << lane) - 1;
    const uint32_t cutoff = load_word(scores, 0);
    const Count quota = load_count(scores, 1);
    const int64_t partitions = (n - 1) / kRadixItems + 1;
    for (int64_t partition = blockIdx.x; partition < partitions; partition += gridDim.x) {
        const int64_t metadata = kMetadataHeader + partition * 8;
        Count preceding_g = load_count(scores, metadata);
        Count preceding_e = load_count(scores, metadata + 4);
#pragma unroll
        for (int batch = 0; batch < kRadixItems / kSelectItems; ++batch) {
            const int64_t base = partition * kRadixItems + batch * kSelectItems;
            unsigned gm[4], em[4];
            uint32_t keys[4];
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int64_t j = base + part * kThreads + tid;
                keys[part] = j < n ? high_key(scores, j) : 0;
                gm[part] = __ballot_sync(kFullWarp, j < n && keys[part] > cutoff);
                em[part] = __ballot_sync(kFullWarp, j < n && keys[part] == cutoff);
                if (lane == 0) {
                    prefix_g[part * 8 + warp] = __popc(gm[part]);
                    prefix_e[part * 8 + warp] = __popc(em[part]);
                }
            }
            __syncthreads();
            if (warp == 0) {
                const uint32_t own_g = prefix_g[lane], own_e = prefix_e[lane];
                uint32_t g = own_g, e = own_e;
#pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    const uint32_t pg = __shfl_up_sync(kFullWarp, g, d);
                    const uint32_t pe = __shfl_up_sync(kFullWarp, e, d);
                    if (lane >= d) { g += pg; e += pe; }
                }
                prefix_g[lane] = g - own_g;
                prefix_e[lane] = e - own_e;
                if (lane == 31) { batch_g = g; batch_e = e; }
            }
            __syncthreads();
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int64_t j = base + part * kThreads + tid;
                const Count g = preceding_g + prefix_g[part * 8 + warp] + __popc(gm[part] & lane_mask);
                const Count e = preceding_e + prefix_e[part * 8 + warp] + __popc(em[part] & lane_mask);
                if (j < n && (keys[part] > cutoff || (keys[part] == cutoff && e < quota)))
                    out[g + (e < quota ? e : quota)] = int32_t(j) + offset;
            }
            preceding_g += batch_g;
            preceding_e += batch_e;
            __syncthreads();
        }
    }
}

// No selector accesses scores after this separate launch starts. Restore only
// the written metadata span; all remaining low halves were never touched.
__host__ __device__ __forceinline__ int64_t metadata_words(int64_t n) {
    const int64_t histograms = 1024 * int64_t(histogram_ctas(n)), records = 8 * partitions_of(n);
    return kMetadataHeader + (histograms > records ? histograms : records);
}
__global__ void restore_scores(float* scores, int64_t n, Live live) {
    if (!partitioned(live, n)) return;
    const int64_t words = metadata_words(n);
    for (int64_t j = int64_t(blockIdx.x) * kThreads + threadIdx.x; j < words;
         j += int64_t(gridDim.x) * kThreads) store_low(scores, j, 0);
}

// n: the length (live.pos == nullptr) or the capacity, which sizes the grids (graph: the live length decides)
void parallel_select(float* scores, int64_t n, int k, int32_t offset,
                     int32_t* out, cudaStream_t stream, Live live = {}) {
    const int hist_parts = histogram_ctas(n);
    parallel_histogram<true><<<hist_parts, kThreads, 0, stream>>>(scores, n, live);
    choose_parallel_byte<true><<<1, kThreads, 0, stream>>>(scores, hist_parts, k, n, live);
    parallel_histogram<false><<<hist_parts, kThreads, 0, stream>>>(scores, n, live);
    choose_parallel_byte<false><<<1, kThreads, 0, stream>>>(scores, hist_parts, k, n, live);
    count_partitions<<<hist_parts, kThreads, 0, stream>>>(scores, n, live);
    prefix_partitions<<<1, kThreads, 0, stream>>>(scores, partitions_of(n), n, live);
    emit_partitions<<<hist_parts, kThreads, 0, stream>>>(scores, n, offset, out, live);
    restore_scores<<<hist_parts, kThreads, 0, stream>>>(scores, n, live);
}

}  // namespace

void indexer_topk(const bf16* q, const bf16* keys, int64_t t, const bf16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores,
                  int32_t* out_idx, cudaStream_t stream) {
    if (t <= 0) return;
    if (t <= 512) {
        small_scores<<<unsigned((t + 7) / 8), kThreads, 0, stream>>>(q, keys, t, w, cand, scores);
    } else {
        tensor_scores<<<unsigned((t - 1) / 64 + 1), 128, 0, stream>>>(q, keys, t, w, cand, scores);
    }
    // Finish all full-width score writes before any selector borrows low halves.
    // Same-stream kernel ordering also applies during every graph replay.
    k = int(std::min<int64_t>(std::max(k, 0), t));
    if (k == t) {
        all_positions<<<unsigned((t + 255) / 256), kThreads, 0, stream>>>(t, offset, out_idx);
    } else if (k > 0 && t >= kRadixItems) {
        parallel_select(scores, t, k, offset, out_idx, stream);
    } else if (k > 0) {
        select_without_scratch<false><<<1, kThreads, 0, stream>>>({scores, t, 1}, t, k, offset, out_idx, nullptr);
    }
    check(cudaGetLastError(), "indexer launch");
}

void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block,
                       uint8_t* cand, cudaStream_t stream) {
    if (t <= 0 || block <= 0) return;
    if (topk_blocks <= 0) {
        check(cudaMemsetAsync(cand, 0, size_t(t), stream), "empty candidates");
    } else {
        const int64_t blocks = (t - 1) / block + 1;
        if (topk_blocks >= blocks) {
            all_candidates<<<unsigned(std::min<int64_t>(256, (blocks - 1) / 256 + 1)), kThreads, 0, stream>>>(scores, t, block, cand);
        } else {
            select_without_scratch<true><<<1, kThreads, 0, stream>>>(
                {scores, t, block}, blocks, topk_blocks, 0, nullptr, cand);
        }
    }
    check(cudaGetLastError(), "candidate launch");
}

void indexer_topk_device(const bf16* q, const bf16* keys, const int* pos_dev,
                         int ratio, int64_t t_cap, const bf16* w, const uint8_t* cand,
                         int k, int32_t offset, float* scores, int32_t* out_idx, cudaStream_t stream) {
    if (ratio <= 0 || t_cap < 0 || !pos_dev) {
        std::fprintf(stderr, "K5: invalid device length parameters\n"); std::abort();
    }
    if (t_cap == 0) return;
    small_scores<<<unsigned((std::min<int64_t>(t_cap, 512) + 7) / 8), kThreads, 0, stream>>>(
        q, keys, t_cap, w, cand, scores, pos_dev, ratio);
    if (t_cap > 512) tensor_scores<<<unsigned((t_cap + 63) / 64), 128, 0, stream>>>(
        q, keys, t_cap, w, cand, scores, pos_dev, ratio);
    const bool partition = t_cap >= kDeviceParallelFrom;
    select_without_scratch<false><<<1, kThreads, 0, stream>>>(
        {scores, t_cap, 1}, t_cap, k, offset, out_idx, nullptr, pos_dev, ratio, partition);
    if (partition) parallel_select(scores, t_cap, k, offset, out_idx, stream, {pos_dev, ratio, k});
    check(cudaGetLastError(), "device indexer launch");
}

void candidate_blocks_device(const float* scores, const int* pos_dev, int ratio, int64_t t_cap,
                              int topk_blocks, int block, uint8_t* cand, cudaStream_t stream) {
    if (ratio <= 0 || t_cap < 0 || !pos_dev) {
        std::fprintf(stderr, "K5: invalid device length parameters\n"); std::abort();
    }
    if (t_cap == 0 || block <= 0) return;
    select_without_scratch<true><<<1, kThreads, 0, stream>>>(
        {scores, t_cap, block}, 0, topk_blocks, 0, nullptr, cand, pos_dev, ratio);
    check(cudaGetLastError(), "device candidate launch");
}

}  // namespace strata::ds41::kernels
