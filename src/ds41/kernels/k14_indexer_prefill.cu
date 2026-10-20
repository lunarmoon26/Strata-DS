// K14-02: head-streamed 16-query x 128-key BF16 tensor-core score tiles.
// All scratch belongs to the caller; no allocation, host sync, or per-query launch.
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"

#include <mma.h>
#include <math_constants.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::ds41::kernels {
namespace {
using bf16 = __nv_bfloat16;
using Count = unsigned long long;
constexpr int kQueryTile = 16;
constexpr int kKeyTile = 128;
constexpr int kHeadGroup = 8;
constexpr int kInputStride = 136;
constexpr int kScoreThreads = 512;
constexpr int kSelectThreads = 256;
constexpr int kBatchRows = 256;
constexpr size_t kAlignment = 256;
constexpr unsigned kFullWarp = 0xffffffffu;

size_t checked_product(size_t a, size_t b) {
    // Pointer differences must also remain representable, not just size_t.
    constexpr size_t limit = size_t(std::numeric_limits<ptrdiff_t>::max());
    if (a && b > limit / a) throw std::invalid_argument("K14: size overflow");
    return a * b;
}

size_t score_bytes(int m, int64_t t_max) {
    if (m < 1 || m > 16384 || t_max < 0)
        throw std::invalid_argument("K14: invalid workspace shape");
    if (uint64_t(t_max) > uint64_t(std::numeric_limits<ptrdiff_t>::max()))
        throw std::invalid_argument("K14: size overflow");
    return checked_product(checked_product(size_t(std::min(m, kBatchRows)), size_t(t_max)), sizeof(bf16));
}

void check_launch(const char* operation) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

__device__ __forceinline__ float rounded(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// Stage eight heads from every query, compute them, then reuse the same
// storage for their FP32 dots. Keys remain resident through all four groups.
// The +8 BF16 input stride rotates consecutive rows across shared banks.
union __align__(32) QueryDots {
    bf16 queries[kQueryTile * kHeadGroup * kInputStride];
    float dots[kQueryTile * kHeadGroup * kKeyTile];
};
struct __align__(32) ScoreTile {
    QueryDots group;
    bf16 keys[kKeyTile * kInputStride];
    bf16 weights[kQueryTile * kHeadGroup];
};
static_assert(sizeof(QueryDots) == 64 * 1024, "query/dot storage size");
static_assert(sizeof(ScoreTile) == 100608, "score tile storage size");
static_assert(sizeof(ScoreTile) <= 99 * 1024, "consumer shared-memory budget");
static_assert(kQueryTile * kKeyTile % kScoreThreads == 0, "whole thread-owned scores");
constexpr int kThreadScores = kQueryTile * kKeyTile / kScoreThreads;

__global__ __launch_bounds__(kScoreThreads) void batched_scores(
        const bf16* q, const bf16* keys, const bf16* w,
        int first, int rows, int pos0, int ratio,
        const uint8_t* cand, int64_t cand_stride,
        bf16* scores, int64_t score_stride) {
    extern __shared__ __align__(32) unsigned char shared[];
    ScoreTile& tile = *reinterpret_cast<ScoreTile*>(shared);
    const int tid = threadIdx.x;
    const int tile_row = int(blockIdx.y) * kQueryTile;
    const int tile_rows = min(kQueryTile, rows - tile_row);
    const int64_t key_base = int64_t(blockIdx.x) * kKeyTile;
    const int64_t tile_end = (int64_t(pos0) + first + tile_row + tile_rows) / ratio;
    if (key_base >= tile_end) return;  // Uniform, before any barrier.

    for (int item = tid; item < kKeyTile * 128; item += kScoreThreads) {
        const int key_row = item / 128, d = item % 128;
        const int64_t j = key_base + key_row;
        tile.keys[key_row * kInputStride + d] = j < tile_end
            ? keys[size_t(j) * 128 + d] : __float2bfloat16_rn(0.0f);
    }
    float total[kThreadScores] = {};
    const int warp = tid >> 5;
    const int query_pair = warp / 2;
    const int key_parity = warp % 2;
    namespace wm = nvcuda::wmma;

    // Ascending groups and ascending heads within each group preserve the
    // reference's FP32 head-sum order, including the group boundaries.
#pragma unroll 1
    for (int head_base = 0; head_base < 32; head_base += kHeadGroup) {
        for (int item = tid; item < kQueryTile * kHeadGroup * 128; item += kScoreThreads) {
            const int qh = item / 128, d = item % 128;
            const int local_row = qh / kHeadGroup, head = qh % kHeadGroup;
            tile.group.queries[qh * kInputStride + d] = local_row < tile_rows
                ? q[(size_t(first + tile_row + local_row) * 32 + head_base + head) * 128 + d]
                : __float2bfloat16_rn(0.0f);
        }
        if (tid < kQueryTile * kHeadGroup) {
            const int local_row = tid / kHeadGroup, head = tid % kHeadGroup;
            tile.weights[tid] = local_row < tile_rows
                ? w[size_t(first + tile_row + local_row) * 32 + head_base + head]
                : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();  // Queries, weights, and the retained keys are ready.

        // Each warp computes two queries x eight heads against four disjoint
        // 16-key groups. All 16 warps together cover the 128x128 dot matrix.
        wm::fragment<wm::matrix_a, 16, 16, 16, bf16, wm::row_major> a;
        wm::fragment<wm::matrix_b, 16, 16, 16, bf16, wm::col_major> b;
        wm::fragment<wm::accumulator, 16, 16, 16, float> dots[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) wm::fill_fragment(dots[part], 0.0f);
#pragma unroll
        for (int d = 0; d < 128; d += 16) {
            wm::load_matrix_sync(a, tile.group.queries + query_pair * 16 * kInputStride + d, kInputStride);
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int key_group = key_parity + part * 2;
                wm::load_matrix_sync(b, tile.keys + key_group * 16 * kInputStride + d, kInputStride);
                wm::mma_sync(dots[part], a, b, dots[part]);
            }
        }
        __syncthreads();  // Retire every query reader before the overlay stores.
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int key_group = key_parity + part * 2;
            wm::store_matrix_sync(tile.group.dots + query_pair * 16 * kKeyTile + key_group * 16,
                                  dots[part], kKeyTile, wm::mem_row_major);
        }
        __syncthreads();  // Every dot is now visible to its owning score thread.
#pragma unroll
        for (int part = 0; part < kThreadScores; ++part) {
            const int output = tid + part * kScoreThreads;
            const int local_row = output / kKeyTile, key_col = output % kKeyTile;
#pragma unroll
            for (int head = 0; head < kHeadGroup; ++head) {
                const float dot = rounded(tile.group.dots[(local_row * kHeadGroup + head) * kKeyTile + key_col]);
                total[part] += rounded(fmaxf(dot, 0.0f)
                    * __bfloat162float(tile.weights[local_row * kHeadGroup + head]));
            }
        }
        __syncthreads();  // Retire dot/weight readers before staging the next group.
    }
#pragma unroll
    for (int part = 0; part < kThreadScores; ++part) {
        const int output = tid + part * kScoreThreads;
        const int local_row = output / kKeyTile, key_col = output % kKeyTile;
        const int row = first + tile_row + local_row;
        const int64_t j = key_base + key_col;
        const int64_t n = (int64_t(pos0) + row + 1) / ratio;
        if (local_row < tile_rows && j < n) {
            const float value = !cand || cand[size_t(row) * size_t(cand_stride) + size_t(j)]
                ? total[part] : -CUDART_INF_F;
            scores[size_t(tile_row + local_row) * size_t(score_stride) + size_t(j)] = __float2bfloat16_rn(value);
        }
    }
}

// Exact selector copied without functional changes from reviewed K14-01,
// commit 1e8f6d22166620116d76d81c1c00851f4b100285, production SHA-256:
// d061923b0df41325df17949a730cfb694b184f13e314f2a7a4def368d818a11e.
// All stored scores and block maxima are BF16 values (or infinity). Float-flip
// therefore needs only two radix bytes. Signed zeros compare equal by contract.
__device__ __forceinline__ uint32_t key_of(float value) {
    const uint32_t bits = value == 0.0f ? 0u : __float_as_uint(value);
    return (bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u)) >> 16;
}

template <bool Candidate>
struct Values {
    const bf16* scores;
    int64_t positions;
    int block_size;
    __device__ __forceinline__ float get(int64_t j) const {
        if constexpr (!Candidate) {
            return __bfloat162float(scores[j]);
        } else {
            if (j == (positions - 1) / block_size) return CUDART_INF_F;
            float value = -CUDART_INF_F;
            const int64_t begin = j * int64_t(block_size);
            for (int64_t d = 0; d < block_size && begin + d < positions; ++d)
                value = fmaxf(value, __bfloat162float(scores[begin + d]));
            return value;
        }
    }
};

struct Selection { uint32_t prefix; Count remaining; };

// K5's ordered cutoff/emission, parallelized over query rows. One CTA owns each
// complete row, so there is no global counter, inter-CTA scan, or hidden scratch.
// Warp-private histogram counters are bounded by ceil(t_i/8)+32 < 2^32 because
// pos0 is an int and m<=16384; their reduction and all positional counts use u64.
template <bool Candidate>
__global__ void select_rows(const bf16* scores, int64_t score_stride, int first,
                            int pos0, int ratio, int k, int32_t offset,
                            int topk_blocks, int block_size, int32_t* out_idx,
                            uint8_t* cand_out, int64_t cand_stride) {
    __shared__ uint32_t warp_bins[8][256];
    __shared__ Count bins[256];
    __shared__ Selection state;
    __shared__ uint32_t greater_prefix[32], equal_prefix[32], batch_greater, batch_equal;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row = first + int(blockIdx.x);
    const int64_t positions = (int64_t(pos0) + row + 1) / ratio;
    const Values<Candidate> values{scores + size_t(blockIdx.x) * size_t(score_stride), positions, block_size};
    int32_t* out = out_idx + size_t(row) * size_t(k);
    uint8_t* candidate = nullptr;
    if constexpr (Candidate) candidate = cand_out + size_t(row) * size_t(cand_stride);
    if constexpr (!Candidate) {
        for (int64_t j = min(int64_t(k), positions) + tid; j < k; j += kSelectThreads) out[j] = -1;
    }
    if (!positions) return;
    const int64_t n = Candidate ? (positions - 1) / block_size + 1 : positions;
    const int64_t wanted = Candidate ? min(int64_t(topk_blocks), n) : min(int64_t(k), n);
    if constexpr (Candidate) {
        if (wanted <= 0) {
            for (int64_t j = tid; j < positions; j += kSelectThreads) candidate[j] = 0;
            return;
        }
    }
    if (wanted == n) {
        for (int64_t j = tid; j < n; j += kSelectThreads) {
            if constexpr (Candidate) {
                const uint8_t keep = values.get(j) != -CUDART_INF_F;
                const int64_t begin = j * int64_t(block_size);
                for (int64_t d = 0; d < block_size && begin + d < positions; ++d) candidate[begin + d] = keep;
            } else {
                out[j] = int32_t(j + int64_t(offset));
            }
        }
        return;
    }

    uint32_t prefix = 0, prefix_mask = 0;
    for (int shift = 8; shift >= 0; shift -= 8) {
#pragma unroll
        for (int w = 0; w < 8; ++w) warp_bins[w][tid] = 0;
        __syncthreads();
        for (int64_t base = 0; base < n; base += kSelectThreads) {
            const int64_t j = base + tid;
            const uint32_t key = j < n ? key_of(values.get(j)) : 0;
            const bool valid = j < n && (key & prefix_mask) == prefix;
            const unsigned active = __ballot_sync(kFullWarp, valid);
            if (valid) {
                const uint32_t bin = (key >> shift) & 255u;
                const unsigned peers = __match_any_sync(active, bin);
                if (lane == __ffs(peers) - 1) atomicAdd(&warp_bins[warp][bin], uint32_t(__popc(peers)));
            }
        }
        __syncthreads();
        Count count = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) count += Count(warp_bins[w][tid]);
        bins[tid] = count;
        __syncthreads();
        if (tid == 0) {
            Count remaining = shift == 8 ? Count(wanted) : state.remaining;
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
    const uint32_t lane_mask = (uint32_t(1) << lane) - 1;
    Count greater_before = 0, equal_before = 0;
    for (int64_t base = 0; base < n; base += 4 * kSelectThreads) {
        unsigned greater_lanes[4], equal_lanes[4];
        uint32_t keys[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int64_t j = base + part * kSelectThreads + tid;
            const uint32_t key = j < n ? key_of(values.get(j)) : 0;
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
            const uint32_t own_greater = greater_prefix[lane], own_equal = equal_prefix[lane];
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
            const int64_t j = base + part * kSelectThreads + tid;
            const Count preceding_greater = greater_before + greater_prefix[part * 8 + warp]
                + __popc(greater_lanes[part] & lane_mask);
            const Count preceding_equal = equal_before + equal_prefix[part * 8 + warp]
                + __popc(equal_lanes[part] & lane_mask);
            const bool selected = j < n && (keys[part] > prefix || (keys[part] == prefix && preceding_equal < quota));
            if constexpr (Candidate) {
                if (j < n) {
                    const uint8_t keep = selected && keys[part] != key_of(-CUDART_INF_F);
                    const int64_t begin = j * int64_t(block_size);
                    for (int64_t d = 0; d < block_size && begin + d < positions; ++d) candidate[begin + d] = keep;
                }
            } else if (selected) {
                out[preceding_greater + (preceding_equal < quota ? preceding_equal : quota)] = int32_t(j + int64_t(offset));
            }
        }
        greater_before += batch_greater;
        equal_before += batch_equal;
        __syncthreads();
    }
}
}  // namespace

size_t indexer_topk_prefill_workspace_bytes(int m, int64_t t_max) {
    const size_t bytes = score_bytes(m, t_max);
    if (!bytes) return 0;
    if (bytes > size_t(std::numeric_limits<ptrdiff_t>::max()) - (kAlignment - 1))
        throw std::invalid_argument("K14: size overflow");
    return bytes + kAlignment - 1;
}

void indexer_topk_prefill(const bf16* q, const bf16* keys, const bf16* w, int m, int pos0,
                          int ratio, const uint8_t* cand, uint8_t* cand_out, int64_t cand_stride,
                          int k, int32_t offset, int topk_blocks, int block, int32_t* out_idx,
                          void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (m < 1 || m > 16384 || ratio < 1 || k < 1 || pos0 < 0 || !out_idx || (cand_out && block < 1))
        throw std::invalid_argument("K14: invalid arguments");
    const int64_t t_max = (int64_t(pos0) + m) / ratio;
    const size_t required = indexer_topk_prefill_workspace_bytes(m, t_max);
    if (workspace_bytes < required || (required && !workspace))
        throw std::invalid_argument("K14: insufficient workspace");
    if (t_max && (!q || !keys || !w)) throw std::invalid_argument("K14: null score input");
    if (t_max && t_max - 1 + int64_t(offset) > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("K14: output index overflow");
    checked_product(size_t(m), checked_product(size_t(k), sizeof(int32_t)));
    checked_product(size_t(t_max), 128 * sizeof(bf16));
    if (cand || cand_out) {
        if (cand_stride < t_max) throw std::invalid_argument("K14: candidate stride is too small");
        const size_t before_last = checked_product(size_t(m - 1), size_t(cand_stride));
        if (before_last > size_t(std::numeric_limits<ptrdiff_t>::max()) - size_t(t_max))
            throw std::invalid_argument("K14: candidate size overflow");
    }
    if (!t_max) {
        const cudaError_t error = cudaMemsetAsync(out_idx, 0xff, size_t(m) * size_t(k) * sizeof(int32_t), stream);
        if (error != cudaSuccess) throw std::runtime_error("K14: output padding failed");
        return;
    }
    // Opt in for this function on the current device, without allocating
    // scratch, querying device properties, or synchronizing a stream. Repeating
    // the same function attribute avoids process-global per-device cache state.
    const cudaError_t attribute_error = cudaFuncSetAttribute(batched_scores,
        cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(ScoreTile)));
    if (attribute_error != cudaSuccess)
        throw std::runtime_error(std::string("K14 shared-memory opt-in: ") + cudaGetErrorString(attribute_error));
    // The API includes alignment slack, so even a byte-aligned workspace works.
    const size_t adjustment = (kAlignment - (reinterpret_cast<uintptr_t>(workspace) % kAlignment)) % kAlignment;
    bf16* scores = reinterpret_cast<bf16*>(static_cast<unsigned char*>(workspace) + adjustment);
    for (int first = 0; first < m; first += kBatchRows) {
        const int rows = std::min(kBatchRows, m - first);
        const int64_t batch_end = (int64_t(pos0) + first + rows) / ratio;
        if (batch_end) {
            const dim3 grid(unsigned((batch_end - 1) / kKeyTile + 1), unsigned((rows + kQueryTile - 1) / kQueryTile));
            batched_scores<<<grid, kScoreThreads, sizeof(ScoreTile), stream>>>(q, keys, w, first, rows, pos0, ratio,
                                                              cand, cand_stride, scores, t_max);
            check_launch("K14 batched scores");
        }
        select_rows<false><<<rows, kSelectThreads, 0, stream>>>(scores, t_max, first, pos0, ratio, k, offset,
                                                               topk_blocks, block, out_idx, nullptr, cand_stride);
        check_launch("K14 top-k rows");
        if (cand_out) {
            select_rows<true><<<rows, kSelectThreads, 0, stream>>>(scores, t_max, first, pos0, ratio, k, offset,
                                                                  topk_blocks, block, out_idx, cand_out, cand_stride);
            check_launch("K14 candidate rows");
        }
    }
}
}  // namespace strata::ds41::kernels
