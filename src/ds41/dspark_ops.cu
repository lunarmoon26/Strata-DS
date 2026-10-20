// src/ds41/dspark_ops.cu - DSpark's own GPU ops. See dspark_ops.hpp.
#include "strata/ds41/dspark_ops.hpp"

#include "strata/ds41/config.hpp"

#include <stdexcept>
#include <string>

namespace strata::ds41::dspark {
namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 dspark: ") + what + ": " + cudaGetErrorString(e));
}

__global__ void hc_mean_k(const bf16* h, bf16* out, int64_t stride) {
    const int r = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kDim) return;
    const bf16* x = h + (size_t) r * kHc * kDim + d;
    float s = 0;
    for (int c = 0; c < kHc; ++c) s += __bfloat162float(x[(size_t) c * kDim]);
    out[r * stride + d] = __float2bfloat16_rn(s / kHc);
}

/// one block per row: each warp scores experts e = warp, warp + nw, ...; thread 0 picks
__global__ void router_small_k(const bf16* x, const bf16* w, const float* bias, int n_exp, int topk, int32_t* ids,
                               float* weights) {
    __shared__ float s[256];
    const int r = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const bf16* xr = x + (size_t) r * kDim;
    for (int e = warp; e < n_exp; e += nw) {
        const bf16* we = w + (size_t) e * kDim;
        float acc = 0;
        for (int d = lane; d < kDim; d += 32) acc += __bfloat162float(xr[d]) * __bfloat162float(we[d]);
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) s[e] = sqrtf(acc > 20.0f ? acc : log1pf(expf(acc)));
    }
    __syncthreads();
    if (threadIdx.x) return;
    int pick[8];
    float sum = 0;
    for (int j = 0; j < topk; ++j) {
        int best = -1;
        float bv = 0;
        for (int e = 0; e < n_exp; ++e) {
            bool used = false;
            for (int q = 0; q < j; ++q) used |= pick[q] == e;
            if (used) continue;
            const float val = s[e] + bias[e];
            if (best < 0 || val > bv) { best = e; bv = val; }
        }
        pick[j] = best;
        sum += s[best];
    }
    for (int j = 0; j < topk; ++j) {
        ids[r * topk + j] = pick[j];
        weights[r * topk + j] = s[pick[j]] / (sum + 1e-20f) * kRouteScale;
    }
}

/// a warp per vocab row: 256 weights, 8 per lane
__global__ void markov_bias_k(const int* token, const bf16* embed, const bf16* head, int64_t n_vocab, float* logits,
                              bf16* embed_out) {
    __shared__ float e[kMarkovRank];
    const bf16* er = embed + (size_t) *token * kMarkovRank;
    for (int i = threadIdx.x; i < kMarkovRank; i += blockDim.x) {
        e[i] = __bfloat162float(er[i]);
        if (blockIdx.x == 0) embed_out[i] = er[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t v = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (v >= n_vocab) return;
    const bf16* hr = head + v * kMarkovRank;
    float acc = 0;
    for (int i = lane; i < kMarkovRank; i += 32) acc += __bfloat162float(hr[i]) * e[i];
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) logits[v] += acc;
}

__global__ void confidence_k(const bf16* x, const bf16* e, const bf16* w, float* out) {
    __shared__ float part[32];
    const int r = blockIdx.x;
    float acc = 0;
    for (int d = threadIdx.x; d < kDim + kMarkovRank; d += blockDim.x) {
        const float a = d < kDim ? __bfloat162float(x[(size_t) r * kDim + d])
                                 : __bfloat162float(e[(size_t) r * kMarkovRank + d - kDim]);
        acc += a * __bfloat162float(w[d]);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x) return;
    float s = 0;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += part[i];
    out[r] = s;
}

__global__ void block_index_k(const int* start_pos, int32_t* idx) {
    const int r = blockIdx.x, j = threadIdx.x;
    if (j >= kWindow + kBlock) return;
    idx[r * (kWindow + kBlock) + j] = j >= kWindow ? j : (j <= *start_pos ? j : -1);
}

}  // namespace

void hc_mean_rows(const bf16* h, int m, bf16* out, int64_t out_stride, cudaStream_t stream) {
    hc_mean_k<<<dim3((kDim + 255) / 256, m), 256, 0, stream>>>(h, out, out_stride);
    check(cudaGetLastError(), "hc_mean_rows");
}

void router_small(const bf16* x, int m, const bf16* w, const float* bias, int n_exp, int topk, int32_t* ids,
                  float* weights, cudaStream_t stream) {
    if (n_exp < 1 || n_exp > 256 || topk < 1 || topk > 8 || topk > n_exp) throw std::invalid_argument("router_small shape");
    router_small_k<<<m, 512, 0, stream>>>(x, w, bias, n_exp, topk, ids, weights);
    check(cudaGetLastError(), "router_small");
}

void markov_bias(const int* token, const bf16* embed, const bf16* head, int64_t n_vocab, float* logits,
                 bf16* embed_out, cudaStream_t stream) {
    constexpr int rows = 8;   // warps per block
    markov_bias_k<<<(unsigned) ((n_vocab + rows - 1) / rows), rows * 32, 0, stream>>>(token, embed, head, n_vocab, logits,
                                                                                     embed_out);
    check(cudaGetLastError(), "markov_bias");
}

void confidence_rows(const bf16* x, const bf16* e, const bf16* w, int m, float* out, cudaStream_t stream) {
    confidence_k<<<m, 256, 0, stream>>>(x, e, w, out);
    check(cudaGetLastError(), "confidence_rows");
}

void block_index(const int* start_pos, int32_t* idx, cudaStream_t stream) {
    block_index_k<<<kBlock, kWindow + kBlock, 0, stream>>>(start_pos, idx);
    check(cudaGetLastError(), "block_index");
}

}  // namespace strata::ds41::dspark
