// src/ds41/prefill_ops.cu - GPU operations that only batched prefill needs. See prefill_ops.hpp.
#include "strata/ds41/prefill_ops.hpp"

#include <algorithm>

#include "strata/ds41/config.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::ds41::prefill {
namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds41 prefill ops: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}
void check(cublasStatus_t e, const char* what) {
    if (e != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "ds41 prefill ops: %s: cuBLAS status %d\n", what, (int) e);
        std::abort();
    }
}
#define LAUNCH_CHECK(what) check(cudaGetLastError(), what)

__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 tobf(float v) { return __float2bfloat16_rn(v); }

int grid(int64_t n, int per_block) { return (int) ((n + per_block - 1) / per_block); }

/// One cuBLAS handle per device, created at the first GEMM, on the legacy default stream
cublasHandle_t handle() {
    static cublasHandle_t h[16] = {};
    int dev = 0;
    check(cudaGetDevice(&dev), "cudaGetDevice");
    if (!h[dev]) check(cublasCreate(&h[dev]), "cublasCreate");
    return h[dev];
}

__global__ void embed_rows_k(const bf16* table, const int32_t* tokens, bf16* h) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kDim) return;
    const bf16 v = table[(int64_t) tokens[blockIdx.y] * kDim + d];
    for (int c = 0; c < kHc; ++c) h[((int64_t) blockIdx.y * kHc + c) * kDim + d] = v;
}

// as ops rope_k, with the row's own position
__global__ void rope_rows_k(bf16* v, int n_vec, int stride, const float* table, int pos0, int pos_step, bool inverse) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;      // one complex pair of row blockIdx.y
    if (i >= n_vec * (kRopeDim / 2)) return;
    const int vec = i / (kRopeDim / 2), p = i % (kRopeDim / 2);
    const float* cs = table + (int64_t) (pos0 + (int) blockIdx.y * pos_step) * kRopeDim;
    bf16* base = v + ((int64_t) blockIdx.y * n_vec + vec) * stride + (stride - kRopeDim) + 2 * p;
    const float re = bf(base[0]), im = bf(base[1]);
    const float c = cs[2 * p], s = inverse ? -cs[2 * p + 1] : cs[2 * p + 1];
    base[0] = tobf(re * c - im * s);
    base[1] = tobf(re * s + im * c);
}

// one warp per row: 12 experts per lane, 6 rounds of a warp argmax (ties: lower id)
__global__ void route_rows_k(const float* logits, const float* bias, int rows, int32_t* ids, float* weights) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    constexpr int kPer = kExperts / 32;
    float s[kPer], key[kPer];
    for (int i = 0; i < kPer; ++i) {
        const int e = lane + 32 * i;
        const float v = logits[(int64_t) row * kExperts + e];
        s[i] = sqrtf(v > 20.0f ? v : log1pf(expf(v)));
        key[i] = s[i] + bias[e];
    }
    float pick_s[kTopK];
    int pick_e[kTopK];
    for (int k = 0; k < kTopK; ++k) {
        float best = -INFINITY;
        int be = 1 << 30;
        for (int i = 0; i < kPer; ++i)
            if (key[i] > best || (key[i] == best && lane + 32 * i < be)) { best = key[i]; be = lane + 32 * i; }
        for (int off = 16; off > 0; off >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, off);
            const int oe = __shfl_xor_sync(0xffffffffu, be, off);
            if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; }
        }
        pick_e[k] = be;
        float sv = 0.0f;
        if ((be & 31) == lane) {
            sv = s[be / 32];
            key[be / 32] = -INFINITY;
        }
        pick_s[k] = __shfl_sync(0xffffffffu, sv, be & 31);
    }
    if (lane == 0) {
        float sum = 0.0f;
        for (int k = 0; k < kTopK; ++k) sum += pick_s[k];
        for (int k = 0; k < kTopK; ++k) {
            ids[row * kTopK + k] = pick_e[k];
            weights[row * kTopK + k] = pick_s[k] / (sum + 1e-20f) * kRouteScale;
        }
    }
}

__global__ void attn_index_rows_k(int p0, int win_base, const int32_t* topk, int k_top, int32_t* idx, int n_idx) {
    const int r = blockIdx.x;
    for (int j = threadIdx.x; j < n_idx; j += blockDim.x) {
        int32_t v;
        if (j < kWindow) v = p0 + r - (kWindow - 1) + j < 0 ? -1 : win_base + r + j;
        else v = topk && j - kWindow < k_top ? topk[(int64_t) r * k_top + (j - kWindow)] : -1;
        idx[(int64_t) r * n_idx + j] = v;
    }
}

__global__ void window_gather_k(const bf16* ring, int p0, int n, bf16* dst) {
    const int k = blockIdx.x;   // dst row k = position p0 - n + k
    const bf16* src = ring + (int64_t) ((p0 - n + k) % kWindow) * kHeadDim;
    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) dst[(int64_t) k * kHeadDim + d] = src[d];
}

__global__ void window_scatter_k(bf16* ring, const bf16* src, int p0, int first) {
    const int r = first + blockIdx.x;   // chunk row r = position p0 + r
    bf16* dst = ring + (int64_t) ((p0 + r) % kWindow) * kHeadDim;
    for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) dst[d] = src[(int64_t) r * kHeadDim + d];
}

__global__ void nll_rows_k(const float* logits, int vocab, const int32_t* target, float* nll) {
    __shared__ float sh[32];
    const int r = blockIdx.x;
    const int t = target[r];
    if (t < 0) {
        if (threadIdx.x == 0) nll[r] = 0.0f;
        return;
    }
    const float* row = logits + (int64_t) r * vocab;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) mx = fmaxf(mx, row[i]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = sh[0];
    for (int i = 1; i < (int) blockDim.x / 32; ++i) mx = fmaxf(mx, sh[i]);
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) sum += expf(row[i] - mx);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < (int) blockDim.x / 32; ++i) total += sh[i];
        nll[r] = logf(total) + mx - row[t];
    }
}

__global__ void round_bf16_k(const float* x, bf16* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tobf(x[i]);
}

void round_bf16(const float* x, bf16* y, int64_t n) {
    if (n > 0) round_bf16_k<<<grid(n, 256), 256>>>(x, y, n);
    LAUNCH_CHECK("round_bf16");
}

}  // namespace

/// rows per launch: grid.y is at most 65535
constexpr int kMaxGridY = 65535;

void embed_rows(const bf16* table, const int32_t* tokens, int rows, bf16* h) {
    for (int r0 = 0; r0 < rows; r0 += kMaxGridY) {
        const int n = std::min(kMaxGridY, rows - r0);
        embed_rows_k<<<dim3(grid(kDim, 256), n), 256>>>(table, tokens + r0, h + (int64_t) r0 * kHc * kDim);
    }
    LAUNCH_CHECK("embed_rows");
}

void rope_rows(bf16* v, int rows, int n_vec, int stride, const float* table, int pos0, int pos_step, bool inverse) {
    for (int r0 = 0; r0 < rows; r0 += kMaxGridY) {
        const int n = std::min(kMaxGridY, rows - r0);
        rope_rows_k<<<dim3(grid((int64_t) n_vec * (kRopeDim / 2), 256), n), 256>>>(
            v + (int64_t) r0 * n_vec * stride, n_vec, stride, table, pos0 + r0 * pos_step, pos_step, inverse);
    }
    LAUNCH_CHECK("rope_rows");
}

void route_rows(const float* logits, const float* bias, int rows, int32_t* ids, float* weights) {
    if (rows > 0) route_rows_k<<<grid(rows, 8), 256>>>(logits, bias, rows, ids, weights);
    LAUNCH_CHECK("route_rows");
}

void attn_index_rows(int rows, int p0, int win_base, const int32_t* topk, int k_top, int32_t* idx, int n_idx) {
    if (rows > 0) attn_index_rows_k<<<rows, 128>>>(p0, win_base, topk, k_top, idx, n_idx);
    LAUNCH_CHECK("attn_index_rows");
}

void window_gather(const bf16* ring, int p0, int n, bf16* dst) {
    if (n > 0) window_gather_k<<<n, 128>>>(ring, p0, n, dst);
    LAUNCH_CHECK("window_gather");
}

void window_scatter(bf16* ring, const bf16* src, int p0, int rows) {
    const int first = rows > kWindow ? rows - kWindow : 0;
    if (rows > 0) window_scatter_k<<<rows - first, 128>>>(ring, src, p0, first);
    LAUNCH_CHECK("window_scatter");
}

void nll_rows(const float* logits, int rows, int vocab, const int32_t* target, float* nll) {
    if (rows > 0) nll_rows_k<<<rows, 1024>>>(logits, vocab, target, nll);
    LAUNCH_CHECK("nll_rows");
}

void bf16_gemm(const bf16* x, const bf16* w, int64_t M, int64_t K, int64_t N, bf16* yb, float* yf, float* tmp) {
    if (M <= 0) return;
    float* out = yf ? yf : tmp;
    if (!out) {
        std::fprintf(stderr, "bf16_gemm: a BF16 output needs the FP32 scratch\n");
        std::abort();
    }
    const float one = 1.0f, zero = 0.0f;
    // column-major view: y^T [N][M] = W^T-as-stored [K][N]^T x x^T [K][M]
    check(cublasGemmEx(handle(), CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) M, (int) K, &one, w, CUDA_R_16BF, (int) K, x,
                       CUDA_R_16BF, (int) K, &zero, out, CUDA_R_32F, (int) N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
          "bf16_gemm");
    if (yb) round_bf16(out, yb, M * N);
}

void wo_a_grouped_rows(const bf16* o, const bf16* wo_a, int64_t M, bf16* y, float* tmp) {
    if (M <= 0) return;
    constexpr int kIn = kHeads * kHeadDim / kOGroups;   // 4096
    const float one = 1.0f, zero = 0.0f;
    check(cublasGemmStridedBatchedEx(handle(), CUBLAS_OP_T, CUBLAS_OP_N, kOLora, (int) M, kIn, &one, wo_a, CUDA_R_16BF,
                                     kIn, (long long) kOLora * kIn, o, CUDA_R_16BF, kHeads * kHeadDim, kIn, &zero, tmp,
                                     CUDA_R_32F, kOGroups * kOLora, kOLora, kOGroups, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT),
          "wo_a_grouped_rows");
    round_bf16(tmp, y, M * kOGroups * kOLora);
}

}  // namespace strata::ds41::prefill
