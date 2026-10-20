// src/ds41/wo_a_fp8.cu - see include/strata/ds41/wo_a_fp8.hpp.
#include "strata/ds41/wo_a_fp8.hpp"

#include "strata/ds41/config.hpp"

#include <cuda_fp8.h>

#include <stdexcept>
#include <string>

namespace strata::ds41 {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kRows = kOGroups * kOLora;              // 8192
constexpr int kIn = kHeads * kHeadDim / kOGroups;     // 4096
constexpr int kBlock = 32;                            // scale block: 32 x 32
constexpr int kUnroll = 8;                            // bytes in flight per lane (RTX 4090: 4 763, 8 873, 16 823 GB/s)

// Match convert.py, including BF16 subnormal rounding.
__device__ __forceinline__ float deq(uint8_t w, uint8_t e) {
    __nv_fp8_e4m3 v;
    v.__x = w;
    const uint32_t bits = e == 0 ? 0x00400000u : (e == 255 ? 0x7FC00000u : (uint32_t) e << 23);
    const float value = float(v) * __uint_as_float(bits);
    // Codes 0..2 can fall between BF16 subnormals. Larger codes are exact in BF16.
    return e <= 2 ? __bfloat162float(__float2bfloat16_rn(value)) : value;
}

// One warp per output row, as ops::wo_a_grouped: lane i adds elements i, i + 32, ... in this order, then the same
// shuffle reduction, so the sum is the BF16 kernel's. kUnroll loads are in flight per lane.
__global__ void wo_a_fp8_k(const bf16* o, const uint8_t* w, const uint8_t* scale, bf16* y) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= kRows) return;
    const bf16* x = o + (row / kOLora) * kIn;
    const uint8_t* wr = w + (int64_t) row * kIn;
    const uint8_t* sr = scale + (row / kBlock) * (kIn / kBlock);
    float acc = 0.0f;
    for (int i = lane; i < kIn; i += kUnroll * 32) {
        uint8_t wv[kUnroll], sv[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = wr[i + 32 * u];
            sv[u] = sr[i / kBlock + u];
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) acc += __bfloat162float(x[i + 32 * u]) * deq(wv[u], sv[u]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = __float2bfloat16_rn(acc);
}

// wo_a_fp8_k for M rows: the same lanes and order per row, each weight byte and scale loaded once for all rows
template <int M>
__global__ void wo_a_fp8_rows_k(const bf16* o, const uint8_t* w, const uint8_t* scale, bf16* y) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= kRows) return;
    const bf16* x = o + (row / kOLora) * kIn;
    const uint8_t* wr = w + (int64_t) row * kIn;
    const uint8_t* sr = scale + (row / kBlock) * (kIn / kBlock);
    float acc[M];
#pragma unroll
    for (int t = 0; t < M; ++t) acc[t] = 0.0f;
    for (int i = lane; i < kIn; i += kUnroll * 32) {
        uint8_t wv[kUnroll], sv[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = wr[i + 32 * u];
            sv[u] = sr[i / kBlock + u];
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const float d = deq(wv[u], sv[u]);
#pragma unroll
            for (int t = 0; t < M; ++t) acc[t] += __bfloat162float(x[t * kHeads * kHeadDim + i + 32 * u]) * d;
        }
    }
#pragma unroll
    for (int t = 0; t < M; ++t) {
        for (int off = 16; off > 0; off >>= 1) acc[t] += __shfl_down_sync(0xffffffffu, acc[t], off);
        if (lane == 0) y[t * kRows + row] = __float2bfloat16_rn(acc[t]);
    }
}

__global__ void dequant_k(const uint8_t* w, const uint8_t* scale, bf16* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) kRows * kIn) return;
    const int r = (int) (i / kIn), c = (int) (i % kIn);
    out[i] = __float2bfloat16_rn(deq(w[i], scale[(r / kBlock) * (kIn / kBlock) + c / kBlock]));
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 wo_a fp8: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

void wo_a_grouped_fp8(const bf16* o, const uint8_t* w, const uint8_t* scale, bf16* y, cudaStream_t stream) {
    wo_a_fp8_k<<<kRows / 8, 256, 0, stream>>>(o, w, scale, y);
    check(cudaGetLastError(), "wo_a_grouped_fp8");
}

void wo_a_grouped_fp8_rows(const bf16* o, int m, const uint8_t* w, const uint8_t* scale, bf16* y, cudaStream_t stream) {
    switch (m) {
        case 1: wo_a_fp8_rows_k<1><<<kRows / 8, 256, 0, stream>>>(o, w, scale, y); break;
        case 2: wo_a_fp8_rows_k<2><<<kRows / 8, 256, 0, stream>>>(o, w, scale, y); break;
        case 3: wo_a_fp8_rows_k<3><<<kRows / 8, 256, 0, stream>>>(o, w, scale, y); break;
        case 4: wo_a_fp8_rows_k<4><<<kRows / 8, 256, 0, stream>>>(o, w, scale, y); break;
        case 5: wo_a_fp8_rows_k<5><<<kRows / 8, 256, 0, stream>>>(o, w, scale, y); break;
        case 6: wo_a_fp8_rows_k<6><<<kRows / 8, 256, 0, stream>>>(o, w, scale, y); break;
        case 7: wo_a_fp8_rows_k<7><<<kRows / 8, 256, 0, stream>>>(o, w, scale, y); break;
        case 8: wo_a_fp8_rows_k<8><<<kRows / 8, 256, 0, stream>>>(o, w, scale, y); break;
        default: throw std::invalid_argument("wo_a_grouped_fp8_rows: m outside 1..8");
    }
    check(cudaGetLastError(), "wo_a_grouped_fp8_rows");
}

void dequant_wo_a(const uint8_t* w, const uint8_t* scale, bf16* out, cudaStream_t stream) {
    const int64_t n = (int64_t) kRows * kIn;
    dequant_k<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(w, scale, out);
    check(cudaGetLastError(), "dequant_wo_a");
}

}  // namespace strata::ds41
