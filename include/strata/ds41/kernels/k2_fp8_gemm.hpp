// include/strata/ds41/kernels/k2_fp8_gemm.hpp - task K2: FP8 block-scaled GEMM for prefill.
// Fixed interface; implementations live in src/ds41/kernels/k2_fp8_gemm.cu. Spec: ds41/tasks/K2.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::ds41::kernels {

/// y [M][N] bf16 = FP8-quantized x [M][K] times W [N][K] (FP8 E4M3, E8M0 32x32 block scales), FP32 accumulation:
/// the same math as strata::ds41::ops::fp8_linear for every row (activation blocks of 32, power-of-two scale).
/// workspace: at least M*K*4 bytes. K % 32 == 0, M <= 16384.
void fp8_block_gemm(const __nv_bfloat16* x, int64_t M, int64_t K, const uint8_t* w, const uint8_t* w_scale,
                    int64_t N, __nv_bfloat16* y, void* workspace, cudaStream_t stream);

}  // namespace strata::ds41::kernels
