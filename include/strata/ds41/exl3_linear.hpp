#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"
#include <cuda_bf16.h>

namespace strata::ds41 {

/// Small speculative windows use native GEMV jobs over shared quantized weights,
/// not reconstruction of a full FP16 matrix. Strides preserve grouped outputs.
size_t exl3_small_rows_workspace_bytes(int rows, int k, int n);
void exl3_linear_small_rows(const __nv_bfloat16* x, const kernels::Exl3Proj& w,
                            __nv_bfloat16* y, float* logits, int rows, int xs, int ys,
                            void* workspace, size_t bytes, cudaStream_t stream);

// One-row, stored EXL3 projection. FP16 input/pre-scale and Hadamard rounding
// follow exllamav3. No FP8 activation QDQ and no reconstructed-weight cache.
// Workspace is per stream; output is BF16, or BF16-rounded FP32 for logits.
size_t exl3_linear_workspace_bytes(int k, int n);
void exl3_linear(const __nv_bfloat16* x, const kernels::Exl3Proj& w,
                 __nv_bfloat16* y, float* logits, void* workspace,
                 size_t workspace_bytes, cudaStream_t stream);
void exl3_input_half(const __nv_bfloat16* x, uint16_t* y, int n, cudaStream_t stream);
size_t exl3_rows_workspace_bytes(int rows, int k, int n);
// Bounded temporary reconstruction of one matrix, reused across rows; not a cache.
void exl3_linear_rows(const __nv_bfloat16* x, const kernels::Exl3Proj& w,
                      __nv_bfloat16* y, int rows, int input_stride, int output_stride,
                      void* workspace, size_t bytes, cudaStream_t stream);

}  // namespace strata::ds41
