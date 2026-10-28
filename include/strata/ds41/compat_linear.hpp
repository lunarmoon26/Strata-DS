#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strata::ds41 {
// Compatibility path: stored FP16 weights are consumed directly, never rounded
// to BF16. BF16 activations, FP32 FMA/reduction; exactly one output is required.
void fp16_weight_linear(const __nv_bfloat16* x, const __half* w, int rows, int k, int n,
                        __nv_bfloat16* y, float* yf, cudaStream_t stream = nullptr);
void fp16_weights_to_float(const __half* x, float* y, int n, cudaStream_t stream = nullptr);
}
