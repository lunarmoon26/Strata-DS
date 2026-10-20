// include/strata/ds41/kernels/k15_hc_prefill.hpp - task K15: hyper-connection mix coefficients plus the collapsed
// input for a prefill sub-batch (many tokens per call). Fixed interface; implementations live in
// src/ds41/kernels/k15_hc_prefill.cu. Spec: ds41/tasks/K15.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::ds41::kernels {

/// Workspace bytes hc_mixes_pre_rows needs for up to m tokens.
size_t hc_mixes_pre_rows_workspace_bytes(int m);

/// K7's hc_mixes_pre for m tokens (1..16384), each token exactly as K7 computes it:
///   x [m][4][5120] bf16 (the hc stream), fn [24][20480] f32, scale [3] f32, base [24] f32, pre_in [m][4] f32
///   y    [m][5120] bf16 = bf16(sum_j pre_in[j] * x[j])                      (ops::hc_pre)
///   pre  [m][4], post [m][4], comb [m][16] f32 = the new coefficients from x  (ops::hc_mixes, Sinkhorn 20 steps)
/// No allocation, no host synchronization (rule 7).
void hc_mixes_pre_rows(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                       const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb, void* workspace,
                       size_t workspace_bytes, cudaStream_t stream);

}  // namespace strata::ds41::kernels
