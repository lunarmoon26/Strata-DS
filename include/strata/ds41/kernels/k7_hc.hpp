// include/strata/ds41/kernels/k7_hc.hpp - task K7: hyper-connection mix coefficients plus the collapsed input.
// Fixed interface; implementations live in src/ds41/kernels/k7_hc.cu. Spec: ds41/tasks/K7.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace strata::ds41::kernels {

/// Allocate maximum scratch for the current device. Call before capture.
/// Idempotent. Scratch lives until process exit. Calls/replays on one device must not overlap.
void hc_init();

/// For m tokens (1..8):
///   x [m][4][5120] bf16 (the hc stream), fn [24][20480] f32, scale [3] f32, base [24] f32, pre_in [m][4] f32
///   y    [m][5120] bf16 = bf16(sum_j pre_in[j] * x[j])                      (ops::hc_pre)
///   pre  [m][4], post [m][4], comb [m][16] f32 = the new coefficients from x  (ops::hc_mixes, Sinkhorn 20 steps)
void hc_mixes_pre(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                  const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb, cudaStream_t stream);
/// hc_mixes_pre for one token with the coefficients off the stream: the partial sums and y on `stream`; pre, post
/// and comb on `side` (forked with `fork` after the partial sums), complete at `done`. Wait for `done` before reading
/// pre, post or comb, and before the next hc_mixes_pre call (the partial sums' scratch is shared). The same
/// arithmetic as hc_mixes_pre.
void hc_mixes_pre_split(const __nv_bfloat16* x, const float* fn, const float* scale, const float* base,
                        const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb,
                        cudaStream_t stream, cudaStream_t side, cudaEvent_t fork, cudaEvent_t done);

}  // namespace strata::ds41::kernels
