// src/ds41/kernels/k15_hc_prefill.cu - task K15 BASELINE: K7 (hc_mixes_pre), 8 tokens per call. Correct and slow at
// prefill sizes: 512 launches for a 4096-token sub-batch. Replace this file to win the task: ds41/tasks/K15.md.
#include "strata/ds41/kernels/k15_hc_prefill.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"

namespace strata::ds41::kernels {

size_t hc_mixes_pre_rows_workspace_bytes(int m) {
    (void) m;
    return 0;
}

void hc_mixes_pre_rows(const __nv_bfloat16* x, int m, const float* fn, const float* scale, const float* base,
                       const float* pre_in, __nv_bfloat16* y, float* pre, float* post, float* comb, void* workspace,
                       size_t workspace_bytes, cudaStream_t stream) {
    (void) workspace, (void) workspace_bytes;
    for (int t = 0; t < m; t += 8)
        hc_mixes_pre(x + (size_t) t * kHc * kDim, m - t < 8 ? m - t : 8, fn, scale, base, pre_in + t * kHc,
                     y + (size_t) t * kDim, pre + t * kHc, post + t * kHc, comb + t * kHc * kHc, stream);
}

}  // namespace strata::ds41::kernels
