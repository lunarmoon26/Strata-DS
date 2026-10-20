// include/strata/ds41/kernels/k3_sparse_attn.hpp - task K3: decode / verify-window sparse attention.
// Fixed interface; implementations live in src/ds41/kernels/k3_sparse_attn.cu. Spec: ds41/tasks/K3.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::ds41::kernels {

/// m queries (1..8), each attending to its own list of n_idx KV rows.
///   q      [m][64][512] bf16          window [128][512] bf16        comp [n_comp][512] bf16
///   idx    [m][n_idx] int32: -1 empty; j < 128 -> window[j]; j >= 128 -> comp[j - 128]
///   sink   [64] f32 (added to each head's softmax denominator only)       o [m][64][512] bf16
/// Same math as strata::ds41::ops::sparse_attn applied to each query. n_idx <= 1024.
void sparse_attn_decode(const __nv_bfloat16* q, const __nv_bfloat16* window, const __nv_bfloat16* comp,
                        const int32_t* idx, int m, int n_idx, const float* sink, float scale,
                        __nv_bfloat16* o, cudaStream_t stream);

/// One decode query. Read t >= 0 from device memory at each execution.
/// n_idx = kWindow + min(kIndexTopK, *t_dev). idx has capacity kWindow + kIndexTopK.
/// Keep pointers and launch dimensions fixed across replays. Unused indices are never read.
void sparse_attn_decode_device(const __nv_bfloat16* q, const __nv_bfloat16* window,
                               const __nv_bfloat16* comp, const int32_t* idx, const int* t_dev,
                               const float* sink, float scale, __nv_bfloat16* o, cudaStream_t stream);

}  // namespace strata::ds41::kernels
