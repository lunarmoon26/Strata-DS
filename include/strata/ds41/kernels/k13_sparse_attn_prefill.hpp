// include/strata/ds41/kernels/k13_sparse_attn_prefill.hpp - task K13: sparse attention for a prefill chunk.
// Fixed interface; implementations live in src/ds41/kernels/k13_sparse_attn_prefill.cu. Spec: ds41/tasks/K13.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::ds41::kernels {

/// m queries (1..16384), each attending to its own list of n_idx rows of one KV buffer:
///   q   [m][64][512] bf16        kv [n_kv][512] bf16        idx [m][n_idx] int32: -1 empty, else a row of kv
///   sink [64] f32 (added to each head's softmax denominator only)          o [m][64][512] bf16
/// Same math as strata::ds41::ops::sparse_attn for each query (every listed row read from kv). n_idx <= 1024.
/// No allocation, no host synchronization (rule 7).
void sparse_attn_prefill(const __nv_bfloat16* q, const __nv_bfloat16* kv, const int32_t* idx, int m, int n_idx,
                         const float* sink, float scale, __nv_bfloat16* o, cudaStream_t stream);

}  // namespace strata::ds41::kernels
