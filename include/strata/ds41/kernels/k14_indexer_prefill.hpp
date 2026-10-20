// include/strata/ds41/kernels/k14_indexer_prefill.hpp - task K14: the indexer (scores, top-k, candidate blocks) for
// a prefill chunk. Fixed interface; implementations live in src/ds41/kernels/k14_indexer_prefill.cu.
// Spec: ds41/tasks/K14.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::ds41::kernels {

/// Workspace bytes indexer_topk_prefill needs for m queries that see at most t_max keys.
size_t indexer_topk_prefill_workspace_bytes(int m, int64_t t_max);

/// m queries (1..16384). Query i sees the keys [0, t_i) with t_i = (pos0 + i + 1) / ratio (the indexer's causal
/// rule: a compressed key exists once its group of `ratio` positions is complete). For every query, K5's
/// indexer_topk with k_i = min(k, t_i):
///   scores_i[j] = bf16(sum_h bf16(relu(bf16(q_ih . key_j)) * w_ih)) (ops::indexer_scores), -inf where cand row i
///   (cand + i * cand_stride) is 0 when cand is non-null;
///   out_idx[i][0 .. k_i) = the k_i best positions (ties: higher score, then lower position), ascending, plus offset;
///   out_idx[i][k_i .. k) = -1.
/// cand_out non-null (the candidate layer): cand_out + i * cand_stride gets K5's candidate_blocks(scores_i, t_i,
/// topk_blocks, block) over [0, t_i); the rest of the row is not written.
///   q [m][32][128] bf16, keys [>= t_max][128] bf16, w [m][32] bf16 (already scaled), out_idx [m][k] int32.
/// No allocation, no host synchronization (rule 7).
void indexer_topk_prefill(const __nv_bfloat16* q, const __nv_bfloat16* keys, const __nv_bfloat16* w, int m, int pos0,
                          int ratio, const uint8_t* cand, uint8_t* cand_out, int64_t cand_stride, int k, int32_t offset,
                          int topk_blocks, int block, int32_t* out_idx, void* workspace, size_t workspace_bytes,
                          cudaStream_t stream);

}  // namespace strata::ds41::kernels
