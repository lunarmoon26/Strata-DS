// include/strata/ds41/kernels/k5_indexer.hpp - task K5: indexer scores, candidate blocks and top-k on the GPU.
// Fixed interface; implementations live in src/ds41/kernels/k5_indexer.cu. Spec: ds41/tasks/K5.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::ds41::kernels {

/// One query against t compressed keys. scores[j] = bf16(sum_h bf16(relu(bf16(q_h . key_j)) * w_h)) as in
/// strata::ds41::ops::indexer_scores; where cand != nullptr and cand[j] == 0 the score is -inf.
///   q [32][128] bf16, keys [t][128] bf16, w [32] bf16, cand [t] uint8 or nullptr
/// Writes scores [t] (f32 holding the bf16 values, -inf where masked) and the k = min(k, t) best positions to
/// out_idx, sorted ascending, each plus `offset`. Ties: higher score first, then lower position.
void indexer_topk(const __nv_bfloat16* q, const __nv_bfloat16* keys, int64_t t, const __nv_bfloat16* w,
                  const uint8_t* cand, int k, int32_t offset, float* scores, int32_t* out_idx, cudaStream_t stream);

/// model.py select_candidate_blocks for one query: score each block of `block` positions by its maximum, always
/// keep the block holding position t-1, keep the best `topk_blocks` blocks (ties: lower block index), drop blocks
/// whose score is -inf. Writes cand [t] = 1 for kept positions, 0 otherwise.
void candidate_blocks(const float* scores, int64_t t, int topk_blocks, int block, uint8_t* cand, cudaStream_t stream);

/// Graph entry points. t = (int64_t(*pos_dev) + 1) / ratio. Require ratio > 0 and 0 <= t <= t_cap.
/// Allocate keys, scores and cand for t_cap entries. Allocate out_idx for min(max(k, 0), t_cap).
/// Only [0, t) scores/candidates and [0, min(max(k, 0), t)) IDs are written. t == 0 writes nothing.
/// Keep t_cap fixed for a graph. One full-context capacity needs only one capture.
/// Optional buckets use max(1, next_power_of_two(t)); each bucket needs its own graph.
/// Both score paths are captured when t_cap > 512. Device guards select the host-equivalent math.
/// Selection: the scratch-free single-CTA radix below 8192 live keys or when k >= t; from 8192 (t_cap >= 8192)
/// the partitioned radix of indexer_topk, whose metadata stays in the low halves of scores [0, t).
/// RTX 5090 Laptop, 262144 keys: 890 us single-CTA, 165 us partitioned.
void indexer_topk_device(const __nv_bfloat16* q, const __nv_bfloat16* keys, const int* pos_dev,
                         int ratio, int64_t t_cap, const __nv_bfloat16* w, const uint8_t* cand,
                         int k, int32_t offset, float* scores, int32_t* out_idx, cudaStream_t stream);
void candidate_blocks_device(const float* scores, const int* pos_dev, int ratio, int64_t t_cap,
                              int topk_blocks, int block, uint8_t* cand, cudaStream_t stream);

}  // namespace strata::ds41::kernels
