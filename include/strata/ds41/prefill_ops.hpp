// include/strata/ds41/prefill_ops.hpp - GPU operations that only batched prefill needs (M3).
//
// Prefill runs T tokens of a chunk through each layer at once. Most decode ops (ops.hpp) already take rows or work
// on flat arrays; these are the ones with no decode counterpart: per-row RoPE positions, embedding of many tokens,
// routing from precomputed logits, the attention index lists of a chunk, and the BF16 GEMMs (cuBLAS).
// All pointers are device pointers; all calls are ordered on the default stream, as the decode ops are.
#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace strata::ds41::prefill {

using bf16 = __nv_bfloat16;

/// h[r][c][d] = table[tokens[r]][d] for the 4 hyper-connection copies; tokens: device int32 [rows]
void embed_rows(const bf16* table, const int32_t* tokens, int rows, bf16* h);

/// RoPE (ops::rope) on `rows` rows of n_vec vectors each (row r is n_vec * stride values), row r at position
/// pos0 + r * pos_step. table: [positions][32] (cos, sin) pairs, as the engine's rope tables.
void rope_rows(bf16* v, int rows, int n_vec, int stride, const float* table, int pos0, int pos_step, bool inverse);

/// Routing from FP32 logits [rows][384] (K8's math after its dot products): s = sqrt(softplus(logit)), the 6
/// experts with the largest s + bias (ties: lower id) in that order, weight = s / (sum of the 6 s + 1e-20) * 1.5.
void route_rows(const float* logits, const float* bias, int rows, int32_t* ids, float* weights);

/// Attention index lists of a chunk for a single KV buffer (K13): the query at chunk row r (position p0 + r) gets
/// [128 window entries, oldest first: position p0 + r - 127 + j at kv row win_base + r + j, -1 before position 0]
/// then [n_idx - 128 entries copied from topk[r] (stride k_top), -1 when topk is null]. idx: [rows][n_idx].
/// The window rows of the KV buffer hold positions p0 - 127 .. p0 + rows - 1 from row win_base on.
void attn_index_rows(int rows, int p0, int win_base, const int32_t* topk, int k_top, int32_t* idx, int n_idx);

/// y [M][N] = x [M][K] times W [N][K] transposed, BF16 inputs, FP32 accumulation (cuBLAS). Output FP32 (yf) or BF16
/// (yb): ops::bf16_linear for every row, up to the summation order. A BF16 output goes through the FP32 scratch `tmp`
/// ([M][N]) and is rounded to nearest by our kernel: cuBLAS's own BF16 output rounds differently (rel. L2 2.7e-3).
void bf16_gemm(const bf16* x, const bf16* w, int64_t M, int64_t K, int64_t N, bf16* yb, float* yf, float* tmp = nullptr);

/// The grouped low-rank output projection (ops::wo_a_grouped) for M rows: o [M][8][4096] x wo_a [8][1024][4096] ->
/// y [M][8 * 1024], BF16, FP32 accumulation (cuBLAS, one strided batch), through the FP32 scratch tmp [M][8192].
void wo_a_grouped_rows(const bf16* o, const bf16* wo_a, int64_t M, bf16* y, float* tmp);

/// The decode window ring [128][512] holds position p at row p % 128. window_gather copies positions
/// p0 - n .. p0 - 1 (n <= 128, all >= 0) to dst rows 0 .. n - 1; window_scatter writes chunk rows src [rows][512]
/// (positions p0 .. p0 + rows - 1) into the ring: the last min(rows, 128) of them.
void window_gather(const bf16* ring, int p0, int n, bf16* dst);
void window_scatter(bf16* ring, const bf16* src, int p0, int rows);

/// Per row: nll[r] = logsumexp(logits[r]) - logits[r][target[r]] (FP32), target from device int32 [rows];
/// rows with target -1 get 0.
void nll_rows(const float* logits, int rows, int vocab, const int32_t* target, float* nll);

}  // namespace strata::ds41::prefill
