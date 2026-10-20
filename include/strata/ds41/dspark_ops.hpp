// include/strata/ds41/dspark_ops.hpp - the GPU ops of DSpark (DeepSeek's MTP drafter) that the backbone has no form
// of. DSpark (model.py DSparkBlock, Transformer.forward_spec) reads the attention input of layers 37-39, runs three
// blocks over a block of 5 rows (the token after the committed ones, then 4 noise tokens), and draws the 5 drafts with
// a Markov bias head; a confidence head scores each draft. Everything else reuses the backbone's ops.
// Drafts only steer which tokens a verify window checks: these ops never change the output.
#pragma once

#include "strata/ds41/ops.hpp"

#include <cstdint>

namespace strata::ds41::dspark {

using ops::bf16;

constexpr int kBlock = 5;          ///< draft rows per pass (dspark_block_size)
constexpr int kStages = 3;         ///< DSpark blocks (n_mtp_layers)
constexpr int kExperts = 128;      ///< routed experts per stage (dspark_n_routed_experts)
constexpr int kTopK = 3;           ///< (dspark_num_experts_per_tok)
constexpr int kTargets = 3;        ///< target layers 37, 38, 39 (dspark_target_layer_ids)
constexpr int kFirstTarget = 37;
constexpr int kMarkovRank = 256;
constexpr int kNoiseToken = 128799;

/// main_hiddens: out[r * out_stride + d] = the mean of the kHc copies of row r of h ([m][kHc][kDim]), rounded to BF16
/// (torch's h.mean(dim=2) on BF16: FP32 sum, one rounding)
void hc_mean_rows(const bf16* h, int m, bf16* out, int64_t out_stride, cudaStream_t stream = 0);

/// Gate.forward for n_exp experts and topk picks (DSpark: 128 and 3): logits in FP32, s = sqrt(softplus(logit)),
/// picks by s + bias (ties: lower id), weight = s / (sum of the picks' s + 1e-20) * 1.5.
///   x [m][kDim], w [n_exp][kDim], bias [n_exp] -> ids [m][topk], weights [m][topk]. n_exp <= 256, topk <= 8.
void router_small(const bf16* x, int m, const bf16* w, const float* bias, int n_exp, int topk, int32_t* ids,
                  float* weights, cudaStream_t stream = 0);

/// DSparkMarkovHead for one draft: e = embed[*token] (kMarkovRank wide, copied to embed_out), then
/// logits[v] += head[v] . e in FP32 for every v < n_vocab
void markov_bias(const int* token, const bf16* embed, const bf16* head, int64_t n_vocab, float* logits,
                 bf16* embed_out, cudaStream_t stream = 0);

/// DSparkConfidenceHead: out[r] = w . [x[r] (kDim), e[r] (kMarkovRank)] in FP32 (the logit; sigmoid on the host)
void confidence_rows(const bf16* x, const bf16* e, const bf16* w, int m, float* out, cudaStream_t stream = 0);

/// get_dspark_topk_idxs as sparse_attn_decode reads it: per row [kWindow + kBlock] ids; j < kWindow is ring slot j when
/// j <= *start_pos (the slots written so far), else -1; then kWindow + i for the block's rows i < kBlock
void block_index(const int* start_pos, int32_t* idx, cudaStream_t stream = 0);

}  // namespace strata::ds41::dspark
