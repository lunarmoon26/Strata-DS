// include/strata/ds41/ops.hpp - the M1 GPU operations for DeepSeek V4.1 Flash decode (one token, batch 1).
//
// M1 is correctness first: every operation restates one line of DeepSeek's model.py (as the prototype runs it
// with proto/torch_kernels.py), including where values are rounded to bf16. Speed comes in M2.
// All pointers are device pointers; all calls use the supplied stream (default 0).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::ds41::ops {

using bf16 = __nv_bfloat16;

/// h[c][d] = embed[token][d] for the 4 hyper-connection copies
void embed(const bf16* table, int token, bf16* h, cudaStream_t stream = 0);
/// RMSNorm: y = bf16(w * x * rsqrt(mean(x^2) + eps)), statistics in fp32; `rows` rows of n, one after the other
void rmsnorm(const bf16* x, const bf16* w, bf16* y, int n, float eps, int rows = 1, cudaStream_t stream = 0);
/// Hyper-connection coefficients from the stream x [4*5120]: pre[4], post[4], comb[4][4] (Sinkhorn)
void hc_mixes(const bf16* x, const float* fn, const float* scale, const float* base,
              float* pre, float* post, float* comb, float* scratch /* >= 25 floats */, cudaStream_t stream = 0);
/// y[d] = bf16(sum_j pre[j] * x[j][d]). Rows (prefill): x [rows][4][5120], pre [rows][4], y [rows][5120]
void hc_pre(const bf16* x, const float* pre, bf16* y, int rows = 1, cudaStream_t stream = 0);
/// y[k][d] = bf16(post[k] * out[d] + sum_j comb[j][k] * res[j][d]); y must not alias res.
/// Rows (prefill): out [rows][5120], res and y [rows][4][5120], post [rows][4], comb [rows][16]
void hc_post(const bf16* out, const bf16* res, const float* post, const float* comb, bf16* y, int rows = 1, cudaStream_t stream = 0);

/// FP8 linear as model.py linear(): activation FP8 block quantized (32, power-of-two scale), weight FP8 E4M3
/// with E8M0 32x32 block scales, FP32 accumulation, BF16 output. `act` is scratch for K floats.
void fp8_linear(const bf16* x, int64_t k, const uint8_t* w, const uint8_t* w_scale, int64_t n, bf16* y,
                float* act, cudaStream_t stream = 0);
/// y = x @ W^T with a BF16 weight, FP32 accumulation; output BF16 (yb) or FP32 (yf), whichever is non-null.
/// x_f32 non-null means the input is FP32 (the compressor and the router take x.float()).
void bf16_linear(const bf16* x, const float* x_f32, const bf16* w, int64_t k, int64_t n, bf16* yb, float* yf, cudaStream_t stream = 0);
/// bf16_linear for m rows (1..8) of x (BF16, k apart) into FP32 rows yf (n apart), the weight read once: every row's
/// output equals bf16_linear's for that row bit for bit (the same lanes, order and reduction)
void bf16_linear_rows(const bf16* x, int m, const bf16* w, int64_t k, int64_t n, float* yf, cudaStream_t stream = 0);

/// RoPE on the last 64 values of each of `n_vec` vectors spaced `stride` apart. cs holds 32 (cos, sin) pairs.
void rope(bf16* v, int n_vec, int stride, const float* cs, bool inverse, cudaStream_t stream = 0);
/// FP8 quantize-dequantize in place, blocks of 32, power-of-two scale (act_quant(..., inplace=True))
void act_quant_inplace(bf16* v, int n, cudaStream_t stream = 0);
/// FP4 E2M1 quantize-dequantize in place (fp4_act_quant): E4M3 scales (block 16) or E8M0 scales (block 32)
void fp4_quant_inplace(bf16* v, int n, int block, bool e4m3_scale, cudaStream_t stream = 0);

/// Decode attention for one query: q [64][512]; kv rows idx < 128 from `window`, others from compressed[idx-128];
/// idx -1 is empty. Softmax with a per-head sink in the denominator; output o [64][512].
void sparse_attn(const bf16* q, const bf16* window, const bf16* compressed, const int32_t* idx, int n_idx,
                 const float* sink, float scale, bf16* o, cudaStream_t stream = 0);
/// The 128 sliding-window entries of the attention index list at position pos, oldest slot first; -1 = empty
void window_index(int pos, int32_t* idx, cudaStream_t stream = 0);
/// Grouped low-rank output projection: o [8][4096] x wo_a [8][1024][4096] -> y [8*1024] (BF16, FP32 accumulate)
void wo_a_grouped(const bf16* o, const bf16* wo_a, bf16* y, cudaStream_t stream = 0);

/// Indexer scores for one query against t compressed keys: score[t] = bf16(sum_h bf16(relu(bf16(q_h.k_t)) * w_h))
/// q [32][128], keys [t][128], w [32] (already scaled, bf16). Output FP32 holding the bf16 values.
void indexer_scores(const bf16* q, const bf16* keys, int64_t t, const bf16* w, float* score, cudaStream_t stream = 0);
/// w[h] = bf16(float(wp[h]) * scale)
void scale_bf16(const bf16* wp, float scale, bf16* w, int n, cudaStream_t stream = 0);

/// Shared expert / SwiGLU core: h = bf16(silu(min(g, lim)) * clamp(u, -lim, lim)) from bf16 inputs
void swiglu(const bf16* g, const bf16* u, float lim, bf16* h, int n, cudaStream_t stream = 0);
/// y = bf16(a_f32 + float(b)) (MoE output: routed fp32 sum plus the shared expert)
void add_f32_bf16(const float* a, const bf16* b, bf16* y, int n, cudaStream_t stream = 0);
/// x_half = fp16(x) after the FP8 activation quantization (the routed-expert input in the prototype)
void to_half_fp8q(const bf16* x, uint16_t* x_half, int n, cudaStream_t stream = 0);
/// Strict resident split: map six router IDs through the layer's 384-entry
/// residency table. A missing/invalid route is a device error, never a CPU miss.
void resident_routes(const int32_t* ids, const int32_t* residency, int32_t* slots, cudaStream_t stream = 0);

/// Compressor pooling of finished groups: out[g][d] = bf16(sum_r kv[g][r][d] * softmax_r(score[g][r][d]))
void compress_pool(const float* kv_state, const float* score_state, int ratio, bf16* out, int groups = 1, cudaStream_t stream = 0);

/// Engram gate (model.py Engram.forward): h[c] += gate_c * value, gate from the normalized dot of h[c] and key[c]
/// Rows (prefill): h [rows][4][5120], kv [rows][5][5120]
void engram_apply(bf16* h, const bf16* kv /* [5*5120]: 4 keys then value */, const bf16* qw, const bf16* kw, float eps,
                  int rows = 1, cudaStream_t stream = 0);
void engram_apply_f32(bf16* h, const bf16* kv, const float* qw, const float* kw, float eps,
                      int rows = 1, cudaStream_t stream = 0);
/// Engram rows: FP8 E4M3 [rows][256] times E8M0 [rows][8] -> BF16 [rows][256]
void engram_dequant(const uint8_t* w, const uint8_t* s, int rows, bf16* out, cudaStream_t stream = 0);

/// Fixed graph parameters. The caller keeps device scalars alive through all replays.
void embed_device(const bf16* table, const int* token_dev, bf16* h, cudaStream_t stream = 0);
/// Identical embedding for 1..8 device token IDs into [rows][kHc][kDim].
void embed_device_rows(const bf16* table, const int* tokens_dev, bf16* h, int rows, cudaStream_t stream = 0);
void window_index_device(const int* pos_dev, int32_t* idx, cudaStream_t stream = 0);
/// cs_table contains rows of kRopeDim floats. Read row *pos_dev + offset.
/// A negative row does no work (the first incomplete compressor group).
void rope_device(bf16* v, int n_vec, int stride, const float* cs_table, const int* pos_dev,
                 int offset, bool inverse, cudaStream_t stream = 0);
/// Copy row_bytes bytes to dst + ((*pos_dev / div) % mod) * row_bytes.
/// pos must be nonnegative; div and mod must be positive. Source and destination must not overlap.
void row_copy_device(void* dst, const void* src, size_t row_bytes, const int* pos_dev,
                     int div, int mod, cudaStream_t stream = 0);
/// Reduce kVocab FP32 logits to one device int. Ties select the lowest index.
/// Match std::max_element: a NaN at index zero wins; other NaNs are ignored.
void argmax_logits(const float* logits, int* token_dev, cudaStream_t stream = 0);

}  // namespace strata::ds41::ops
