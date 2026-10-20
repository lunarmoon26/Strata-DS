// include/strata/ds41/kernels/k12_exl3_moe_prefill.hpp - task K12: routed experts for a prefill chunk (many tokens),
// rows grouped by expert, from EXL3 (mul1) weights.
// Fixed interface; implementation in src/ds41/kernels/k12_exl3_moe_prefill.cu (+ src/ds41/kernels/k12/).
// Spec: ds41/tasks/K12.md
#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"

namespace strata::ds41::kernels {

/// Workspace bytes that exl3_moe_prefill needs for calls with at most max_rows rows and max_groups groups.
size_t exl3_moe_prefill_workspace_bytes(int max_rows, int max_groups);

/// The caller sorts the (token, slot) assignments of a chunk by expert. Row r is one assignment: token tok[r],
/// routing weight w[r]. Group g (0 <= g < n_groups) covers rows [off[g], off[g + 1]) and uses expert experts[g].
/// For every row r of every group, with e = experts[g] and x_t = x[tok[r]], the math of K10 (exl3_moe_decode):
///   g = W1 x_t, u = W3 x_t (FP32); g = min(g, 10); u = clamp(u, -10, 10)
///   h = bf16(silu(g) * u * w[r]); h = FP8 block-32 quantize-dequantize(h); fp16(h)
///   out[tok[r]] += W2 h (FP32; the order of the additions is not specified)
/// x [T][5120] fp16 (already FP8-quantized by the caller). tok [rows] int32 and w [rows] f32 in device memory,
/// indexed by the row numbers in off. off: HOST array of n_groups + 1 non-decreasing entries; off[0] can be above 0
/// (the call covers a slice of a larger sorted list) and groups can be empty. experts: device array of n_groups.
/// out [T][5120] f32 is accumulated into; rows of tokens without an assignment in this call stay unchanged.
/// Limits: off[n_groups] - off[0] <= max_rows and n_groups <= max_groups of the workspace size.
/// One-time setup per device at the first call is allowed (kernel attributes, library handles). After that: no
/// allocation, no host synchronization (CUDA-graph capturable).
void exl3_moe_prefill(const __half* x, const int32_t* tok, const float* w, const int32_t* off, int n_groups,
                      const Exl3Expert* experts, float* out, void* workspace, size_t workspace_bytes,
                      cudaStream_t stream);

}  // namespace strata::ds41::kernels
