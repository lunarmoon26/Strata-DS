// include/strata/ds41/kernels/k10_exl3_moe.hpp - task K10: GPU routed experts, EXL3 mul1 or resident-split MCG K2.
// Fixed interface; implementation in src/ds41/kernels/k10_exl3_moe.cu (+ src/ds41/kernels/k10/). Spec: ds41/tasks/K10.md
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::ds41::kernels {

/// One EXL3 projection in device memory, exactly as the pack stores it (tools/ds41/pack.py): trellis is
/// [k/16][n/16][tile_w] uint16 (tile_w = 16*bits, or 16*bits+8 for a half-integer rate); suh [k] and svh [n] fp16.
struct Exl3Proj {
    const uint16_t* trellis;
    const __half* suh;
    const __half* svh;
    int k, n, tile_w;
    int codebook = 2;  ///< 2: original mul1; 1: MCG (resident split K2 only)
};
/// One routed expert: w1 (gate, 5120 -> 2304), w3 (up, 5120 -> 2304), w2 (down, 2304 -> 5120).
struct Exl3Expert {
    Exl3Proj w1, w3, w2;
    bool native_qdq = true;  ///< false: published full-EXL3 FP16 activation semantics
};

/// m tokens (1..8), topk slots each. For every token t and slot j with sel[t][j] >= 0, e = experts[sel[t][j]]:
///   g = W1 x_t, u = W3 x_t (FP32; exllamav3 LinearEXL3 semantics); g = min(g, 10); u = clamp(u, -10, 10)
///   h = bf16(silu(g) * u * w[t][j]); h = FP8 block-32 quantize-dequantize(h) (as ops::act_quant_inplace); fp16(h)
///   out[t] += W2 h (FP32)
/// With native_qdq=false, h is rounded directly to FP16 without BF16/FP8 QDQ;
/// the caller also supplies ordinary FP16 input for that published EXL3 profile.
/// x [m][5120] fp16 (already FP8-quantized by the caller), sel [m][topk] int32, w [m][topk] f32, experts: device array.
/// out [m][5120] f32 is accumulated into. workspace: device memory of workspace_bytes (at least 64 MiB).
/// No allocation, no synchronization (CUDA-graph capturable).
void exl3_moe_decode(const __half* x, int m, const int32_t* sel, const float* w, int topk, const Exl3Expert* experts,
                     float* out, void* workspace, size_t workspace_bytes, cudaStream_t stream);

}  // namespace strata::ds41::kernels
