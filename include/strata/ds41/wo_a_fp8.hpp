// include/strata/ds41/wo_a_fp8.hpp - the attention output projection wo_a kept as FP8 (half the bytes of BF16).
//
// The pack stores attn.wo_a as DeepSeek publishes it: FP8 E4M3 [8 * 1024][4096] with E8M0 scales per 32 x 32
// block [256][128]. inference/convert.py dequantizes it to BF16 (w * scale, rounded to BF16). That product is exact
// (the scale is a power of two), so these functions give the same numbers as the BF16 path, bit for bit:
//   wo_a_grouped_fp8 = ops::wo_a_grouped on the dequantized weight (decode, one token), and
//   dequant_wo_a     = the dequantized BF16 weight (prefill uses it with prefill::wo_a_grouped_rows).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::ds41 {

/// y [8 * 1024] = per group g: wo_a[g] [1024][4096] x o[g * 4096 ...]; o [64 * 512], y BF16.
void wo_a_grouped_fp8(const __nv_bfloat16* o, const uint8_t* w, const uint8_t* scale, __nv_bfloat16* y,
                      cudaStream_t stream = 0);
/// wo_a_grouped_fp8 for m rows (1..8): o [m][64 * 512], y [m][8 * 1024], the weight read once; every row equals
/// wo_a_grouped_fp8 on that row bit for bit (the verify window)
void wo_a_grouped_fp8_rows(const __nv_bfloat16* o, int m, const uint8_t* w, const uint8_t* scale, __nv_bfloat16* y,
                           cudaStream_t stream = 0);
/// out [8 * 1024][4096] BF16 = w * scale, rounded to BF16
void dequant_wo_a(const uint8_t* w, const uint8_t* scale, __nv_bfloat16* out, cudaStream_t stream = 0);

}  // namespace strata::ds41
