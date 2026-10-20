// include/strata/ds41/fp8_gemv.hpp - block-32 E4M3/E8M0 decode GEMV and its observable quantizer.
#pragma once

#include <cstdint>
#include <cstring>

namespace strata::ds41 {

// Device pointers: x [m,k] BF16 bits, w [n,k] E4M3, w_scale [ceil(n/32),k/32] E8M0,
// y [m,n] BF16 bits. 1 <= m <= 8, k > 0 divisible by 32, n > 0; buffers must not overlap.
// Compatibility wrapper: allocates [m,k] floats from CUDA's stream-ordered pool, calls the two
// entry points below, then frees the scratch. Asynchronous even on the default stream.
void fp8_block_gemv(const uint16_t* x, int m, int64_t k,
                    const uint8_t* w, const uint8_t* w_scale, int64_t n,
                    uint16_t* y, void* stream);

// Caller-owned device x_deq [m,k] floats: exactly decode(E4M3(x/s))*s, with K1 block-32 scales.
// Same m/k constraints as above. x and x_deq must not overlap.
// No allocation or synchronization, even on the default stream. Capturable on CUDA capture-capable streams.
void fp8_quantize_activation_f32(const uint16_t* x, int m, int64_t k,
                                 float* x_deq, void* stream);

// x_deq is the output of fp8_quantize_activation_f32; reuse it across weights with the same k.
// Same geometry and non-overlap rules as fp8_block_gemv. Only natural pointer alignment is required;
// 16-byte-aligned x_deq and w select vector loads. The caller keeps all buffers alive until completion.
// No allocation or synchronization, even on the default stream. Capturable on CUDA capture-capable streams.
void fp8_block_gemv_q(const float* x_deq, int m, int64_t k,
                      const uint8_t* w, const uint8_t* w_scale, int64_t n,
                      uint16_t* y, void* stream);

// The production quantizer, with exported xq [m,k] bytes and x_scale [m,k/32] FP32 powers of two.
// All pointers are device pointers; asynchronous on stream. Used for bitwise parity checks.
void fp8_quantize_activation(const uint16_t* x, int m, int64_t k,
                             uint8_t* xq, float* x_scale, void* stream);

// These scalar conversions also compile on the host so their edge cases can be checked without CUDA.
#if defined(__CUDACC__)
#define STRATA_DS41_HD __host__ __device__
#else
#define STRATA_DS41_HD
#endif
namespace detail {

// Shared dispatch policy lets the host test model the same lane ownership as the CUDA kernel.
inline int gemv_split_warps(int64_t n) { return n <= 2048 ? 4 : (n <= 8192 ? 2 : 1); }
inline int gemv_rows_per_group(int64_t n) { return n >= 1024 ? 2 : 1; }

STRATA_DS41_HD inline uint32_t float_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

STRATA_DS41_HD inline float from_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

STRATA_DS41_HD inline float decode_e4m3(uint8_t q) {
    const uint32_t a = q & 127u;
    const uint32_t sign = uint32_t(q & 128u) << 24;
    if (a == 127) return from_bits(sign | 0x7fc00000u);
    const float v = a < 8 ? float(a) * 0x1p-9f : from_bits((a + 960u) << 20);
    return from_bits(float_bits(v) | sign);
}

STRATA_DS41_HD inline uint8_t encode_e4m3(float x) {
    const uint32_t bits = float_bits(x);
    const uint8_t sign = uint8_t((bits >> 24) & 128u);
    const float a = from_bits(bits & 0x7fffffffu);
    if (a != a) return uint8_t(sign | 127u);
    if (a >= 448.0f) return uint8_t(sign | 126u);
    if (a < 0x1p-6f) {
        const float scaled = a * 512.0f;
        unsigned q = unsigned(scaled);
        const float rem = scaled - float(q);
        q += rem > 0.5f || (rem == 0.5f && (q & 1u));
        return uint8_t(sign | q);
    }
    const uint32_t u = bits & 0x7fffffffu;
    const uint32_t rounded = u + 0x7ffffu + ((u >> 20) & 1u);
    return uint8_t(sign | ((rounded >> 20) - 960u));
}

// amax >= 1e-4 makes amax/448 normal; integer exponent rounding preserves exact powers of two.
STRATA_DS41_HD inline float activation_scale(float amax) {
    const float a = (amax < 1e-4f ? 1e-4f : amax) * (1.0f / 448.0f);
    return from_bits((float_bits(a) + 0x7fffffu) & 0x7f800000u);
}

STRATA_DS41_HD inline float decode_e8m0(uint8_t q) {
    return from_bits(q == 0 ? 0x00400000u : uint32_t(q) << 23);
}

}  // namespace detail
#undef STRATA_DS41_HD
}  // namespace strata::ds41
