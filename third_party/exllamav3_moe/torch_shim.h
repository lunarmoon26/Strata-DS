// third_party/exllamav3_moe/torch_shim.h - the three PyTorch names moe_mul1 uses, without PyTorch.
//
// moe_mul1 needs only an IEEE fp16 storage type that converts to float (c10::Half / at::Half) and an
// argument check that throws (TORCH_CHECK). Strata-DS builds without PyTorch, so this header provides both.
// The tensor entry points of moe_mul1 stay behind EXL3_MOE_WITH_TORCH; Strata-DS uses the raw-pointer ones.
#pragma once

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef EXL3_MOE_WITH_TORCH

namespace c10 {

/// IEEE 754 binary16 storage. Conversion to float is exact (every fp16 value is a float value).
struct Half {
    uint16_t x = 0;
    struct from_bits_t {};
    static constexpr from_bits_t from_bits() { return from_bits_t(); }
    Half() = default;
    constexpr Half(uint16_t bits, from_bits_t) : x(bits) {}

    operator float() const {
        const uint32_t sign = (uint32_t) (x & 0x8000u) << 16;
        const uint32_t exp = (x >> 10) & 0x1Fu;
        uint32_t man = x & 0x3FFu;
        uint32_t bits;
        if (exp == 0x1Fu) {
            bits = sign | 0x7F800000u | (man << 13);                 // inf or nan
        } else if (exp != 0) {
            bits = sign | ((exp + 112u) << 23) | (man << 13);        // normal: rebias 15 -> 127
        } else if (man == 0) {
            bits = sign;                                             // signed zero
        } else {                                                     // subnormal: normalize
            int e = -1;
            do { man <<= 1; ++e; } while ((man & 0x400u) == 0);
            bits = sign | ((uint32_t) (112 - e) << 23) | ((man & 0x3FFu) << 13);
        }
        float f;
        std::memcpy(&f, &bits, sizeof f);
        return f;
    }
};

}  // namespace c10

namespace at {
using Half = c10::Half;
}  // namespace at

namespace exl3_shim {
inline void append(std::ostringstream&) {}
template <typename T, typename... R>
void append(std::ostringstream& s, const T& v, const R&... r) { s << v; append(s, r...); }
template <typename... A>
std::string message(const A&... a) { std::ostringstream s; append(s, a...); return s.str(); }
}  // namespace exl3_shim

#define TORCH_CHECK(cond, ...) \
    do { if (!(cond)) throw std::runtime_error(exl3_shim::message("moe_mul1: ", __VA_ARGS__)); } while (0)

#endif  // EXL3_MOE_WITH_TORCH
