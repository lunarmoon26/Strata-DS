#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::ds41::kernels::k12 {

constexpr int H = 5120;
constexpr int F = 2304;
constexpr int ROW_TILE = 512;
constexpr float HAD_SCALE = 0.088388347648f;
constexpr size_t MATRIX_ELEMENTS = size_t(H) * F;
constexpr size_t BLAS_BYTES = 4ull << 20;

// The three reconstructed matrices survive all row tiles of one expert. All
// other buffers are reused in stream order. No scratch depends on group count.
// Include 255 bytes of slack so even an unaligned caller pointer can be used.
struct Layout {
    size_t trellis = 0, matrices, input, gu, down_input, down, blas, end;
    int rows;

    explicit Layout(int max_rows) : rows(max_rows < ROW_TILE ? max_rows : ROW_TILE) {
        if (max_rows < 1) throw std::invalid_argument("K12: layout needs rows");
        matrices = 256;  // three reconstruction jobs, padded to 256 bytes
        input = matrices + 3 * MATRIX_ELEMENTS * sizeof(uint16_t);
        gu = input + size_t(2) * rows * H * sizeof(uint16_t);
        down_input = gu + size_t(2) * rows * F * sizeof(float);
        down = down_input + size_t(rows) * F * sizeof(uint16_t);
        blas = down + size_t(rows) * H * sizeof(float);
        end = blas + BLAS_BYTES;
    }
    size_t bytes() const { return end + 255; }
};

}  // namespace strata::ds41::kernels::k12
