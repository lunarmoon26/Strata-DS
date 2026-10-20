#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata_exl3 {

// One row, already input-Hadamard transformed. C is FP32 before output Hadamard.
// Stored on device; B == nullptr skips a slot without touching A or C.
struct GemvJob {
    const half* A;
    const uint16_t* B;
    float* C;
    int k;
    int n;
    int bits;
    int codebook = 2;
};

// K10: integer K1..K6 mul1 or K2 MCG; one row per job. max_n sizes grid.x;
// each job's n must be a multiple of 128 and no larger than max_n.
void gemv_mul1(const GemvJob* jobs, int count, int max_n, cudaStream_t stream);
// One shared weight matrix and 1..8 contiguous activation/output rows. The MMA
// lanes compute all rows using each decoded trellis tile once, preserving the
// one-row fold/reduction order. No FP16 weight reconstruction is needed.
void gemv_mul1_rows(const GemvJob* job, int rows, int n, int bits, cudaStream_t stream);

}  // namespace strata_exl3
