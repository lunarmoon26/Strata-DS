#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata_exl3 {

// One projection, prepared on the device. No descriptor readback is needed.
struct ReconstructJob {
    const uint16_t* packed;
    int bits;
};

// job is a DEVICE pointer. Output is row-major [k][n] fp16 in the Hadamard
// basis, without scales or Hadamards. k % 16 == 0 and n % 128 == 0.
void reconstruct_mul1(half* unpacked, const ReconstructJob* job,
                       int k, int n, cudaStream_t stream);

}  // namespace strata_exl3
