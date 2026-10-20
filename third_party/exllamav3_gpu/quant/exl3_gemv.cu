// Strata raw-pointer adapter for the upstream small-row GEMV kernel.
#include "exl3_gemv.cuh"
#include <cassert>
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"

namespace strata_exl3 {

// Each block reads one job. The branch is uniform before any barrier.
// Retain K10's two-CTA bound for the inlined K3 body.
__global__ __launch_bounds__(512, 2)
void gemv_mul1_jobs(const GemvJob* jobs)
{
    const GemvJob job = jobs[blockIdx.y];
    if (!job.B) return;
    if (job.codebook == 1) {
        if (job.bits == 2) exl3_gemv_kernel<2, true, 1, 0, 0, false>(jobs);
        else assert(false && "MCG adapter supports K2 only");
        return;
    }
    assert(job.codebook == 2 && "unknown EXL3 codebook");
    switch (job.bits)
    {
        case 3: exl3_gemv_kernel<3, true, 2, 0, 0, false>(jobs); break;
        case 1: exl3_gemv_kernel<1, true, 2, 0, 0, false>(jobs); break;
        case 2: exl3_gemv_kernel<2, true, 2, 0, 0, false>(jobs); break;
        case 4: exl3_gemv_kernel<4, true, 2, 0, 0, false>(jobs); break;
        case 5: exl3_gemv_kernel<5, true, 2, 0, 0, false>(jobs); break;
        case 6: exl3_gemv_kernel<6, true, 2, 0, 0, false>(jobs); break;
        default: assert(false && "K10: expected integer K1..K6");
    }
}

void gemv_mul1(const GemvJob* jobs, int count, int max_n, cudaStream_t stream)
{
    gemv_mul1_jobs<<<dim3(max_n / 32, count), 512, 0, stream>>>(jobs);
}

}  // namespace strata_exl3
