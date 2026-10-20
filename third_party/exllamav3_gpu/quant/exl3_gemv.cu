// Strata raw-pointer adapter for the upstream small-row GEMV kernel.
#include "exl3_gemv.cuh"
#include <cassert>
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"

namespace strata_exl3 {

template<int bits> __global__ __launch_bounds__(512,2)
void gemv_mul1_shared_rows(const GemvJob* job,int rows) {
    assert(job->codebook==2);
    exl3_gemv_kernel<bits,true,2,1,0,false>(job,rows);
}
void gemv_mul1_rows(const GemvJob* job,int rows,int n,int bits,cudaStream_t stream) {
    assert(rows>=1 && rows<=8 && n%128==0);
    const dim3 grid(n/32,1);
    switch(bits) {
        case 1: gemv_mul1_shared_rows<1><<<grid,512,0,stream>>>(job,rows);break;
        case 2: gemv_mul1_shared_rows<2><<<grid,512,0,stream>>>(job,rows);break;
        case 3: gemv_mul1_shared_rows<3><<<grid,512,0,stream>>>(job,rows);break;
        case 4: gemv_mul1_shared_rows<4><<<grid,512,0,stream>>>(job,rows);break;
        case 5: gemv_mul1_shared_rows<5><<<grid,512,0,stream>>>(job,rows);break;
        case 6: gemv_mul1_shared_rows<6><<<grid,512,0,stream>>>(job,rows);break;
        default: assert(false && "unsupported shared-row EXL3 rate");
    }
}

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
