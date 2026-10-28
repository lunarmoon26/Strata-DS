// Strata raw-pointer adapter for the upstream small-row GEMV kernel.
#include "exl3_gemv.cuh"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"

namespace strata_exl3 {

template<int bits> __global__ __launch_bounds__(512,2)
void gemv_mul1_shared_rows(const GemvJob* job,int rows) {
    assert(job[blockIdx.y].codebook==2);
    exl3_gemv_kernel<bits,true,2,1,0,false>(job,rows);
}
template<int bits> __global__ __launch_bounds__(512,2)
void gemv_mul1_register_rows(const GemvJob* job,int rows) {
    exl3_gemv_kernel<bits,true,2,1,0,false,false,true>(job,rows);
}
template<int bits> __global__ __launch_bounds__(512,1)
void gemv_mul1_relaxed_rows(const GemvJob* job,int rows) {
    exl3_gemv_kernel<bits,true,2,1,0,false>(job,rows);
}
void gemv_mul1_rows(const GemvJob* job,int rows,int n,int bits,cudaStream_t stream,int count) {
    assert(rows>=1 && rows<=8 && n%128==0);
    const dim3 grid(n/32,count);
    static const bool registers=[] {
        const char* value=std::getenv("DS41_EXPERIMENTAL_REG56");
        if(!value || !std::strcmp(value,"0")) return false;
        if(std::strcmp(value,"1")) throw std::invalid_argument("DS41_EXPERIMENTAL_REG56 must be 0 or 1");
        return true;
    }();
    static const bool relaxed=[] {
        const char* value=std::getenv("DS41_EXPERIMENTAL_DENSE_OCCUPANCY1");
        if(!value || !std::strcmp(value,"0")) return false;
        if(std::strcmp(value,"1")) throw std::invalid_argument("DS41_EXPERIMENTAL_DENSE_OCCUPANCY1 must be 0 or 1");
        return true;
    }();
    if(relaxed && registers) throw std::invalid_argument("dense occupancy and REG56 experiments must be isolated");
    if(relaxed && bits==5) { gemv_mul1_relaxed_rows<5><<<grid,512,0,stream>>>(job,rows);return; }
    if(relaxed && bits==6) { gemv_mul1_relaxed_rows<6><<<grid,512,0,stream>>>(job,rows);return; }
    if(registers && bits==5) { gemv_mul1_register_rows<5><<<grid,512,0,stream>>>(job,rows);return; }
    if(registers && bits==6) { gemv_mul1_register_rows<6><<<grid,512,0,stream>>>(job,rows);return; }
    switch(bits) {
        case 1: gemv_mul1_shared_rows<1><<<grid,512,0,stream>>>(job,rows);break;
        case 2: gemv_mul1_shared_rows<2><<<grid,512,0,stream>>>(job,rows);break;
        case 3: gemv_mul1_shared_rows<3><<<grid,512,0,stream>>>(job,rows);break;
        case 4: gemv_mul1_shared_rows<4><<<grid,512,0,stream>>>(job,rows);break;
        case 5: gemv_mul1_shared_rows<5><<<grid,512,0,stream>>>(job,rows);break;
        case 6: gemv_mul1_shared_rows<6><<<grid,512,0,stream>>>(job,rows);break;
        case 7: gemv_mul1_shared_rows<7><<<grid,512,0,stream>>>(job,rows);break;
        case 8: gemv_mul1_shared_rows<8><<<grid,512,0,stream>>>(job,rows);break;
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
        // Dense K7/K8 use the separate host-dispatched shared-row entry points.
        // Do not inflate this mixed expert kernel's static shared memory.
        default: assert(false && "K10 experts: expected integer K1..K6");
    }
}

void gemv_mul1(const GemvJob* jobs, int count, int max_n, cudaStream_t stream)
{
    gemv_mul1_jobs<<<dim3(max_n / 32, count), 512, 0, stream>>>(jobs);
}

__device__ bool same_weights(const GemvJob& a, const GemvJob& b) {
    return a.B == b.B && a.k == b.k && a.n == b.n &&
           a.bits == b.bits && a.codebook == b.codebook;
}

__global__ void group_jobs(const GemvJob* jobs, int count, int* groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int* group = groups + 9 * i;
    group[0] = 0;
    const GemvJob job = jobs[i];
    if (!job.B) return;
    int rank = 0;
    for (int j = 0; j < i; ++j) rank += same_weights(job, jobs[j]);
    if (rank % 8) return;
    int rows = 0;
    for (int j = i; j < count && rows < 8; ++j)
        if (same_weights(job, jobs[j])) group[1 + rows++] = j;
    group[0] = rows;
}

__global__ __launch_bounds__(512, 2)
void gemv_grouped_jobs(const GemvJob* jobs, const int* groups) {
    const int* group = groups + 9 * blockIdx.y;
    const int rows = group[0];
    if (!rows) return;
    const GemvJob job = jobs[blockIdx.y];
    if (job.codebook == 1) {
        assert(job.bits == 2);
        exl3_gemv_kernel<2,true,1,1,0,false,false,false,true>(jobs,rows,group+1);
        return;
    }
    assert(job.codebook == 2);
    switch (job.bits) {
        case 1: exl3_gemv_kernel<1,true,2,1,0,false,false,false,true>(jobs,rows,group+1); break;
        case 2: exl3_gemv_kernel<2,true,2,1,0,false,false,false,true>(jobs,rows,group+1); break;
        case 3: exl3_gemv_kernel<3,true,2,1,0,false,false,false,true>(jobs,rows,group+1); break;
        case 4: exl3_gemv_kernel<4,true,2,1,0,false,false,false,true>(jobs,rows,group+1); break;
        case 5: exl3_gemv_kernel<5,true,2,1,0,false,false,false,true>(jobs,rows,group+1); break;
        case 6: exl3_gemv_kernel<6,true,2,1,0,false,false,false,true>(jobs,rows,group+1); break;
        default: assert(false && "grouped experts: expected integer K1..K6");
    }
}

void gemv_mul1_grouped(const GemvJob* jobs, int count, int max_n, int* groups, cudaStream_t stream) {
    group_jobs<<<(count+127)/128,128,0,stream>>>(jobs,count,groups);
    gemv_grouped_jobs<<<dim3(max_n/32,count),512,16*8*32*sizeof(float),stream>>>(jobs,groups);
}

}  // namespace strata_exl3
