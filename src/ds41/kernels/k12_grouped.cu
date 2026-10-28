// Native resident grouped EXL3 prefill. Packed pretrained bytes never change.
// Tiled MMA loop adapted from Diffbot xmoe_do/do_gemm.cu, copyright 2026 Diffbot,
// MIT: third_party/diffbot_exl3/LICENSE. Inspected source revision:
// diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000@58799d945d0f836fdb575e3a39a662b0a4d1543c
// Adaptation: raw descriptors, independent K1..K6 projections, FP32 outputs,
// existing K12 activation rounding, GPU job planning and ordered reduction.
#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "../../../third_party/exllamav3_gpu/util.cuh"
#include "../../../third_party/exllamav3_gpu/ptx.cuh"
#include "../../../third_party/exllamav3_gpu/quant/exl3_dq.cuh"
#include "../../../third_party/exllamav3_gpu/quant/hadamard_inner.cuh"
#include "k12/workspace.hpp"
#include <cassert>
#include <string>

namespace strata::ds41::kernels {
namespace grouped {
using namespace k12;
constexpr int M=64,SK=128,STRIDE=SK+8;
__device__ inline void check_proj(const Exl3Proj& p,int k,int n) {
    assert(p.k==k && p.n==n && p.codebook==2 && p.shard_count==1);
    assert(p.tile_w%16==0 && p.tile_w>=16 && p.tile_w<=96);
    assert(p.trellis && p.suh && p.svh);
}
__device__ inline float round_pow2(float a) {
    return __uint_as_float((__float_as_uint(a)+0x007fffffu)&0x7f800000u);
}
struct Job { int expert,first,rows; };
struct Buffers {
    int *counts,*offsets,*cursor,*slots,*reverse,*labels;
    Job* jobs;
    half *input,*hidden;
    float *gu,*down;
    size_t bytes=0;
    int max_jobs;
    Buffers(void* base,int tokens,int topk,int groups,int capacity) {
        const size_t rows=size_t(tokens)*topk;
        max_jobs=(capacity+M-1)/M+groups;
        auto take=[&](size_t n,size_t width) {
            bytes=(bytes+255)&~size_t(255);
            void* p=base?static_cast<char*>(base)+bytes:nullptr;bytes+=n*width;return p;
        };
        counts=static_cast<int*>(take(groups,sizeof(int)));
        offsets=static_cast<int*>(take(groups+1,sizeof(int)));
        cursor=static_cast<int*>(take(groups,sizeof(int)));
        slots=static_cast<int*>(take(capacity,sizeof(int)));
        reverse=static_cast<int*>(take(rows,sizeof(int)));
        labels=static_cast<int*>(take(capacity,sizeof(int)));
        jobs=static_cast<Job*>(take(max_jobs,sizeof(Job)));
        input=static_cast<half*>(take(size_t(2)*capacity*H,sizeof(half)));
        // Gate/up inputs die before the down projection. Their combined FP16
        // arena exactly fits the FP32 down output; stream order owns reuse.
        down=reinterpret_cast<float*>(input);
        gu=static_cast<float*>(take(size_t(2)*capacity*F,sizeof(float)));
        hidden=static_cast<half*>(take(size_t(capacity)*F,sizeof(half)));
        bytes=(bytes+255)&~size_t(255);
    }
};
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(std::string("K12 grouped: ")+cudaGetErrorString(e)); }
__global__ void remap_routes(const int* ids,const int* map,int rows,int first,int groups,int* filtered) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows) return;
    const int e=ids[row];
    if(e==-1) { filtered[row]=-1;return; }
    if(e<0 || e>=384) { __trap();return; }
    const int local=map[e]-first;
    filtered[row]=local>=0 && local<groups ? local : -1;
}
__global__ void count_routes(const int* ids,int rows,int groups,int* counts) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows) return;
    const int e=ids[r];assert(e>=-1 && e<groups);if(e>=0) atomicAdd(counts+e,1);
}
__global__ void plan(int groups,const int* counts,int* offsets,int* cursor,Job* jobs,int max_jobs,int* active,int capacity) {
    if(threadIdx.x) return;
    int total=0;for(int e=0;e<groups;++e) total+=counts[e];
    if(total>capacity) { __trap();return; }
    int first=0,j=0,n=0;
    for(int e=0;e<groups;++e) {
        offsets[e]=cursor[e]=first;const int count=counts[e];if(count) ++n;
        for(int r=0;r<count;r+=M) { assert(j<max_jobs);jobs[j++]={e,first+r,min(M,count-r)}; }
        first+=count;
    }
    offsets[groups]=first;
    for(;j<max_jobs;++j) jobs[j]={0,0,0};
    if(active) *active+=n;
}
__global__ void scatter_routes(const int* ids,int rows,int* cursor,int* slots,int* reverse,int* labels) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows) return;
    const int e=ids[r];if(e<0) { reverse[r]=-1;return; }
    const int at=atomicAdd(cursor+e,1);slots[at]=r;reverse[r]=at;labels[at]=e;
}
__device__ const Exl3Expert& expert_at(const Exl3Expert* experts,const int* residency,int e) {
    const int slot=residency?residency[e]:e;assert(slot>=0);return experts[slot];
}
__global__ void gather_had(const half* x,const int* slots,const int* labels,const int* offsets,int groups,int topk,
                           const Exl3Expert* experts,const int* residency,half* input,int capacity) {
    const int row=blockIdx.x/2,side=blockIdx.x%2,off=blockIdx.y*128;
    if(row>=offsets[groups]) return;
    const auto& e=expert_at(experts,residency,labels[row]);const auto p=side?e.w3:e.w1;
    check_proj(p,H,F);assert(p.codebook==2);
    had_hf_r_128_inner<true,false>(x+size_t(slots[row]/topk)*H+off,
        input+(size_t(side)*capacity+row)*H+off,p.suh,HAD_SCALE);
}

template<int Bits>
__device__ __forceinline__ void mma_rows(const half* A,float* C,const uint16_t* trellis,
                                        int K,int N,int row0,int rows,half* sa) {
    constexpr int KB=SK/16,MC=M/16;
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    const int nb=blockIdx.x*8+warp,mch=(rows+15)/16,slabs=K/SK;
    FragC acc[MC][2];
    #pragma unroll
    for(int i=0;i<MC;++i) { acc[i][0]={};acc[i][1]={}; }
    auto load_a=[&](int buf,int slab,int mc) {
        #pragma unroll
        for(int i=threadIdx.x;i<16*SK/8;i+=256) {
            const int r=i/(SK/8),c8=i%(SK/8),grow=mc*16+r;
            half* dst=sa+buf*16*STRIDE+r*STRIDE+c8*8;
            if(grow<rows) cp_async(dst,A+size_t(row0+grow)*K+size_t(slab)*SK+c8*8);
            else *reinterpret_cast<int4*>(dst)=make_int4(0,0,0,0);
        }
        cp_async_fence();
    };
    FragB weights[KB][2];int buf=0;load_a(0,0,0);
    for(int s=0;s<slabs;++s) {
        #pragma unroll
        for(int kb=0;kb<KB;++kb) {
            const auto* packed=reinterpret_cast<const uint32_t*>(trellis+(size_t(s*KB+kb)*(N/16)+nb)*16*Bits);
            dq_dispatch<Bits,2>(packed,lane<<3,weights[kb][0],weights[kb][1]);
        }
        #pragma unroll
        for(int mc=0;mc<MC;++mc) {
            if(mc>=mch) break;
            int nmc=mc+1,ns=s;if(nmc>=mch) { nmc=0;ns=s+1; }
            if(ns<slabs) load_a(buf^1,ns,nmc);else cp_async_fence();
            cp_async_wait<1>();__syncthreads();
            const int r=(lane%8)+8*((lane/8)%2);
            #pragma unroll
            for(int kb=0;kb<KB;++kb) {
                FragA a;ldsm4(a,sa+buf*16*STRIDE+r*STRIDE+kb*16+(lane/16)*8);
                ptx_mma_m16n8k16(a,weights[kb][0],acc[mc][0]);
                ptx_mma_m16n8k16(a,weights[kb][1],acc[mc][1]);
            }
            __syncthreads();buf^=1;
        }
    }
    const int col0=nb*16+2*(lane%4);
    #pragma unroll
    for(int mc=0;mc<MC;++mc) {
        if(mc>=mch) break;
        #pragma unroll
        for(int h=0;h<2;++h) {
            const float* v=reinterpret_cast<const float*>(&acc[mc][h]);
            const int ra=mc*16+lane/4,rb=ra+8,col=col0+h*8;
            if(ra<rows) *reinterpret_cast<float2*>(C+size_t(row0+ra)*N+col)=make_float2(v[0],v[1]);
            if(rb<rows) *reinterpret_cast<float2*>(C+size_t(row0+rb)*N+col)=make_float2(v[2],v[3]);
        }
    }
}
template<int Projection>
__global__ __launch_bounds__(256,2) void project(const half* input,float* output,const Job* jobs,
                                               const Exl3Expert* experts,const int* residency) {
    const Job job=jobs[blockIdx.y];if(!job.rows) return;
    const auto& e=expert_at(experts,residency,job.expert);
    const auto p=Projection==0?e.w1:Projection==1?e.w3:e.w2;
    check_proj(p,Projection==2?F:H,Projection==2?H:F);assert(p.codebook==2);
    __shared__ __align__(16) half shared[2*16*STRIDE];
    #define RUN(B) mma_rows<B>(input,output,p.trellis,p.k,p.n,job.first,job.rows,shared);break
    switch(p.tile_w/16) { case 1:RUN(1);case 2:RUN(2);case 3:RUN(3);case 4:RUN(4);case 5:RUN(5);case 6:RUN(6);default:assert(false); }
    #undef RUN
}
__global__ void activate(const int* slots,const int* labels,const int* offsets,int groups,const float* weights,
                         const Exl3Expert* experts,const int* residency,const float* gu,half* hidden,int capacity) {
    const int row=blockIdx.x,off=blockIdx.y*128,lane=threadIdx.x;if(row>=offsets[groups]) return;
    const auto& e=expert_at(experts,residency,labels[row]);
    __shared__ __align__(16) float gate[128],up[128];
    __shared__ __align__(16) half values[128];
    had_ff_r_128_inner<false,true>(gu+size_t(row)*F+off,gate,e.w1.svh,HAD_SCALE);
    had_ff_r_128_inner<false,true>(gu+size_t(capacity+row)*F+off,up,e.w3.svh,HAD_SCALE);__syncwarp();
    for(int i=lane;i<128;i+=32) {
        const float g=fminf(gate[i],10.f),u=fminf(fmaxf(up[i],-10.f),10.f);
        const float raw=(g/(1.f+expf(-g))*u)*weights[slots[row]];
        if(!e.native_qdq) { values[i]=__float2half_rn(raw);continue; }
        const float h=__bfloat162float(__float2bfloat16_rn(raw));float a=fabsf(h);
        for(int d=16;d;d>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffffu,a,d));
        const float scale=round_pow2(fmaxf(a,1e-4f)*(1.f/448.f));
        const __nv_fp8_e4m3 q(fminf(fmaxf(h/scale,-448.f),448.f));
        values[i]=__float2half_rn(__bfloat162float(__float2bfloat16_rn(float(q)*scale)));
    }
    __syncwarp();had_hf_r_128_inner<true,false>(values,hidden+size_t(row)*F+off,e.w2.suh,HAD_SCALE);
}
__global__ void finish_down(const int* labels,const int* offsets,int groups,const Exl3Expert* experts,
                            const int* residency,float* down) {
    const int row=blockIdx.x,off=blockIdx.y*128;if(row>=offsets[groups]) return;
    const auto& e=expert_at(experts,residency,labels[row]);__shared__ __align__(16) float result[128];
    had_ff_r_128_inner<false,true>(down+size_t(row)*H+off,result,e.w2.svh,HAD_SCALE);__syncwarp();
    for(int i=threadIdx.x;i<128;i+=32) down[size_t(row)*H+off+i]=result[i];
}
__global__ void reduce(const int* ids,const int* reverse,int topk,const float* down,float* out) {
    const int token=blockIdx.x,off=blockIdx.y*128,lane=threadIdx.x;
    __shared__ int order[6],count;
    if(!lane) {
        count=0;
        for(int j=0;j<topk;++j) if(ids[token*topk+j]>=0) {
            int at=count++;
            while(at>0 && ids[token*topk+order[at-1]]>ids[token*topk+j]) { order[at]=order[at-1];--at; }
            order[at]=j;
        }
    }
    __syncthreads();
    if(!count) return;
    for(int d=lane;d<128;d+=32) {
        float value=out[size_t(token)*H+off+d];
        for(int j=0;j<count;++j) value+=down[size_t(reverse[token*topk+order[j]])*H+off+d];
        out[size_t(token)*H+off+d]=value;
    }
}
} // namespace grouped

void exl3_moe_remap_routes(const int32_t* ids,const int32_t* map,int rows,int first,int groups,
                           int32_t* filtered,cudaStream_t stream) {
    if(!ids || !map || !filtered || rows<1 || first<0 || first>=384 || groups<1 || groups>384-first)
        throw std::invalid_argument("K12 grouped: invalid expert wave");
    grouped::remap_routes<<<(rows+255)/256,256,0,stream>>>(ids,map,rows,first,groups,filtered);
    grouped::check(cudaGetLastError());
}

size_t exl3_moe_grouped_workspace_bytes(int tokens,int topk,int groups,int active_rows) {
    if(tokens<1 || tokens>1024 || topk<1 || topk>6 || groups<1 || groups>384)
        throw std::invalid_argument("K12 grouped: expected tokens 1..1024, topk 1..6, groups 1..384");
    if(active_rows<0 || active_rows>tokens*topk) throw std::invalid_argument("K12 grouped: invalid active row capacity");
    return grouped::Buffers(nullptr,tokens,topk,groups,active_rows?active_rows:tokens*topk).bytes;
}
void exl3_moe_prefill_grouped(const __half* x,const int32_t* ids,const float* weights,int tokens,int topk,
                             const Exl3Expert* experts,const int32_t* residency,int groups,float* out,
                              void* workspace,size_t bytes,int32_t* active_groups,cudaStream_t stream,int active_rows) {
    const size_t required=exl3_moe_grouped_workspace_bytes(tokens,topk,groups,active_rows);
    if(!x || !ids || !weights || !experts || !out || !workspace || reinterpret_cast<uintptr_t>(workspace)%256 || bytes<required)
        throw std::invalid_argument("K12 grouped: invalid buffers or workspace");
    const int rows=tokens*topk,capacity=active_rows?active_rows:rows;
    grouped::Buffers b(workspace,tokens,topk,groups,capacity);
    grouped::check(cudaMemsetAsync(b.counts,0,groups*sizeof(int),stream));
    grouped::count_routes<<<(rows+255)/256,256,0,stream>>>(ids,rows,groups,b.counts);
    grouped::plan<<<1,1,0,stream>>>(groups,b.counts,b.offsets,b.cursor,b.jobs,b.max_jobs,active_groups,capacity);
    grouped::scatter_routes<<<(rows+255)/256,256,0,stream>>>(ids,rows,b.cursor,b.slots,b.reverse,b.labels);
    grouped::gather_had<<<dim3(capacity*2,grouped::H/128),32,0,stream>>>(x,b.slots,b.labels,b.offsets,groups,topk,experts,residency,b.input,capacity);
    grouped::project<0><<<dim3(grouped::F/128,b.max_jobs),256,0,stream>>>(b.input,b.gu,b.jobs,experts,residency);
    grouped::project<1><<<dim3(grouped::F/128,b.max_jobs),256,0,stream>>>(b.input+size_t(capacity)*grouped::H,b.gu+size_t(capacity)*grouped::F,b.jobs,experts,residency);
    grouped::activate<<<dim3(capacity,grouped::F/128),32,0,stream>>>(b.slots,b.labels,b.offsets,groups,weights,experts,residency,b.gu,b.hidden,capacity);
    grouped::project<2><<<dim3(grouped::H/128,b.max_jobs),256,0,stream>>>(b.hidden,b.down,b.jobs,experts,residency);
    grouped::finish_down<<<dim3(capacity,grouped::H/128),32,0,stream>>>(b.labels,b.offsets,groups,experts,residency,b.down);
    grouped::reduce<<<dim3(tokens,grouped::H/128),32,0,stream>>>(ids,b.reverse,topk,b.down,out);
    grouped::check(cudaGetLastError());
}
} // namespace strata::ds41::kernels
