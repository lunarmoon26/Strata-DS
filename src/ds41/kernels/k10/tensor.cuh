#pragma once
namespace strata::ds41::kernels::k10 {
__global__ void tensor_input_had(const half* x,const int32_t* selected,int topk,
                                 const Exl3Expert* experts,half* input,float* gu,Job* jobs) {
    const int job=blockIdx.x,slot=job/2,id=selected[slot],off=blockIdx.y*128;
    if(id<0) { if(!off && !threadIdx.x) jobs[job]=Job{}; return; }
    const auto p=(job&1)?experts[id].w3:experts[id].w1;
    check_proj(p,H,F/2);
    half* a=input+size_t(job)*H;
    if(!off && !threadIdx.x) jobs[job]=Job{a,p.trellis,gu+size_t(job)*(F/2),H,F/2,p.tile_w/16,p.codebook};
    had_hf_r_128_inner<true,false>(x+size_t(slot/topk)*H+off,a+off,p.suh,HAD_SCALE);
}
__global__ void tensor_activate(const int32_t* selected,const float* weights,const Exl3Expert* experts,
                                const float* gu,half* hidden,float* down,Job* jobs,int rank) {
    const int slot=blockIdx.x,id=selected[slot],off=blockIdx.y*128,lane=threadIdx.x;
    if(id<0) { if(!off && !lane) jobs[slot]=Job{}; return; }
    const auto e=experts[id];check_proj(e.w2,F,H/2);assert(!e.native_qdq);
    __shared__ __align__(16) float gate[128],up[128];
    __shared__ __align__(16) half activation[128];
    had_ff_r_128_inner<false,true>(gu+size_t(2*slot)*(F/2)+off,gate,e.w1.svh,HAD_SCALE);
    had_ff_r_128_inner<false,true>(gu+size_t(2*slot+1)*(F/2)+off,up,e.w3.svh,HAD_SCALE);
    __syncwarp();
    for(int i=lane;i<128;i+=32) {
        const float g=fminf(gate[i],10.0f),u=fminf(fmaxf(up[i],-10.0f),10.0f);
        activation[i]=__float2half_rn((g/(1.0f+expf(-g))*u)*weights[slot]);
    }
    __syncwarp();
    half* a=hidden+size_t(slot)*F;
    had_hf_r_128_inner<true,false>(activation,a+rank*(F/2)+off,e.w2.suh+rank*(F/2),HAD_SCALE);
    if(!off && !lane) jobs[slot]=Job{a,e.w2.trellis,down+size_t(slot)*(H/2),F,H/2,e.w2.tile_w/16,e.w2.codebook};
}
__global__ void tensor_output(const int32_t* selected,int topk,const Exl3Expert* experts,
                              const float* down,float* output,int rank) {
    const int token=blockIdx.x,off=blockIdx.y*128,lane=threadIdx.x;
    __shared__ __align__(16) float values[128];
    float acc[4];
    for(int i=0;i<4;++i) acc[i]=output[size_t(token)*H+rank*(H/2)+off+lane+32*i];
    for(int j=0;j<topk;++j) {
        const int slot=token*topk+j,id=selected[slot];if(id<0) continue;
        had_ff_r_128_inner<false,true>(down+size_t(slot)*(H/2)+off,values,experts[id].w2.svh,HAD_SCALE);
        __syncwarp();
        for(int i=0;i<4;++i) acc[i]+=values[lane+32*i];
        __syncwarp();
    }
    for(int i=0;i<4;++i) output[size_t(token)*H+rank*(H/2)+off+lane+32*i]=acc[i];
}
}
