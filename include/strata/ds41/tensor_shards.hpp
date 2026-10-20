#pragma once
#include "strata/ds41/bounded_upload.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/pack.hpp"
#include "strata/ds41/kernels/k10_exl3_moe.hpp"

namespace strata::ds41 {
inline size_t tensor_expert_bytes(const ExpertSlot& slot) {
    size_t bytes=0;
    for(int p=0;p<3;++p) {
        const int c=p*4,k=p==2?kMoeInter:kDim,n=p==2?kDim:kMoeInter;
        const uint64_t tiles=uint64_t(k/16)*(n/16);
        if(slot.codebook[p]!=2 || slot.comp_bytes[c]%(tiles*2) ||
            slot.comp_bytes[c]/(tiles*2)<16 || slot.comp_bytes[c]/(tiles*2)>96 ||
            slot.comp_bytes[c]/(tiles*2)%16 || slot.comp_bytes[c+1]!=uint64_t(k)*2 ||
            slot.comp_bytes[c+2]!=uint64_t(n)*2)
            throw std::invalid_argument("TP2 expert requires aligned integer mul1 EXL3 components");
        for(size_t nbytes:{size_t(slot.comp_bytes[c]/2),size_t(k*2),size_t(n)})
            bytes+=(nbytes+255)/256*256;
    }
    return bytes;
}
inline kernels::Exl3Expert upload_tensor_expert(BoundedUpload& file,const ExpertSlot& slot,
                                                uint8_t* dst,int rank) {
    if(rank!=0 && rank!=1) throw std::invalid_argument("TP2 expert rank");
    tensor_expert_bytes(slot);
    kernels::Exl3Expert result; result.native_qdq=false;
    kernels::Exl3Proj* projections[]={&result.w1,&result.w3,&result.w2};
    size_t off=0;
    for(int p=0;p<3;++p) {
        const int c=p*4,k=p==2?kMoeInter:kDim,n=p==2?kDim:kMoeInter;
        auto& w=*projections[p];w.k=k;w.n=n/2;w.codebook=2;
        w.tile_w=int(slot.comp_bytes[c]/(uint64_t(k/16)*(n/16)*2));
        if(file.scalar_u32(slot.offset+slot.comp_off[c+3])!=0x83DCD12Du)
            throw std::runtime_error("TP2 expert mul1 marker mismatch");
        const size_t row=size_t(n/16)*w.tile_w*2,half=row/2;
        w.trellis=reinterpret_cast<const uint16_t*>(dst+off);
        file.copy_columns(dst+off,slot.offset+slot.comp_off[c],k/16,row,size_t(rank)*half,half);
        off+=(slot.comp_bytes[c]/2+255)/256*256;
        w.suh=reinterpret_cast<const __half*>(dst+off);
        file.copy(dst+off,slot.offset+slot.comp_off[c+1],size_t(k)*2);off+=(size_t(k)*2+255)/256*256;
        w.svh=reinterpret_cast<const __half*>(dst+off);
        file.copy(dst+off,slot.offset+slot.comp_off[c+2]+size_t(rank)*n,size_t(n));off+=(size_t(n)+255)/256*256;
        w.shard_rank=rank;w.shard_count=2;
    }
    return result;
}
}
