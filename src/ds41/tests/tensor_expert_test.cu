#include "bench_util.hpp"
#include "strata/ds41/tensor_parallel.hpp"
#include "strata/ds41/tensor_shards.hpp"
#include <future>
#include <fcntl.h>
#include <unistd.h>

using namespace strata::ds41;
using namespace ds41test;
namespace {
struct Fixture {
    std::string path;
    ExpertSlot slot;
    Fixture() {
        char name[]=".ds41-tensor-expert-XXXXXX";int fd=mkstemp(name);
        if(fd<0) throw std::runtime_error("create tensor expert fixture");path=name;
        std::mt19937 rng(1329);
        uint64_t offset=0;
        for(int p=0;p<3;++p) {
            const int c=4*p,k=p==2?kMoeInter:kDim,n=p==2?kDim:kMoeInter,bits=p+2;
            slot.codebook[p]=2;
            std::vector<uint16_t> packed(size_t(k/16)*(n/16)*bits*16);
            for(auto& value:packed) value=uint16_t(rng());
            std::vector<half> suh(k),svh(n);
            for(auto& value:suh) value=__float2half_rn(rng()&1?.5f:-.5f);
            for(auto& value:svh) value=__float2half_rn(rng()&1?.02f:-.02f);
            const uint32_t marker=0x83DCD12Du;
            const void* data[]={packed.data(),suh.data(),svh.data(),&marker};
            const size_t bytes[]={packed.size()*2,size_t(k)*2,size_t(n)*2,4};
            for(int j=0;j<4;++j) {
                slot.comp_off[c+j]=offset;slot.comp_bytes[c+j]=bytes[j];
                if(pwrite(fd,data[j],bytes[j],offset)!=ssize_t(bytes[j])) throw std::runtime_error("write tensor expert fixture");
                offset+=(bytes[j]+255)/256*256;
            }
        }
        slot.bytes=offset;
        if(ftruncate(fd,offset)) { close(fd);throw std::runtime_error("size tensor expert fixture"); }
        close(fd);
    }
    ~Fixture() { unlink(path.c_str()); }
};
}
int main() {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || count<2) return 77;
    try {
        Fixture fixture;TensorParallel world({0,1});
        for(int m:{1,4,8}) for(int topk:{1,6}) {
            std::vector<float> reference(size_t(m)*kDim);
            {
                DeviceScope scope(0);BoundedUpload file(fixture.path);Dev<uint8_t> data(fixture.slot.bytes);
                file.copy(data.p,0,fixture.slot.bytes);
                kernels::Exl3Expert descriptor{};descriptor.native_qdq=false;
                kernels::Exl3Proj* projections[]={&descriptor.w1,&descriptor.w3,&descriptor.w2};
                for(int p=0;p<3;++p) {
                    const int c=p*4,k=p==2?kMoeInter:kDim,n=p==2?kDim:kMoeInter;
                    *projections[p]={reinterpret_cast<uint16_t*>(data.p+fixture.slot.comp_off[c]),
                        reinterpret_cast<half*>(data.p+fixture.slot.comp_off[c+1]),
                        reinterpret_cast<half*>(data.p+fixture.slot.comp_off[c+2]),k,n,(p+2)*16,2};
                }
                Dev<kernels::Exl3Expert> descriptors(std::vector<kernels::Exl3Expert>{descriptor});
                std::vector<half> input(size_t(m)*kDim);for(size_t i=0;i<input.size();++i) input[i]=__float2half_rn(float(int(i%31)-15)/64);
                Dev<half> x(input);Dev<int32_t> selected(std::vector<int32_t>(m*topk,0));
                std::vector<float> weights(m*topk);for(int i=0;i<m*topk;++i) weights[i]=float(i%topk+1)/32;
                Dev<float> w(weights),out(reference.size());Dev<uint8_t> ws(64ull<<20);
                ck(cudaMemset(out.p,0,out.n*sizeof(float)),"reference clear");
                kernels::exl3_moe_decode(x.p,m,selected.p,w.p,topk,descriptors.p,out.p,ws.p,ws.n,0);
                ck(cudaDeviceSynchronize(),"reference experts");reference=out.down();
            }
            auto run=[&](int rank) {
                DeviceScope scope(rank);BoundedUpload file(fixture.path);Dev<uint8_t> data(tensor_expert_bytes(fixture.slot));
                auto descriptor=upload_tensor_expert(file,fixture.slot,data.p,rank);
                Dev<kernels::Exl3Expert> descriptors(std::vector<kernels::Exl3Expert>{descriptor});
                std::vector<half> input(size_t(m)*kDim);for(size_t i=0;i<input.size();++i) input[i]=__float2half_rn(float(int(i%31)-15)/64);
                Dev<half> x(input);Dev<int32_t> selected(std::vector<int32_t>(m*topk,0));
                std::vector<float> weights(m*topk);for(int i=0;i<m*topk;++i) weights[i]=float(i%topk+1)/32;
                Dev<float> w(weights),out(reference.size());Dev<uint8_t> ws(64ull<<20);
                cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"expert tensor stream");
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed),"expert tensor capture");
                ck(cudaMemsetAsync(out.p,0,out.n*sizeof(float),stream),"tensor clear");
                kernels::exl3_moe_tensor_up(x.p,m,selected.p,w.p,topk,descriptors.p,ws.p,ws.n,rank,stream);
                world.gather(rank,1,kernels::exl3_moe_tensor_hidden(ws.p,m,topk),m*topk,kMoeInter,2,stream,0,false);
                kernels::exl3_moe_tensor_down(m,selected.p,topk,descriptors.p,out.p,ws.p,ws.n,rank,stream);
                world.gather(rank,2,out.p,m,kDim,4,stream,0,false);
                world.barrier(rank,3,stream); // protect out before the next replay clears it
                cudaGraph_t graph;cudaGraphExec_t executable;
                ck(cudaStreamEndCapture(stream,&graph),"expert tensor end capture");
                ck(cudaGraphInstantiate(&executable,graph,0),"expert tensor instantiate");
                for(int replay=0;replay<3;++replay) ck(cudaGraphLaunch(executable,stream),"expert tensor replay");
                ck(cudaStreamSynchronize(stream),"expert tensor complete");
                const auto got=out.down();
                if(std::memcmp(reference.data(),got.data(),got.size()*sizeof(float))) throw std::runtime_error("TP2 expert differs from full EXL3");
                cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);cudaStreamDestroy(stream);
            };
            auto a=std::async(std::launch::async,run,0),b=std::async(std::launch::async,run,1);a.get();b.get();
        }
        std::puts("PASS TP2 mixed-rate expert shards: full-logit bit parity, rows 1/4/8, topk 1/6, repeated routes, captured replays");
    } catch(const std::exception& ex) { std::fprintf(stderr,"FAIL %s\n",ex.what());return 1; }
}
