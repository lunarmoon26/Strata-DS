#include "strata/ds41/tensor_parallel.hpp"
#include "strata/ds41/device_scope.hpp"
#include <cmath>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <vector>

using namespace strata::ds41;
namespace {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void fill(float* out,int rank,int count,int columns,int stride,const int* iteration) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(rank && blockIdx.x==0 && threadIdx.x==0 && *iteration%3==0) {
        const auto start=clock64();while(clock64()-start<20000) __nanosleep(100);
    }
    __syncthreads();
    if(i<count*columns/2) out[(i/(columns/2))*stride+rank*columns/2+i%(columns/2)]=float(*iteration*2+rank+1);
}
__global__ void validate(const float* out,int count,int columns,int stride,int* iteration,int* faults) {
    bool failed=false;
    for(int i=threadIdx.x;i<count*stride;i+=blockDim.x) {
        const int column=i%stride;
        const float expected=column>=columns?0:float(*iteration*2+(column<columns/2?1:2));
        failed |= out[i]!=expected;
    }
    if(failed) atomicAdd(faults,1);
    __syncthreads();if(!threadIdx.x) ++*iteration;
}
}
int main() {
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices<2) return 77;
    int peer=0; check(cudaDeviceCanAccessPeer(&peer,0,1)); if(!peer) return 77;
    try {
        TensorParallel world({0,1});
        struct Case {
            int rows;
            float* output[2]={};cudaStream_t streams[2]={};cudaGraphExec_t graphs[2]={};int* state[2]={};
        };
        std::vector<Case> cases;
        for(int rows:{1,4,8,32,64}) cases.push_back(Case{rows});
        constexpr int columns=8192,stride=8320;
        // Capture every shape on rank 0 before rank 1. A mutable "latest layout"
        // would accidentally bind the second rank's small graphs to the big slot.
        for(int rank=0;rank<2;++rank) for(auto& c:cases) {
                const int rows=c.rows;
                auto& output=c.output;auto& streams=c.streams;auto& graphs=c.graphs;auto& state=c.state;
                DeviceScope scope(rank);
                check(cudaMalloc(&output[rank],rows*stride*sizeof(float)));
                check(cudaMemset(output[rank],0,rows*stride*sizeof(float)));
                check(cudaMalloc(&state[rank],2*sizeof(int)));check(cudaMemset(state[rank],0,2*sizeof(int)));
                check(cudaStreamCreateWithFlags(&streams[rank],cudaStreamNonBlocking));
                check(cudaStreamBeginCapture(streams[rank],cudaStreamCaptureModeRelaxed));
                fill<<<(rows*columns/2+255)/256,256,0,streams[rank]>>>(output[rank],rank,rows,columns,stride,state[rank]);
                world.gather(rank,0,output[rank],rows,columns,4,streams[rank],stride,rows==1);
                validate<<<1,256,0,streams[rank]>>>(output[rank],rows,columns,stride,state[rank],state[rank]+1);
                if(rows==1) world.barrier(rank,1,streams[rank]);
                cudaGraph_t graph;
                check(cudaStreamEndCapture(streams[rank],&graph));
                check(cudaGraphInstantiate(&graphs[rank],graph,0)); check(cudaGraphDestroy(graph));
        }
        bool bounded=false;
        try { world.gather(0,2,cases[0].output[0],64,1<<30,4,cases[0].streams[0]); }
        catch(const std::runtime_error&) { bounded=true; }
        if(!bounded) throw std::runtime_error("TP mailbox did not enforce its fixed allocation ceiling");
        for(int pass=1;pass<=2;++pass) for(auto& c:cases) {
            const int rows=c.rows;
            auto& output=c.output;auto& streams=c.streams;auto& graphs=c.graphs;auto& state=c.state;
            auto run=[&](int rank) {
                DeviceScope scope(rank);
                for(int iteration=0;iteration<100;++iteration) check(cudaGraphLaunch(graphs[rank],streams[rank]));
                check(cudaStreamSynchronize(streams[rank]));
                int counts[2];check(cudaMemcpy(counts,state[rank],sizeof(counts),cudaMemcpyDeviceToHost));
                if(counts[0]!=pass*100 || counts[1]) throw std::runtime_error("TP mailbox mixed replay epochs or changed padding");
                if(world.agree(rank==0) || !world.agree(true)) throw std::runtime_error("TP cache working-set agreement differs");
                std::vector<float> result(rows*stride);
                check(cudaMemcpy(result.data(),output[rank],result.size()*sizeof(float),cudaMemcpyDeviceToHost));
                for(int row=0;row<rows;++row) for(int c=0;c<columns;++c)
                    if(result[row*stride+c]!=float(pass*200-2+(c<columns/2?1:2))) throw std::runtime_error("TP peer gather differs");
            };
            auto first=std::async(std::launch::async,run,0);
            auto last=std::async(std::launch::async,run,1);
            first.get();last.get();
        }
        for(auto& c:cases) {
            for(int rank=0;rank<2;++rank) {
                DeviceScope scope(rank);check(cudaGraphExecDestroy(c.graphs[rank]));check(cudaStreamDestroy(c.streams[rank]));
                check(cudaFree(c.output[rank]));
                check(cudaFree(c.state[rank]));
            }
        }
        world.abort();
        bool refused=false;
        try { world.agree(true); } catch(const std::runtime_error&) { refused=true; }
        if(!refused) throw std::runtime_error("TP scheduling did not observe abort");
        std::puts("PASS TP2 mailbox: strided rows 1/4/8/32/64, changing/skewed epochs, stable grown layouts, acknowledgements on/off");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
