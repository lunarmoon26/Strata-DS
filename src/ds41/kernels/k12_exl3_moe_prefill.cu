// K12: grouped prefill with upstream EXL3 reconstruction and cuBLAS GEMM.
#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"

// CMake intentionally builds only this translation unit.
#include "../../../third_party/exllamav3_gpu/quant/reconstruct.cu"
#include "k12/pipeline.cuh"
#include "k12/blas.cuh"
#include <algorithm>

namespace strata::ds41::kernels {

namespace expert_graph {
struct Request { const Exl3Expert* expert; int first; };
struct State { Exl3Expert expert; int first; };
__global__ void prepare(Request request,State* state,strata_exl3::ReconstructJob* jobs) {
    const auto e=*request.expert;state->expert=e;state->first=request.first;
    k12::check_proj(e.w1,k12::H,k12::F);k12::check_proj(e.w3,k12::H,k12::F);k12::check_proj(e.w2,k12::F,k12::H);
    jobs[0]={e.w1.trellis,e.w1.tile_w/16};jobs[1]={e.w3.trellis,e.w3.tile_w/16};jobs[2]={e.w2.trellis,e.w2.tile_w/16};
}
__global__ void input(const half* x,const int32_t* tok,const State* state,int rows,half* out) {
    const int job=blockIdx.x,row=job%rows,off=blockIdx.y*128;
    const auto p=job<rows?state->expert.w1:state->expert.w3;
    had_hf_r_128_inner<true,false>(x+size_t(tok[state->first+row])*k12::H+off,
                                  out+size_t(job)*k12::H+off,p.suh,k12::HAD_SCALE);
}
__global__ void activate(const float* weights,const State* state,int rows,const float* gu,half* out) {
    const int row=blockIdx.x;
    k12::activate_down_had_inner(weights[state->first+row],state->expert,gu+size_t(row)*k12::F,
                                 gu+size_t(rows+row)*k12::F,out+size_t(row)*k12::F);
}
__global__ void scatter(const int32_t* tok,const State* state,const float* down,float* out) {
    const int row=blockIdx.x,off=blockIdx.y*128,lane=threadIdx.x;
    __shared__ __align__(16) float result[128];
    had_ff_r_128_inner<false,true>(down+size_t(row)*k12::H+off,result,state->expert.w2.svh,k12::HAD_SCALE);
    __syncwarp();
    #pragma unroll
    for(int i=0;i<4;++i) atomicAdd(out+size_t(tok[state->first+row])*k12::H+off+lane+32*i,result[lane+32*i]);
}
} // namespace expert_graph

struct Exl3PrefillGraphs::Impl {
    struct Entry { cudaGraphExec_t exec=nullptr; cudaGraphNode_t prepare=nullptr; cudaGraph_t graph=nullptr; };
    std::map<int,Entry> entries;
    const half* x;const int32_t* tok;const float* weights;float* out;void* workspace;size_t bytes;
    std::unique_ptr<k12::Workspace> ws;
    expert_graph::State* state=nullptr;
    cudaStream_t capture=nullptr,user=nullptr;bool bound=false;int device=0;
    uint64_t n_captures=0,n_replays=0,n_eager=0;
    Impl(const half* x_,const int32_t* tok_,const float* weights_,float* out_,void* workspace_,size_t bytes_)
        :x(x_),tok(tok_),weights(weights_),out(out_),workspace(workspace_),bytes(bytes_) {
        if(!x || !tok || !weights || !out || !workspace || bytes<k12::Layout(k12::ROW_TILE).bytes())
            throw std::invalid_argument("K12 expert graphs: invalid buffers/workspace");
        ws=std::make_unique<k12::Workspace>(workspace,k12::Layout(k12::ROW_TILE));
        k12::check_cuda(cudaGetDevice(&device),"expert graph device");
        try {
            k12::check_cuda(cudaStreamCreateWithFlags(&capture,cudaStreamNonBlocking),"expert capture stream");
            k12::check_cuda(cudaMalloc(&state,sizeof(*state)),"expert graph state");
        } catch(...) { if(capture) cudaStreamDestroy(capture);throw; }
    }
    ~Impl() {
        int previous=device;cudaGetDevice(&previous);cudaSetDevice(device);
        if(bound) cudaStreamSynchronize(user);
        for(auto& [count,entry]:entries) {
            if(entry.exec) cudaGraphExecDestroy(entry.exec);
            if(entry.graph) cudaGraphDestroy(entry.graph);
        }
        if(state) cudaFree(state);
        if(capture) cudaStreamDestroy(capture);
        cudaSetDevice(previous);
    }
    void enqueue(expert_graph::Request request,int count,cublasHandle_t handle,cudaStream_t stream) {
        expert_graph::prepare<<<1,1,0,stream>>>(request,state,ws->trellis);
        strata_exl3::reconstruct_mul1(ws->matrices,ws->trellis,k12::H,k12::F,stream);
        strata_exl3::reconstruct_mul1(ws->matrices+k12::MATRIX_ELEMENTS,ws->trellis+1,k12::H,k12::F,stream);
        strata_exl3::reconstruct_mul1(ws->matrices+2*k12::MATRIX_ELEMENTS,ws->trellis+2,k12::F,k12::H,stream);
        expert_graph::input<<<dim3(2*count,k12::H/128),32,0,stream>>>(x,tok,state,count,ws->input);
        k12::gemm(handle,ws->input,ws->matrices,ws->gu,count,k12::H,k12::F);
        k12::gemm(handle,ws->input+size_t(count)*k12::H,ws->matrices+k12::MATRIX_ELEMENTS,
                   ws->gu+size_t(count)*k12::F,count,k12::H,k12::F);
        expert_graph::activate<<<dim3(count,k12::F/128),32,0,stream>>>(weights,state,count,ws->gu,ws->down_input);
        k12::gemm(handle,ws->down_input,ws->matrices+2*k12::MATRIX_ELEMENTS,ws->down,count,k12::F,k12::H);
        expert_graph::scatter<<<dim3(count,k12::H/128),32,0,stream>>>(tok,state,ws->down,out);
        k12::check_cuda(cudaPeekAtLastError(),"expert graph enqueue");
    }
    void run(const int32_t* offsets,int groups,const Exl3Expert* experts,cudaStream_t stream) {
        if(groups<0 || (groups && (!offsets || offsets[0]<0 || !experts)))
            throw std::invalid_argument("K12 expert graphs: invalid groups/offsets");
        for(int g=0;g<groups;++g) if(offsets[g+1]<offsets[g]) throw std::invalid_argument("K12 expert graphs: decreasing offsets");
        int current;k12::check_cuda(cudaGetDevice(&current),"expert graph current device");
        if(current!=device) throw std::invalid_argument("K12 expert graph used on another device");
        cudaStreamCaptureStatus status;k12::check_cuda(cudaStreamIsCapturing(stream,&status),"expert graph caller capture");
        if(status!=cudaStreamCaptureStatusNone) throw std::invalid_argument("K12 expert graph cannot nest inside another capture");
        if(bound && stream!=user) throw std::invalid_argument("K12 expert graphs require one serialized stream");
        user=stream;bound=true;
        for(int g=0;g<groups;++g) {
            const int count=offsets[g+1]-offsets[g];if(!count) continue;
            if(count>k12::ROW_TILE || (!entries.count(count) && entries.size()>=128)) {
                exl3_moe_prefill(x,tok,weights,offsets+g,1,experts+g,out,workspace,bytes,stream);++n_eager;continue;
            }
            const expert_graph::Request request{experts+g,offsets[g]};
            auto found=entries.find(count);
            if(found==entries.end() || !found->second.exec) {
                auto& registry=k12::blas_registry();std::lock_guard<std::mutex> lock(registry.mutex);
                const auto handle=registry.get(device);
                const bool warm=found==entries.end();const auto target=warm?stream:capture;
                k12::check_blas(cublasSetStream(handle,target),"expert graph cuBLAS stream");
                k12::check_blas(cublasSetWorkspace(handle,ws->blas,k12::BLAS_BYTES),"expert graph cuBLAS workspace");
                if(warm) { enqueue(request,count,handle,stream);entries.emplace(count,Entry{});++n_eager;continue; }
                cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
                k12::check_cuda(cudaStreamBeginCapture(capture,cudaStreamCaptureModeThreadLocal),"expert begin capture");
                cudaGraphNode_t root=nullptr;
                try {
                    enqueue(request,count,handle,capture);
                    k12::check_cuda(cudaStreamEndCapture(capture,&graph),"expert end capture");
                    size_t roots=1;k12::check_cuda(cudaGraphGetRootNodes(graph,&root,&roots),"expert graph root");
                    if(roots!=1) throw std::runtime_error("K12 expert graph must have one preparation root");
                    cudaGraphNodeType type;k12::check_cuda(cudaGraphNodeGetType(root,&type),"expert root type");
                    if(type!=cudaGraphNodeTypeKernel) throw std::runtime_error("K12 expert graph preparation must be a kernel");
                    k12::check_cuda(cudaGraphInstantiate(&exec,graph,0),"expert graph instantiate");
                } catch(...) {
                    if(!graph) cudaStreamEndCapture(capture,&graph);
                    if(exec) cudaGraphExecDestroy(exec);
                    if(graph) cudaGraphDestroy(graph);
                    throw;
                }
                // Node handles must outlive updates to the executable.
                found->second={exec,root,graph};++n_captures;
            }
            auto mutable_request=request;auto* metadata=state;auto* jobs=ws->trellis;
            void* args[]={&mutable_request,&metadata,&jobs};cudaKernelNodeParams params{};
            params.func=reinterpret_cast<void*>(expert_graph::prepare);params.gridDim=dim3(1);params.blockDim=dim3(1);params.kernelParams=args;
            k12::check_cuda(cudaGraphExecKernelNodeSetParams(found->second.exec,found->second.prepare,&params),"expert graph parameters");
            k12::check_cuda(cudaGraphLaunch(found->second.exec,stream),"expert graph launch");++n_replays;
        }
    }
};

Exl3PrefillGraphs::Exl3PrefillGraphs(const half* x,const int32_t* tok,const float* weights,float* out,void* workspace,size_t bytes)
    :impl_(std::make_unique<Impl>(x,tok,weights,out,workspace,bytes)) {}
Exl3PrefillGraphs::~Exl3PrefillGraphs()=default;
void Exl3PrefillGraphs::run(const int32_t* offsets,int groups,const Exl3Expert* experts,cudaStream_t stream) { impl_->run(offsets,groups,experts,stream); }
uint64_t Exl3PrefillGraphs::captures() const { return impl_->n_captures; }
uint64_t Exl3PrefillGraphs::replays() const { return impl_->n_replays; }
uint64_t Exl3PrefillGraphs::eager_calls() const { return impl_->n_eager; }
size_t Exl3PrefillGraphs::shapes() const { return impl_->entries.size(); }

namespace batched {
// Passed by value: no borrowed host offsets survive an asynchronous enqueue.
struct Wave { int offsets[17]; int groups; };
struct Layout {
    size_t matrices,input,gu,hidden,down,blas,end;
    Layout(int rows,int groups) {
        size_t pos=0;
        auto take=[&](size_t count,size_t width) { pos=(pos+255)&~size_t(255);const size_t at=pos;pos+=count*width;return at; };
        matrices=take(size_t(groups)*3*k12::MATRIX_ELEMENTS,sizeof(half));
        input=take(size_t(rows)*2*k12::H,sizeof(half));
        gu=take(size_t(rows)*2*k12::F,sizeof(float));
        hidden=take(size_t(rows)*k12::F,sizeof(half));
        down=take(size_t(rows)*k12::H,sizeof(float));
        blas=take(k12::BLAS_BYTES,1);end=pos;
    }
};
__device__ int expert_for(int row,const Wave& wave) {
    int g=0;while(g+1<wave.groups && row>=wave.offsets[g+1]) ++g;return g;
}
__global__ __launch_bounds__(256) void reconstruct(const Exl3Expert* experts,Wave wave,half* matrices) {
    const int g=blockIdx.z/3,projection=blockIdx.z%3;if(wave.offsets[g]==wave.offsets[g+1]) return;
    const auto e=experts[g];
    const auto p=projection==0?e.w1:projection==1?e.w3:e.w2;
    const int K=projection==2?k12::F:k12::H,N=projection==2?k12::H:k12::F;
    k12::check_proj(p,K,N);
    assert(p.shard_count==1 && p.codebook==2);
    half* out=matrices+(size_t(g)*3+projection)*k12::MATRIX_ELEMENTS;
    const int tiles_n=N/128,x=blockIdx.x%tiles_n,y=blockIdx.x/tiles_n;
    #define RECON(B) reconstruct_tile<B,2,false>(out,p.trellis,N/16,0,x,y,tiles_n);break
    switch(p.tile_w/16) { case 1:RECON(1);case 2:RECON(2);case 3:RECON(3);case 4:RECON(4);case 5:RECON(5);case 6:RECON(6);default:assert(false); }
    #undef RECON
}
__global__ void input_had(const half* x,const int32_t* tok,const Exl3Expert* experts,Wave wave,half* input) {
    const int rows=wave.offsets[wave.groups],row=blockIdx.x%rows,side=blockIdx.x/rows,off=blockIdx.y*128;
    const auto e=experts[expert_for(row,wave)];const auto p=side?e.w3:e.w1;
    had_hf_r_128_inner<true,false>(x+size_t(tok[row])*k12::H+off,
        input+(size_t(side)*rows+row)*k12::H+off,p.suh,k12::HAD_SCALE);
}
__global__ void activate(const float* weights,const Exl3Expert* experts,Wave wave,const float* gu,half* hidden) {
    const int row=blockIdx.x,rows=wave.offsets[wave.groups];
    k12::activate_down_had_inner(weights[row],experts[expert_for(row,wave)],gu+size_t(row)*k12::F,
                                 gu+size_t(rows+row)*k12::F,hidden+size_t(row)*k12::F);
}
} // namespace batched

size_t exl3_moe_batched_workspace_bytes(int max_rows,int groups_per_wave) {
    if(max_rows<0 || groups_per_wave<1 || groups_per_wave>16)
        throw std::invalid_argument("K12 batched: expected nonnegative rows and 1..16 groups/wave");
    if(!max_rows) return 0;
    return std::max(batched::Layout(max_rows,groups_per_wave).end+255,k12::Layout(max_rows).bytes());
}

void exl3_moe_prefill_batched(const __half* x,const int32_t* tok,const float* weights,
                            const int32_t* offsets,int groups,const Exl3Expert* experts,float* out,
                            void* workspace,size_t bytes,int groups_per_wave,cudaStream_t stream) {
    if(groups<0 || groups_per_wave<1 || groups_per_wave>16)
        throw std::invalid_argument("K12 batched: invalid group count/wave");
    if(!groups) return;
    if(!offsets || offsets[0]<0) throw std::invalid_argument("K12 batched: invalid offsets");
    bool tiled=false;
    for(int g=0;g<groups;++g) {
        if(offsets[g+1]<offsets[g]) throw std::invalid_argument("K12 batched: decreasing offsets");
        tiled|=offsets[g+1]-offsets[g]>k12::ROW_TILE;
    }
    const int rows=offsets[groups]-offsets[0];if(!rows) return;
    if(!x || !tok || !weights || !experts || !out || !workspace || bytes<exl3_moe_batched_workspace_bytes(rows,groups_per_wave))
        throw std::invalid_argument("K12 batched: invalid buffers/workspace");
    // The reference tiles rows at 512. Do not silently choose another GEMM
    // shape for heavily skewed larger passes; keep its exact tiled path.
    if(tiled) { exl3_moe_prefill(x,tok,weights,offsets,groups,experts,out,workspace,bytes,stream);return; }
    const batched::Layout layout(rows,groups_per_wave);
    auto* base=reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(workspace)+255)&~uintptr_t(255));
    auto* matrices=reinterpret_cast<half*>(base+layout.matrices);
    auto* input=reinterpret_cast<half*>(base+layout.input);
    auto* gu=reinterpret_cast<float*>(base+layout.gu);
    auto* hidden=reinterpret_cast<half*>(base+layout.hidden);
    auto* down=reinterpret_cast<float*>(base+layout.down);
    auto& registry=k12::blas_registry();std::lock_guard<std::mutex> lock(registry.mutex);
    int device;k12::check_cuda(cudaGetDevice(&device),"batched device");const auto handle=registry.get(device);
    k12::check_blas(cublasSetStream(handle,stream),"batched stream");
    k12::check_blas(cublasSetWorkspace(handle,base+layout.blas,k12::BLAS_BYTES),"batched cuBLAS workspace");
    for(int first=0;first<groups;first+=groups_per_wave) {
        batched::Wave wave{};wave.groups=std::min(groups_per_wave,groups-first);
        const int start=offsets[first];
        for(int g=0;g<=wave.groups;++g) wave.offsets[g]=offsets[first+g]-start;
        const int count=wave.offsets[wave.groups];if(!count) continue;
        const auto* descriptors=experts+first;
        // H*F has the same tile count in all projections. Logical tile
        // coordinates avoid both padding CTAs and separate prepare/launches.
        batched::reconstruct<<<dim3(k12::H/16*(k12::F/128),1,3*wave.groups),256,0,stream>>>(descriptors,wave,matrices);
        batched::input_had<<<dim3(2*count,k12::H/128),32,0,stream>>>(x,tok+start,descriptors,wave,input);
        for(int g=0;g<wave.groups;++g) {
            const int at=wave.offsets[g],n=wave.offsets[g+1]-at;if(!n) continue;
            k12::gemm(handle,input+size_t(at)*k12::H,matrices+size_t(g)*3*k12::MATRIX_ELEMENTS,
                       gu+size_t(at)*k12::F,n,k12::H,k12::F);
            k12::gemm(handle,input+size_t(count+at)*k12::H,matrices+(size_t(g)*3+1)*k12::MATRIX_ELEMENTS,
                       gu+size_t(count+at)*k12::F,n,k12::H,k12::F);
        }
        batched::activate<<<dim3(count,k12::F/128),32,0,stream>>>(weights+start,descriptors,wave,gu,hidden);
        for(int g=0;g<wave.groups;++g) {
            const int at=wave.offsets[g],n=wave.offsets[g+1]-at;if(!n) continue;
            k12::gemm(handle,hidden+size_t(at)*k12::F,matrices+(size_t(g)*3+2)*k12::MATRIX_ELEMENTS,
                       down+size_t(at)*k12::H,n,k12::F,k12::H);
            k12::output_had_add<<<dim3(n,k12::H/128),32,0,stream>>>(tok+start+at,descriptors+g,
                                                                 down+size_t(at)*k12::H,out);
        }
        k12::check_cuda(cudaPeekAtLastError(),"batched expert wave");
    }
}

size_t exl3_moe_prefill_workspace_bytes(int max_rows, int max_groups) {
    if (max_rows < 0 || max_groups < 0)
        throw std::invalid_argument("K12: negative workspace limits");
    if (max_rows == 0 || max_groups == 0) return 0;
    return k12::Layout(max_rows).bytes();
}

void exl3_moe_prefill(const __half* x, const int32_t* tok, const float* w,
                      const int32_t* off, int n_groups, const Exl3Expert* experts,
                      float* out, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (n_groups < 0) throw std::invalid_argument("K12: negative group count");
    if (n_groups == 0) return;
    if (!off || off[0] < 0) throw std::invalid_argument("K12: invalid offsets");
    for (int g = 0; g < n_groups; ++g)
        if (off[g + 1] < off[g]) throw std::invalid_argument("K12: decreasing offsets");
    const int rows = off[n_groups] - off[0];
    if (rows == 0) return;  // Do not even initialize cuBLAS for an empty call.
    if (!x || !tok || !w || !experts || !out || !workspace)
        throw std::invalid_argument("K12: null argument");
    const k12::Layout layout(rows);
    if (workspace_bytes < layout.bytes())
        throw std::invalid_argument("K12: insufficient workspace");
    const k12::Workspace ws(workspace, layout);

    auto& registry = k12::blas_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    int device;
    k12::check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    const cublasHandle_t handle = registry.get(device);
    k12::check_blas(cublasSetStream(handle, stream), "cublasSetStream");
    // SetStream resets the library workspace. Restore caller-owned scratch
    // afterward, preventing cuBLAS allocation nodes in captured graphs.
    k12::check_blas(cublasSetWorkspace(handle, ws.blas, k12::BLAS_BYTES), "cublasSetWorkspace");

    for (int g = 0; g < n_groups; ++g) {
        if (off[g] == off[g + 1]) continue;
        const Exl3Expert* expert = experts + g;
        k12::prepare<<<1, 1, 0, stream>>>(expert, ws.trellis);
        strata_exl3::reconstruct_mul1(ws.matrices, ws.trellis, k12::H, k12::F, stream);
        strata_exl3::reconstruct_mul1(ws.matrices + k12::MATRIX_ELEMENTS, ws.trellis + 1,
                                          k12::H, k12::F, stream);
        strata_exl3::reconstruct_mul1(ws.matrices + 2 * k12::MATRIX_ELEMENTS, ws.trellis + 2,
                                          k12::F, k12::H, stream);
        k12::check_cuda(cudaPeekAtLastError(), "reconstruct launches");

        for (int first = off[g]; first < off[g + 1];) {
            const int remaining = off[g + 1] - first;
            const int count = remaining < layout.rows ? remaining : layout.rows;
            k12::input_had<<<dim3(2 * count, k12::H / 128), 32, 0, stream>>>
                (x, tok + first, expert, count, ws.input);
            k12::check_cuda(cudaPeekAtLastError(), "input Hadamard launch");
            k12::gemm(handle, ws.input, ws.matrices, ws.gu, count, k12::H, k12::F);
            k12::gemm(handle, ws.input + size_t(count) * k12::H,
                       ws.matrices + k12::MATRIX_ELEMENTS, ws.gu + size_t(count) * k12::F,
                       count, k12::H, k12::F);
            k12::activate_down_had<<<dim3(count, k12::F / 128), 32, 0, stream>>>
                (w + first, expert, count, ws.gu, ws.down_input);
            k12::check_cuda(cudaPeekAtLastError(), "activation launch");
            k12::gemm(handle, ws.down_input, ws.matrices + 2 * k12::MATRIX_ELEMENTS,
                       ws.down, count, k12::F, k12::H);
            k12::output_had_add<<<dim3(count, k12::H / 128), 32, 0, stream>>>
                (tok + first, expert, ws.down, out);
            k12::check_cuda(cudaPeekAtLastError(), "scatter launch");
            first += count;
        }
    }
}

}  // namespace strata::ds41::kernels
