#include "strata/ds41/tensor_parallel.hpp"
#include "strata/ds41/device_scope.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <cuda/atomic>

namespace strata::ds41 {
namespace {
void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("TP2: ") + what + ": " + cudaGetErrorString(error));
}
struct alignas(64) Rendezvous {
    uint32_t sent, ready, done;
    uint4 shape[2];
};

// Push owned channels with posted P2P stores instead of non-posted remote reads.
// Each rank receives into a private ping-pong inbox, then copies locally into its
// output. Reaching epoch N+1 proves the peer consumed N-1, so two inboxes protect
// reuse even when one rank advances a generation ahead. Captured layouts never
// move: growing a slot takes a fresh region from the bounded preallocated arena.
__global__ void gather_peer(Rendezvous* mine, volatile Rendezvous* peer,
                            uint8_t* rows, int count, int columns, int bytes, int stride, int rank,
                            const volatile uint32_t* aborted, bool acknowledge,
                            uint8_t* inbox, uint8_t* peer_inbox, size_t phase_stride) {
    __shared__ uint32_t epoch;
    if(threadIdx.x==0) epoch=mine->sent+1;
    __syncthreads();
    const size_t half=size_t(columns/2)*bytes,pitch=size_t(stride)*bytes,vectors=half/16;
    const size_t phase=size_t(epoch&1u)*phase_stride;
    for(size_t i=threadIdx.x;i<size_t(count)*vectors;i+=blockDim.x) {
        const size_t src=(i/vectors)*pitch+size_t(rank)*half+(i%vectors)*16;
        *reinterpret_cast<uint4*>(peer_inbox+phase+i*16)=*reinterpret_cast<const uint4*>(rows+src);
    }
    __threadfence_system(); // order every writer's posted stores before ready
    __syncthreads();
    if(threadIdx.x==0) {
        mine->sent=epoch;
        auto* shape=const_cast<uint4*>(&peer->shape[epoch&1u]);
        *shape=make_uint4(count,columns,bytes,stride);
        auto& ready=const_cast<uint32_t&>(peer->ready);
        // Post the notification into the receiver's memory: spin locally rather
        // than issuing a non-posted PCIe read on every acquire poll.
        cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(ready).store(epoch,cuda::memory_order_release);
        const auto start=clock64();
        while(int32_t(cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(mine->ready).load(cuda::memory_order_acquire)-epoch)<0) {
            if(*aborted || clock64()-start>120000000000ull) asm volatile("trap;");
            __nanosleep(200);
        }
        const auto received=__ldcv(&mine->shape[epoch&1u]);
        if(received.x!=count || received.y!=columns || received.z!=bytes || received.w!=stride)
            asm volatile("trap;");
    }
    __syncthreads();
    for(size_t i=threadIdx.x;i<size_t(count)*vectors;i+=blockDim.x) {
        const size_t dst=(i/vectors)*pitch+size_t(1-rank)*half+(i%vectors)*16;
        *reinterpret_cast<uint4*>(rows+dst)=__ldcv(reinterpret_cast<const uint4*>(inbox+phase+i*16));
    }
    __syncthreads();
    if(threadIdx.x==0 && acknowledge) {
        auto& done=const_cast<uint32_t&>(peer->done);
        cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(done).store(epoch,cuda::memory_order_release);
        const auto start=clock64();
        while(int32_t(cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(mine->done).load(cuda::memory_order_acquire)-epoch)<0) {
            if(*aborted || clock64()-start>120000000000ull) asm volatile("trap;");
            __nanosleep(200);
        }
    }
}
}

struct TensorParallel::Impl {
    int devices[2];
    Rendezvous* flags[2] = {};
    uint32_t* abort_host = nullptr;
    uint32_t* abort_device[2] = {};
    static constexpr size_t arena_bytes=64ull<<20;
    uint8_t* inbox[2] = {};
    size_t used = 0;
    struct Layout { size_t offset=0,phase=0;int rows=0,columns=0,bytes=0; };
    std::vector<Layout> layouts[kSlots];
    std::mutex mutex;
    std::condition_variable changed;
    int arrivals = 0;
    uint64_t generation = 0;
    bool pending = true, decision = false;
};

TensorParallel::TensorParallel(const std::vector<int>& devices): impl_(new Impl) {
    if (devices.size()!=2 || devices[0]<0 || devices[1]<0 || devices[0]==devices[1])
        throw std::invalid_argument("TP2 requires two distinct CUDA devices");
    impl_->devices[0]=devices[0]; impl_->devices[1]=devices[1];
    try {
        check(cudaHostAlloc(&impl_->abort_host,sizeof(uint32_t),cudaHostAllocMapped|cudaHostAllocPortable),"abort flag");
        *impl_->abort_host=0;
        for (int rank=0;rank<2;++rank) {
            DeviceScope scope(devices[rank]);
            check(cudaHostGetDevicePointer(&impl_->abort_device[rank],impl_->abort_host,0),"abort flag alias");
            int peer=0;
            check(cudaDeviceCanAccessPeer(&peer,devices[rank],devices[1-rank]),"peer capability");
            if (!peer) throw std::invalid_argument("TP2 requires bidirectional peer access; use layer split on this topology");
            auto status=cudaDeviceEnablePeerAccess(devices[1-rank],0);
            if (status==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else check(status,"enable peer access");
            check(cudaMalloc(&impl_->flags[rank],sizeof(Rendezvous)*kSlots),"collective flags");
            check(cudaMemset(impl_->flags[rank],0,sizeof(Rendezvous)*kSlots),"clear collective flags");
            check(cudaMalloc(&impl_->inbox[rank],Impl::arena_bytes),"bounded collective inbox arena");
        }
    } catch (...) {
        for (int rank=0;rank<2;++rank) {
            DeviceScope scope(devices[rank],std::nothrow);
            if(impl_->flags[rank]) cudaFree(impl_->flags[rank]);
            if(impl_->inbox[rank]) cudaFree(impl_->inbox[rank]);
        }
        if(impl_->abort_host) cudaFreeHost(impl_->abort_host);
        throw;
    }
}
TensorParallel::~TensorParallel() {
    for (int rank=0;rank<2;++rank) {
        DeviceScope scope(impl_->devices[rank],std::nothrow);
        cudaFree(impl_->flags[rank]);cudaFree(impl_->inbox[rank]);
    }
    cudaFreeHost(impl_->abort_host);
}
void TensorParallel::abort() noexcept {
    { std::lock_guard<std::mutex> lock(impl_->mutex);__atomic_store_n(impl_->abort_host,1u,__ATOMIC_RELEASE); }
    impl_->changed.notify_all();
}
bool TensorParallel::agree(bool local) const {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    const auto generation=impl_->generation;
    impl_->pending &= local;
    if(++impl_->arrivals==2) {
        impl_->decision=impl_->pending;impl_->pending=true;impl_->arrivals=0;++impl_->generation;
        impl_->changed.notify_all();
    } else impl_->changed.wait(lock,[&] { return impl_->generation!=generation || __atomic_load_n(impl_->abort_host,__ATOMIC_ACQUIRE); });
    if(__atomic_load_n(impl_->abort_host,__ATOMIC_ACQUIRE)) throw std::runtime_error("TP2 draft scheduling aborted");
    return impl_->decision;
}
void TensorParallel::barrier(int rank,int slot,cudaStream_t stream) const {
    if(rank<0 || rank>1 || slot<0 || slot>=kSlots) throw std::invalid_argument("TP2 barrier rank/slot");
    gather_peer<<<1,256,0,stream>>>(impl_->flags[rank]+slot,impl_->flags[1-rank]+slot,
                                  nullptr,0,16,2,16,rank,impl_->abort_device[rank],false,nullptr,nullptr,0);
    check(cudaGetLastError(),"barrier launch");
}
void TensorParallel::gather(int rank,int slot,void* rows,int count,int columns,
                             int bytes,cudaStream_t stream,int stride,bool acknowledge) const {
    if (rank<0 || rank>1 || slot<0 || slot>=kSlots || !rows || count<1 || count>64 ||
        columns<2 || columns%2 || (bytes!=2 && bytes!=4) || (stride && stride<columns) ||
        (size_t(columns/2)*bytes)%16 || (size_t(stride?stride:columns)*bytes)%16 || reinterpret_cast<uintptr_t>(rows)%16)
        throw std::invalid_argument("TP2 all-gather: invalid shape, rank or collective slot");
    Impl::Layout layout;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto& choices=impl_->layouts[slot];
        int capacity=8;while(capacity<count) capacity*=2;
        const auto found=std::find_if(choices.begin(),choices.end(),[&](const auto& choice) {
            return choice.rows==capacity && choice.columns==columns && choice.bytes==bytes;
        });
        if(found==choices.end()) {
            const size_t phase=(size_t(capacity)*(columns/2)*bytes+255)&~size_t(255);
            if(2*phase>Impl::arena_bytes-impl_->used)
                throw std::runtime_error("TP2 collective inbox budget exhausted");
            layout={impl_->used,phase,capacity,columns,bytes};impl_->used+=2*phase;
            choices.push_back(layout);
        } else layout=*found;
    }
    gather_peer<<<1,256,0,stream>>>(impl_->flags[rank]+slot,impl_->flags[1-rank]+slot,
                                   static_cast<uint8_t*>(rows),count,columns,bytes,stride?stride:columns,rank,
                                  impl_->abort_device[rank],acknowledge,
                                  impl_->inbox[rank]+layout.offset,impl_->inbox[1-rank]+layout.offset,layout.phase);
    check(cudaGetLastError(),"all-gather launch");
}
}
