#include "strata/ds41/stage_transfer.hpp"
#include "strata/ds41/config.hpp"

namespace strata::ds41 {

StageTransfer::StageTransfer(int source, int destination, size_t capacity, bool force_host)
    : source_(source), destination_(destination), capacity_(capacity) {
    if (!capacity) throw std::invalid_argument("ds41 transfer: zero capacity");
    DeviceScope scope(destination_);
    int can = 0;
    if (!force_host && source_ != destination_)
        device_check(cudaDeviceCanAccessPeer(&can, destination_, source_), "peer capability");
    if (can) {
        const auto e = cudaDeviceEnablePeerAccess(source_, 0);
        if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) {
            if (e != cudaSuccess) cudaGetLastError();
            peer_ = true;  // shared CUDA-context capability; do not revoke another user's access at destruction
        } else {
            cudaGetLastError();  // denied peer access is supported via the pinned fallback
        }
    }
    if (!peer_) device_check(cudaHostAlloc(&host_, capacity_, cudaHostAllocPortable), "handoff staging");
}

StageTransfer::~StageTransfer() { if (host_) cudaFreeHost(host_); }

void StageTransfer::copy_state(void* dst_hidden, const void* src_hidden, float* dst_pre, const float* src_pre,
                               cudaStream_t producer, cudaStream_t consumer) {
    copy(dst_hidden, src_hidden, kHc * kDim * sizeof(uint16_t), producer, consumer);
    copy(dst_pre, src_pre, kHc * sizeof(float), producer, consumer);
}

void StageTransfer::copy(void* dst, const void* src, size_t bytes, cudaStream_t producer, cudaStream_t consumer) {
    if (!src || !dst || !bytes || bytes > capacity_) throw std::invalid_argument("ds41 transfer: bad copy range");
    {
        DeviceScope scope(source_);
        device_check(cudaStreamSynchronize(producer), "handoff producer");
        if (!peer_) {
            device_check(cudaMemcpyAsync(host_, src, bytes, cudaMemcpyDeviceToHost, producer), "handoff download");
            device_check(cudaStreamSynchronize(producer), "handoff download wait");
        }
    }
    DeviceScope scope(destination_);
    if (peer_) device_check(cudaMemcpyPeerAsync(dst, destination_, src, source_, bytes, consumer), "peer handoff");
    else device_check(cudaMemcpyAsync(dst, host_, bytes, cudaMemcpyHostToDevice, consumer), "host handoff");
    device_check(cudaStreamSynchronize(consumer), "handoff consumer");
}

}  // namespace strata::ds41
