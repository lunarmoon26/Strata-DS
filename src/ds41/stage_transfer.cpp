#include "strata/ds41/stage_transfer.hpp"
#include "strata/ds41/config.hpp"
#include <algorithm>

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

StageTransfer::~StageTransfer() {
    { DeviceScope scope(source_, std::nothrow); if (ready_) cudaEventDestroy(ready_); }
    { DeviceScope scope(destination_, std::nothrow); if (done_) cudaEventDestroy(done_); }
    if (host_) cudaFreeHost(host_);
}

void StageTransfer::copy_state(void* dst_hidden, const void* src_hidden, float* dst_pre, const float* src_pre,
                               cudaStream_t producer, cudaStream_t consumer) {
    copy(dst_hidden, src_hidden, kHc * kDim * sizeof(uint16_t), producer, consumer);
    copy(dst_pre, src_pre, kHc * sizeof(float), producer, consumer);
}

void StageTransfer::copy(void* dst, const void* src, size_t bytes, cudaStream_t producer, cudaStream_t consumer) {
    if (!src || !dst || !bytes || bytes > capacity_) throw std::invalid_argument("ds41 transfer: bad copy range");
    stats_.bytes += bytes; ++stats_.groups;
    {
        DeviceScope scope(source_);
        device_check(cudaStreamSynchronize(producer), "handoff producer");
        ++stats_.host_waits;
        if (!peer_) {
            device_check(cudaMemcpyAsync(host_, src, bytes, cudaMemcpyDeviceToHost, producer), "handoff download");
            device_check(cudaStreamSynchronize(producer), "handoff download wait");
            ++stats_.copy_calls; ++stats_.host_waits;
        }
    }
    DeviceScope scope(destination_);
    if (peer_) device_check(cudaMemcpyPeerAsync(dst, destination_, src, source_, bytes, consumer), "peer handoff");
    else device_check(cudaMemcpyAsync(dst, host_, bytes, cudaMemcpyHostToDevice, consumer), "host handoff");
    device_check(cudaStreamSynchronize(consumer), "handoff consumer");
    ++stats_.copy_calls; ++stats_.host_waits;
}

void StageTransfer::copy_group(std::initializer_list<Buffer> buffers, cudaStream_t producer, cudaStream_t consumer) {
    copy_group(buffers.begin(), buffers.size(), producer, consumer);
}

void StageTransfer::copy_group(const Buffer* buffers, size_t count, cudaStream_t producer, cudaStream_t consumer) {
    if (!buffers || !count) throw std::invalid_argument("ds41 transfer: empty buffer group");
    for (size_t i = 0; i < count; ++i) {
        const auto& b = buffers[i];
        if (!b.src || !b.dst || !b.bytes) throw std::invalid_argument("ds41 transfer: bad grouped copy range");
    }
    try {
        {
            DeviceScope scope(source_);
            if (!ready_) device_check(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming), "handoff ready event");
            device_check(cudaEventRecord(ready_, producer), "handoff producer event");
        }
        DeviceScope scope(destination_);
        if (!done_) device_check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming), "handoff done event");
        ++stats_.groups;
        if (peer_) {
            device_check(cudaStreamWaitEvent(consumer, ready_, 0), "handoff consumer dependency");
            ++stats_.event_waits;
            for (size_t i = 0; i < count; ++i) {
                const auto& b = buffers[i];
                device_check(cudaMemcpyPeerAsync(b.dst, destination_, b.src, source_, b.bytes, consumer), "bulk peer handoff");
                ++stats_.copy_calls; stats_.bytes += b.bytes;
            }
        } else {
            // A single bounded host buffer cannot be refilled until H2D has
            // consumed it. These per-piece waits are intentional fallback only.
            device_check(cudaEventSynchronize(ready_), "host handoff producer wait");
            ++stats_.host_waits;
            for (size_t i = 0; i < count; ++i) for (size_t off = 0; off < buffers[i].bytes; off += capacity_) {
                const auto& b = buffers[i];
                const size_t n = std::min(capacity_, b.bytes - off);
                {
                    DeviceScope source(source_);
                    device_check(cudaMemcpyAsync(host_, static_cast<const uint8_t*>(b.src) + off, n, cudaMemcpyDeviceToHost, producer), "grouped host download");
                    device_check(cudaStreamSynchronize(producer), "grouped host download wait");
                }
                device_check(cudaMemcpyAsync(static_cast<uint8_t*>(b.dst) + off, host_, n, cudaMemcpyHostToDevice, consumer), "grouped host upload");
                device_check(cudaStreamSynchronize(consumer), "grouped host reuse fence");
                stats_.copy_calls += 2; stats_.host_waits += 2; stats_.bytes += n;
            }
        }
        device_check(cudaEventRecord(done_, consumer), "handoff completion event");
        device_check(cudaEventSynchronize(done_), "handoff group completion");
        ++stats_.host_waits;
    } catch (...) {
        // Even a partially queued group may still own source/destination or
        // pinned staging. Drain both streams before caller cleanup can free it.
        { DeviceScope scope(source_, std::nothrow); cudaStreamSynchronize(producer); }
        { DeviceScope scope(destination_, std::nothrow); cudaStreamSynchronize(consumer); }
        throw;
    }
}

}  // namespace strata::ds41
