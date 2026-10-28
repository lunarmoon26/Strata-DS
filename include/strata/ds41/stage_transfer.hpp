#pragma once

#include "strata/ds41/device_scope.hpp"
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace strata::ds41 {

// Synchronous stage boundary. The producer and consumer streams are explicitly
// ordered; the pinned buffer is never reused until its H2D copy has completed.
class StageTransfer {
public:
    StageTransfer(int source, int destination, size_t capacity, bool force_host = false);
    ~StageTransfer();
    StageTransfer(const StageTransfer&) = delete;
    StageTransfer& operator=(const StageTransfer&) = delete;
    void copy(void* dst, const void* src, size_t bytes, cudaStream_t producer, cudaStream_t consumer);
    void copy_state(void* dst_hidden, const void* src_hidden, float* dst_pre, const float* src_pre,
                    cudaStream_t producer, cudaStream_t consumer);
    struct Buffer { void* dst; const void* src; size_t bytes; };
    struct Stats {
        uint64_t bytes = 0, copy_calls = 0, host_waits = 0, event_waits = 0, groups = 0;
    };
    // Whole peer buffers, one producer dependency and one completion fence.
    // The host path still chunks through capacity_ bytes of pinned staging.
    // Return (or error unwind) ends every use of the caller's buffers.
    void copy_group(std::initializer_list<Buffer> buffers, cudaStream_t producer, cudaStream_t consumer);
    void copy_group(const Buffer* buffers, size_t count, cudaStream_t producer, cudaStream_t consumer);
    void reset_stats() { stats_ = Stats{}; }
    const Stats& stats() const { return stats_; }
    bool peer() const { return peer_; }
private:
    int source_, destination_;
    size_t capacity_;
    bool peer_ = false;
    void* host_ = nullptr;
    cudaEvent_t ready_ = nullptr, done_ = nullptr;
    Stats stats_;
};

}  // namespace strata::ds41
