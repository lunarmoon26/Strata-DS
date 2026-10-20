#pragma once

#include "strata/ds41/device_scope.hpp"
#include <cstddef>

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
    bool peer() const { return peer_; }
private:
    int source_, destination_;
    size_t capacity_;
    bool peer_ = false;
    void* host_ = nullptr;
};

}  // namespace strata::ds41
