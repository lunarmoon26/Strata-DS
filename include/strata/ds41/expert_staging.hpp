// Fixed storage and a graph-capturable copy branch for mapped RAM experts.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::ds41 {

// Immutable metadata from the pack. first_trellis is relative to the start of the blob.
struct ExpertBlob { size_t bytes, first_trellis; };
// Both addresses must be 16-byte aligned. bytes need not be aligned.
struct ExpertCopy { const uint8_t* src; uint8_t* dst; size_t bytes; };

class ExpertStaging {
public:
    ExpertStaging(int max_jobs, size_t max_expert_bytes);
    ~ExpertStaging();
    ExpertStaging(const ExpertStaging&) = delete;
    ExpertStaging& operator=(const ExpertStaging&) = delete;

    int capacity() const { return capacity_; }
    size_t stride() const { return stride_; }
    uint8_t* data() const { return data_; }
    ExpertCopy* jobs() const { return jobs_; }
    int* count() const { return count_; }

    // Publish jobs and count on main first. Run independent work on main between these calls.
    // Join before reading the staging bytes or reusing any call buffer. All calls are capturable.
    void fork_copy(cudaStream_t main);
    void join(cudaStream_t main);

private:
    int capacity_;
    size_t stride_;
    uint8_t* data_ = nullptr;
    ExpertCopy* jobs_ = nullptr;
    int* count_ = nullptr;
    cudaStream_t copy_ = nullptr;
    cudaEvent_t ready_ = nullptr, done_ = nullptr;
};

} // namespace strata::ds41
