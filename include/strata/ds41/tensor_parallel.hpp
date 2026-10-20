#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <vector>

namespace strata::ds41 {
/// Two-rank, intra-process peer all-gather. Each rank writes its disjoint output
/// channels; the collective materializes the full activation on both ranks.
/// Flags and addresses are device-resident, so CUDA graph replay needs no host
/// rendezvous and never changes model values. Requires bidirectional CUDA P2P.
class TensorParallel {
public:
    static constexpr int kSlots = 1024;
    explicit TensorParallel(const std::vector<int>& devices);
    ~TensorParallel();
    TensorParallel(const TensorParallel&) = delete;
    TensorParallel& operator=(const TensorParallel&) = delete;
    void gather(int rank, int slot, void* rows, int count, int columns,
                int element_bytes, cudaStream_t stream, int stride = 0, bool acknowledge = true) const;
    // The private ping-pong inbox protects source reuse without acknowledgement.
    // acknowledge=true adds a consumption rendezvous for callers that need it.
    void abort() noexcept;
    void barrier(int rank,int slot,cudaStream_t stream) const;
    bool agree(bool local) const; ///< eager host-side AND for bounded draft-cache scheduling
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
