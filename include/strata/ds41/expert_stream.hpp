// include/strata/ds41/expert_stream.hpp - prefill's expert stream (upstream's Stager, issuer and ring:
// src/prefill/prefill.cpp, struct Stager and the D-5 issuer thread).
//
// Prefill computes every routed expert on the GPU. Experts that are not in the VRAM tier are copied into a ring of
// device slots while the GPU computes earlier ones. Three roles, each waiting only for what it needs:
//   - reader threads take the jobs in order and put job j's bytes into pinned host buffer j % host_buffers (pread
//     from the pack, or nothing when the expert is in the RAM tier: its slot is pinned already); a buffer is
//     refilled once the DMA of the job before it in that buffer is done;
//   - the issuer thread copies job j into ring slot j % slots on its own copy stream, in order, once the job is read
//     and job j - slots is released; the slot's reuse waits on the GPU (cudaStreamWaitEvent on the consumer's
//     "used" event), not on the host;
//   - the consumer waits on the host only until job j's copy is issued, and makes its stream wait for the copy on
//     the GPU (wait); release records that its stream is done with the slot.
//
// File cache, as upstream's file tier (v0.1.40, #286 / #577 / #773): the hottest experts have a RAM copy (the RAM
// tier, HostExperts: no read at all); the rest are read through the OS file cache, or - when the cache could not
// keep them anyway (unbuffered) - with O_DIRECT, so a 190 GB stream does not push the RAM copy's neighbours and
// decode's pages out of RAM for nothing.
#pragma once

#include "strata/ds41/pack.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::ds41 {

class HostExperts;

class ExpertStream {
public:
    /// ring: `slots` device slots of `slot_bytes` each. host: the RAM tier or null. readers: reader threads.
    /// host_buffers: pinned staging buffers of slot_bytes each (how far the reads can run ahead of the copies).
    /// unbuffered: read the pack with O_DIRECT (4 KiB-aligned windows); false, or O_DIRECT refused: plain reads.
    ExpertStream(const Pack& pack, const HostExperts* host, uint8_t* ring, int slots, size_t slot_bytes, int readers,
                 int host_buffers, bool unbuffered = false);
    bool unbuffered() const { return direct_; }
    ~ExpertStream();
    ExpertStream(const ExpertStream&) = delete;
    ExpertStream& operator=(const ExpertStream&) = delete;

    /// Append jobs (layer, expert) in the order the consumer will take them. Returns the index of the first.
    int64_t push(const std::vector<std::pair<int, int>>& jobs);
    /// Blocks until job j's copy is issued, then makes `stream` wait for it (on the GPU). Returns the slot. Throws
    /// if a read or copy failed.
    uint8_t* wait(int64_t j, cudaStream_t stream);
    /// The consumer is done with job j once the work enqueued on `stream` so far has run: job j + slots may then
    /// overwrite the slot. Release jobs in order.
    void release(int64_t j, cudaStream_t stream);
    /// Blocks until every pushed job is copied and released, then forgets the jobs (job numbers restart at 0).
    void drain();

    struct Stats {
        /// experts by source; unbuffered reads all count as from_ssd (they read the drive whatever the cache holds)
        int64_t jobs = 0, from_ram = 0, from_cache = 0, from_ssd = 0;
        double consumer_wait_ms = 0;                                     ///< time wait() blocked
    };
    Stats take_stats();
    int slots() const { return slots_; }

private:
    void cleanup() noexcept;
    void reader();
    void issuer();
    void fail(const std::string& what);

    const Pack& pack_;
    const HostExperts* host_;
    bool direct_ = false;
    uint8_t* ring_;
    int slots_, n_host_;
    size_t slot_bytes_;
    int fd_ = -1;
    int device_ = 0;
    cudaStream_t copy_ = nullptr;
    std::vector<uint8_t*> host_buf_;          ///< pinned staging buffers
    std::vector<cudaEvent_t> dma_done_;       ///< per host buffer: its last DMA is done
    std::vector<cudaEvent_t> copied_;         ///< per slot: the copy into it is done
    std::vector<cudaEvent_t> used_;           ///< per slot: the consumer is done with it
    std::vector<std::thread> threads_;

    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::string error_;
    std::vector<std::pair<int, int>> jobs_;
    std::vector<const uint8_t*> src_;         ///< per job: the bytes to copy (a host buffer or a RAM-tier slot)
    std::vector<uint8_t> read_;               ///< per job: src_ is ready
    int64_t next_read_ = 0;                   ///< next job a reader takes
    int64_t issued_ = 0;                      ///< jobs [0, issued_) are copied on the copy stream
    int64_t released_ = 0;                    ///< jobs [0, released_) are released
    std::atomic<int64_t> n_ram_{0}, n_cache_{0}, n_ssd_{0};
    double wait_ms_ = 0;
};

}  // namespace strata::ds41
