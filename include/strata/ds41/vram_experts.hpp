// include/strata/ds41/vram_experts.hpp - the VRAM expert tier (upstream include/strata/core/expert_cache.hpp).
//
// Some routed experts live in VRAM slots and the GPU computes them (task K10); the CPU computes only the misses.
// The slots are filled at start from a profile (tools/ds41/make_profile.py, upstream's STRP format) in rank order.
// Experts may differ in size (SAGE 1.59bpw): as upstream's sized slots (src/core/expert_cache.cpp open_sized), the
// slots are compact, each as large as the expert that first filled it, in one arena.
// The residency table maps (layer, expert) to a slot or -1; the doorbell publish reads it on the device, so the
// GPU and the CPU split every routed expert the same way.
//
// The adaptive tier (upstream src/program/generate.cpp, adapt()) makes the slots follow the conversation: decayed
// routing counts per (layer, expert); every few steps, per layer, the most-routed missing experts replace the
// least-routed resident ones when they were routed clearly more often. With a RAM tier a swap runs in three copies
// switched in between steps (copy_worker), so the CPU never reads a swapped expert from the file. The copies run on their own thread and
// stream, because the pack is mmap'ed (pageable) and a pageable copy holds the calling thread. A swap needs the new
// expert to fit the old one's slot (upstream swaps only within a layer, whose experts have one size there; here a
// layer mixes sizes, so the plan checks it).
#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"
#include "strata/ds41/pack.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::ds41 {

class HostExperts;
class BoundedUpload;

/// Strict opt-in MiB budget, unset/0 disables; rejects signs, suffixes and >8192.
size_t experimental_decode_cache_bytes(const char* value);
/// Total pinned RAM budget, even integer MiB in 0..16384; 0/unset disables.
size_t experimental_ram_cache_bytes(const char* value);
/// Optional logical stage/rank split A,B in MiB; sum must equal total_bytes.
size_t experimental_ram_cache_stage_bytes(const char* split, size_t total_bytes, int rank);
/// Optional exact static counts by logical stage/rank, for matched experiments.
/// Unset gives -1 (automatic); supplied counts must both be in 1..15360.
int experimental_resident_slot_limit(const char* value, int rank);

/// Reads a STRP profile: the ranked (layer, expert) pairs. Throws when the file is not a profile of this shape.
std::vector<std::pair<int, int>> read_expert_profile(const std::string& path, int n_layers, int n_experts);

struct ExpertSwap {
    float gain;
    int32_t layer, in, out;   ///< `in` (missing) takes the slot of `out` (resident), same layer
};
/// Upstream's swap choice. usage and res: [n_layers][n_experts]. Per layer, candidates are missing experts with
/// usage >= 2, victims the resident ones; the most-used candidates first, each pairs with the least-used victim
/// not taken yet that it fits (fits(layer, in, out); empty: every expert fits), if it leads that victim by 1.5 or
/// more. With one expert size this is upstream's pairing of the i-th candidate with the i-th victim. All layers'
/// swaps, largest gain first, at most max_swaps.
std::vector<ExpertSwap> plan_expert_swaps(const std::vector<float>& usage, const std::vector<int32_t>& res,
                                          int n_layers, int n_experts, int max_swaps,
                                          const std::function<bool(int, int, int)>& fits = {});

class VramExperts {
public:
    /// Workspace the K10 kernel gets (its interface asks for at least 64 MiB)
    static constexpr size_t kWorkspaceBytes = 64ull << 20;

    struct Adapt {
        int every = 4;          ///< steps between adaptations; 0 = static residency
        float decay = 0.7f;     ///< usage counts are multiplied by this after each adaptation
        int max_swaps = 96;
    };

    /// Fills `n_slots` slots with the profile's first pairs, copied from the mapped pack. n_slots < 0: the profile's
    /// first pairs while their bytes fit in the free VRAM after keeping `reserve_bytes` free (and the workspace), as
    /// upstream (it stops at the first that does not fit). 0 slots is valid (no tier).
    VramExperts(const Pack& pack, const std::string& profile_path, int64_t n_slots, size_t reserve_bytes,
                Adapt adapt, int first_layer = 0, int end_layer = -1, bool require_all = false, int tensor_rank = -1,
                bool bounded_misses = false, size_t decode_cache_bytes = 0, size_t ram_cache_bytes = 0,
                bool async_cached_routes = false);
    ~VramExperts();
    VramExperts(const VramExperts&) = delete;
    VramExperts& operator=(const VramExperts&) = delete;

    int slots() const { return slots_; }
    /// the arena: the slots' capacities summed
    size_t arena_bytes() const { return off_.back(); }
    /// the largest slot
    size_t max_slot_bytes() const { return max_slot_bytes_; }
    /// the bytes of the last n slots (what lend(n) gives)
    size_t tail_bytes(int n) const { return off_[slots_] - off_[slots_ - n]; }
    /// [n_layers][n_experts] slot or -1, on the device; layer l starts at res_dev() + l * n_experts
    const int32_t* res_dev() const { return res_dev_; }
    const std::vector<int32_t>& res_host() const { return res_host_; }
    /// [slots] expert descriptors on the device (pointers into the slots)
    const kernels::Exl3Expert* experts_dev() const { return experts_dev_; }
    void* workspace() const { return ws_; }
    /// Experimental eager decode only: stage every missing selected expert, then
    /// publish a complete GPU descriptor selection. No CPU expert computation.
    void prepare_decode(int layer, const int32_t* ids, int topk, int32_t* selected, cudaStream_t stream,
                        bool resident_fastpath = false);
    /// Hit-only 1..6-row selection. False means no cache/LRU/selection changes;
    /// caller stages rows normally. One stream fence protects prior consumers;
    /// the selection is published by-value, never from a borrowed host buffer.
    bool try_cached_rows(int layer,const int32_t* ids,int rows,int topk,int32_t* selected,cudaStream_t stream);
    int64_t cached_row_batches() const { return cached_row_batches_; }
    /// Optional two-phase equivalent. One owned pinned read may be in flight;
    /// finish preserves try_cached_rows' validation, miss and LRU semantics.
    void begin_cached_rows(int layer,const int32_t* ids,int rows,int topk,int32_t* selected,cudaStream_t stream);
    bool finish_cached_rows();
    void cancel_cached_rows() noexcept;
    int64_t async_cached_row_reads() const { return async_cached_row_reads_; }
    struct PrefillPrime { int gpu_fills=0,ram_fills=0;uint64_t bytes=0; };
    /// Idle-engine operation only. Fills empty slots; retains every valid entry.
    /// Lookups remain immutable until the caller joins every prefill reader.
    PrefillPrime prime_prefill_cache();
    int prefill_gpu_slot(int layer,int expert) const;
    const uint8_t* prefill_ram_source(int layer,int expert) const;
    bool fully_resident_layer(int layer) const { return !lent_ && full_layers_.at(layer); }
    bool streaming() const { return bool(miss_upload_); }
    int64_t streamed_decode_experts() const { return streamed_decode_; }
    int decode_cache_slots() const { return int(miss_cache_.size()); }
    int64_t decode_cache_hits() const { return cache_hits_; }
    int64_t decode_cache_evictions() const { return cache_evictions_; }
    int ram_cache_slots() const { return int(ram_cache_.size()); }
    size_t ram_cache_bytes() const { return ram_cache_.size() * ram_slot_bytes_; }
    int64_t ram_cache_hits() const { return ram_hits_; }
    int64_t ram_cache_fills() const { return ram_fills_; }
    uint64_t ram_cache_read_bytes() const { return ram_read_bytes_; }

    /// Count one step's routing: routes [n_layers][topk], host memory.
    void count(const int32_t* routes, int topk);
    /// Call between steps, with no work of the tier in flight on the device. Commits swaps whose copies have
    /// landed; every `every` calls plans new swaps, evicts their victims and starts the copies. Returns the
    /// number of swaps committed now.
    int between_steps();
    int64_t swaps_total() const { return swaps_total_; }
    /// With a RAM tier: an expert swapped out of VRAM takes the RAM slot of the one swapped in (upstream), so swaps
    /// read nothing from the file. Set before the first step.
    void set_host(HostExperts* host);

    /// Prefill borrows cache slots, as upstream does for its expert ring and scratch: the last n slots (contiguous)
    /// leave the residency table and their memory goes to the caller until restore(). Call between steps; adaptive
    /// copies in flight are finished and committed first. Returns the first lent byte.
    uint8_t* lend(int n);
    /// lend() of the fewest last slots that hold `bytes`. Throws if all slots together hold less.
    uint8_t* lend_bytes(size_t bytes);
    /// Copies the lent slots' experts back from the pack and enters them in the table again.
    void restore();
    int lent() const { return lent_; }
    /// The descriptor of slot s (valid while the slot is not lent)
    const kernels::Exl3Expert& desc(int slot) const { return desc_host_[slot]; }
    /// The descriptor of (layer, expert) whose pack bytes sit at `dev` in device memory
    static kernels::Exl3Expert describe_at(const Pack& pack, int layer, int expert, const uint8_t* dev);
    /// table[idx[i]] = desc[i] for i < n on the legacy default stream (idx, desc: host memory mapped to the device)
    static void scatter_descriptors(kernels::Exl3Expert* table, const int32_t* idx, const kernels::Exl3Expert* desc,
                                    int n);

private:
    void cleanup() noexcept;
    bool resolve_cached_rows(int layer,const int32_t* routes,int n,int32_t* selected,cudaStream_t stream);
    kernels::Exl3Expert describe(int layer, int expert, int slot) const;
    void upload_res();
    const uint8_t* ram_payload(int layer, int expert);
    int commit_pending(bool wait);   ///< commits finished adaptive copies (waits for them when `wait`); returns count
    struct Pending {
        int32_t layer, in, out;
        int32_t vram_slot;   ///< the slot `out` leaves and `in` takes
        int32_t ram_slot;    ///< `in`'s RAM slot, which `out` takes; -1: `in` came from the file
        size_t staged;       ///< offset of `out` in the swap buffer (with a RAM slot)
    };
    void copy_worker(std::vector<Pending> work, int phase);
    void start_phase(int phase);

    const Pack& pack_;
    Adapt adapt_;
    int device_ = 0;
    int first_layer_ = 0, end_layer_ = 0;
    int slots_ = 0;
    std::vector<size_t> off_{0};       ///< [slots + 1] slot offsets in the arena, 256 B aligned
    size_t max_slot_bytes_ = 0;
    uint8_t* arena_ = nullptr;
    int32_t* res_dev_ = nullptr;
    std::vector<int32_t> res_host_;
    std::vector<uint8_t> full_layers_;
    kernels::Exl3Expert* experts_dev_ = nullptr;
    std::vector<kernels::Exl3Expert> desc_host_;
    void* ws_ = nullptr;
    std::unique_ptr<BoundedUpload> miss_upload_;
    uint8_t* miss_arena_ = nullptr;
    int tensor_rank_ = -1;
    int64_t streamed_decode_ = 0;
    struct CacheEntry { int owner = -1; uint64_t age = 0; };
    std::vector<CacheEntry> miss_cache_;
    std::vector<int32_t> miss_cache_lookup_;
    uint8_t* miss_cache_arena_ = nullptr;
    uint64_t cache_clock_ = 0;
    bool frequency_cache_ = false;
    std::vector<uint64_t> route_frequency_;
    int64_t cache_hits_ = 0, cache_evictions_ = 0;
    int64_t cached_row_batches_ = 0;
    int32_t* async_routes_ = nullptr;  ///< owned pinned read target; never stack memory
    bool async_routes_failed_ = false; ///< failed drain poisons this tier, never reuses the host buffer
    struct PendingRows {
        int layer = 0, count = 0;
        int32_t* selected = nullptr;
        cudaStream_t stream = nullptr;
        bool active = false;
    } pending_rows_;
    int64_t async_cached_row_reads_ = 0;
    uint8_t* ram_arena_ = nullptr;
    size_t ram_slot_bytes_ = 0;
    std::vector<CacheEntry> ram_cache_;
    std::vector<int32_t> ram_lookup_;
    uint64_t ram_clock_ = 0, ram_read_bytes_ = 0;
    int64_t ram_hits_ = 0, ram_fills_ = 0;
    // adaptive tier
    std::vector<float> usage_;
    int64_t calls_ = 0, swaps_total_ = 0;
    cudaStream_t copy_stream_ = nullptr;
    std::thread copier_;
    std::atomic<bool> copies_done_{false};
    bool copy_error_ = false;
    std::vector<Pending> pending_;     ///< swaps in flight
    int phase_ = 0;                    ///< 0: none in flight; 1, 2, 3: the copy running (see copy_worker)
    HostExperts* host_ = nullptr;
    uint8_t* staging_ = nullptr;       ///< pinned swap buffer: the evicted experts on their way from VRAM to RAM
    size_t staging_bytes_ = 0;
    int lent_ = 0;                     ///< slots lent to prefill: the last lent_ slots
    std::vector<std::pair<int, int>> lent_owner_;   ///< (layer, expert) of each lent slot, -1 for an empty slot
};

}  // namespace strata::ds41
