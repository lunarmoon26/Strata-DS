// include/strata/ds41/doorbell.hpp - the per-layer handoff between the decode stream and the CPU expert thread.
//
// Upstream Strata (src/core/verify.cpp) runs the routed experts that miss the VRAM cache on the CPU, in place in
// RAM, while the GPU runs everything else. The two sides meet in mapped (zero-copy) host memory:
//   1. the GPU routes, then a publish kernel writes the expert input, the miss ids and the weights to mapped
//      memory and raises `seq` to the round number;
//   2. a CPU thread spins on `seq`, computes the misses, writes the rows to mapped memory and raises `done`;
//   3. the GPU, after its own work (shared expert, VRAM hits), spins on `done` and adds the CPU rows.
// No host synchronization and no allocation happens on the GPU side, so a decode step stays capturable as one
// CUDA graph. Round numbers are fixed per layer (1, 2, ...) and the host resets both words before each step.
#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"
#include "strata/ds41/expert_staging.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>

namespace strata::ds41 {

/// Mapped host buffers for one handoff of up to `max_m` tokens x `topk` experts, `dim` wide.
class ExpertDoorbell {
public:
    ExpertDoorbell(int max_m, int topk, int dim);
    ~ExpertDoorbell();
    ExpertDoorbell(const ExpertDoorbell&) = delete;
    ExpertDoorbell& operator=(const ExpertDoorbell&) = delete;

    int max_m() const { return max_m_; }
    int topk() const { return topk_; }
    int dim() const { return dim_; }

    // ---- GPU side (stream-ordered, graph-capturable)
    /// Publish x [m][dim] fp16 bits, CPU ids, and weights. Then raise seq to round.
    /// res is this layer's VRAM slot table, or null. GPU-assigned CPU ids become -1.
    /// Legacy mode (vram and ram null): gpu_sel holds VRAM slot indices or -1.
    /// With descriptors supplied: gpu_sel holds i or -1 into gpu_experts() [m][topk].
    /// Supply vram whenever res is supplied in descriptor mode. ram is this layer's RAM table.
    /// quota is one device int, capped to [0, topk]. Null means zero.
    /// Each token sends its first quota eligible RAM misses to K10, in routing order.
    /// Update tables and quota only between steps. Buffers must survive the graph.
    /// With stage: blobs is this layer's device metadata. Publish writes copy jobs and rebases RAM descriptors.
    /// Call stage->fork_copy(stream), then stage->join(stream) before K10. Storage must cover m * topk slots.
    /// Each blob must fit a staging slot. Its mapped base must be 16-byte aligned.
    void publish(const uint16_t* x, const int32_t* ids, const float* w, int m, const int32_t* res, int32_t* gpu_sel,
                 uint32_t round, cudaStream_t stream,
                 const kernels::Exl3Expert* vram = nullptr, const kernels::Exl3Expert* ram = nullptr,
                 const int* quota = nullptr, ExpertStaging* stage = nullptr, const ExpertBlob* blobs = nullptr,
                 const int32_t* pf_ids = nullptr, const kernels::Exl3Expert* pf_desc = nullptr, int pf_n = 0,
                 int tail_rank = 0, int ssd_rank = 0);
    /// Per-call descriptors. Consume on the publish stream before the next publish.
    const kernels::Exl3Expert* gpu_experts() const { return gpu_experts_; }
    /// prefetched: misses the GPU computes from the prefetch buffer (pf_ids / pf_desc of publish), outside the quota.
    /// skipped: with tail_rank R > 0 (upstream's STRATA_ROUTE_TAIL_SKIP), the uses of an expert outside VRAM that every
    /// token of the call routes at rank R or lower (ids in router order, 0 = best): neither the GPU nor the CPU computes
    /// them, they add nothing, the other experts keep their weights. Only for calls of up to 5 tokens (one pass).
    /// ssd_rank S > 0 skips the same way an expert in neither VRAM nor `ram` (read from the SSD) routed at rank S or lower.
    struct Counts { int vram, zero_copy, cpu, prefetched, skipped; };
    /// Read after wait_published, before mark_done. Counts refer to this call.
    Counts counts() const { return *h_counts_; }
    /// Wait until the CPU raised done to `round`, then out[i] += its rows [m][dim] (FP32).
    void wait_add(float* out, int m, uint32_t round, cudaStream_t stream);

    // ---- host side
    /// Zero both words. Call only while no step is in flight.
    void reset();
    /// Spin until the GPU published `round` (acquire). Returns false if `stop` became true first.
    bool wait_published(uint32_t round, const std::atomic<bool>& stop) const;
    const uint16_t* x() const { return h_x_; }
    const int32_t* ids() const { return h_ids_; }
    const float* w() const { return h_w_; }
    float* y() { return h_y_; }
    /// Make the rows in y() visible to the GPU (release), as round `round`.
    void mark_done(uint32_t round);

private:
    int max_m_, topk_, dim_;
    kernels::Exl3Expert* gpu_experts_ = nullptr;
    Counts *h_counts_ = nullptr, *d_counts_ = nullptr;
    void* host_ = nullptr;   // one cudaHostAlloc block holding every buffer below
    uint32_t *h_seq_ = nullptr, *d_seq_ = nullptr;
    uint32_t *h_done_ = nullptr, *d_done_ = nullptr;
    uint16_t *h_x_ = nullptr, *d_x_ = nullptr;
    int32_t *h_ids_ = nullptr, *d_ids_ = nullptr;
    float *h_w_ = nullptr, *d_w_ = nullptr;
    float *h_y_ = nullptr, *d_y_ = nullptr;
};

/// The tail_rank and ssd_rank of publish (0 = off, valid 1..5). route_env / ssd_env: DS41_ROUTE_TAIL_SKIP /
/// DS41_SSD_TAIL_SKIP or null. Defaults, with a VRAM tier under 90% of the experts (upstream #1884): tail 4 (ranks 5
/// and 6 of 6) and SSD 3 (ranks 4..6 of experts read from the SSD; laptop, 2 rounds: code 42.1 -> 39.2 ms/token,
/// 4K 50.3 -> 46.6, short prompts neutral; NLL code +0.006). The SSD default follows the tail skip: tail 0 turns it off.
struct SkipRanks { int tail, ssd; };
SkipRanks skip_ranks(const char* route_env, const char* ssd_env, bool has_vram, double cached_share);

}  // namespace strata::ds41
