// include/strata/ds41/expert_prefetch.hpp - decode prefetch of the next layer's experts.
//
// While layer l computes, layer l+1's router applied to layer l's expert input guesses layer l+1's experts (upstream's
// router lookahead, docs/DETAILS.md, here on the GPU). The guesses outside VRAM that the RAM tier holds are copied to
// a VRAM buffer, one per layer parity. The guesses and the copy run on a stream of their own, beside layer l's shared
// expert, its experts and the wait for the CPU: on the main stream they cost 32 + 10 us per layer and the copy waited
// for layer l's experts (RTX 5090 Laptop, nsys, 2026-10-10). When layer l+1 routes, a miss that was guessed is
// computed by the GPU from the buffer (the doorbell's publish gets the guessed ids and their descriptors), so its
// copy is off the critical path.
// DMA mode: the copy kernel reads host memory with SM loads and slows every main-stream kernel while it runs (about
// 190 us per layer at 4 guesses). In DMA mode the plan writes the copy list to mapped host memory instead; a host
// thread copies it with the copy engine (28.7 GB/s on the laptop's PCIe 5.0 x8) and then raises a flag, which a
// one-thread kernel on main waits for before the layer's experts (as the doorbell's wait).
// Measured on SAGE 1.59bpw (RTX 3090 box, DS41_PREDICT_STATS): 9 guesses hold 77-79% of the misses, 6 hold 66%.
// Everything is enqueued on the device (capturable): no host wait.
#pragma once
#include "strata/ds41/expert_staging.hpp"
#include "strata/ds41/kernels/k10_exl3_moe.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace strata::ds41 {

namespace detail {
/// DMA mode: the layer the copy thread copies next: the lowest layer of this step (`epoch`) above `last` in the two
/// parity tags (read(p) loads tag p: epoch * 64 + layer), or -1. Once a candidate is seen, the other parity's tag is
/// read again: plans run in layer order on one stream and each tag follows a system fence, so that second read sees
/// an earlier layer that the first read missed (reading the tags once let a descheduled thread take layer l + 1 before
/// layer l, which was then never copied). No later plan can overwrite that tag first: it waits for layer l's copy.
template <class Read>
int next_copy_layer(Read read, unsigned long long epoch, int last) {
    int layer = -1;
    for (int p = 0; p < 2; ++p) {
        const unsigned long long t = read(p);
        if (t / 64 != epoch) continue;
        const int l = (int) (t % 64);
        if (l > last && (layer < 0 || l < layer)) layer = l;
    }
    if (layer < 0) return -1;
    const unsigned long long t = read((layer & 1) ^ 1);
    if (t / 64 == epoch) {
        const int l = (int) (t % 64);
        if (l > last && l < layer) layer = l;
    }
    return layer;
}
}  // namespace detail

class ExpertPrefetch {
public:
    static constexpr int kMaxGuesses = 16;

    /// `guesses` per layer (1..kMaxGuesses), a buffer of `buffer_bytes` per layer parity, routers of n_experts x dim;
    /// dma: copy with the copy engine from a host thread (the RAM tier must be pinned host memory)
    ExpertPrefetch(int guesses, size_t buffer_bytes, int n_experts, int dim, bool dma = false);
    ~ExpertPrefetch();
    ExpertPrefetch(const ExpertPrefetch&) = delete;
    ExpertPrefetch& operator=(const ExpertPrefetch&) = delete;

    int guesses() const { return guesses_; }
    bool dma() const { return dma_; }
    /// DMA mode: a copy failed (its layer's experts were then computed from a stale buffer)
    bool failed() const { return failed_.load(); }
    /// Before a step's first plan(), with the device idle: in DMA mode a new step's layers are copied again (the layer
    /// numbers restart). Nothing in the copy-kernel mode.
    void begin_step();

    /// On the prefetch stream, after the work enqueued on `main` so far: layer `layer`'s guesses from x (bf16 [dim],
    /// the previous layer's expert input) and its router (w [n_experts][dim] bf16, bias [n_experts]); the copy plan: in
    /// rank order, each guess outside VRAM (res[id] < 0) that the RAM tier holds (ram[id].w1.trellis set) while the
    /// buffer has room (res, ram, blobs: the layer's rows); then the copy. The buffer of this parity must be free.
    void plan(int layer, const __nv_bfloat16* x, const __nv_bfloat16* w, const float* bias, const int32_t* res,
              const kernels::Exl3Expert* ram, const ExpertBlob* blobs, cudaStream_t main);
    /// plan() for m rows of x ([m][dim], a verify window, 1 <= m <= 4): min(guesses, kMaxGuesses / m) guesses per row,
    /// merged in rank order (every row's best, then every row's second, ...), each expert once, then the same copy
    /// plan. ids() then holds up to kMaxGuesses entries. m = 1 is plan().
    void plan_rows(int layer, const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias,
                   const int32_t* res, const kernels::Exl3Expert* ram, const ExpertBlob* blobs, cudaStream_t main);
    /// `main` waits for the guesses of `layer` (ids() and descs()); x may be written after this.
    void ready(int layer, cudaStream_t main);
    /// `main` waits for the copy of `layer` (before the GPU computes the layer's experts).
    void join(int layer, cudaStream_t main);

    /// The layer's copied guesses [guesses] (-1 after the last) and their descriptors in the buffer (device).
    /// Valid for the layer planned last with this parity.
    const int32_t* ids(int layer) const { return ids_ + (size_t) (layer & 1) * kMaxGuesses; }
    const kernels::Exl3Expert* descs(int layer) const { return descs_ + (size_t) (layer & 1) * kMaxGuesses; }
    /// The guesses in rank order, before the residency filter (device; tests)
    const int32_t* ranked(int layer) const { return ranked_ + (size_t) (layer & 1) * kMaxGuesses; }

private:
    int guesses_, n_experts_, dim_;
    size_t buffer_bytes_;
    uint8_t* buffer_ = nullptr;                 ///< [2][buffer_bytes]
    float* logits_ = nullptr;                   ///< [4][n_experts] (plan_rows: one row per window token)
    int32_t* ids_ = nullptr;                    ///< [2][kMaxGuesses]
    int32_t* ranked_ = nullptr;                 ///< [2][kMaxGuesses]
    kernels::Exl3Expert* descs_ = nullptr;      ///< [2][kMaxGuesses]
    ExpertCopy* jobs_ = nullptr;                ///< [2][kMaxGuesses]
    int* count_ = nullptr;                      ///< [2]
    cudaStream_t copy_ = nullptr;
    cudaEvent_t ready_[2] = {}, planned_[2] = {}, done_[2] = {};

    /// DMA mode: mapped host memory shared by the plan (GPU), the copy thread and the wait (GPU). Tags are
    /// epoch * 64 + layer, so a value of an earlier step never matches.
    struct Shared {
        unsigned long long epoch;                 ///< written by begin_step
        unsigned long long tag[2];                ///< per parity: the layer planned (written by plan_k after its jobs)
        unsigned long long done;                  ///< the last layer copied (written by the copy thread)
        int count[2];
        ExpertCopy jobs[2][kMaxGuesses];
    };
    void copier();
    bool dma_ = false;
    int device_ = 0;
    Shared* shared_ = nullptr;                    ///< host address
    Shared* shared_dev_ = nullptr;                ///< its device alias
    cudaStream_t dma_stream_ = nullptr;
    std::atomic<bool> stop_{false}, failed_{false};
    std::thread thread_;
};

}  // namespace strata::ds41
