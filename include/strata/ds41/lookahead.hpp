// include/strata/ds41/lookahead.hpp - router lookahead: warm the next layer's experts in the file cache
// (upstream docs/DETAILS.md, "A RAM budget": "while the CPU works on a layer, a thread applies the next layer's
// router to this layer's input and asks the OS for the pages of the predicted experts that neither the GPU nor the
// RAM budget holds").
//
// Only pages are requested (madvise MADV_WILLNEED); the experts computed are the ones the GPU router picks, so the
// output does not change. The prediction is a guess: layer l+1's real input differs from layer l's. The engine
// counts how many of the warmed experts were then used (`useful`), so the guess is judged by measurement.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::ds41 {

/// DeepSeek V4.1's routing on the CPU, as kernels::router_topk computes it: logits = x . w_e in FP32 (fp16 x, bf16
/// w), s_e = sqrt(softplus(logit_e)), the k largest s_e + bias_e (ties: lower id). x [dim] fp16 bits, w [n][dim]
/// bf16 bits, bias [n]. Writes the k ids, best first.
void cpu_router_topk(const uint16_t* x, const uint16_t* w, const float* bias, int n, int dim, int k, int32_t* ids);

class RouterLookahead {
public:
    /// router_w[l]: [n_experts][dim] bf16 bits of layer l, bias[l]: [n_experts]. `warm(layer, expert)` is called for
    /// each predicted expert of the next layer; it decides whether the expert needs warming (file tier) and does it,
    /// returning true when it asked the OS for pages.
    /// `fetch(layer, ids, n)`, when given, then gets the experts warm() accepted, in prediction order, once per
    /// prediction and without the residency lock (warm() runs under it). `ahead`: layer l's input predicts layer
    /// l + ahead (1: the next layer).
    RouterLookahead(std::vector<std::vector<uint16_t>> router_w, std::vector<std::vector<float>> bias, int n_experts,
                    int dim, int k, std::function<bool(int, int)> warm,
                    std::function<void(int, const int32_t*, int)> fetch = nullptr, int ahead = 1);
    ~RouterLookahead();
    RouterLookahead(const RouterLookahead&) = delete;
    RouterLookahead& operator=(const RouterLookahead&) = delete;

    /// Layer `layer`'s expert input x (fp16 bits) is ready: predict layer + ahead in the background. Never blocks for
    /// long; a request that arrives while the previous one still runs replaces it.
    void post(int layer, const uint16_t* x);
    /// The experts layer `layer` really routed to the CPU's file tier: counts how many had been warmed.
    void observe(int layer, const int32_t* file_ids, int n);

    /// The router of layer `layer` on x, now (on the calling thread): the k best ids (k <= n_experts)
    void predict_now(int layer, const uint16_t* x, int k, int32_t* ids) const;

    struct Stats { int64_t predicted = 0, warmed = 0, useful = 0; };
    Stats take_stats();   ///< since the last call

private:
    void run();

    std::vector<std::vector<uint16_t>> w_;
    std::vector<std::vector<float>> bias_;
    int n_experts_, dim_, k_, ahead_;
    std::function<bool(int, int)> warm_;
    std::function<void(int, const int32_t*, int)> fetch_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false, have_ = false;
    int layer_ = -1;
    std::vector<uint16_t> x_;
    std::vector<std::vector<int32_t>> warmed_;   ///< [layer] experts warmed for it (the last prediction)
    std::atomic<int64_t> predicted_{0}, warmed_n_{0}, useful_{0};
    std::thread th_;
};

}  // namespace strata::ds41
