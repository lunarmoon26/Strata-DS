// src/ds41/lookahead.cpp - see include/strata/ds41/lookahead.hpp.
#include "strata/ds41/lookahead.hpp"
#include "strata/ds41/residency.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace strata::ds41 {

namespace {

float f16(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, b;
    if (e == 0) {
        if (m == 0) b = s;
        else {   // subnormal
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            b = s | (e << 23) | ((m & 0x3ff) << 13);
        }
    } else if (e == 31) {
        b = s | 0x7f800000u | (m << 13);
    } else {
        b = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

float bf16(uint16_t v) {
    const uint32_t b = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

}  // namespace

void cpu_router_topk(const uint16_t* x, const uint16_t* w, const float* bias, int n, int dim, int k, int32_t* ids) {
    std::vector<float> xf(dim), biased(n);
    for (int i = 0; i < dim; ++i) xf[i] = f16(x[i]);
    // 16 partial sums: independent lanes the compiler vectorizes (one running sum is a 4-cycle add chain per
    // element, about 1.6 ms per layer on the laptop, longer than the layer the prediction must run ahead of)
    constexpr int kLanes = 16;
    for (int e = 0; e < n; ++e) {
        const uint16_t* r = w + (size_t) e * dim;
        float lane[kLanes] = {};
        int i = 0;
        for (; i + kLanes <= dim; i += kLanes)
            for (int j = 0; j < kLanes; ++j) lane[j] += xf[i + j] * bf16(r[i + j]);
        float acc = 0.f;
        for (int j = 0; j < kLanes; ++j) acc += lane[j];
        for (; i < dim; ++i) acc += xf[i] * bf16(r[i]);
        const float sp = acc > 20.f ? acc : std::log1p(std::exp(acc));
        biased[e] = std::sqrt(sp) + bias[e];
    }
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) { return biased[a] > biased[b] || (biased[a] == biased[b] && a < b); });
    for (int i = 0; i < k; ++i) ids[i] = order[i];
}

RouterLookahead::RouterLookahead(std::vector<std::vector<uint16_t>> router_w, std::vector<std::vector<float>> bias,
                                 int n_experts, int dim, int k, std::function<bool(int, int)> warm,
                                 std::function<void(int, const int32_t*, int)> fetch, int ahead)
    : w_(std::move(router_w)), bias_(std::move(bias)), n_experts_(n_experts), dim_(dim), k_(k), ahead_(ahead),
      warm_(std::move(warm)), fetch_(std::move(fetch)), x_(dim), warmed_(w_.size()) {
    th_ = std::thread([this] { run(); });
}

RouterLookahead::~RouterLookahead() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    th_.join();
}

void RouterLookahead::post(int layer, const uint16_t* x) {
    if (layer + ahead_ >= (int) w_.size()) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        std::memcpy(x_.data(), x, (size_t) dim_ * 2);
        layer_ = layer;
        have_ = true;
    }
    cv_.notify_one();
}

void RouterLookahead::run() {
    std::vector<uint16_t> x(dim_);
    std::vector<int32_t> ids(k_);
    for (;;) {
        int layer;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || have_; });
            if (stop_) return;
            have_ = false;
            layer = layer_;
            x = x_;
        }
        const int next = layer + ahead_;
        cpu_router_topk(x.data(), w_[next].data(), bias_[next].data(), n_experts_, dim_, k_, ids.data());
        std::vector<int32_t> warmed;
        for (int i = 0; i < k_; ++i) {
            std::lock_guard<std::mutex> lock(residency_mutex());
            if (warm_(next, ids[i])) warmed.push_back(ids[i]);
        }
        if (fetch_ && !warmed.empty()) fetch_(next, warmed.data(), (int) warmed.size());
        predicted_ += k_;
        warmed_n_ += (int64_t) warmed.size();
        std::lock_guard<std::mutex> lk(mu_);
        warmed_[next] = std::move(warmed);
    }
}

void RouterLookahead::predict_now(int layer, const uint16_t* x, int k, int32_t* ids) const {
    cpu_router_topk(x, w_[layer].data(), bias_[layer].data(), n_experts_, dim_, k, ids);
}

void RouterLookahead::observe(int layer, const int32_t* file_ids, int n) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& w = warmed_[layer];
    for (int i = 0; i < n; ++i)
        if (std::find(w.begin(), w.end(), file_ids[i]) != w.end()) ++useful_;
    w.clear();
}

RouterLookahead::Stats RouterLookahead::take_stats() {
    Stats s;
    s.predicted = predicted_.exchange(0);
    s.warmed = warmed_n_.exchange(0);
    s.useful = useful_.exchange(0);
    return s;
}

}  // namespace strata::ds41
