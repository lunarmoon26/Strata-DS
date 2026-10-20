#pragma once
#include <stdexcept>
#include <vector>
namespace strata::ds41 {
// CPU multi-row kernels above four rows are being fixed in a separate worktree.
inline constexpr int kVerifyMaxTokens = 4;
struct VerifyResult {
    bool graph_reused = false;             // True when this call replayed an existing window graph.
    std::vector<int> next;                 // next[t] follows inputs [0, t].
    std::vector<std::vector<float>> logits; // Empty unless requested.
};
// Keep the seed and all matching draft inputs. The correction stays pending.
inline int accepted_inputs(const std::vector<int>& window, const std::vector<int>& next) {
    if (window.empty() || window.size() != next.size())
        throw std::invalid_argument("acceptance requires equal nonempty rows");
    int keep = 1;
    while (keep < int(window.size()) && window[keep] == next[keep - 1]) ++keep;
    return keep;
}
} // namespace strata::ds41
