// include/strata/ds41/config.hpp - DeepSeek V4.1 Flash geometry, from its inference/config.json.
//
// These are kernel contracts, not tuning knobs. The pack tool copies config.json into the pack and the loader
// checks the numbers here against it, so a different model fails at load instead of producing plausible logits.
#pragma once

#include <array>
#include <cstdint>

namespace strata::ds41 {

constexpr int kDim = 5120;
constexpr int kLayers = 40;
constexpr int kVocab = 129280;
constexpr int kHeads = 64;
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;               // the last 64 of each 512-wide head carry RoPE
constexpr int kQLora = 1280;
constexpr int kOGroups = 8;
constexpr int kOLora = 1024;
constexpr int kWindow = 128;               // sliding-window KV slots per layer
constexpr int kExperts = 384;
constexpr int kTopK = 6;
constexpr int kMoeInter = 2304;
constexpr float kRouteScale = 1.5f;
constexpr float kSwigluLimit = 10.0f;
constexpr int kHc = 4;                     // hyper-connection copies of the residual stream
constexpr int kHcMix = (2 + kHc) * kHc;    // 24 coefficients: pre[4], post[4], comb[4x4]
constexpr int kSinkhornIters = 20;
constexpr float kHcEps = 1e-6f;
constexpr float kNormEps = 1e-20f;
constexpr int kIndexHeads = 32;
constexpr int kIndexDim = 128;
constexpr int kIndexTopK = 512;
constexpr int kCandidateLayer = 20;
constexpr int kCandidateBlocks = 2048;
constexpr int kCandidateBlock = 8;
constexpr double kRopeTheta = 10000.0;
constexpr double kCompressRopeTheta = 160000.0;
constexpr double kRopeFactor = 16.0;
constexpr int kBetaFast = 32;
constexpr int kBetaSlow = 1;
constexpr int kOriginalSeqLen = 65536;

/// 0 = sliding window only; r = KV compressed r-to-1 (one entry per backbone layer)
constexpr std::array<int, kLayers> kCompressRatio = {
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

/// Layers that compress their own KV (and own the indexer keys); the layers after each read its cache.
constexpr bool is_kv_source(int l) { return l == 2 || l == 8 || l == 14 || l == 20; }
/// Layers that run their own indexer; the layers after each reuse its top-k.
constexpr bool is_index_source(int l) {
    return l == 2 || l == 8 || l == 14 || l == 20 || l == 24 || l == 28 || l == 32 || l == 36;
}
constexpr bool is_engram_layer(int l) { return l == 1 || l == 14; }

}  // namespace strata::ds41
