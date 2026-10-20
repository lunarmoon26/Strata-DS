// src/ds41/tests/fake_pack.hpp - a fake ds41 pack for tests without the model: the full 40 x 384 expert geometry,
// tiny experts of random bytes, the index entries the loader checks, no dense weights and no engram tables.
#pragma once

#include "strata/ds41/config.hpp"

#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ds41test {

constexpr int L = strata::ds41::kLayers, E = strata::ds41::kExperts;
/// every fake expert has this size: 12 components of 256 bytes, 4 KiB in all; consecutive experts are kStride
/// apart, so most do not start on a 4 KiB boundary (as in a real pack): O_DIRECT readers must cover them
constexpr uint64_t kExpertBytes = 4096, kStride = kExpertBytes + 256;

/// mixed packs (as SAGE 1.59bpw, whose experts differ in size): expert (l, e) is fake_mult(l, e) times larger, its
/// components too; the experts follow each other 256 bytes apart
inline int fake_mult(int l, int e) { return 1 + (l * E + e) % 3; }
inline uint64_t fake_bytes(int l, int e, bool mixed) { return mixed ? kExpertBytes * fake_mult(l, e) : kExpertBytes; }
/// gap: bytes between two experts (256: most experts start off a 4 KiB boundary; a multiple of 4096: every one starts
/// on one, as in a real pack)
inline uint64_t fake_offset(int l, int e, bool mixed, uint64_t gap = 256) {
    if (!mixed) return ((uint64_t) l * E + e) * (kExpertBytes + gap);
    const uint64_t i = (uint64_t) l * E + e, cycles = i / 3;   // a cycle of 3 experts: 1 + 2 + 3 units, 3 gaps
    uint64_t off = cycles * (6 * kExpertBytes + 3 * gap);
    for (uint64_t j = 0; j < i % 3; ++j) off += kExpertBytes * (1 + j) + gap;
    return off;
}

inline void write_fake_pack(const std::string& dir, bool mixed = false, uint64_t gap = 256) {
    mkdir(dir.c_str(), 0755);
    {
        std::ofstream f(dir + "/index.txt");
        f << "embed.weight bf16 2 " << strata::ds41::kVocab << " " << strata::ds41::kDim << " 0 " << (uint64_t) strata::ds41::kVocab * strata::ds41::kDim * 2 << "\n";
        f << "head.weight bf16 2 " << strata::ds41::kVocab << " " << strata::ds41::kDim << " 0 " << (uint64_t) strata::ds41::kVocab * strata::ds41::kDim * 2 << "\n";
        f << "layers.0.attn.wq_b.weight f8e4m3 2 " << strata::ds41::kHeads * strata::ds41::kHeadDim << " " << strata::ds41::kQLora << " 0 "
          << (uint64_t) strata::ds41::kHeads * strata::ds41::kHeadDim * strata::ds41::kQLora << "\n";
        f << "layers.0.attn.wo_a.weight bf16 2 " << strata::ds41::kOGroups * strata::ds41::kOLora << " " << strata::ds41::kHeads * strata::ds41::kHeadDim / strata::ds41::kOGroups
          << " 0 " << (uint64_t) strata::ds41::kOGroups * strata::ds41::kOLora * (strata::ds41::kHeads * strata::ds41::kHeadDim / strata::ds41::kOGroups) * 2 << "\n";
        f << "layers.0.hc_attn_fn f32 2 " << strata::ds41::kHcMix << " " << strata::ds41::kHc * strata::ds41::kDim << " 0 "
          << (uint64_t) strata::ds41::kHcMix * strata::ds41::kHc * strata::ds41::kDim * 4 << "\n";
    }
    {
        std::ofstream f(dir + "/experts.txt");
        const char* comp[12] = {"w1.trellis", "w1.suh", "w1.svh", "w1.mul1", "w3.trellis", "w3.suh",
                                "w3.svh",     "w3.mul1", "w2.trellis", "w2.suh", "w2.svh", "w2.mul1"};
        for (int l = 0; l < L; ++l)
            for (int e = 0; e < E; ++e) {
                const int m = mixed ? fake_mult(l, e) : 1;
                f << l << " " << e << " " << fake_offset(l, e, mixed, gap) << " " << fake_bytes(l, e, mixed) << " 3 3 3";
                for (int c = 0; c < 12; ++c) f << " " << comp[c] << ":" << c * 256 * m << ":" << 256 * m;
                f << "\n";
            }
    }
    {
        std::vector<uint8_t> b((size_t) (fake_offset(L - 1, E - 1, mixed, gap) + fake_bytes(L - 1, E - 1, mixed)) + 8192);
        std::mt19937 g(1);
        for (auto& x : b) x = (uint8_t) g();
        std::ofstream f(dir + "/experts.bin", std::ios::binary);
        f.write((const char*) b.data(), (std::streamsize) b.size());
    }
    std::ofstream(dir + "/engram.txt");
    std::ofstream(dir + "/engram_hash.txt");
    {
        std::vector<int32_t> tm(strata::ds41::kVocab, 0);
        std::ofstream f(dir + "/engram_tokenmap.bin", std::ios::binary);
        f.write((const char*) tm.data(), (std::streamsize) (tm.size() * 4));
    }
    std::ofstream(dir + "/pack_info.txt") << "layers " << L << "\nexperts " << E << "\nfinished 1\n";
}

/// STRP profile of the ranked pairs
inline void write_profile(const std::string& path, const std::vector<std::pair<int, int>>& pairs) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    const uint32_t n = (uint32_t) pairs.size();
    const uint32_t h[5] = {1, (uint32_t) L, (uint32_t) E, n, n};
    std::fwrite("STRP", 1, 4, f);
    std::fwrite(h, 4, 5, f);
    for (auto [l, e] : pairs) {
        const uint16_t p[2] = {(uint16_t) l, (uint16_t) e};
        std::fwrite(p, 2, 2, f);
    }
    std::vector<int32_t> table((size_t) L * E, -1);
    for (uint32_t i = 0; i < n; ++i) table[(size_t) pairs[i].first * E + pairs[i].second] = (int32_t) i;
    std::fwrite(table.data(), 4, table.size(), f);
    std::fclose(f);
}

}  // namespace ds41test
