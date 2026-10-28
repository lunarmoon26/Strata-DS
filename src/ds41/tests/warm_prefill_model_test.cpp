// Explicit real-model regression: decode/draft -> incremental prefill -> next request.
// Not an automatic CTest. Keeps one 524288-token session and all backbone experts resident.
#include "strata/ds41/engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

using namespace strata::ds41;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void finite(const Engine& e) {
    require(e.last_logits().size() == kVocab, "logit count");
    for (float x : e.last_logits()) require(std::isfinite(x), "nonfinite logits");
}
void equal(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), "logits differ");
}
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        std::string pack, ids;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, "missing value"); const std::string key = argv[i];
            if (key == "--pack") pack = argv[i + 1];
            else if (key == "--ids-file") ids = argv[i + 1];
            else throw std::invalid_argument("unknown option " + key);
        }
        std::ifstream input(ids); require(bool(input), "missing prompt");
        std::vector<int> prompt; std::string value;
        while (std::getline(input, value, ',')) if (!value.empty()) prompt.push_back(std::stoi(value));
        require(prompt.size() >= 1024 && prompt.size() <= 16384, "use 1K..16K fixed prompt");
        EngineOptions o; o.devices = {0, 1}; o.split_layer = 18; o.max_seq = 524288;
        o.ram_budget_gib = 0; o.adapt_every = 0; o.cpu_threads = 2;
        o.prefill_chunk = 1024; o.prefill_batch = 256; o.snapshots = 4;
        o.dspark = true; o.draft_cache_bytes = 1ull << 30; o.vram_reserve_bytes = 1ull << 30;
        Engine e(pack, o); require(e.vram_expert_slots() == kLayers * kExperts, "missing backbone experts");
        for (int turn = 0; turn < 2; ++turn) {
            e.reset();
            e.prefill(std::vector<int>(prompt.begin(), prompt.end() - 1), 0);
            e.save_snapshot(0); e.prefill({prompt.back()}, e.position()); finite(e);
            const int pos = e.position(); e.save_snapshot(1);
            std::vector<int> fed; std::vector<std::vector<float>> reference;
            for (int i = 0; i < 96; ++i) {
                fed.push_back(prompt[(size_t(turn) * 96 + i) % prompt.size()]);
                e.step(fed.back(), e.position()); finite(e); reference.push_back(e.last_logits());
            }
            e.restore_snapshot(1);
            int uses = 0, hits = 0;
            for (int i = 0; i < 96; ++i) {
                auto draft = e.draft(fed[i]); require(draft.n >= 3 && draft.uses == 45, "no actual published draft");
                uses += draft.uses; hits += draft.vram_hits;
                std::vector<int> window{fed[i], draft.tokens[0], draft.tokens[1], draft.tokens[2]};
                const auto checked = e.verify(window, e.position(), true);
                equal(reference[i], checked.logits[0]); e.commit(1); equal(reference[i], e.last_logits());
            }
            require(hits > 0, "draft cache never warmed");
            require(e.position() == pos + 96, "wrong committed draft position");
            std::printf("WARM turn=%d draft_uses=%d cache_hits=%d position=%d\n", turn, uses, hits, e.position());
            std::vector<int> follow;
            for (int i = 0; i < 751; ++i) follow.push_back(prompt[(153 + i) % prompt.size()]);
            e.prefill(std::vector<int>(follow.begin(), follow.end() - 1), e.position());
            require(e.last_prefill().streamed == 0, "backbone streaming");
            e.save_snapshot(2); const int next = e.prefill({follow.back()}, e.position()); finite(e);
            require(e.position() == pos + 96 + 751, "incremental prefill position");
            e.save_snapshot(3); const int at = e.position(); const int after = e.step(next, at);
            const auto logits = e.last_logits(); e.restore_snapshot(3);
            require(e.step(next, at) == after, "snapshot token mismatch"); equal(logits, e.last_logits());
            e.restore_snapshot(3);
            const auto draft = e.draft(next);
            const auto checked = e.verify({next, draft.tokens[0], draft.tokens[1]}, at, true);
            equal(logits, checked.logits[0]); e.commit(1); equal(logits, e.last_logits());
            require(e.vram_expert_slots() == kLayers * kExperts, "backbone evicted");
            std::printf("PASS turn=%d warm incremental prefill, last-token prefill and snapshot continuation\n", turn);
        }
        std::puts("RESULT pass warm draft/decode to prefill at 524288 capacity");
    } catch (const std::exception& error) { std::fprintf(stderr, "RESULT fail %s\n", error.what()); return 1; }
}
