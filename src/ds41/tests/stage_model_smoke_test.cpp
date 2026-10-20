// Opt-in real-weight check. Never registered as a weight-free CTest.
#include "strata/ds41/engine.hpp"
#include "strata/ds41/config.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace strata::ds41;
namespace {
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void check_step(Engine& engine) {
    const auto& logits = engine.last_logits();
    require(logits.size() == kVocab, "wrong logit count");
    for (float v : logits) require(std::isfinite(v), "nonfinite logit");
    const auto& t = engine.last_timing();
    require(t.expert_total == kLayers * kTopK && t.expert_hits == t.expert_total,
            "not all six routed experts per layer executed in VRAM");
    require(!t.skipped && !t.cpu_experts() && !t.zero_copy_experts(), "expert fallback or skip occurred");
}
void same(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
            "reset/prefill/snapshot logits differ bitwise");
}
void save(const std::string& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    require(bool(out), "cannot write logit receipt");
}
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        std::string pack, ids, out;
        int split_layer = 20;
        bool force_host = false;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, "missing option value");
            const std::string a = argv[i];
            if (a == "--pack") pack = argv[i + 1];
            else if (a == "--ids-file") ids = argv[i + 1];
            else if (a == "--out") out = argv[i + 1];
            else if (a == "--split-layer") split_layer = std::stoi(argv[i + 1]);
            else if (a == "--force-host") force_host = std::stoi(argv[i + 1]) != 0;
            else throw std::invalid_argument("unknown argument: " + a);
        }
        require(!pack.empty() && !ids.empty() && !out.empty(), "needs --pack, --ids-file and --out");
        std::ifstream input(ids);
        require(bool(input), "cannot open prompt ids");
        std::string text((std::istreambuf_iterator<char>(input)), {}), item;
        std::istringstream fields(text);
        std::vector<int> prompt;
        while (std::getline(fields, item, ',')) if (!item.empty()) prompt.push_back(std::stoi(item));
        require(!prompt.empty() && prompt.size() + 16 < 256, "prompt must fit short smoke-test context");
        for (int token : prompt) require(token >= 0 && token < kVocab, "invalid prompt id");
        EngineOptions opt;
        opt.devices = {0, 1}; opt.max_seq = 256; opt.cpu_threads = 2;
        opt.split_layer = split_layer; opt.force_host_handoff = force_host;
        opt.ram_budget_gib = 0; opt.adapt_every = 0; opt.prefill_chunk = 0;
        opt.vram_expert_slots = -1; opt.vram_reserve_bytes = 2ull << 30; opt.snapshots = 2;
        const auto begin = std::chrono::steady_clock::now();
        Engine engine(pack, opt);
        std::printf("LOADED seconds=%.3f resident_experts=%d\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(), engine.vram_expert_slots());
        require(engine.vram_expert_slots() == kLayers * kExperts, "not all experts resident");
        int next = -1;
        for (size_t pos = 0; pos < prompt.size(); ++pos) {
            next = engine.step(prompt[pos], int(pos)); check_step(engine);
            std::printf("STEP pos=%zu next=%d finite=1 vram=%d\n", pos, next, engine.last_timing().expert_hits);
        }
        const auto first = engine.last_logits();
        save(out + "/logits-first.bin", first);
        std::vector<int> generated;
        for (int i = 0; i < 16; ++i) {
            generated.push_back(next);
            if (next == 1) break;
            next = engine.step(next, int(prompt.size()) + i); check_step(engine);
        }
        std::printf("GENERATED:"); for (int token : generated) std::printf(" %d", token); std::puts("");
        engine.reset(); require(engine.position() == 0, "reset did not rewind both stages");
        next = engine.prefill(prompt, 0); check_step(engine); same(first, engine.last_logits());
        std::puts("PASS reset and token-by-token prefill: bitwise logits");
        const int pos = engine.position();
        engine.save_snapshot(0);
        const int after = engine.step(next, pos); check_step(engine);
        const auto reference = engine.last_logits();
        save(out + "/logits-after.bin", reference);
        require(engine.restore_snapshot(0) == pos, "snapshot did not rewind both stages");
        require(engine.step(next, pos) == after, "snapshot next token differs"); check_step(engine);
        same(reference, engine.last_logits());
        require(engine.restore_snapshot(0) == pos, "second restore failed");
        require(engine.prefill({next}, pos) == after, "snapshot prefill next token differs"); check_step(engine);
        same(reference, engine.last_logits());
        std::puts("PASS repeat snapshot/restore and prefill: bitwise logits");
        // Save after an odd number of tokens too: the next step completes a
        // ratio-2 compressed row, which the layer-18 bridge must replay.
        if (engine.position() % 2 == 0) engine.step(after, engine.position());
        const int odd_pos = engine.position();
        engine.save_snapshot(1);
        const int odd_next = engine.step(after, odd_pos); check_step(engine);
        const auto odd_logits = engine.last_logits();
        engine.step(odd_next, engine.position()); check_step(engine);
        require(engine.restore_snapshot(1) == odd_pos, "odd snapshot restore failed");
        require(engine.step(after, odd_pos) == odd_next, "odd snapshot next token differs");
        check_step(engine); same(odd_logits, engine.last_logits());
        std::puts("PASS odd-position compressor/snapshot replay: bitwise logits");
        std::puts("RESULT pass real-weight split smoke; reference-model logit parity and quality are not established");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RESULT fail real-weight split smoke: %s\n", error.what()); return 1;
    }
}
