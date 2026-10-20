#include "strata/ds41/engine.hpp"
#include <cstdio>
#include <functional>

using namespace strata::ds41;
namespace {
int checks = 0;
void require(bool value) {
    ++checks;
    if (!value) throw std::runtime_error("stage plan check " + std::to_string(checks));
}
void rejects(const std::function<void()>& fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}
}

int main() {
    try {
        const auto ranges = split_stages({3, 1});
        require(ranges[0].device == 3 && ranges[1].device == 1);
        for (int layer = 0; layer < kLayers; ++layer) {
            const std::string name = "layers." + std::to_string(layer) + ".attn.wq_a.weight";
            require(ranges[0].owns(layer) != ranges[1].owns(layer));
            for (const auto& stage : ranges) require(stage.owns_dense(name) == stage.owns(layer));
        }
        require(ranges[0].owns_dense("embed.weight") && !ranges[1].owns_dense("embed.weight"));
        for (const char* name : {"head.weight", "norm.weight", "head.trellis", "head.suh", "head.svh", "head.mul1"})
            require(!ranges[0].owns_dense(name) && ranges[1].owns_dense(name));
        require(!ranges[0].owns_dense("mtp.0.weight") && !ranges[1].owns_dense("mtp.0.weight"));
        // A stage cannot read KV, keys, a reused top-k or candidate mask from a prior stage.
        for (const auto& stage : ranges) {
            int kv = -1, index = -1;
            bool candidate = false;
            for (int l = stage.begin; l < stage.end; ++l) {
                if (is_kv_source(l)) kv = l;
                if (is_index_source(l)) index = l;
                if (l == kCandidateLayer) candidate = true;
                if (kCompressRatio[l]) require(stage.owns(kv) && stage.owns(index));
                if (l > kCandidateLayer) require(candidate);
            }
        }
        for (auto ids : std::vector<std::vector<int>>{{}, {0}, {0, 0}, {-1, 1}, {0, 1, 2}})
            rejects([&] { split_stages(ids); });
        for (const char* text : {"", "0", "0,", "0,,1", "0,1,", "-1,0", "0,1x", "0, 1", "0,0"})
            rejects([&] { parse_split_devices(text); });
        require(parse_split_devices("1,0") == std::vector<int>({1, 0}));
        const auto balanced = split_stages({1, 0}, 18);
        for (int l = 0; l < kLayers; ++l) {
            require(balanced[0].owns(l) == (l < 18));
            require(balanced[1].owns(l) == (l >= 18));
        }
        require(!is_kv_source(18) && !is_index_source(18));
        for (int boundary : {0, 14, 17, 19, 21, 40}) rejects([&] { split_stages({0, 1}, boundary); });
        rejects([&] { ranges[0].owns_dense("layers.40.w"); });
        rejects([&] { ranges[1].owns_dense("layers.-1.w"); });
        EngineOptions opt;
        opt.validate_split(); // unchanged single GPU defaults
        opt.devices = {0, 1};
        rejects([&] { opt.validate_split(); });
        opt.ram_budget_gib = 0; opt.adapt_every = 0; opt.prefill_chunk = 0;
        opt.validate_split();
        opt.prefill_chunk = 512; opt.validate_split();
        opt.prefill_chunk = -1; rejects([&] { opt.validate_split(); }); opt.prefill_chunk = 0;
        opt.batch_slots = 2; opt.validate_split(); opt.batch_slots = 5;
        rejects([&] { opt.validate_split(); }); opt.batch_slots = 0;
        opt.dspark = true; opt.validate_split(); opt.dspark = false;
        opt.vram_expert_slots = 0; rejects([&] { opt.validate_split(); });
        std::printf("PASS stage plan: %d checks (no CUDA, files or weights)\n", checks);
    } catch (const std::exception& ex) { std::fprintf(stderr, "%s\n", ex.what()); return 1; }
}
