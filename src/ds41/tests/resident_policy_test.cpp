// Host-only refusal paths: do not open a pack, allocate a CUDA context or load weights.
#include "strata/ds41/engine.hpp"
#include "strata/ds41/expert_rate_hint.hpp"
#include "../../../third_party/exllamav3_moe/moe_mul1.h"
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static void test_expert_rate_hints() {
    using namespace strata::ds41;
    constexpr uint64_t tile_bytes = uint64_t(kDim / 16) * (kMoeInter / 16) * 2;
    std::vector<ExpertSlot> slots(3);
    for (auto& slot : slots) for (int p = 0; p < 3; ++p) {
        slot.bits[p] = 3;
        slot.codebook[p] = 2;
        slot.comp_bytes[4*p] = tile_bytes * 48;
    }
    auto hint = [&] { return uniform_expert_integer_rate(int(slots.size()),
        [&](int i) -> const ExpertSlot& { return slots[i]; }); };
    if (hint() != 3) throw std::runtime_error("uniform expert rate not detected");
    auto invalid = [&](const char* reason) {
        if (hint() != 0) throw std::runtime_error(reason);
    };
    slots[1].bits[0] = 2.5f; invalid("fractional rate accepted"); slots[1].bits[0] = 3;
    slots[1].bits[1] = 2.5f; invalid("fractional projection accepted"); slots[1].bits[1] = 3;
    slots[2].bits[2] = 2; invalid("mixed rates accepted"); slots[2].bits[2] = 3;
    slots[1].codebook[0] = 1; invalid("mixed codebook accepted"); slots[1].codebook[0] = 2;
    slots[2].comp_bytes[4] += 16; invalid("incorrect trellis size accepted"); slots[2].comp_bytes[4] -= 16;
    slots[0].bits[0] = std::numeric_limits<float>::quiet_NaN(); invalid("NaN rate accepted"); slots[0].bits[0] = 3;
    if (hint() != 3 || uniform_expert_integer_rate(0, [&](int i) -> const ExpertSlot& { return slots[i]; }) != 0)
        throw std::runtime_error("uniform or empty expert rates mishandled");
}

int main() {
    try {
        test_expert_rate_hints();
        strata::ds41::EngineOptions options;
        try {
            strata::ds41::Engine engine("/nonexistent-pack", options);
            throw std::runtime_error("single-device construction unexpectedly succeeded");
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("requires --devices") == std::string::npos) throw;
        }
        options.devices = {0, 1};
        // The default RAM budget violates resident policy, also before any CUDA call.
        try {
            strata::ds41::Engine engine("/nonexistent-pack", options);
            throw std::runtime_error("invalid resident policy unexpectedly succeeded");
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("--ram-budget-gib 0") == std::string::npos) throw;
        }
        options.ram_budget_gib=0;options.adapt_every=0;options.tensor_parallel=true;
        for(int invalid=0;invalid<2;++invalid) {
            options.split_layer=invalid?20:18;options.force_host_handoff=invalid;
            try {
                strata::ds41::Engine engine("/nonexistent-pack",options);
                throw std::runtime_error("invalid TP2 configuration unexpectedly succeeded");
            } catch(const std::invalid_argument& e) {
                if(std::string(e.what()).find("TP2 requires")==std::string::npos) throw;
            }
        }
        try {
            exl3_moe_cpu_forward_raw(-1, nullptr, nullptr, nullptr, nullptr, 1, 6, 1);
            throw std::runtime_error("CPU fallback silently returned");
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("CPU expert fallback is not available") == std::string::npos) throw;
        }
        std::cout << "resident policy refuses unsupported paths before pack/CUDA access\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
