// Host-only refusal paths: do not open a pack, allocate a CUDA context or load weights.
#include "strata/ds41/engine.hpp"
#include "../../../third_party/exllamav3_moe/moe_mul1.h"
#include <iostream>
#include <string>

int main() {
    try {
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
