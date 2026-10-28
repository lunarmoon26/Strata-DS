// Explicit real-weight compatibility test. Not registered as an automatic CTest.
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
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void finite(const Engine& e) {
    require(e.last_logits().size() == kVocab, "logit count");
    for (float value : e.last_logits()) require(std::isfinite(value), "nonfinite logit");
}
void same(const std::vector<float>& a, const std::vector<float>& b, const char* phase) {
    if (a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float))) return;
    size_t changed = 0; double maximum = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        changed += a[i] != b[i]; maximum = std::max(maximum, std::abs(double(a[i]) - b[i]));
    }
    std::fprintf(stderr, "REPLAY phase=%s changed=%zu max_abs=%.9g\n", phase, changed, maximum);
    throw std::runtime_error(std::string("exact replay mismatch: ") + phase);
}
void save(const std::string& name, const std::vector<float>& values) {
    std::ofstream out(name, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    require(bool(out), "cannot save logits");
}
uint64_t fingerprint(const std::vector<float>& values) {
    uint64_t result = 14695981039346656037ull;
    for (float value : values) {
        uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
        result = (result ^ bits) * 1099511628211ull;
    }
    return result;
}
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        std::string pack, ids, out, parallelism = "layer";
        int reserve_gib = 12;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, "missing argument value");
            const std::string key = argv[i];
            if (key == "--pack") pack = argv[i + 1];
            else if (key == "--ids-file") ids = argv[i + 1];
            else if (key == "--out") out = argv[i + 1];
            else if (key == "--parallelism") parallelism = argv[i + 1];
            else if (key == "--reserve-gib") reserve_gib = std::stoi(argv[i + 1]);
            else throw std::invalid_argument("unknown argument " + key);
        }
        require(!pack.empty() && !ids.empty() && !out.empty(), "requires pack, IDs and private output directory");
        std::ifstream input(ids); require(bool(input), "missing frozen IDs");
        std::string text((std::istreambuf_iterator<char>(input)), {}), field;
        std::istringstream fields(text); std::vector<int> prompt;
        while (std::getline(fields, field, ',')) if (!field.empty()) prompt.push_back(std::stoi(field));
        require(!prompt.empty() && prompt.size() <= 1024, "initial compatibility test needs 1..1024 input tokens");
        EngineOptions options;
        options.devices = {0, 1}; options.split_layer = 18; options.max_seq = 524288;
        require(parallelism == "layer" || parallelism == "tensor", "unknown parallelism");
        options.tensor_parallel = parallelism == "tensor";
        if (options.tensor_parallel) options.split_layer = 20;
        options.cpu_threads = 2; options.ram_budget_gib = 0; options.adapt_every = 0;
        options.prefill_chunk = 128; options.prefill_batch = 64; options.snapshots = 2;
        require(reserve_gib >= 2 && reserve_gib <= 24 && (options.tensor_parallel || reserve_gib >= 12), "unsafe experimental reserve");
        options.vram_reserve_bytes = size_t(reserve_gib) << 30;
        const auto start = std::chrono::steady_clock::now();
        Engine e(pack, options);
        require(!e.has_dspark() && e.vram_expert_slots() < 40 * 384, "experimental stream profile not active");
        std::printf("LOADED seconds=%.3f resident_slots=%d capacity=524288 spec=none parallelism=%s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), e.vram_expert_slots(), parallelism.c_str());
        int reference_next = -1;
        int64_t streamed = 0;
        std::vector<uint64_t> sequential_fingerprints;
        const auto sequential_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompt.size(); ++i) {
            reference_next = e.step(prompt[i], int(i)); finite(e);
            const auto& timing = e.last_timing();
            require(timing.expert_total == 240 && timing.expert_hits + timing.streamed_experts == 240 &&
                    !timing.cpu_experts() && !timing.skipped && !timing.zero_copy_experts(), "incomplete expert execution");
            streamed += timing.streamed_experts;
            if (options.tensor_parallel) sequential_fingerprints.push_back(fingerprint(e.last_logits()));
        }
        const double sequential_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - sequential_start).count();
        require(streamed > 0, "no decode streaming exercised");
        const auto reference = e.last_logits(); save(out + "/sequential.bin", reference);
        e.reset();
        int first_difference = -1;
        if (options.tensor_parallel) e.set_prefill_progress([&](int done, int) {
            if (fingerprint(e.last_logits()) != sequential_fingerprints.at(done - 1) && first_difference < 0) {
                first_difference = done - 1;
                std::printf("TP_REPLAY first_different_position=%d\n", first_difference);
            }
            return true;
        });
        int next = e.prefill(prompt, 0); finite(e);
        e.set_prefill_progress(nullptr);
        std::printf("TIMING input_tokens=%zu sequential_seconds=%.6f sequential_tok_s=%.3f prefill_seconds=%.6f prefill_tok_s=%.3f reserve_gib=%d\n",
                    prompt.size(), sequential_seconds, prompt.size() / sequential_seconds,
                    e.last_prefill().total_ms / 1000., prompt.size() * 1000. / e.last_prefill().total_ms, reserve_gib);
        const auto batched = e.last_logits(); save(out + "/prefill.bin", batched);
        double error = 0, norm = 0, kl = 0, asum = 0, bsum = 0;
        const double amax = *std::max_element(reference.begin(), reference.end());
        const double bmax = *std::max_element(batched.begin(), batched.end());
        for (size_t i = 0; i < reference.size(); ++i) {
            const double d = reference[i] - batched[i]; error += d*d; norm += double(reference[i])*reference[i];
            asum += std::exp(reference[i] - amax); bsum += std::exp(batched[i] - bmax);
        }
        const double az = amax + std::log(asum), bz = bmax + std::log(bsum);
        for (size_t i = 0; i < reference.size(); ++i)
            kl += std::exp(reference[i] - az) * ((reference[i] - az) - (batched[i] - bz));
        const double relative = std::sqrt(error / std::max(norm, 1e-30));
        std::printf("COMPARE sequential_next=%d prefill_next=%d rel_l2=%.9g KL=%.9g streamed_decode=%lld streamed_prefill=%lld\n",
                    reference_next, next, relative, kl, (long long)streamed, (long long)e.last_prefill().streamed);
        if (options.tensor_parallel) {
            require(first_difference < 0, "token-wise TP reset replay diverged");
            same(reference, batched, "TP token-wise reset replay");
        }
        require(next == reference_next && kl < .02 && relative < .25, "prefill/sequential predictive gate failed");
        require(e.last_prefill().streamed > 0, "no prefill streaming exercised");
        std::vector<int> generated;
        const auto decode_start = std::chrono::steady_clock::now();
        int decode_steps = 0;
        for (int i = 0; i < 24; ++i) {
            generated.push_back(next); if (next == 1) break;
            next = e.step(next, e.position()); finite(e);
            ++decode_steps;
        }
        const double decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
        std::printf("DECODE_TIMING steps=%d seconds=%.6f tok_s=%.3f warmup=0 repeats=1\n", decode_steps, decode_seconds, decode_steps / decode_seconds);
        std::printf("GENERATED:"); for (int token : generated) std::printf(" %d", token); std::puts("");
        const int pos = e.position(); e.save_snapshot(0);
        const int after = e.step(next, pos); finite(e); const auto saved = e.last_logits();
        require(e.restore_snapshot(0) == pos && e.step(next, pos) == after, "snapshot continuation token mismatch");
        same(saved, e.last_logits(), "snapshot decode");
        const std::vector<int> extension(prompt.end() - std::min<size_t>(17, prompt.size()), prompt.end());
        const int warm_pos = e.position(); e.save_snapshot(1);
        const int warm = e.prefill(extension, warm_pos); finite(e); const auto warm_logits = e.last_logits();
        require(e.restore_snapshot(1) == warm_pos && e.prefill(extension, warm_pos) == warm, "warm prefill continuation mismatch");
        same(warm_logits, e.last_logits(), "warm prefill snapshot");
        e.reset(); require(e.prefill(prompt, 0) == reference_next, "reset next differs"); same(batched, e.last_logits(), "reset prefill");
        e.reset(); e.set_prefill_progress([](int, int) { return false; });
        bool cancelled = false;
        try { e.prefill(prompt, 0); } catch (const PrefillCancelled&) { cancelled = true; }
        require(cancelled && e.position() == 0, "cancel/reset failed");
        e.set_prefill_progress(nullptr);
        require(e.prefill(prompt, 0) == reference_next, "cancel recovery next differs"); same(batched, e.last_logits(), "cancel recovery prefill");
        std::puts("RESULT pass experimental 2.9 short-input compatibility, snapshots, reset, cancel and warm prefill; not broad quality or actual-524K qualification");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "RESULT fail experimental 2.9: %s\n", error.what()); return 1; }
}
