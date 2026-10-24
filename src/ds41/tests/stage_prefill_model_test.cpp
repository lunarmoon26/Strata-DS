// Opt-in real-weight resident batched prefill / continuation / snapshot test.
#include "strata/ds41/engine.hpp"
#include "strata/ds41/config.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#ifdef __linux__
#include <sys/resource.h>
#endif

using namespace strata::ds41;
void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
void finite(const Engine& e) {
    require(e.last_logits().size() == kVocab, "wrong logit count");
    for (float f : e.last_logits()) require(std::isfinite(f), "nonfinite logits");
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        std::string pack, ids, out;
        int maxseq = 4096, chunk = 512, batch = 256, sequential = 1, force_host = 0, dspark = 0;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, "missing value"); const std::string a = argv[i];
            if (a == "--pack") pack = argv[i + 1];
            else if (a == "--ids-file") ids = argv[i + 1];
            else if (a == "--out") out = argv[i + 1];
            else if (a == "--max-seq") maxseq = std::stoi(argv[i + 1]);
            else if (a == "--chunk") chunk = std::stoi(argv[i + 1]);
            else if (a == "--batch") batch = std::stoi(argv[i + 1]);
            else if (a == "--sequential") sequential = std::stoi(argv[i + 1]);
            else if (a == "--force-host") force_host = std::stoi(argv[i + 1]);
            else if (a == "--dspark") dspark = std::stoi(argv[i + 1]);
            else throw std::invalid_argument("unknown option " + a);
        }
        require(!out.empty(), "needs output directory");
        std::ifstream f(ids); require(bool(f), "prompt missing"); std::string text((std::istreambuf_iterator<char>(f)), {}), s;
        std::istringstream stream(text); std::vector<int> prompt;
        while (std::getline(stream, s, ',')) if (!s.empty()) prompt.push_back(std::stoi(s));
        require(prompt.size() > 0 && prompt.size() + 32 <= size_t(maxseq), "prompt requires 32-token decode headroom");
        EngineOptions opt; opt.devices = {0, 1}; opt.split_layer = 18; opt.max_seq = maxseq;
        opt.cpu_threads = 2; opt.ram_budget_gib = 0; opt.adapt_every = 0; opt.prefill_chunk = chunk;
        opt.prefill_batch = batch; opt.vram_reserve_bytes = 2ull << 30; opt.snapshots = 2; opt.force_host_handoff = force_host;
        require(dspark == 0 || dspark == 1, "dspark must be 0 or 1");
        opt.dspark = dspark;
        Engine e(pack, opt); require(e.vram_expert_slots() == kLayers * kExperts, "missing experts");
        require(e.has_dspark() == bool(dspark), "DSpark profile mismatch");
        std::vector<float> ref; int ref_next = -1;
        if (sequential) {
            StepDump dump;
            for (size_t p = 0; p < prompt.size(); ++p)
                ref_next = e.step(prompt[p], int(p), p + 1 == prompt.size() ? &dump : nullptr);
            std::ofstream hidden(out + "/hidden-sequential.bin", std::ios::binary);
            hidden.write(reinterpret_cast<const char*>(dump.hidden.data()), dump.hidden.size() * 2); hidden.close();
            std::ofstream batched(out + "/hidden-batched.bin", std::ios::binary);
            batched.write(reinterpret_cast<const char*>(dump.hidden.data()), dump.hidden.size() * 2); batched.close();
            setenv("DS41_PF_HIDDEN_FILE", (out + "/hidden-batched.bin").c_str(), 1);
            finite(e); ref = e.last_logits(); e.reset();
        }
        e.set_prefill_progress([](int done, int total) {
            std::printf("PREFILL_PROGRESS %d/%d\n", done, total); return true;
        });
        const auto start = std::chrono::steady_clock::now();
        int next = e.prefill(prompt, 0); finite(e);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto prefill_logits = e.last_logits();
        require(e.position() == int(prompt.size()), "wrong final position");
        const auto pt = e.last_prefill();
        require(pt.chunks > 0 && pt.streamed == 0, "not resident batched prefill");
        std::printf("PREFILL_DONE tokens=%zu seconds=%.6f tok_s=%.3f chunks=%d chunk=%d batch=%d resident_groups=%lld streamed=%lld\n",
                    prompt.size(), seconds, prompt.size() / seconds, pt.chunks, pt.chunk_tokens, pt.sub_batch,
                    (long long) pt.vram_experts, (long long) pt.streamed);
        if (sequential) {
            double error = 0, norm = 0, mx = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double d = prefill_logits[i] - ref[i]; error += d*d; norm += double(ref[i])*ref[i]; mx = std::max(mx, std::abs(d));
            }
            const double rel = std::sqrt(error / std::max(norm, 1e-30));
            const double amax = *std::max_element(ref.begin(), ref.end());
            const double bmax = *std::max_element(prefill_logits.begin(), prefill_logits.end());
            double asum = 0, bsum = 0;
            for (size_t i = 0; i < ref.size(); ++i) { asum += std::exp(ref[i] - amax); bsum += std::exp(prefill_logits[i] - bmax); }
            const double az = amax + std::log(asum), bz = bmax + std::log(bsum);
            double kl = 0;
            for (size_t i = 0; i < ref.size(); ++i)
                kl += std::exp(ref[i] - az) * ((ref[i] - az) - (prefill_logits[i] - bz));
            std::printf("COMPARE sequential_next=%d batched_next=%d rel_l2=%.9g max_abs=%.9g kl_decode_to_prefill=%.9g\n", ref_next, next, rel, mx, kl);
            std::ofstream saved(out + "/logits-sequential.bin", std::ios::binary);
            saved.write(reinterpret_cast<const char*>(ref.data()), kVocab * 4); saved.close();
            std::ofstream saved_pf(out + "/logits-prefill.bin", std::ios::binary);
            saved_pf.write(reinterpret_cast<const char*>(prefill_logits.data()), kVocab * 4); saved_pf.close();
            // GEMM and GEMV are not bitwise equivalent: their summation and
            // rounding differences propagate through 40 layers and routing.
            // Preserve raw errors in the receipt; gate the actual predictive
            // distribution as well as argmax, rather than a raw-logit norm alone.
            require(next == ref_next && std::isfinite(kl) && kl < .02 && rel < .25,
                    "batched/sequential predictive distribution disagrees");
        }
        std::ofstream lg(out + "/logits-prefill.bin", std::ios::binary);
        lg.write(reinterpret_cast<const char*>(prefill_logits.data()), kVocab * sizeof(float)); lg.close();
        e.save_snapshot(0); const int pos = e.position();
        const int after = e.step(next, pos); finite(e); const auto after_logits = e.last_logits();
        require(e.restore_snapshot(0) == pos, "restore position");
        require(e.step(next, pos) == after, "restore next token"); finite(e);
        require(std::memcmp(after_logits.data(), e.last_logits().data(), kVocab * 4) == 0, "snapshot logits mismatch");
        std::vector<int> generated{next}; next = after;
        for (int i = 1; i < 24 && generated.back() != 1; ++i) {
            generated.push_back(next); if (next == 1) break;
            next = e.step(next, e.position()); finite(e);
            const auto& t = e.last_timing(); require(t.expert_hits == 240 && !t.cpu_experts() && !t.skipped, "expert fallback");
        }
        std::printf("GENERATED:"); for (int token : generated) std::printf(" %d", token); std::puts("");
        require(e.position() > pos && e.position() <= maxseq, "continuation outside context capacity");
        std::printf("CONTINUATION prefill_position=%d final_position=%d capacity=%d dspark=%d\n", pos, e.position(), maxseq, dspark);
#ifdef __linux__
        struct rusage usage{}; require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage");
        std::printf("MEMORY process_peak_rss_kib=%ld reused_prompt_tokens=0\n", usage.ru_maxrss);
#endif
        if (sequential) {
            e.reset(); const int replay = e.prefill(prompt, 0); finite(e);
            require(replay == generated[0], "reset prefill next differs");
            require(std::memcmp(prefill_logits.data(), e.last_logits().data(), kVocab * 4) == 0, "reset prefill logits differ");
            std::puts("PASS reset/batched-prefill deterministic replay");
            e.reset();
            e.set_prefill_progress([](int, int) { return false; });
            bool cancelled = false;
            try { e.prefill(prompt, 0); } catch (const PrefillCancelled&) { cancelled = true; }
            require(cancelled && e.position() == 0, "cancel must reset both stages");
            e.set_prefill_progress(nullptr);
            require(e.prefill(prompt, 0) == replay, "post-cancel prefill token differs"); finite(e);
            require(std::memcmp(prefill_logits.data(), e.last_logits().data(), kVocab * 4) == 0, "post-cancel prefill differs");
            std::puts("PASS cancellation resets both stages and retry reproduces logits");
        }
        std::puts("RESULT pass resident batched prefill and decode continuation");
        return 0;
    } catch (const std::exception& ex) { std::fprintf(stderr, "RESULT fail: %s\n", ex.what()); return 1; }
}
