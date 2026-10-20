// Opt-in full-model contract checks; no downloads or model transformation.
#include "strata/ds41/engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace strata::ds41;
namespace {
void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
void finite(const Engine& e) {
    require(e.last_logits().size() == kVocab, "logit count");
    for (float f : e.last_logits()) require(std::isfinite(f), "nonfinite logits");
}
void equal(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), "logits differ");
}
void rejects(const std::function<void()>& fn) {
    bool caught = false; try { fn(); } catch (const std::logic_error&) { caught = true; }
    require(caught, "pending window operation was not rejected");
}
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        std::string pack, ids, reference_out, reference_in; int dspark = 0, host = 0, maxseq = 4096, tensor = 0;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, "missing value"); std::string a = argv[i];
            if (a == "--pack") pack = argv[i + 1];
            else if (a == "--ids-file") ids = argv[i + 1];
            else if (a == "--dspark") dspark = std::stoi(argv[i + 1]);
            else if (a == "--force-host") host = std::stoi(argv[i + 1]);
            else if (a == "--max-seq") maxseq = std::stoi(argv[i + 1]);
            else if (a == "--tensor") tensor = std::stoi(argv[i + 1]);
            else if (a == "--reference-out") reference_out = argv[i+1];
            else if (a == "--reference-in") reference_in = argv[i+1];
            else throw std::invalid_argument("unknown option " + a);
        }
        std::ifstream file(ids); require(bool(file), "missing prompt"); std::string item;
        std::vector<int> prompt; while (std::getline(file, item, ',')) if (!item.empty()) prompt.push_back(std::stoi(item));
        require(prompt.size() >= 18 && prompt.size() + 64 < size_t(maxseq), "prompt size");
        EngineOptions o; o.devices = {0, 1}; o.split_layer = 18; o.max_seq = maxseq;
        o.ram_budget_gib = 0; o.adapt_every = 0; o.prefill_chunk = 512; o.prefill_batch = 256;
        o.snapshots = 2; o.batch_slots = 2; o.dspark = dspark; o.force_host_handoff = host;
        o.tensor_parallel=tensor; if(tensor) o.split_layer=20;
        o.vram_reserve_bytes = 1ull << 30; o.cpu_threads = 2;
        Engine e(pack, o); require(e.vram_expert_slots() == 15360, "not all backbone experts resident");
        int next = e.prefill(prompt, 0); finite(e); const int pos = e.position();
        e.save_snapshot(0); const auto original = e.last_logits();
        std::vector<int> tokens; std::vector<std::vector<float>> reference;
        for (int i = 0; i < 8; ++i) {
            tokens.push_back(next); next = e.step(next, e.position()); finite(e); reference.push_back(e.last_logits());
        }
        for (int keep = 1; keep <= 4; ++keep) {
            require(e.restore_snapshot(0) == pos, "restore");
            const std::vector<int> window(tokens.begin(), tokens.begin() + 4);
            const auto verified = e.verify(window, pos, true);
            require(e.position() == pos && verified.next.size() == 4, "verify modified committed position");
            rejects([&] { e.step(tokens[0], pos); });
            rejects([&] { e.prefill({tokens[0]}, pos); });
            rejects([&] { e.copy_to_slot(0); });
            rejects([&] { e.draft(tokens[0]); });
            for (int i = 0; i < 4; ++i) equal(reference[i], verified.logits[i]);
            e.commit(keep); require(e.position() == pos + keep, "commit position");
            equal(reference[keep - 1], e.last_logits());
            e.step(tokens[keep], e.position()); equal(reference[keep], e.last_logits());
        }
        std::puts("PASS verify full logits, keep=1..4, rejected-tail rollback and continuation");
        for(int boundary:{127,128,511,512}) {
            e.restore_snapshot(0);int boundary_next=tokens[0];
            while(e.position()<boundary) boundary_next=e.step(boundary_next,e.position());
            e.save_snapshot(1);
            std::vector<int> window;std::vector<std::vector<float>> wanted;
            for(int i=0;i<5;++i) {
                window.push_back(boundary_next);boundary_next=e.step(boundary_next,e.position());
                wanted.push_back(e.last_logits());
            }
            for(int keep=1;keep<=4;++keep) {
                e.restore_snapshot(1);
                const auto checked=e.verify(std::vector<int>(window.begin(),window.begin()+4),boundary,true);
                for(int i=0;i<4;++i) equal(wanted[i],checked.logits[i]);
                e.commit(keep);equal(wanted[keep-1],e.last_logits());
                e.step(window[keep],e.position());equal(wanted[keep],e.last_logits());
            }
            // Rejected proposals must actually differ from the reference path;
            // truncating an otherwise correct greedy window would miss stale
            // compressor/index rows with the wrong contents.
            e.restore_snapshot(1);
            std::vector<int> wrong(window.begin(),window.begin()+4);
            for(int i=1;i<4;++i) wrong[i]=(wrong[i]+123+i)%kVocab;
            const auto checked=e.verify(wrong,boundary,true);
            equal(wanted[0],checked.logits[0]);e.commit(1);
            for(int i=1;i<5;++i) { e.step(window[i],e.position());equal(wanted[i],e.last_logits()); }
        }
        std::puts("PASS ring/index/compressor boundaries 127/128/511/512, prefix commits and rejected-tail continuation");
        e.restore_snapshot(0); e.step(tokens[0], pos); const auto main = e.last_logits();
        e.copy_to_slot(0); const int pos_a = e.position();
        e.reset(); const std::vector<int> other(prompt.begin(), prompt.begin() + 17);
        const int b = e.prefill(other, 0); e.copy_to_slot(1);
        e.step(b, e.position()); const auto reference_b = e.last_logits();
        e.copy_from_slot(0); equal(main, e.last_logits());
        const auto batch = e.step_slots({0, 1}, {tokens[1], b});
        require(batch.size() == 2 && e.position() == pos_a, "batch changed main position"); equal(main, e.last_logits());
        require(!std::memcmp(e.slot_logits(0), reference[1].data(), kVocab * sizeof(float)), "slot A contaminated");
        require(!std::memcmp(e.slot_logits(1), reference_b.data(), kVocab * sizeof(float)), "slot B contaminated");
        require(e.slot_position(0) == pos_a + 1 && e.slot_position(1) == 18, "slot positions");
        e.copy_from_slot(0); e.step(tokens[2], e.position()); equal(reference[2], e.last_logits());
        e.copy_from_slot(1); equal(reference_b, e.last_logits());
        std::puts("PASS isolated slots, distinct lengths, preserved main state and full-logit continuation");
        if (dspark) {
            require(e.has_dspark(), "missing DSpark");
            const auto before = e.last_logits(); const int p = e.position();
            e.copy_to_slot(1);
            auto first = e.draft(b);
            require(first.n == 5 && first.uses == 45, "no actual published drafter execution");
            for (int i = 0; i < first.n; ++i)
                require(first.tokens[i] >= 0 && first.tokens[i] < kVocab && std::isfinite(first.accept[i]), "invalid draft");
            equal(before, e.last_logits()); require(e.position() == p, "draft changed committed position");
            e.copy_from_slot(1); auto replay = e.draft(b);
            require(first.tokens == replay.tokens && first.accept == replay.accept, "draft session replay differs");
            std::printf("DSPARK tokens:"); for (int t : first.tokens) std::printf(" %d", t);
            std::printf(" ms=%.3f uses=%d cpu_ms=%.3f\n", first.ms, first.uses, first.cpu_experts_ms);
            std::puts("PASS published DSpark execution, finite confidence, session replay, no backbone mutation");
        }
        e.reset(); bool cancelled = false;
        e.set_prefill_progress([](int, int) { return false; });
        try { e.prefill(other, 0); } catch (const PrefillCancelled&) { cancelled = true; }
        require(cancelled && e.position() == 0, "cancel did not reset");
        e.set_prefill_progress(nullptr); e.prefill(other, 0); e.step(b, e.position()); equal(reference_b, e.last_logits());
        std::puts("PASS cancellation, reset and deterministic retry");
        if(!reference_out.empty() || !reference_in.empty()) {
            std::ofstream output; std::ifstream input;
            if(!reference_out.empty()) { output.open(reference_out,std::ios::binary);require(bool(output),"reference output"); }
            if(!reference_in.empty()) { input.open(reference_in,std::ios::binary);require(bool(input),"reference input"); }
            e.reset();int token=0;
            for(int t:prompt) token=e.step(t,e.position());
            for(int i=0;i<16;++i) {
                const auto& logits=e.last_logits();
                if(output.is_open()) output.write(reinterpret_cast<const char*>(logits.data()),kVocab*sizeof(float));
                if(input.is_open()) {
                    std::vector<float> expected(kVocab);input.read(reinterpret_cast<char*>(expected.data()),kVocab*sizeof(float));
                    require(bool(input),"reference truncated");equal(expected,logits);
                }
                token=e.step(token,e.position());
            }
            std::puts("PASS sequential PP/TP full-logit reference (16 generated tokens)");
        }
        std::puts("RESULT pass resident serving model contract");
    } catch (const std::exception& ex) { std::fprintf(stderr, "RESULT fail: %s\n", ex.what()); return 1; }
}
