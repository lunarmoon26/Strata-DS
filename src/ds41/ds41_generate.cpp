// src/ds41/ds41_generate.cpp - run the M1 engine on token ids: greedy generation, timings, optional dump.
//
//   ds41_generate --pack DIR --ids 0,128000,... [--gen 32] [--threads 8] [--dump steps.bin] [--force-ids FILE]
//                 [--expert-profile ds41/data/expert-profile.bin [--vram-slots N] [--adapt-every 4] [--adapt-swaps 96]
//                  [--ram-budget-gib N (-1 available RAM less 8 GiB: default; less 24 GiB with DS41_RAM_ADAPT=0; 0 none)]]
//                 [--prefill [--prefill-chunk 65536] [--prefill-batch 4096] [--prefill-ring 256] [--prefill-threads 8]]
//
// --prefill runs the prompt (or, with --force-ids, the whole forced sequence for its nll) through the batched
// prefill (M3) instead of step() token by token; generation then continues with step().
// --force-ids feeds a fixed token sequence (from the oracle) instead of the engine's own predictions, so a
// per-layer comparison stays aligned even after the first differing token. Tokenization stays in Python.
// --force-list FILE runs one forced sequence per line, the engine reset before each, step() token by token (the
// decode path, where the tail skips act); --logprobs-out FILE also writes every position's fp16 log-softmax, so two
// engine settings can be compared by KL divergence (format: force_list()).
// --continue-ids FILE decodes as usual (prompt, then step() per token, --step-log) but feeds the file's tokens after
// the prompt instead of the engine's own predictions: every run of a speed A/B then routes the same experts.
// --spec oracle --continue-ids FILE measures the verify window: every window is --spec-max tokens, its drafts are the
// file's next tokens and every row is committed (the cost at full acceptance; the file is a greedy run's output).
// --spec dspark drafts with DSpark (DeepSeek's MTP drafter; the pack needs pack.py --mtp-experts). Per round it
// verifies the k drafts with the most expected tokens per millisecond: E(k) = 1 + S1 + .. + Sk, Si the product of the
// confidence head's first i probabilities, over the measured time of rounds with k drafts (DeepSeek's scheduler, and
// upstream's DraftPolicy). --draft-k K verifies K drafts every round instead (raw acceptance).
#include "strata/ds41/config.hpp"
#include "strata/ds41/engine.hpp"
#include "strata/ds41/suffix_drafter.hpp"

#include <algorithm>
#include <chrono>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::ds41;

/// comma-separated ids, or @FILE holding them (a 32K-token prompt is longer than one command-line argument may be)
/// IEEE binary16 of a float, rounded to nearest even
static uint16_t to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000, m = x & 0x7fffff;
    const int exp = int((x >> 23) & 0xff);
    if (exp == 0xff) return uint16_t(sign | 0x7c00 | (m ? 0x200 : 0));
    const int e = exp - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7c00);
    auto round = [](uint32_t kept, uint32_t rest, uint32_t half) {
        return kept + (rest > half || (rest == half && (kept & 1)));
    };
    if (e <= 0) {   // subnormal half (or zero)
        if (e < -10) return uint16_t(sign);
        const int shift = 14 - e;
        const uint32_t full = m | 0x800000;
        return uint16_t(sign | round(full >> shift, full & ((1u << shift) - 1), 1u << (shift - 1)));
    }
    return uint16_t(sign | round((uint32_t(e) << 10) | (m >> 13), m & 0x1fff, 0x1000));   // a carry fixes the exponent
}

static std::vector<int> parse_ids(const std::string& arg) {
    std::string s = arg;
    if (!s.empty() && s[0] == '@') {
        std::ifstream f(s.substr(1));
        if (!f) throw std::runtime_error("cannot open " + s.substr(1));
        s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
    return v;
}

/// The prefill line the speed scripts read (m3_context.sh): every field of PrefillTiming
static void print_prefill(const Engine& engine, size_t n) {
    const auto& p = engine.last_prefill();
    std::printf("prefill_tokens %zu ms %.1f tok_s %.1f chunks %d chunk_tokens %d sub_batch %d engram_ms %.1f stream_wait_ms %.1f"
                " vram_experts %lld streamed %lld (ram %lld cache %lld ssd %lld) engram_rows %lld unique %lld\n",
                n, p.total_ms, n / (std::max(p.total_ms, 1e-9) / 1000.0), p.chunks, p.chunk_tokens, p.sub_batch,
                p.engram_ms, p.stream_wait_ms, (long long) p.vram_experts, (long long) p.streamed,
                (long long) p.from_ram, (long long) p.from_cache, (long long) p.from_ssd, (long long) p.engram_rows,
                (long long) p.engram_unique);
}

// Output history contains raw IDs. It includes the pending target prediction.
/// --step-log: one line per decode step (where the step's time went), for the decode profile
static std::FILE* step_log = nullptr;

/// DSpark's policy (DeepSeek's scheduler and upstream's DraftPolicy: expected tokens per measured millisecond).
/// Before a round: draft only while drafting rounds commit more tokens per ms than plain steps (EMAs); after a losing
/// round, plain steps for a backoff (4, doubling to 64 while drafting keeps losing), then one drafting round again.
/// The first 16 tokens are plain steps.
/// After a draft (its time is spent): the k drafts with the most E(k) / verify(k), E(k) = 1 + S1 + .. + Sk, Si the
/// product of the first i confidence probabilities. Verify times are EMAs per k; a k not measured yet is the plain
/// step's time times 1 + 0.45 k (measured on the laptop: each verify row adds ~45% of a step).
struct DraftScheduler {
    static constexpr int kMax = 8;
    double vms[kMax] = {}, vn[kMax] = {};
    double step_ms = 0, spec_rate = 0;   // EMAs: a plain step's time, a drafting round's tokens per ms
    // the first 16 tokens are plain steps: they measure the step and keep drafting out of the cold start, when the
    // tiers are empty and every extra verify row reads its experts from the SSD (a 4-row window took 2.7 s)
    int backoff = 4, plain_left = 16;
    double verify_cost(int k) const {
        if (vn[k] > 0) return vms[k];
        const double base = step_ms > 0 ? step_ms : 50.0;
        return base * (1.0 + 0.45 * k);
    }
    bool want_draft() {
        if (plain_left > 0) { --plain_left; return false; }
        return true;
    }
    int choose(const Engine::Draft& d, int k_max) const {
        int best = 0;
        double e = 1, s = 1, best_rate = 1 / verify_cost(0);
        for (int k = 1; k <= std::min(k_max, d.n); ++k) {
            s *= d.accept[k - 1];
            e += s;
            if (e / verify_cost(k) > best_rate) { best_rate = e / verify_cost(k); best = k; }
        }
        return best;
    }
    void observe_step(double ms) { step_ms = step_ms > 0 ? 0.8 * step_ms + 0.2 * ms : ms; }
    void observe_round(int k, double verify_ms, double round_ms, int tokens) {
        vms[k] = vn[k] > 0 ? 0.8 * vms[k] + 0.2 * verify_ms : verify_ms;
        vn[k] += 1;
        const double rate = tokens / round_ms;
        spec_rate = spec_rate > 0 ? 0.7 * spec_rate + 0.3 * rate : rate;
        if (step_ms > 0 && spec_rate < 1.0 / step_ms) {   // drafting loses: plain steps for a while
            plain_left = backoff;
            backoff = std::min(backoff * 2, 64);
        } else {
            backoff = 4;
        }
    }
};

static void generate(Engine& engine, const EngineOptions& opt, const std::vector<int>& prompt,
                     int count, bool batched, bool suffix, int max_t, int eos, const std::vector<int>& cont = {},
                     bool oracle = false, bool dspark = false, int draft_k = -1) {
    if (prompt.empty()) throw std::invalid_argument("generation needs a nonempty prompt");
    int next = -1;
    if (batched) {
        next = engine.prefill(prompt, 0);
        print_prefill(engine, prompt.size());
    } else {
        for (size_t p = 0; p < prompt.size(); ++p) next = engine.step(prompt[p], int(p));
    }
    SuffixDrafter drafter(3, 64, size_t(opt.max_seq)+8);
    for (int token : prompt) drafter.append(token);
    std::vector<int> out;
    if (!cont.empty()) next = cont[0];   // --continue-ids: the file's tokens replace the predictions
    if (count > 0) { out.push_back(next); drafter.append(next); }
    int pos = int(prompt.size()), rounds = 0, accepted = 0;
    double step_ms = 0;            // plain decode: the engine's step times, as the forced path reports them
    int differ = 0;                // oracle: rows whose prediction differs from the file
    int64_t hits = 0, routed = 0, skipped = 0;
    DraftScheduler sched;
    double draft_ms = 0, draft_cpu_ms = 0;
    int spec_rounds = 0;   // DSpark: rounds that drafted (the others were plain steps)
    int64_t draft_hits = 0, draft_uses = 0;
    int64_t pos_tried[5] = {}, pos_ok[5] = {};   // DSpark: drafts verified and accepted, by draft position
    double pos_conf[5] = {};                     // the confidence head's probabilities of the verified drafts
    const auto begin = std::chrono::steady_clock::now();
    while (int(out.size()) < count && next != eos) {
        const int limit = std::min({max_t, count-int(out.size()), opt.max_seq-pos});
        if (limit < 1) throw std::runtime_error("generation exceeds max_seq");
        if (!suffix || (dspark && draft_k < 0 && !sched.want_draft())) {
            next = engine.step(next, pos++);
            if (dspark) sched.observe_step(engine.last_timing().total_ms);
            const auto& tm = engine.last_timing();
            if (step_log)
                std::fprintf(step_log, "%d %.3f %.3f %.3f %.3f %d %d %d %d %d %d %d %d %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                             pos - 1, tm.total_ms, tm.gpu_ms, tm.cpu_experts_ms, tm.engram_ms, tm.expert_total,
                             tm.expert_hits, tm.zero_copy_experts(), tm.ram_experts, tm.file_experts, tm.ssd_experts,
                             tm.vram_swaps, tm.warmed_useful, tm.prefetched, tm.worker_lead_ms, tm.worker_span_ms,
                             tm.worker_wait_ms, tm.worker_first_wait_ms, tm.admit_ms, tm.swaps_ms, tm.end_ms);
            step_ms += tm.total_ms;
            skipped += tm.skipped;
            hits += tm.expert_hits;
            routed += tm.expert_total;
            if (!cont.empty()) next = cont[out.size()];
            out.push_back(next); drafter.append(next); ++rounds;
            continue;
        }
        int32_t draft[8];
        const auto r0 = std::chrono::steady_clock::now();
        Engine::Draft dd;
        int proposed = 0;
        if (dspark) {
            dd = engine.draft(next);
            draft_ms += dd.ms;
            draft_cpu_ms += dd.cpu_experts_ms;
            draft_hits += dd.vram_hits;
            draft_uses += dd.uses;
            proposed = draft_k >= 0 ? std::min({draft_k, dd.n, limit-1}) : sched.choose(dd, std::min(limit-1, dd.n));
            ++spec_rounds;
            for (int i = 0; i < proposed; ++i) draft[i] = dd.tokens[i];
        } else {
            proposed = oracle ? limit-1 : drafter.propose(limit-1, draft);
        }
        if (oracle)
            for (int i = 0; i < proposed; ++i) draft[i] = cont[out.size() + i];
        std::vector<int> window{next};
        window.insert(window.end(), draft, draft+proposed);
        const auto v0 = std::chrono::steady_clock::now();
        auto result = engine.verify(window, pos);
        const double verify_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - v0).count();
        int keep = oracle ? int(window.size()) : accepted_inputs(window, result.next);
        for (int i = 0; i < keep && !oracle; ++i)
            if (result.next[i] == eos) { keep = i+1; break; }
        engine.commit(keep);
        const auto& tm = engine.last_timing();
        if (dspark) {
            sched.observe_round(proposed, verify_ms,
                                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count(),
                                keep);
            if (proposed == 0) sched.observe_step(verify_ms);   // a one-row window costs about a step
            for (int i = 0; i < proposed; ++i) {
                ++pos_tried[i];
                pos_conf[i] += dd.accept[i];
                if (i < keep - 1) ++pos_ok[i];
            }
        }
        std::printf("spec_window pos %d T %zu accepted %d emitted %d ms %.1f engram_ms %.2f cpu_experts_ms %.2f "
                    "hits %d of %d wait_ms %.2f admit_ms %.2f ssd %d ram %d", pos, window.size(), keep-1, keep,
                    verify_ms, tm.engram_ms, tm.cpu_experts_ms, tm.expert_hits, tm.expert_total, tm.worker_wait_ms,
                    tm.admit_ms, tm.ssd_experts, tm.ram_experts);
        if (dspark)
            std::printf(" draft_ms %.2f draft_cpu_ms %.2f conf %.2f %.2f %.2f %.2f %.2f", dd.ms, dd.cpu_experts_ms,
                        dd.accept[0], dd.accept[1], dd.accept[2], dd.accept[3], dd.accept[4]);
        std::printf("\n");
        skipped += tm.skipped;
        routed += tm.expert_total;
        for (int i = 0; i < keep; ++i) {
            const int t = oracle ? cont[out.size()] : result.next[i];
            if (oracle && result.next[i] != t) ++differ;
            out.push_back(t); drafter.append(t);
        }
        accepted += keep-1; ++rounds;
        next = out.back(); pos += keep;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    const int decoded = std::max(0, int(out.size())-1);
    std::printf("generated:");
    for (int token : out) std::printf(" %d", token);
    std::printf("\ndecode_tokens %d windows %d accepted_drafts %d seconds %.6f tok_s %.3f\n",
                decoded, rounds, accepted, seconds, decoded/std::max(seconds, 1e-9));
    // the lines the speed scripts read (m3_context.sh); speculation: wall time per emitted token
    const double ms = suffix ? seconds * 1000.0 : step_ms;
    std::printf("decode_ms_per_token %.1f over %d steps\n", decoded > 0 ? ms / decoded : 0.0, decoded);
    if (oracle) std::printf("oracle_rows_differing %d of %d\n", differ, decoded);
    if (dspark) {
        std::printf("dspark draft_ms %.2f cpu_ms %.2f vram_hit %.3f per round; by draft position: accepted/verified "
                    "(mean confidence)", spec_rounds ? draft_ms / spec_rounds : 0.0,
                    spec_rounds ? draft_cpu_ms / spec_rounds : 0.0,
                    draft_uses ? (double) draft_hits / draft_uses : 0.0);
        for (int i = 0; i < 5; ++i)
            std::printf(" %lld/%lld (%.2f)", (long long) pos_ok[i], (long long) pos_tried[i],
                        pos_tried[i] ? pos_conf[i] / pos_tried[i] : 0.0);
        std::printf("\nmean_accepted_per_window %.3f drafting_rounds %d of %d\n",
                    spec_rounds ? (double) accepted / spec_rounds : 0.0, spec_rounds, rounds);
    }
    std::printf("tail_skipped %lld of %lld routed uses\n", (long long) skipped, (long long) routed);
    if (!suffix)
        std::printf("vram_expert_slots %d hit_rate %.4f\n", engine.vram_expert_slots(),
                    routed ? (double) hits / routed : 0.0);
}

/// --force-list: one forced sequence per line (blank lines skipped), the engine reset before each. Prints each
/// sequence's mean nll and the overall one. --logprobs-out writes int32 {0x31504c44, vocab}, then for every scored
/// position int32 {sequence, pos, next forced id} and the fp16 log-softmax of the whole vocabulary.
static void force_list(Engine& engine, const std::string& list_path, const std::string& lp_path) {
    std::ifstream in(list_path);
    if (!in) throw std::runtime_error("cannot read --force-list " + list_path);
    std::FILE* out = lp_path.empty() ? nullptr : std::fopen(lp_path.c_str(), "wb");
    if (!lp_path.empty() && !out) throw std::runtime_error("cannot open --logprobs-out " + lp_path);
    auto put = [&](const void* p, size_t size, size_t n) {
        if (std::fwrite(p, size, n, out) != n) throw std::runtime_error("cannot write --logprobs-out " + lp_path);
    };
    if (out) {
        const int32_t hdr[2] = {0x31504c44, kVocab};
        put(hdr, 4, 2);
    }
    std::vector<uint16_t> row(kVocab);
    std::string line;
    int seq = 0, steps = 0;
    long long scored = 0;
    double total = 0, ms = 0;
    while (std::getline(in, line)) {
        const std::vector<int> ids = parse_ids(line);
        if (ids.empty()) continue;
        for (int t : ids)
            if (t < 0 || t >= kVocab) throw std::invalid_argument("token id " + std::to_string(t) + " is outside the vocabulary");
        engine.reset();
        double nll = 0;
        for (int pos = 0; pos + 1 < (int) ids.size(); ++pos) {   // the last id is only a target
            engine.step(ids[pos], pos);
            ms += engine.last_timing().total_ms;
            ++steps;
            const auto& lg = engine.last_logits();
            double mx = -1e300, se = 0;
            for (float v : lg) mx = std::max(mx, (double) v);
            for (float v : lg) se += std::exp((double) v - mx);
            const double lse = mx + std::log(se);
            nll += lse - lg[ids[pos + 1]];
            if (out) {
                const int32_t rec[3] = {seq, pos, ids[pos + 1]};
                put(rec, 4, 3);
                for (int v = 0; v < kVocab; ++v) row[v] = to_half(float(lg[v] - lse));
                put(row.data(), 2, kVocab);
            }
        }
        const int n = (int) ids.size() - 1;
        std::printf("force_list sequence %d tokens %zu mean_nll %.6f\n", seq, ids.size(), n > 0 ? nll / n : 0.0);
        total += nll;
        scored += n;
        ++seq;
    }
    if (out && std::fclose(out) != 0) throw std::runtime_error("cannot write --logprobs-out " + lp_path);
    std::printf("generated:\nforce_list sequences %d scored %lld mean_nll %.6f\ndecode_ms_per_token %.1f over %d steps\n",
                seq, scored, scored ? total / scored : 0.0, steps ? ms / steps : 0.0, steps);
}

int main(int argc, char** argv) {
    std::string pack, ids_s, dump_path, force_path, cont_path, list_path, logprobs_path, spec = "none";
    int spec_max = kVerifyMaxTokens, eos = -1, draft_k = -1;
    int gen = 32;
    bool batched = false;
    EngineOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--pack") pack = next();
        else if (a == "--devices") {
            try { opt.devices = parse_split_devices(next()); }
            catch (const std::exception& ex) { std::fprintf(stderr, "%s\n", ex.what()); return 2; }
        }
        else if (a == "--ids") ids_s = next();
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--spec") spec = next();
        else if (a == "--spec-max") spec_max = std::stoi(next());
        else if (a == "--eos-id") eos = std::stoi(next());
        else if (a == "--draft-k") draft_k = std::stoi(next());
        else if (a == "--threads") opt.cpu_threads = std::stoi(next());
        else if (a == "--max-seq") opt.max_seq = std::stoi(next());
        else if (a == "--expert-profile") opt.expert_profile = next();
        else if (a == "--vram-slots") opt.vram_expert_slots = std::stoll(next());
        else if (a == "--vram-reserve-mib") opt.vram_reserve_bytes = (size_t) std::stoll(next()) << 20;
        else if (a == "--adapt-every") opt.adapt_every = std::stoi(next());
        else if (a == "--adapt-swaps") opt.adapt_swaps = std::stoi(next());
        else if (a == "--ram-budget-gib") opt.ram_budget_gib = std::stod(next());
        else if (a == "--prefill") batched = true;
        else if (a == "--split-layer") opt.split_layer = std::stoi(next());
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--prefill-ring") opt.prefill_ring = std::stoi(next());
        else if (a == "--prefill-batch") opt.prefill_batch = std::stoi(next());
        else if (a == "--prefill-threads") opt.prefill_threads = std::stoi(next());
        else if (a == "--dump") dump_path = next();
        else if (a == "--step-log") {
            const std::string f = next();
            step_log = std::fopen(f.c_str(), "w");
            if (!step_log) { std::fprintf(stderr, "cannot write --step-log %s\n", f.c_str()); return 2; }
            std::fprintf(step_log, "# pos total_ms gpu_ms cpu_experts_ms engram_ms routed vram_hits zero_copy ram_cpu "
                                   "file_cpu ssd swaps warmed_useful prefetched lead_ms span_ms wait_ms first_wait_ms admit_ms "
                                   "swaps_ms end_ms\n");
        }
        else if (a == "--force-ids") force_path = next();
        else if (a == "--force-list") list_path = next();
        else if (a == "--logprobs-out") logprobs_path = next();
        else if (a == "--continue-ids") cont_path = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (pack.empty() || (ids_s.empty() && force_path.empty() && list_path.empty())) {
        std::fprintf(stderr, "usage: ds41_generate --pack DIR --ids a,b,c [--gen N] [--threads T] [--dump F] [--force-ids F] [--spec none|suffix|oracle|dspark] [--spec-max 1..8] [--draft-k 0..5] [--eos-id ID]\n");
        return 2;
    }
    std::vector<int> prompt = parse_ids(ids_s), forced;
    if (!force_path.empty()) {
        std::ifstream f(force_path);
        std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        forced = parse_ids(all);
    }
    std::vector<int> cont;
    if (!cont_path.empty()) {
        std::ifstream f(cont_path);
        std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        cont = parse_ids(all);
    }
    try {
        // the teacher-forced path reads logits at the next forced id before the engine sees it
        for (const auto* ids : {&prompt, &forced, &cont})
            for (int t : *ids)
                if (t < 0 || t >= kVocab) throw std::invalid_argument("token id " + std::to_string(t) + " is outside the vocabulary");
        if (gen < 0 || spec_max < 1 || spec_max > 8 ||
            (spec != "none" && spec != "suffix" && spec != "oracle" && spec != "dspark"))
            throw std::invalid_argument("use --gen >= 0, --spec none|suffix|oracle|dspark, --spec-max 1..8");
        if (draft_k > 5 || (draft_k >= 0 && spec != "dspark")) throw std::invalid_argument("--draft-k 0..5 needs --spec dspark");
        if (spec == "dspark") opt.dspark = true;
        if (spec != "none" && (!force_path.empty() || !dump_path.empty()))
            throw std::invalid_argument("--spec cannot be combined with --force-ids or --dump");
        if (spec == "oracle" && (int) cont.size() < gen + 8)
            throw std::invalid_argument("--spec oracle needs --continue-ids with at least --gen + 8 tokens");
        if (!cont_path.empty() && ((int) cont.size() < gen || spec == "suffix" || spec == "dspark" || !force_path.empty() || !dump_path.empty()))
            throw std::invalid_argument("--continue-ids needs at least --gen tokens, without --spec suffix or dspark, --force-ids, --dump");
        if (spec_max > kVerifyMaxTokens) {
            std::fprintf(stderr, "spec: cap T at %d until CPU rows 5..8 are validated\n", kVerifyMaxTokens);
            spec_max = kVerifyMaxTokens;
        }
        if (!logprobs_path.empty() && list_path.empty()) throw std::invalid_argument("--logprobs-out needs --force-list");
        if (!list_path.empty() && (!force_path.empty() || !dump_path.empty() || !cont_path.empty() || spec != "none" || batched))
            throw std::invalid_argument("--force-list runs step() alone: without --force-ids, --dump, --continue-ids, --spec, --prefill");
        Engine engine(pack, opt);
        if (!list_path.empty()) {
            force_list(engine, list_path, logprobs_path);
            return 0;
        }
        if (force_path.empty() && dump_path.empty()) {
            generate(engine, opt, prompt, gen, batched, spec != "none", spec_max, eos, cont, spec == "oracle",
                     spec == "dspark", draft_k);
            return 0;
        }
        if (batched) {   // prefill the prompt (or the forced sequence) in one call, then decode with step()
            const std::vector<int>& pre = forced.empty() ? prompt : forced;
            std::vector<float> nll;
            int next = engine.prefill(pre, 0, forced.empty() ? nullptr : &nll);
            print_prefill(engine, pre.size());
            if (!nll.empty()) {
                double sum = 0;
                for (float v : nll) sum += v;
                std::printf("teacher_forced_mean_nll %.6f ppl %.4f over %zu tokens\n", sum / nll.size(),
                            std::exp(sum / nll.size()), nll.size());
            }
            if (forced.empty()) {
                std::vector<int> out = {next};
                double ms = 0;
                for (int i = 1; i < gen; ++i) {
                    next = engine.step(next, (int) prompt.size() + i - 1);
                    ms += engine.last_timing().total_ms;
                    out.push_back(next);
                }
                std::printf("generated:");
                for (int v : out) std::printf(" %d", v);
                std::printf("\ndecode_ms_per_token %.1f over %d steps\n", gen > 1 ? ms / (gen - 1) : 0.0, gen - 1);
            }
            return 0;
        }
        std::FILE* dump = dump_path.empty() ? nullptr : std::fopen(dump_path.c_str(), "wb");
        if (!dump_path.empty() && !dump) throw std::runtime_error("cannot open --dump " + dump_path);
        auto put = [&](const void* p, size_t size, size_t n) {
            if (std::fwrite(p, size, n, dump) != n) throw std::runtime_error("cannot write --dump " + dump_path);
        };
        StepDump sd;
        const int total = forced.empty() ? (int) prompt.size() + std::max(0, gen-1) : (int) forced.size();
        int next = -1;
        std::vector<int> out;
        double decode_ms = 0, nll_sum = 0, cpu_ms = 0;
        long long hits = 0, routed = 0, ram = 0, file = 0, ssd = 0, warmed = 0, useful = 0;
        int decode_steps = 0, nll_n = 0;
        for (int pos = 0; pos < total; ++pos) {
            const int tok = !forced.empty() ? forced[pos] : pos < (int) prompt.size() ? prompt[pos] : next;
            next = engine.step(tok, pos, dump ? &sd : nullptr);
            const auto& t = engine.last_timing();
            if (!forced.empty() && pos + 1 < (int) forced.size()) {    // teacher-forced -log p(next token)
                const auto& lg = engine.last_logits();
                double mx = -1e300, se = 0;
                for (float v : lg) mx = std::max(mx, (double) v);
                for (float v : lg) se += std::exp((double) v - mx);
                nll_sum += mx + std::log(se) - lg[forced[pos + 1]];
                ++nll_n;
            }
            if (pos >= (int) prompt.size() - 1) { decode_ms += t.total_ms; cpu_ms += t.cpu_experts_ms; ++decode_steps; }
            hits += t.expert_hits;
            ram += t.ram_experts;
            file += t.file_experts;
            ssd += t.ssd_experts;
            warmed += t.warmed;
            useful += t.warmed_useful;
            routed += t.expert_total;
            if (pos >= (int) prompt.size() - 1 && forced.empty() && gen > 0) out.push_back(next);
            std::fprintf(stderr,
                         "pos %d tok %d -> %d  total %.1f ms (engram reads %.1f, layers %.1f, of which cpu experts %.1f)"
                         "  vram hits %d/%d swaps %d  zero-copy %d cpu %d: ram %d file %d (ssd %d)  warmed %d useful %d\n",
                         pos, tok, next, t.total_ms, t.engram_ms, t.gpu_ms, t.cpu_experts_ms, t.expert_hits,
                         t.expert_total, t.vram_swaps, t.zero_copy_experts(), t.cpu_experts(), t.ram_experts, t.file_experts,
                         t.ssd_experts, t.warmed, t.warmed_useful);
            if (dump) {
                const int32_t hdr[2] = {tok, next};
                put(hdr, 4, 2);
                put(sd.hidden.data(), 2, sd.hidden.size());
                for (int l = 0; l < kLayers; ++l) put(sd.routes[l].data(), 4, kTopK);
                for (int l = 0; l < kLayers; ++l) put(sd.weights[l].data(), 4, kTopK);
                for (auto& p : sd.top_logits) put(&p.first, 4, 1);
                for (auto& p : sd.top_logits) put(&p.second, 4, 1);
            }
        }
        if (dump && std::fclose(dump) != 0) throw std::runtime_error("cannot write --dump " + dump_path);
        std::printf("generated:");
        for (int v : out) std::printf(" %d", v);
        if (nll_n) std::printf("\nteacher_forced_mean_nll %.6f ppl %.4f over %d tokens", nll_sum / nll_n,
                               std::exp(nll_sum / nll_n), nll_n);
        std::printf("\ndecode_ms_per_token %.1f over %d steps (cpu experts %.1f)\n",
                    decode_steps ? decode_ms / decode_steps : 0.0, decode_steps,
                    decode_steps ? cpu_ms / decode_steps : 0.0);
        std::printf("vram_expert_slots %d hit_rate %.4f\n", engine.vram_expert_slots(),
                    routed ? (double) hits / routed : 0.0);
        std::printf("tiers_share vram %.4f ram %.4f file %.4f ssd %.4f\n", routed ? (double) hits / routed : 0.0,
                    routed ? (double) ram / routed : 0.0, routed ? (double) file / routed : 0.0,
                    routed ? (double) ssd / routed : 0.0);
        std::printf("lookahead warmed %lld useful %lld (%.3f)\n", warmed, useful, warmed ? (double) useful / warmed : 0.0);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
