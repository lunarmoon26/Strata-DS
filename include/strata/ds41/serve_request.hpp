// include/strata/ds41/serve_request.hpp - ds41_serve's line protocol, the host-only parts: the GEN line, which part
// of the session (or which snapshot) a prompt reuses, and the DONE line. The formats are upstream's `strata --serve`
// (src/program/generate.cpp), so serve/server.py drives both engines the same way.
#pragma once

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::ds41::serve {

/// One GEN request. Absent keys keep upstream's defaults: greedy, no penalties.
struct Request {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 20;   ///< upstream's sampler default; the sampled path needs 1..64
    int penalty_last_n = 0;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    unsigned long long seed = 0;   ///< 0: a seed from the clock
    std::vector<int> ids;

    bool penalties() const {
        return penalty_last_n > 0 && (penalty_repeat != 1.0f || penalty_freq != 0.0f || penalty_present != 0.0f);
    }
    /// the engine's own argmax is the answer: no temperature and nothing that changes the logits
    bool greedy() const { return temperature <= 0.0f && !penalties(); }
};

/// "GEN <max_new> [key=value ...] id,id,..." -> r. Returns "" or what is wrong. Unknown keys are skipped (upstream:
/// the ids start at the first token without '='), so a newer server can send keys this engine does not use.
inline std::string parse_gen(const std::string& line, Request& r) {
    r = Request{};
    if (line.rfind("GEN ", 0) != 0) return "expected: GEN <max_new> <id,id,...>";
    const char* p = line.c_str() + 4;
    char* end = nullptr;
    errno = 0;
    r.max_new = std::strtoll(p, &end, 10);
    if (end == p || errno != 0 || r.max_new < 1 || (*end != ' ' && *end != '\0')) return "max_new";
    p = end;
    for (;;) {
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        if (p == start) break;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            p = start;
            break;
        }
        const std::string key = tok.substr(0, eq);
        const char* v = tok.c_str() + eq + 1;
        const float fv = std::strtof(v, nullptr);
        if (key == "temperature") r.temperature = fv;
        else if (key == "top_p") r.top_p = fv;
        else if (key == "top_k") r.top_k = std::atoi(v);
        else if (key == "min_p") r.min_p = fv;
        else if (key == "penalty_last_n") r.penalty_last_n = std::atoi(v);
        else if (key == "penalty_repeat") r.penalty_repeat = fv;
        else if (key == "penalty_freq") r.penalty_freq = fv;
        else if (key == "penalty_present") r.penalty_present = fv;
        else if (key == "seed") r.seed = std::strtoull(v, nullptr, 10);
    }
    while (*p == ' ') ++p;
    if (*p == '\0') return "no token ids";
    for (;;) {
        errno = 0;
        const long long v = std::strtoll(p, &end, 10);
        if (end == p || errno != 0 || v < INT_MIN || v > INT_MAX)
            return "token id " + std::to_string(r.ids.size()) + " is not a number";
        r.ids.push_back((int) v);
        p = end;
        if (*p == '\0' || *p == ' ' || *p == '\r') break;
        if (*p != ',') return "token id " + std::to_string(r.ids.size()) + " is followed by '" + std::string(1, *p) + "'";
        ++p;
    }
    return "";
}

/// How many tokens of the session (`live`, every token fed so far) the prompt reuses. The session can only grow, so
/// it is all of it when the prompt starts with it, else none (the engine starts over at position 0). Upstream's rule:
/// at most n - 1, the prompt's last token is always read again (its logits pick the first output).
inline int64_t reusable(const std::vector<int>& live, const std::vector<int>& prompt) {
    const size_t L = live.size();
    if (prompt.empty() || L < 1 || L > prompt.size() - 1) return 0;
    for (size_t i = 0; i < L; ++i)
        if (live[i] != prompt[i]) return 0;
    return (int64_t) L;
}

/// Which snapshot a prompt goes back to, or -1. snaps[i]: the tokens fed before slot i's position (empty: no
/// snapshot). A snapshot counts when the session still holds its tokens (they were fed before its position and not
/// replaced since) and the prompt starts with them; it must beat the session's own reuse (`live_reuse`) and leave the
/// prompt's last token to read. The longest one wins.
inline int pick_snapshot(const std::vector<std::vector<int>>& snaps, const std::vector<int>& live,
                         const std::vector<int>& prompt, int64_t live_reuse) {
    int best = -1;
    size_t best_n = 0;
    for (size_t i = 0; i < snaps.size(); ++i) {
        const std::vector<int>& t = snaps[i];
        const size_t n = t.size();
        if (n == 0 || (int64_t) n <= live_reuse || n <= best_n || n + 1 > prompt.size() || n > live.size()) continue;
        bool same = true;
        for (size_t j = 0; j < n && same; ++j) same = t[j] == live[j] && t[j] == prompt[j];
        if (same) {
            best = (int) i;
            best_n = n;
        }
    }
    return best;
}

/// The fields of the DONE line
struct DoneStats {
    long long generated = 0, prompt = 0;
    double prompt_ms = 0, decode_ms = 0;
    std::string finish = "length";   ///< stop | length | cancel
    long long accepted = 0, offered = 0, reused = 0;
    long long hits = 0, lookups = 0;   ///< decode's routed experts from VRAM slots, of all routed uses
    long long ram = 0, file = 0;       ///< the CPU's experts from the RAM copy and from the mapped pack
    double file_mb = 0;
    long long read = 0;                ///< prompt tokens read (fewer than prompt - reused when a cancel stopped it)
    long long offloaded = 0;           ///< routed experts the GPU read over PCIe (zero copy)
};

/// DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <accepted> <offered> <reused> <hits> <lookups> <ram>
///      <file> <file MB> <read> <offloaded> - upstream's order (generate.cpp), so the server parses it unchanged
inline std::string done_line(const DoneStats& d) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld",
                  d.generated, d.prompt, d.prompt_ms, d.decode_ms, d.finish.c_str(), d.accepted, d.offered, d.reused,
                  d.hits, d.lookups, d.ram, d.file, d.file_mb, d.read, d.offloaded);
    return buf;
}

}  // namespace strata::ds41::serve
