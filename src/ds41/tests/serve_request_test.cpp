// src/ds41/tests/serve_request_test.cpp - ds41_serve's line protocol on the host: the GEN line, the prefix reuse,
// the snapshot choice and the DONE line. No GPU and no pack.
//   c++ -std=c++17 -Iinclude src/ds41/tests/serve_request_test.cpp -o /tmp/srt && /tmp/srt
#include "strata/ds41/serve_request.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace strata::ds41::serve;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

}  // namespace

int main() {
    {
        Request r;
        const std::string err = parse_gen("GEN 16 1,2,3", r);
        check(err.empty(), "a plain GEN line parses");
        check(r.max_new == 16 && r.ids == std::vector<int>{1, 2, 3}, "max_new and the ids");
        check(r.temperature == 0.0f && r.top_k == 20 && r.top_p == 1.0f && r.seed == 0, "absent keys: greedy defaults");
        check(r.greedy(), "no keys: greedy");
    }
    {
        Request r;
        const std::string err = parse_gen(
            "GEN 8 temperature=0.6 top_p=0.95 top_k=40 min_p=0.05 penalty_last_n=64 penalty_repeat=1.1 "
            "penalty_freq=0.2 penalty_present=0.3 seed=42 pcie_frac=0.5 cvec=1 future_key=7 5,6",
            r);
        check(err.empty(), "a GEN line with every sampling key parses");
        check(r.temperature == 0.6f && r.top_p == 0.95f && r.top_k == 40 && r.min_p == 0.05f, "the filters");
        check(r.penalty_last_n == 64 && r.penalty_repeat == 1.1f && r.penalty_freq == 0.2f &&
                  r.penalty_present == 0.3f,
              "the penalties");
        check(r.seed == 42 && r.ids == std::vector<int>{5, 6}, "the seed; unknown keys are skipped");
        check(!r.greedy(), "a temperature: sampled");
    }
    {
        Request r;
        parse_gen("GEN 4 temperature=0 penalty_last_n=64 penalty_repeat=1.2 1", r);
        Request q;
        parse_gen("GEN 4 temperature=0 penalty_repeat=1.2 1", q);
        check(q.greedy(), "a penalty without penalty_last_n is off (upstream: 0 disables)");
        check(!r.greedy(), "temperature 0 with a penalty still goes through the sampler");
    }
    {
        Request r;
        check(parse_gen("GEN 0 1,2", r) == "max_new", "max_new 0 is refused");
        check(parse_gen("GEN 4", r) == "no token ids", "no ids are refused");
        check(parse_gen("GEN 4 1,,2", r).find("token id") != std::string::npos, "an empty id is refused");
        check(parse_gen("GEN 4 1,x", r).find("token id") != std::string::npos, "a non-number id is refused");
        check(parse_gen("GEN 4 99999999999", r).find("token id") != std::string::npos, "an id past int is refused");
        check(parse_gen("GEN x 1", r) == "max_new", "a non-number max_new is refused");
    }
    {
        const std::vector<int> live = {1, 2, 3, 4};
        check(reusable(live, {1, 2, 3, 4, 5, 6}) == 4, "a prompt that continues the session reuses all of it");
        check(reusable(live, {1, 2, 3, 4}) == 0, "the same prompt: the last token must be read again, start over");
        check(reusable(live, {1, 2, 9, 4, 5}) == 0, "a prompt that differs inside the session starts over");
        check(reusable(live, {1, 2}) == 0, "a shorter prompt starts over");
        check(reusable({}, {1, 2}) == 0, "an empty session reuses nothing");
    }
    {
        // snapshots: tokens fed before each saved position; the session is 1..6 now
        const std::vector<int> live = {1, 2, 3, 4, 5, 6};
        const std::vector<std::vector<int>> snaps = {{}, {1, 2, 3}, {1, 2, 3, 4}, {1, 9}};
        check(pick_snapshot(snaps, live, {1, 2, 3, 4, 7, 8}, 0) == 2,
              "the longest snapshot the prompt starts with (the last answer's start changed)");
        check(pick_snapshot(snaps, live, {1, 2, 3, 7}, 0) == 1, "a shorter one when the longer does not match");
        check(pick_snapshot(snaps, live, {1, 2, 3, 4, 5, 6, 7}, 6) == -1, "none when the session itself reuses more");
        check(pick_snapshot(snaps, live, {1, 2, 3, 4}, 0) == 1,
              "never all of the prompt: its last token must be read again");
        check(pick_snapshot(snaps, {1, 2}, {1, 2, 3, 4, 7}, 0) == -1,
              "none past the session (the session was restarted since they were saved)");
        check(pick_snapshot(snaps, {1, 2, 8, 4, 5}, {1, 2, 3, 4, 7}, 0) == -1,
              "none whose tokens the session no longer has");
        check(pick_snapshot({{1, 9}}, {1, 9, 4}, {1, 9, 5}, 0) == 0, "a two-token snapshot");
    }
    {
        DoneStats d;
        d.generated = 5;
        d.prompt = 100;
        d.prompt_ms = 812.04;
        d.decode_ms = 201.55;
        d.finish = "stop";
        d.reused = 40;
        d.hits = 30;
        d.lookups = 48;
        d.ram = 10;
        d.file = 2;
        d.read = 60;
        d.offloaded = 6;
        check(done_line(d) == "DONE 5 100 812.0 201.6 stop 0 0 40 30 48 10 2 0.0 60 6", "the DONE line, upstream's order");
    }
    std::printf("RESULT %s serve_request\n", failures ? "fail" : "pass");
    return failures ? 1 : 0;
}
