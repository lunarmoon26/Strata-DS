#include "strata/ds41/suffix_drafter.hpp"
#include "strata/ds41/verify.hpp"
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
using namespace strata::ds41;
static void require(bool ok) { if (!ok) throw std::runtime_error("suffix assertion"); }
int main() {
    SuffixDrafter d;
    int32_t out[8]{};
    require(d.propose(7, out) == 0);
    for (int x : {1, 2, 3, 4, 5, 1, 2}) d.append(x);
    require(d.propose(7, out) == 0);
    d.append(3);
    require(d.propose(2, out) == 2 && out[0] == 4 && out[1] == 5);
    require(d.last_match() == 3);
    require(d.propose(0, out) == 0);
    d.reset();
    for (int x : {9, 1, 2, 3, 4, 8, 1, 2, 3, 7, 9, 1, 2, 3}) d.append(x);
    require(d.propose(1, out) == 1 && out[0] == 4 && d.last_match() == 4);
    d.reset();
    for (int x : {1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3}) d.append(x);
    require(d.propose(1, out) == 1 && out[0] == 5); // Most recent tie.
    d.reset();
    for (int i = 0; i < 20; ++i) d.append(7);
    require(d.propose(7, out) == 1 && out[0] == 7); // Overlap is legal.
    require(accepted_inputs({10}, {11}) == 1);
    require(accepted_inputs({10, 11, 12, 13}, {99, 12, 13, 14}) == 1);
    require(accepted_inputs({10, 11, 12, 13}, {11, 12, 99, 14}) == 3);
    require(accepted_inputs({10, 11, 12, 13}, {11, 12, 13, 14}) == 4);
    bool threw = false;
    try { accepted_inputs({}, {}); } catch (const std::invalid_argument&) { threw = true; }
    require(threw);
    // Rejected IDs are never appended. The next seed is a target prediction.
    d.reset();
    for (int x : {1, 2, 3, 4, 5, 1, 2, 3}) d.append(x);
    const std::vector<int> window{3, 4, 99}, target{4, 5, 6};
    const int keep = accepted_inputs(window, target);
    for (int i = 0; i < keep; ++i) d.append(target[i]);
    require(keep == 2 && d.size() == 10);
    // Constructor bounds still enforce a three-token match.
    SuffixDrafter short_match(1, 1, 32);
    for (int x : {128000, 128001, 128002, 17, 128000, 128001, 128002}) short_match.append(x);
    require(short_match.propose(1, out) == 1 && out[0] == 17);
    // A full hash table may miss proposals. It must never invent a continuation.
    SuffixDrafter small(3, 64, 4);
    std::mt19937 rng(159);
    std::vector<int> hist;
    for (int n = 0; n < 5000; ++n) {
        int x = int(rng()%7); hist.push_back(x); small.append(x);
        const int k = small.propose(7, out);
        if (!k) continue;
        require(k <= 7 && small.last_match() >= 3);
        bool found = false;
        for (int p = 2; p < n; ++p) {
            if (p+k > n) continue;
            bool match = true;
            for (int j = 0; j < 3; ++j) match &= hist[p-j] == hist[n-j];
            for (int j = 0; j < k; ++j) match &= out[j] == hist[p+j+1];
            found |= match;
        }
        require(found);
    }
    std::puts("RESULT pass suffix_drafter acceptance=zero,partial,full overlap=1 longest=1 ties=1");
}
