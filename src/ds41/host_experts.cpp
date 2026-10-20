// src/ds41/host_experts.cpp - see include/strata/ds41/host_experts.hpp.
#include "strata/ds41/host_experts.hpp"
#include "strata/ds41/parallel.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/residency.hpp"
#include "strata/ds41/vram_experts.hpp"

#include "moe_mul1.h"   // third_party/exllamav3_moe

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace strata::ds41 {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ds41 RAM tier: ") + what + ": " + cudaGetErrorString(e));
}

/// a "key: N kB" line of /proc/meminfo, or a "key N" line of a cgroup stat file, in bytes; 0 when absent
uint64_t read_field(const char* path, const std::string& key, uint64_t scale) {
    std::ifstream f(path);
    std::string k;
    uint64_t v = 0;
    while (f >> k >> v) {
        if (k == key || k == key + ":") return v * scale;
        std::string rest;
        std::getline(f, rest);
    }
    return 0;
}

uint64_t read_number(const char* path) {
    std::ifstream f(path);
    std::string s;
    if (!(f >> s) || s == "max") return 0;
    return std::strtoull(s.c_str(), nullptr, 10);
}

}  // namespace

std::vector<std::pair<int, int>> plan_ram_tier(const std::vector<std::pair<int, int>>& ranked,
                                               const std::vector<int32_t>& vram_res, int n_experts,
                                               const std::vector<uint64_t>& bytes, uint64_t budget) {
    std::vector<std::pair<int, int>> out;
    uint64_t at = 0;
    for (const auto& [l, e] : ranked) {
        const size_t i = (size_t) l * n_experts + e;
        if (vram_res[i] >= 0 || bytes[i] > budget - at) continue;   // upstream: skip it, smaller ones may fit
        out.emplace_back(l, e);
        at += bytes[i];
    }
    return out;
}

size_t auto_ram_budget(size_t headroom) {
    uint64_t avail = read_field("/proc/meminfo", "MemAvailable", 1024);
    uint64_t limit = read_number("/sys/fs/cgroup/memory.max"), anon = 0;
    if (limit) {
        anon = read_field("/sys/fs/cgroup/memory.stat", "anon", 1);
    } else {   // cgroup v1 (MemAvailable is then the host's): its limit, "no limit" being a huge number
        limit = read_number("/sys/fs/cgroup/memory/memory.limit_in_bytes");
        if (limit >= (1ull << 60)) limit = 0;
        anon = read_field("/sys/fs/cgroup/memory/memory.stat", "total_rss", 1);
    }
    if (limit) {   // a container: its limit counts the file cache too, which the arena may displace
        const uint64_t room = limit > anon ? limit - anon : 0;
        avail = avail ? std::min(avail, room) : room;
    }
    return avail > headroom ? (size_t) (avail - headroom) : 0;
}

HostExperts::HostExperts(const Pack& pack, const std::vector<std::pair<int, int>>& ranked,
                         const std::vector<int32_t>& vram_res, size_t budget, const std::vector<int64_t>& cpu_handles,
                         int threads)
    : pack_(pack), handles_(cpu_handles), n_experts_(pack.n_experts()) {
    const int L = pack.n_layers();
    slot_.assign((size_t) L * n_experts_, -1);
    off_.assign(1, 0);
    // each slot as large as its expert, 4 KiB aligned (a pack's experts are already)
    std::vector<uint64_t> bytes((size_t) L * n_experts_);
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < n_experts_; ++e) bytes[(size_t) l * n_experts_ + e] = (pack.expert(l, e).bytes + 4095) / 4096 * 4096;
    const auto plan = plan_ram_tier(ranked, vram_res, n_experts_, bytes, budget);
    slots_ = (int) plan.size();
    if (slots_ == 0) return;
    busy_.assign(slots_, 0);   // VRAM swap locks count from the start, so enable_adapt() sees swaps in flight
    for (const auto& [l, e] : plan) {
        const uint64_t b = bytes[(size_t) l * n_experts_ + e];
        off_.push_back(off_.back() + b);
        max_slot_bytes_ = std::max<size_t>(max_slot_bytes_, b);
    }
    arena_bytes_ = off_.back();
    void* p = mmap(nullptr, arena_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("ds41 RAM tier: cannot reserve " + std::to_string(arena_bytes_) + " B");
    arena_ = (uint8_t*) p;
    holder_ = plan;
    // Read the slots with O_DIRECT, 16 readers: the file cache cannot keep a pack beside a RAM tier this large, and
    // filling the tier through it (page faults on the mapped pack) ran at 0.5 GB/s on a 120 GiB container whose SSD
    // reads 5.2 GB/s with 8 parallel O_DIRECT readers (RTX 3090 + EPYC 7D12 box, 2026-10-07). Slots and expert
    // offsets are 4 KiB aligned in a pack (an expert that is not is copied) and slot lengths 4 KiB multiples. DS41_FILL_DIRECT=0, or a file
    // system that refuses O_DIRECT, copies from the mapped pack as before.
    const char* fill_env = std::getenv("DS41_FILL_DIRECT");
    const int dfd = fill_env && fill_env[0] == '0' ? -1 : open((pack.dir() + "/experts.bin").c_str(), O_RDONLY | O_DIRECT);
    filled_direct_ = dfd >= 0;
    const uint8_t* base = pack.expert_base();
    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    const int readers = filled_direct_ ? std::max(threads, 16) : std::max(threads, 1);
    run_parallel((size_t) readers, [&](size_t) {
        for (int s; (s = next++) < slots_ && !failed;) {
            const ExpertSlot& x = pack.expert(holder_[s].first, holder_[s].second);
            if (filled_direct_ && x.offset % 4096 == 0) {   // an unaligned expert (not in a real pack): copied
                const size_t len = (size_t) (off_[s + 1] - off_[s]);   // the slot: the expert rounded to 4 KiB
                size_t got = 0;
                while (got < x.bytes) {
                    const ssize_t r = pread(dfd, arena_ + off_[s] + got, len - got, (off_t) (x.offset + got));
                    if (r <= 0) break;
                    got += (size_t) r;
                }
                if (got < x.bytes) failed = true;
                continue;
            }
            std::memcpy(arena_ + off_[s], base + x.offset, x.bytes);
            // the file pages are not needed any more: unmap them, so the OS reclaims them first
            const uintptr_t a = ((uintptr_t) (base + x.offset)) & ~(uintptr_t) 4095;
            madvise((void*) a, (uintptr_t) (base + x.offset + x.bytes) - a, MADV_DONTNEED);
        }
    });
    if (dfd >= 0) close(dfd);
    if (failed) {
        munmap(arena_, arena_bytes_);
        arena_ = nullptr;
        throw std::runtime_error("ds41 RAM tier: a read of " + pack.dir() + "/experts.bin failed");
    }
    // Map the anonymous arena only. Pageable file experts must stay CPU-only. Pinning fails when memory is short at
    // that moment (the laptop: a 50 GiB tier of 60 GB, right after another process ended); an unpinned tier is paged
    // out and the GPU cannot read it (a token then took 86 instead of 54 ms). So the lowest-ranked slots at the
    // arena's end leave, in doubling steps (below), until the pin fits; at most half goes.
    auto pin = [&] {
        const size_t limit = detail::register_limit().load();
        if (limit && arena_bytes_ > limit) return false;   // tests: a pin that does not fit
        if (cudaHostRegister(arena_, arena_bytes_, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess)
            return true;
        cudaGetLastError();
        return false;
    };
    bool pinned = pin();
    const size_t full_bytes = arena_bytes_;
    // steps of 512 MiB, doubling (0.5, 1, 2, 4 GiB ...): a pin that fails now and then costs 0.5 GiB of tier (it
    // failed and then fit 0.5 GiB lower in 3 of 6 laptop starts; a failed attempt takes 1.4-3.9 s), one that keeps
    // failing still ends within ~7 attempts. The last step stops at half the arena.
    size_t step = std::max<size_t>(std::min<size_t>(full_bytes / 64, 512ull << 20), 1);
    while (!pinned && slots_ > 1 && arena_bytes_ > full_bytes / 2) {
        const size_t target = arena_bytes_ - std::min(step, arena_bytes_ - full_bytes / 2);
        int keep = slots_;
        while (keep > 1 && off_[keep] > target) --keep;
        if (off_[keep] < full_bytes / 2 && keep < slots_) ++keep;   // whole slots: never below half
        if (keep == slots_) break;   // nothing left to drop above half
        munmap(arena_ + off_[keep], arena_bytes_ - off_[keep]);   // slot offsets are 4 KiB aligned
        slots_ = keep;
        off_.resize((size_t) keep + 1);
        holder_.resize((size_t) keep);
        busy_.resize((size_t) keep);
        arena_bytes_ = off_.back();
        max_slot_bytes_ = 0;
        for (int s = 0; s < slots_; ++s) max_slot_bytes_ = std::max<size_t>(max_slot_bytes_, off_[s + 1] - off_[s]);
        pinned = pin();
        if (pinned)
            std::fprintf(stderr, "ds41 RAM tier: pinned after dropping to %.1f GiB (%d experts)\n",
                         arena_bytes_ / 1073741824.0, slots_);
        step *= 2;
    }
    if (pinned) {
        locked_ = registered_ = true;
        void* alias = nullptr;
        if (cudaHostGetDevicePointer(&alias, arena_, 0) == cudaSuccess)
            device_alias_ = static_cast<uint8_t*>(alias);
        else
            cudaGetLastError();
    } else {
        locked_ = mlock(arena_, arena_bytes_) == 0;
    }
    if (!locked_)
        std::fprintf(stderr, "ds41 RAM tier: could not lock %.1f GiB (memlock limit?); it may be paged out\n",
                     arena_bytes_ / 1073741824.0);
    if (!device_alias_)
        std::fprintf(stderr, "ds41 RAM tier: GPU mapping unavailable; RAM experts stay on the CPU\n");
    if (device_alias_) {
        try {
            std::vector<kernels::Exl3Expert> desc((size_t) L * n_experts_);
            for (int s = 0; s < slots_; ++s) {
                const auto [l, e] = holder_[s];
                desc[(size_t) l * n_experts_ + e] =
                    VramExperts::describe_at(pack_, l, e, device_alias_ + off_[s]);
            }
            ck(cudaMalloc(&experts_dev_, desc.size() * sizeof(desc[0])), "descriptor allocation");
            ck(cudaMemcpy(experts_dev_, desc.data(), desc.size() * sizeof(desc[0]), cudaMemcpyHostToDevice),
               "descriptor upload");
            // The next step may use a nonblocking stream.
            ck(cudaStreamSynchronize(nullptr), "descriptor publication");
        } catch (...) {
            cudaFree(experts_dev_);
            cudaHostUnregister(arena_);
            munmap(arena_, arena_bytes_);
            throw;
        }
    }
    for (int s = 0; s < slots_; ++s) {
        publish_residency(slot_[(size_t) holder_[s].first * n_experts_ + holder_[s].second], s);
        point(holder_[s].first, holder_[s].second, slot_ptr(s));
    }
}

HostExperts::~HostExperts() {
    if (!pf_threads_.empty()) {
        {
            std::lock_guard<std::mutex> lk(admit_mu_);
            pf_stop_ = true;
        }
        pf_cv_.notify_all();
        for (auto& t : pf_threads_) t.join();
    }
    if (dfd_ >= 0) close(dfd_);
    cudaFree(experts_dev_);
    if (pend_idx_) cudaFreeHost(pend_idx_);
    if (pend_desc_) cudaFreeHost(pend_desc_);
    if (!arena_) return;
    if (registered_) cudaHostUnregister(arena_);
    else if (locked_) munlock(arena_, arena_bytes_);
    munmap(arena_, arena_bytes_);
}

void HostExperts::point(int layer, int expert, const uint8_t* bytes) {
    if (handles_.empty()) return;   // no CPU kernel (tests)
    const ExpertSlot& s = pack_.expert(layer, expert);
    auto desc = [&](int c0, int k_tiles, int n_tiles) {
        MoeCpuMatrixDesc d;
        d.trellis = (const uint16_t*) (bytes + s.comp_off[c0]);
        d.suh = (const at::Half*) (bytes + s.comp_off[c0 + 1]);
        d.svh = (const at::Half*) (bytes + s.comp_off[c0 + 2]);
        d.k_tiles = k_tiles;
        d.n_tiles = n_tiles;
        d.tile_w = (int) (s.comp_bytes[c0] / ((uint64_t) k_tiles * n_tiles * 2));
        return d;
    };
    const MoeCpuMatrixDesc g = desc(0, kDim / 16, kMoeInter / 16), u = desc(4, kDim / 16, kMoeInter / 16),
                           d = desc(8, kMoeInter / 16, kDim / 16);
    exl3_moe_cpu_set_expert_raw(handles_[layer], expert, &g, &u, &d, 0);
}

void HostExperts::reserve_pending(int n) {
    if (n <= cap_pend_) return;
    const int cap = std::max(n, std::max(2 * cap_pend_, 256));
    int32_t* idx = nullptr;
    kernels::Exl3Expert* desc = nullptr;
    ck(cudaHostAlloc((void**) &idx, (size_t) cap * sizeof(int32_t), cudaHostAllocMapped), "pending descriptors");
    if (cudaHostAlloc((void**) &desc, (size_t) cap * sizeof(kernels::Exl3Expert), cudaHostAllocMapped) != cudaSuccess) {
        cudaFreeHost(idx);
        ck(cudaErrorMemoryAllocation, "pending descriptors");
    }
    if (n_pend_) {
        std::memcpy(idx, pend_idx_, (size_t) n_pend_ * sizeof(int32_t));
        std::memcpy((void*) desc, pend_desc_, (size_t) n_pend_ * sizeof(kernels::Exl3Expert));
    }
    if (pend_idx_) cudaFreeHost(pend_idx_);
    if (pend_desc_) cudaFreeHost(pend_desc_);
    pend_idx_ = idx;
    pend_desc_ = desc;
    cap_pend_ = cap;
}

void HostExperts::defer_descriptors() {
    if (!experts_dev_) return;
    if (pend_pos_.empty()) pend_pos_.assign(slot_.size(), -1);
    defer_ = true;
}

void HostExperts::flush_descriptors() {
    if (!defer_) return;
    defer_ = false;
    if (!n_pend_) return;
    VramExperts::scatter_descriptors(experts_dev_, pend_idx_, pend_desc_, n_pend_);
    ck(cudaStreamSynchronize(nullptr), "descriptor publication");   // as publish_descriptor: before any step
    for (int i = 0; i < n_pend_; ++i) pend_pos_[pend_idx_[i]] = -1;
    n_pend_ = 0;
}

void HostExperts::publish_descriptor(int layer, int expert, int slot) {
    if (!experts_dev_) return;
    kernels::Exl3Expert desc{};
    if (slot >= 0) desc = VramExperts::describe_at(pack_, layer, expert, device_alias_ + off_[slot]);
    if (defer_) {
        const int32_t i = (int32_t) ((size_t) layer * n_experts_ + expert);
        if (pend_pos_[i] < 0) {
            reserve_pending(n_pend_ + 1);
            pend_pos_[i] = n_pend_;
            pend_idx_[n_pend_++] = i;
        }
        pend_desc_[pend_pos_[i]] = desc;
        return;
    }
    ck(cudaMemcpy(experts_dev_ + (size_t) layer * n_experts_ + expert, &desc, sizeof(desc),
                  cudaMemcpyHostToDevice), "descriptor update");
    ck(cudaStreamSynchronize(nullptr), "descriptor publication");
}

void HostExperts::point_to(int layer, int expert, const uint8_t* bytes) {
    publish_descriptor(layer, expert, -1);
    point(layer, expert, bytes);
    if (held_.empty()) held_.assign(slot_.size(), 0);
    held_[(size_t) layer * n_experts_ + expert] = 1;
}

void HostExperts::point_to_file(int layer, int expert) {
    if (!held_.empty()) held_[(size_t) layer * n_experts_ + expert] = 0;
    publish_descriptor(layer, expert, -1);
    point(layer, expert, pack_.expert_base() + pack_.expert(layer, expert).offset);
    // Read from the file now. Keep its slot reserved until assign().
    publish_residency(slot_[(size_t) layer * n_experts_ + expert], -1);
}

void HostExperts::assign(int slot, int layer, int expert) {
    if (slot < 0 || slot >= slots_) throw std::invalid_argument("HostExperts::assign: bad slot");
    if (pack_.expert(layer, expert).bytes > slot_capacity(slot))
        throw std::invalid_argument("HostExperts::assign: the expert is larger than the slot");
    const auto [ol, oe] = holder_[slot];
    if (ol >= 0) {   // (-1, -1): a free slot of the adaptive tier
        publish_descriptor(ol, oe, -1);
        publish_residency(slot_[(size_t) ol * n_experts_ + oe], -1);
    }
    holder_[slot] = {layer, expert};
    if (!held_.empty()) held_[(size_t) layer * n_experts_ + expert] = 0;
    publish_residency(slot_[(size_t) layer * n_experts_ + expert], slot);
    point(layer, expert, slot_ptr(slot));
    publish_descriptor(layer, expert, slot);
    if (!age_.empty()) {   // adaptive: the expert comes from VRAM, where it was in use until now
        age_[slot] = clock_;
        free_[slot] = 0;
    }
}

// ------------------------------------------------------------------------------------------------ adaptive tier

void HostExperts::free_slot(int slot) {
    const auto [l, e] = holder_[slot];
    // point_to_file only when the table still names this slot (a VRAM swap may have moved the expert on)
    if (l >= 0 && slot_of(l, e) == slot) point_to_file(l, e);
    holder_[slot] = {-1, -1};
    free_[slot] = 1;
}

namespace {
constexpr size_t kReaders = 8;   // admit()'s reader threads, the caller included
constexpr int kPrefetchReaders = 4;
}  // namespace

void HostExperts::enable_adapt(int reserve) {
    if (reserve <= 0 || slots_ == 0) return;
    if (reserve_ > 0) throw std::logic_error("HostExperts::enable_adapt: already enabled");
    // a VRAM swap in flight reads or writes its RAM slot: freeing that slot now would let a miss overwrite it
    if (std::find(busy_.begin(), busy_.end(), 1) != busy_.end())
        throw std::logic_error("HostExperts::enable_adapt: a VRAM swap is in flight");
    // O_DIRECT where the file system allows it: a miss is read once and kept here, so the file cache would only hold
    // a second copy (on the laptop the cache kept about 8 of its 24 free GiB useful)
    dfd_ = open((pack_.dir() + "/experts.bin").c_str(), O_RDONLY | O_DIRECT);
    // persistent readers: new threads for every admit made a 7 MB read 17% slower (laptop SSD, 0.26 -> 0.216 ms/MB)
    readers_ = std::make_unique<ThreadPool>(kReaders);
    // prefetch: 4 threads read the queued parts (one each in flight), beside admit()'s 8 readers
    for (int i = 0; i < kPrefetchReaders; ++i) pf_threads_.emplace_back([this] { prefetch_loop(); });
    free_.assign(slots_, 0);
    age_.assign(slots_, 0);
    std::map<size_t, std::vector<int>> by_capacity;
    for (int s = 0; s < slots_; ++s) by_capacity[slot_capacity(s)].push_back(s);
    for (auto& [cap, v] : by_capacity) classes_.push_back(std::move(v));
    reserve_ = reserve;
    // the slots follow the profile's rank: a capacity's last slots hold its lowest-ranked experts
    for (const auto& v : classes_)
        for (int k = 0; k < reserve && k < (int) v.size(); ++k) free_slot(v[v.size() - 1 - k]);
}

int HostExperts::free_slots() const {
    int n = 0;
    for (uint8_t f : free_) n += f;
    return n;
}

void HostExperts::lock(int slot) {
    if (!busy_.empty()) busy_.at(slot) = 1;
}

void HostExperts::unlock(int slot) {
    if (!busy_.empty()) busy_.at(slot) = 0;
}

void HostExperts::release(int layer, int expert) {
    if (!reserve_) return;
    const int s = slot_of(layer, expert);
    if (s >= 0 && !busy_[s]) free_slot(s);
}

int HostExperts::pick_free(uint64_t bytes) const {
    for (const auto& v : classes_) {   // the smallest capacity with a free slot that holds it
        if (slot_capacity(v[0]) < bytes) continue;
        for (int s : v)
            if (free_[s] && !busy_[s]) return s;
    }
    return -1;
}

int HostExperts::admit(int layer, const int32_t* ids, int n, bool* ok) {
    for (int i = 0; i < n; ++i) ok[i] = false;
    if (!reserve_ || n <= 0) return 0;
    std::vector<int> pick(n, -1), claim(n, -1);   // claim: the prefetch flight of ids[i]
    {
        std::lock_guard<std::mutex> lk(admit_mu_);
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0) continue;
            for (size_t f = 0; f < flights_.size() && claim[i] < 0; ++f) {
                const Prefetch& x = flights_[f];
                if (x.layer == layer && x.expert == ids[i] && !x.claimed && x.state != Flight::failed) claim[i] = (int) f;
            }
            if (claim[i] >= 0) flights_[claim[i]].claimed = true;
        }
        // the claimed parts go first; the unclaimed guesses of this layer and earlier ones were wrong: their parts
        // not yet read are dropped, which leaves the SSD to these reads
        std::deque<PrefetchPart> first, later;
        for (const PrefetchPart& p : pf_parts_) {
            Prefetch& f = flights_[p.flight];
            if (f.claimed) first.push_back(p);
            else if (f.layer > layer) later.push_back(p);
            else {
                f.bad = true;
                if (--f.parts_left == 0) finish_flight(f);
            }
        }
        first.insert(first.end(), later.begin(), later.end());
        pf_parts_.swap(first);
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0 || claim[i] >= 0) continue;
            pick[i] = pick_free(pack_.expert(layer, ids[i]).bytes);
            if (pick[i] >= 0) free_[pick[i]] = 0;
        }
        ++admitting_;   // the prefetch readers read only claimed parts until these reads are done
    }
    pf_cv_.notify_all();
    struct Admitting {
        HostExperts* h;
        ~Admitting() {
            {
                std::lock_guard<std::mutex> lk(h->admit_mu_);
                --h->admitting_;
            }
            h->pf_cv_.notify_all();
        }
    } admitting{this};
    auto read = [&](const std::vector<int>& slots) {
        try {
            return read_picked(layer, ids, n, slots, ok);
        } catch (...) {   // the picked slots that did not get their expert are free again; no pointer moved for them
            std::lock_guard<std::mutex> lk(admit_mu_);
            for (int i = 0; i < n; ++i)
                if (slots[i] >= 0 && !ok[i]) free_[slots[i]] = 1;
            throw;
        }
    };
    int got = read(pick);
    // then the claimed reads: take the ones read; a failed one is read here
    std::vector<int> again(n, -1);
    bool any_again = false;
    {
        std::unique_lock<std::mutex> lk(admit_mu_);
        for (int i = 0; i < n; ++i) {
            if (claim[i] < 0) continue;
            const size_t f = (size_t) claim[i];   // an index: prefetch() may grow flights_ meanwhile
            pf_cv_.wait(lk, [&] { return flights_[f].state != Flight::queued; });
            if (flights_[f].state == Flight::read) {
                flights_[f].taken = true;
                ok[i] = true;
                ++got;
                ++prefetch_used_total_;
                admitted_.push_back({flights_[f].slot, {layer, ids[i]}});
                point(layer, ids[i], slot_ptr(flights_[f].slot));
            } else if ((again[i] = pick_free(pack_.expert(layer, ids[i]).bytes)) >= 0) {
                free_[again[i]] = 0;
                any_again = true;
            }
        }
    }
    if (any_again) got += read(again);
    return got;
}

void HostExperts::read_into(int layer, const int32_t* ids, int n, const std::vector<int>& pick,
                            std::atomic<bool>* failed, ThreadPool* pool) {
    // the reads: admit_part_bytes() parts of every picked expert on the pool's threads (admit(): 8), so several
    // reads are in flight (the laptop's SSD reads 4.7-5.3 GB/s with parallel O_DIRECT readers)
    struct Part { int i; size_t at, len; };
    std::vector<Part> parts;
    const size_t kPart = detail::admit_part_bytes().load();
    for (int i = 0; i < n; ++i) {
        if (pick[i] < 0) continue;
        const ExpertSlot& x = pack_.expert(layer, ids[i]);
        if (dfd_ < 0 || x.offset % 4096 != 0) {   // copied from the mapped pack (an unaligned expert: tests)
            parts.push_back({i, 0, 0});
            continue;
        }
        for (size_t at = 0; at < x.bytes; at += kPart) parts.push_back({i, at, std::min(kPart, (size_t) x.bytes - at)});
    }
    std::atomic<size_t> next{0};
    auto read_parts = [&](size_t) {
        if (detail::admit_fault().exchange(false)) throw std::runtime_error("ds41 RAM tier: test fault in admit");
        for (size_t k; (k = next++) < parts.size();) {
            const Part& p = parts[k];
            if (!read_part(layer, ids[p.i], slot_ptr(pick[p.i]), p.at, p.len)) failed[p.i] = true;
        }
    };
    const size_t threads = std::min<size_t>(parts.size(), pool ? pool->size() : kReaders);
    if (pool) pool->run(threads, read_parts);
    else run_parallel(threads, read_parts);
}

bool HostExperts::read_part(int layer, int expert, uint8_t* dst, size_t at, size_t len) {
    const ExpertSlot& x = pack_.expert(layer, expert);
    if (len == 0) {
        std::memcpy(dst, pack_.expert_base() + x.offset, x.bytes);
        return true;
    }
    // O_DIRECT needs 4 KiB multiples: the last part reads up to the next boundary (the slot holds it)
    const size_t want = (len + 4095) / 4096 * 4096;
    size_t got = 0;
    while (got < len) {
        const ssize_t r = pread(dfd_, dst + at + got, want - got, (off_t) (x.offset + at + got));
        if (r <= 0) break;
        got += (size_t) r;
    }
    return got >= len;
}

int HostExperts::read_picked(int layer, const int32_t* ids, int n, const std::vector<int>& pick, bool* ok) {
    std::unique_ptr<std::atomic<bool>[]> failed(new std::atomic<bool>[n]);
    for (int i = 0; i < n; ++i) failed[i] = false;
    read_into(layer, ids, n, pick, failed.get(), readers_.get());
    int read = 0;
    std::lock_guard<std::mutex> lk(admit_mu_);
    for (int i = 0; i < n; ++i) {
        if (pick[i] < 0) continue;
        if (failed[i]) {   // the CPU kernel keeps reading it from the file
            free_[pick[i]] = 1;
            continue;
        }
        // recorded before the CPU kernel is pointed at it: end_step() then always knows the slot's expert
        admitted_.push_back({pick[i], {layer, ids[i]}});
        ok[i] = true;
        ++read;
        point(layer, ids[i], slot_ptr(pick[i]));
    }
    return read;
}

void HostExperts::open_prefetch() {
    std::lock_guard<std::mutex> lk(admit_mu_);
    pf_open_ = true;
}

int HostExperts::prefetch(int layer, const int32_t* ids, int n) {
    if (!reserve_ || n <= 0) return 0;
    int started = 0;
    {
        std::lock_guard<std::mutex> lk(admit_mu_);
        if (!pf_open_ || pf_threads_.empty()) return 0;
        for (int i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || in_memory(layer, e)) continue;
            bool known = false;   // on its way, or admitted this step
            for (const auto& f : flights_) known |= f.layer == layer && f.expert == e;
            for (const auto& a : admitted_) known |= a.second == std::make_pair(layer, (int) e);
            if (known) continue;
            const int s = pick_free(pack_.expert(layer, e).bytes);
            if (s < 0) continue;
            free_[s] = 0;
            const size_t k = flights_.size();
            flights_.push_back({layer, e, s, Flight::queued, false, false, 0, false});
            const ExpertSlot& x = pack_.expert(layer, e);
            const size_t kPart = detail::admit_part_bytes().load();
            if (dfd_ < 0 || x.offset % 4096 != 0) {   // copied from the mapped pack (an unaligned expert: tests)
                pf_parts_.push_back({k, 0, 0});
                flights_[k].parts_left = 1;
            } else {
                for (size_t at = 0; at < x.bytes; at += kPart) {
                    pf_parts_.push_back({k, at, std::min(kPart, (size_t) x.bytes - at)});
                    ++flights_[k].parts_left;
                }
            }
            ++started;
        }
    }
    if (started) {
        prefetched_total_ += started;
        pf_cv_.notify_all();
    }
    return started;
}

void HostExperts::prefetch_loop() {
    std::unique_lock<std::mutex> lk(admit_mu_);
    for (;;) {
        pf_cv_.wait(lk, [&] {
            return pf_stop_ || (!pf_parts_.empty() && (admitting_ == 0 || flights_[pf_parts_.front().flight].claimed));
        });
        if (pf_stop_) return;
        const PrefetchPart p = pf_parts_.front();
        pf_parts_.pop_front();
        const int layer = flights_[p.flight].layer, expert = flights_[p.flight].expert, slot = flights_[p.flight].slot;
        lk.unlock();
        const bool ok = !detail::admit_fault().exchange(false) && read_part(layer, expert, slot_ptr(slot), p.at, p.len);
        lk.lock();
        // flights_ keeps the flight: only settle_prefetches() clears it, after every read is done
        Prefetch& f = flights_[p.flight];
        f.bad |= !ok;
        if (--f.parts_left == 0) finish_flight(f);
    }
}

void HostExperts::finish_flight(Prefetch& f) {
    f.state = f.bad ? Flight::failed : Flight::read;
    if (f.bad) free_[f.slot] = 1;   // a claimed one: admit() reads the expert itself
    pf_cv_.notify_all();
}

void HostExperts::settle_prefetches(std::unique_lock<std::mutex>& lk) {
    pf_cv_.wait(lk, [&] {
        for (const auto& f : flights_)
            if (f.state == Flight::queued) return false;
        return true;
    });
    // a guess no admit() took leaves: kept, it would enter as recently used and push out experts the steps use
    // (laptop, 16 GiB tier, code: SSD experts per token 14.0 -> 18.3 when the wrong guesses stayed)
    for (const auto& f : flights_)
        if (f.state == Flight::read && !f.taken) free_[f.slot] = 1;
    flights_.clear();
}

int HostExperts::end_step(const int32_t* routes, int topk) {
    if (!reserve_) return 0;
    defer_descriptors();
    struct Flush {   // when a call below throws: publish what was collected, keep the first error
        HostExperts* h;
        bool armed = true;
        ~Flush() {
            if (armed) try { h->flush_descriptors(); } catch (...) {}
        }
    } flush{this};
    ++clock_;
    {
        std::unique_lock<std::mutex> lk(admit_mu_);
        pf_open_ = false;   // a late guess of this step starts nothing until the next step opens
        settle_prefetches(lk);
        for (const auto& [slot, le] : admitted_) {
            const auto [l, e] = le;
            holder_[slot] = {l, e};
            publish_residency(slot_[(size_t) l * n_experts_ + e], slot);
            publish_descriptor(l, e, slot);
            age_[slot] = clock_;
            ++admitted_total_;
        }
        admitted_.clear();
    }
    for (int l = 0; l < pack_.n_layers(); ++l)
        for (int k = 0; k < topk; ++k) {
            const int32_t e = routes[(size_t) l * topk + k];
            if (e < 0) continue;
            const int32_t s = slot_[(size_t) l * n_experts_ + e];
            if (s >= 0) age_[s] = clock_;
        }
    int evicted = 0;
    for (const auto& v : classes_) {
        int nfree = 0;
        for (int s : v) nfree += free_[s];
        while (nfree < reserve_) {
            int victim = -1;   // least recently used; ties: the later slot (the lower-ranked expert)
            for (int s : v)
                if (!free_[s] && !busy_[s] && (victim < 0 || age_[s] <= age_[victim])) victim = s;
            if (victim < 0) break;
            free_slot(victim);
            ++nfree;
            ++evicted;
        }
    }
    evicted_total_ += evicted;
    flush.armed = false;
    flush_descriptors();
    return evicted;
}

}  // namespace strata::ds41
