// src/ds41/vram_experts.cu - see include/strata/ds41/vram_experts.hpp.
#include "strata/ds41/vram_experts.hpp"
#include "strata/ds41/bounded_upload.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/residency.hpp"
#include "strata/ds41/host_experts.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>

namespace strata::ds41 {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 vram experts: ") + what + ": " + cudaGetErrorString(e));
}

/// After copies from the mapped pack: every page of experts.bin leaves this process (the copied experts' pages and the
/// neighbours the page faults mapped with them), so the OS reclaims them first; a later read faults them in again.
/// Mapped, the 11.8 GiB of a laptop's VRAM fill made the kernel page the RAM tier (unlocked while it fills) out to
/// zram instead: fill 20 -> 40 s in half the starts (60 GB laptop, 2 rounds x 3 prompts: mean 31.9 -> 20.9 s).
void unmap_pack_pages(const Pack& pack) {
    uint64_t end = 0;
    for (int l = 0; l < pack.n_layers(); ++l)
        for (int e = 0; e < pack.n_experts(); ++e) end = std::max(end, pack.expert(l, e).offset + pack.expert(l, e).bytes);
    if (pack.expert_base() && end) madvise((void*) pack.expert_base(), end, MADV_DONTNEED);
}

}  // namespace

std::vector<std::pair<int, int>> read_expert_profile(const std::string& path, int n_layers, int n_experts) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open expert profile " + path);
    const std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    uint32_t h[5];
    if (blob.size() < 24 || blob.compare(0, 4, "STRP") != 0) throw std::runtime_error(path + ": not a Strata profile");
    std::memcpy(h, blob.data() + 4, sizeof h);
    if (h[0] != 1 || (int) h[1] != n_layers || (int) h[2] != n_experts)
        throw std::runtime_error(path + ": profile version " + std::to_string(h[0]) + ", " + std::to_string(h[1]) + "x" +
                                 std::to_string(h[2]) + ", not 1, " + std::to_string(n_layers) + "x" +
                                 std::to_string(n_experts));
    const uint32_t n = h[4];
    if (blob.size() < 24 + (size_t) n * 4) throw std::runtime_error(path + ": profile truncated");
    std::vector<std::pair<int, int>> ranked;
    ranked.reserve(n);
    std::vector<uint8_t> seen((size_t) n_layers * n_experts, 0);
    for (uint32_t i = 0; i < n; ++i) {
        uint16_t p[2];
        std::memcpy(p, blob.data() + 24 + (size_t) i * 4, 4);
        if (p[0] >= n_layers || p[1] >= n_experts || seen[(size_t) p[0] * n_experts + p[1]]++)
            throw std::runtime_error(path + ": bad or repeated pair at rank " + std::to_string(i));
        ranked.emplace_back(p[0], p[1]);
    }
    return ranked;
}

std::vector<ExpertSwap> plan_expert_swaps(const std::vector<float>& usage, const std::vector<int32_t>& res,
                                          int n_layers, int n_experts, int max_swaps,
                                          const std::function<bool(int, int, int)>& fits) {
    std::vector<ExpertSwap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    std::vector<uint8_t> taken;
    for (int l = 0; l < n_layers; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage.data() + (size_t) l * n_experts;
        const int32_t* r = res.data() + (size_t) l * n_experts;
        for (int32_t e = 0; e < n_experts; ++e) {
            if (r[e] < 0) {
                if (u[e] >= 2.0f) cand.emplace_back(u[e], e);
            } else {
                vict.emplace_back(u[e], e);
            }
        }
        if (cand.empty() || vict.empty()) continue;
        // ties: lower expert first, so the plan does not depend on the sort's stability
        std::sort(cand.begin(), cand.end(),
                  [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        std::sort(vict.begin(), vict.end(),
                  [](auto& a, auto& b) { return a.first < b.first || (a.first == b.first && a.second < b.second); });
        taken.assign(vict.size(), 0);
        for (const auto& c : cand) {
            // the least-used victim not taken yet whose slot holds the candidate
            size_t j = 0;
            while (j < vict.size() && (taken[j] || (fits && !fits(l, c.second, vict[j].second)))) ++j;
            if (j == vict.size() || c.first < vict[j].first + 1.5f) continue;
            taken[j] = 1;
            swaps.push_back({c.first - vict[j].first, (int32_t) l, c.second, vict[j].second});
        }
    }
    std::stable_sort(swaps.begin(), swaps.end(), [](const ExpertSwap& a, const ExpertSwap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) std::max(max_swaps, 0));
    return swaps;
}

VramExperts::VramExperts(const Pack& pack, const std::string& profile_path, int64_t n_slots, size_t reserve_bytes,
                         Adapt adapt, int first_layer, int end_layer, bool require_all)
    : pack_(pack), adapt_(adapt) {
    try {
        const int L = pack.n_layers(), E = pack.n_experts();
        if (end_layer < 0) end_layer = L;
        if (first_layer < 0 || end_layer > L || first_layer >= end_layer)
            throw std::invalid_argument("ds41 vram: invalid stage layer range");
        if (require_all && (adapt.every != 0 || n_slots != -1))
            throw std::invalid_argument("ds41 vram: resident stage requires all slots and static residency");
        ck(cudaGetDevice(&device_), "cudaGetDevice");
        res_host_.assign((size_t) L * E, -1);
        usage_.assign((size_t) L * E, 0.0f);
        ck(cudaMalloc(&res_dev_, res_host_.size() * sizeof(int32_t)), "residency table");
        ck(cudaMalloc(&ws_, kWorkspaceBytes), "K10 workspace");
        ck(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "copy stream");
        auto slot_size = [&](int l, int e) { return (size_t) (pack.expert(l, e).bytes + 255) / 256 * 256; };
        if (require_all || (n_slots != 0 && !profile_path.empty())) {
            std::vector<std::pair<int, int>> ranked;
            if (require_all) {
                for (int l = first_layer; l < end_layer; ++l)
                    for (int e = 0; e < E; ++e) ranked.emplace_back(l, e);
            } else {
                for (auto pair : read_expert_profile(profile_path, L, E))
                    if (pair.first >= first_layer && pair.first < end_layer) ranked.push_back(pair);
            }
            if (n_slots < 0) {   // by bytes, in rank order, up to the first that does not fit (upstream)
                size_t free_b = 0, total_b = 0;
                ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
                const size_t descs = (size_t) L * E * sizeof(kernels::Exl3Expert);
                const size_t room = free_b > reserve_bytes && free_b - reserve_bytes > descs
                                        ? free_b - reserve_bytes - descs : 0;
                size_t used = 0;
                n_slots = 0;
                for (const auto& [l, e] : ranked) {
                    if (slot_size(l, e) > room - used) {
                        if (require_all) throw std::runtime_error("ds41 vram: all stage experts do not fit with the VRAM reserve; refusing CPU fallback");
                        break;
                    }
                    used += slot_size(l, e);
                    ++n_slots;
                }
            }
            slots_ = (int) std::min<int64_t>(n_slots, (int64_t) ranked.size());
            for (int s = 0; s < slots_; ++s) {
                off_.push_back(off_.back() + slot_size(ranked[s].first, ranked[s].second));
                max_slot_bytes_ = std::max(max_slot_bytes_, off_.back() - off_[s]);
            }
            if (slots_ > 0) {
                ck(cudaMalloc(&arena_, off_.back()), "expert slots");
                desc_host_.resize(slots_);
                const uint8_t* base = pack.expert_base();
                std::unique_ptr<BoundedUpload> upload;
                if (require_all) upload = std::make_unique<BoundedUpload>(pack.dir() + "/experts.bin");
                for (int s = 0; s < slots_; ++s) {
                    const auto [l, e] = ranked[s];
                    const ExpertSlot& x = pack.expert(l, e);
                    for (int p = 0; p < 3; ++p) if (x.codebook[p] == 1) {
                        if (!upload || upload->scalar_u32(x.offset + x.comp_off[p * 4 + 3]) != 0xCBAC1FEDu)
                            throw std::runtime_error("ds41 vram: unknown MCG marker or non-resident MCG loading");
                    }
                    if (upload) upload->copy(arena_ + off_[s], x.offset, x.bytes);
                    else ck(cudaMemcpy(arena_ + off_[s], base + x.offset, x.bytes, cudaMemcpyHostToDevice), "expert copy");
                    desc_host_[s] = describe(l, e, s);
                    publish_residency(res_host_[(size_t) l * E + e], s);
                }
                if (!upload) unmap_pack_pages(pack);
                ck(cudaMalloc(&experts_dev_, desc_host_.size() * sizeof(desc_host_[0])), "expert descriptors");
                ck(cudaMemcpy(experts_dev_, desc_host_.data(), desc_host_.size() * sizeof(desc_host_[0]),
                              cudaMemcpyHostToDevice),
                   "expert descriptors");
            }
        }
        upload_res();
    } catch (...) {
        cleanup();
        throw;
    }
}

VramExperts::~VramExperts() { cleanup(); }

void VramExperts::cleanup() noexcept {
    if (copier_.joinable()) copier_.join();
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    if (staging_) cudaFreeHost(staging_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    cudaFree(arena_);
    cudaFree(res_dev_);
    cudaFree(experts_dev_);
    cudaFree(ws_);
}

namespace {
__global__ void scatter_descriptors_k(kernels::Exl3Expert* table, const int32_t* idx, const kernels::Exl3Expert* desc,
                                      int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) table[idx[i]] = desc[i];
}
}  // namespace

void VramExperts::scatter_descriptors(kernels::Exl3Expert* table, const int32_t* idx, const kernels::Exl3Expert* desc,
                                      int n) {
    if (n <= 0) return;
    int32_t* d_idx = nullptr;
    kernels::Exl3Expert* d_desc = nullptr;
    ck(cudaHostGetDevicePointer((void**) &d_idx, (void*) idx, 0), "descriptor index alias");
    ck(cudaHostGetDevicePointer((void**) &d_desc, (void*) desc, 0), "descriptor alias");
    scatter_descriptors_k<<<(n + 127) / 128, 128>>>(table, d_idx, d_desc, n);
    ck(cudaGetLastError(), "descriptor scatter");
}

kernels::Exl3Expert VramExperts::describe(int layer, int expert, int slot) const {
    return describe_at(pack_, layer, expert, arena_ + off_[slot]);
}

kernels::Exl3Expert VramExperts::describe_at(const Pack& pack, int layer, int expert, const uint8_t* dst) {
    const ExpertSlot& x = pack.expert(layer, expert);
    auto proj = [&](int c0, int k, int n) {
        kernels::Exl3Proj p;
        p.trellis = (const uint16_t*) (dst + x.comp_off[c0]);
        p.suh = (const __half*) (dst + x.comp_off[c0 + 1]);
        p.svh = (const __half*) (dst + x.comp_off[c0 + 2]);
        p.k = k;
        p.n = n;
        p.tile_w = (int) (x.comp_bytes[c0] / ((uint64_t) (k / 16) * (n / 16) * 2));
        p.codebook = x.codebook[c0 / 4];
        if (p.codebook == 1 && (x.comp_bytes[c0] != uint64_t(k / 16) * (n / 16) * 64 ||
                                x.comp_bytes[c0 + 1] != uint64_t(k) * 2 || x.comp_bytes[c0 + 2] != uint64_t(n) * 2))
            throw std::runtime_error("ds41 vram: MCG metadata does not describe a K2 projection");
        return p;
    };
    kernels::Exl3Expert d;
    d.w1 = proj(0, kDim, kMoeInter);
    d.w3 = proj(4, kDim, kMoeInter);
    d.w2 = proj(8, kMoeInter, kDim);
    d.native_qdq = !pack.has_exl3_dense();
    return d;
}

void VramExperts::upload_res() {
    ck(cudaMemcpy(res_dev_, res_host_.data(), res_host_.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
       "residency table");
    // Publish before a nonblocking decode stream or the background copier can use the table.
    // This function runs only during initialization or between steps.
    ck(cudaStreamSynchronize(nullptr), "residency publication");
}

void VramExperts::count(const int32_t* routes, int topk) {
    const int L = pack_.n_layers(), E = pack_.n_experts();
    for (int l = 0; l < L; ++l)
        for (int j = 0; j < topk; ++j) {
            const int32_t e = routes[l * topk + j];
            if (e >= 0 && e < E) usage_[(size_t) l * E + e] += 1.0f;
        }
}

void VramExperts::set_host(HostExperts* host) {
    host_ = host;
    if (host_ && host_->slots() > 0 && !staging_) {
        // the swap buffer: the evicted experts on their way from VRAM to RAM (a batch's evicted bytes fit in it)
        staging_bytes_ = std::min<size_t>((size_t) std::max(adapt_.max_swaps, 1) * max_slot_bytes_, 1ull << 30);
        staging_bytes_ = std::max(staging_bytes_, max_slot_bytes_);
        ck(cudaMallocHost((void**) &staging_, staging_bytes_), "swap staging");
    }
}

// A swap with a RAM tier runs in three copies, each started between two steps and switched in only between steps,
// so the CPU never reads a swapped expert from the file (on a 128 GB PC those page faults went to the SSD and the
// GPU waited for them):
//   phase 1  `out`: its VRAM slot -> the swap buffer (both stay valid; the GPU keeps computing `out`)
//   phase 2  `out` leaves VRAM, the CPU reads it from the swap buffer; `in`: its RAM slot -> the VRAM slot (the CPU
//            keeps reading `in` from its RAM slot)
//   phase 3  `in` enters VRAM (its RAM slot is free now); `out`: the swap buffer -> that RAM slot
//   done     the CPU reads `out` from its RAM slot
// Without a RAM tier (`in` from the file), phase 1 copies nothing, `out` goes to the file in phase 2, and `in` is
// copied from the mapped pack.
void VramExperts::copy_worker(std::vector<Pending> work, int phase) {
    bool ok = cudaSetDevice(device_) == cudaSuccess;
    const uint8_t* base = pack_.expert_base();
    for (size_t i = 0; ok && i < work.size(); ++i) {
        const Pending& w = work[i];
        uint8_t* vslot = arena_ + off_[w.vram_slot];
        const ExpertSlot& xin = pack_.expert(w.layer, w.in);
        const ExpertSlot& xout = pack_.expert(w.layer, w.out);
        if (phase == 1 && w.ram_slot >= 0) {
            ok = cudaMemcpyAsync(staging_ + w.staged, vslot, xout.bytes, cudaMemcpyDeviceToHost, copy_stream_) ==
                 cudaSuccess;
        } else if (phase == 2) {
            const uint8_t* src = w.ram_slot >= 0 ? host_->slot_ptr(w.ram_slot) : base + xin.offset;
            ok = cudaMemcpyAsync(vslot, src, xin.bytes, cudaMemcpyHostToDevice, copy_stream_) == cudaSuccess &&
                 cudaMemcpyAsync(experts_dev_ + w.vram_slot, &desc_host_[w.vram_slot], sizeof(kernels::Exl3Expert),
                                 cudaMemcpyHostToDevice, copy_stream_) == cudaSuccess;
        } else if (phase == 3 && w.ram_slot >= 0) {
            std::memcpy(host_->slot_ptr(w.ram_slot), staging_ + w.staged, xout.bytes);
        }
    }
    ok = ok && cudaStreamSynchronize(copy_stream_) == cudaSuccess;
    copy_error_ = !ok;
    copies_done_.store(true, std::memory_order_release);
}

void VramExperts::start_phase(int phase) {
    phase_ = phase;
    copies_done_.store(false, std::memory_order_relaxed);
    copier_ = std::thread(&VramExperts::copy_worker, this, pending_, phase);
}

int VramExperts::commit_pending(bool wait) {
    int committed = 0;
    while (phase_ != 0) {
        if (!wait && !copies_done_.load(std::memory_order_acquire)) return committed;   // the next step checks again
        copier_.join();
        if (copy_error_) throw std::runtime_error("ds41 vram experts: an adaptive expert copy failed");
        const int E = pack_.n_experts();
        // the RAM descriptors change together below, published before the phase's next copy; on a throw the
        // guard publishes what was collected
        if (host_) host_->defer_descriptors();
        struct Batch {
            HostExperts* h;
            ~Batch() {
                if (h) try { h->flush_descriptors(); } catch (...) {}
            }
        } batch{host_};
        if (phase_ == 1) {
            // `out` leaves VRAM: the CPU reads it from the swap buffer (or the file); the slot now describes `in`
            for (const Pending& w : pending_) {
                publish_residency(res_host_[(size_t) w.layer * E + w.out], -1);
                desc_host_[w.vram_slot] = describe(w.layer, w.in, w.vram_slot);
                if (w.ram_slot >= 0) host_->point_to(w.layer, w.out, staging_ + w.staged);
                else if (host_) host_->point_to_file(w.layer, w.out);
            }
            if (host_) host_->flush_descriptors();
            upload_res();   // before the copy into the slots: no step reads a slot that is being overwritten
            start_phase(2);
        } else if (phase_ == 2) {
            // `in` enters VRAM; its RAM slot is free (a VRAM hit is never read from RAM)
            for (const Pending& w : pending_) {
                publish_residency(res_host_[(size_t) w.layer * E + w.in], w.vram_slot);
                if (w.ram_slot >= 0) host_->point_to_file(w.layer, w.in);   // revokes its RAM descriptor
                else if (host_) host_->release(w.layer, w.in);   // the adaptive RAM tier read it in meanwhile
            }
            if (host_) host_->flush_descriptors();
            upload_res();
            committed += (int) pending_.size();
            swaps_total_ += (int) pending_.size();
            start_phase(3);
        } else {
            for (const Pending& w : pending_)
                if (w.ram_slot >= 0) {
                    host_->assign(w.ram_slot, w.layer, w.out);   // the CPU reads `out` from RAM
                    host_->unlock(w.ram_slot);
                }
            if (host_) host_->flush_descriptors();
            pending_.clear();
            phase_ = 0;
        }
        if (!wait) return committed;   // one phase per call: each switch happens between two steps
    }
    return committed;
}

int VramExperts::between_steps() {
    if (slots_ == 0 || adapt_.every <= 0) return 0;
    if (lent_) throw std::logic_error("ds41 vram experts: a step while slots are lent to prefill");
    const int L = pack_.n_layers(), E = pack_.n_experts();
    const int committed = commit_pending(false);
    if (phase_ != 0) return committed;   // a batch is still on its way
    if (++calls_ % adapt_.every != 0) return committed;
    // `in` must fit the VRAM slot of `out`; with a RAM tier, `out` then takes the RAM slot of `in`, and must fit it
    auto fits = [&](int l, int in, int out) {
        const size_t vcap = off_[res_host_[(size_t) l * E + out] + 1] - off_[res_host_[(size_t) l * E + out]];
        if (pack_.expert(l, in).bytes > vcap) return false;
        const int32_t ram = host_ ? host_->slot_of(l, in) : -1;
        // With the adaptive RAM tier, `in` comes from its RAM slot only. An expert the tier did not admit (a skipped SSD
        // expert, say) was copied from the mapped pack: page faults to the SSD during decode, 1.5 GiB of file pages in
        // 255 steps of a 128K prompt. Laptop, 2 rounds x 3 prompts: no-stall decode 33.7 -> 32.9 ms/token.
        if (ram < 0 && host_ && host_->reserve() > 0) return false;
        return ram < 0 || pack_.expert(l, out).bytes <= host_->slot_capacity(ram);
    };
    const auto swaps = plan_expert_swaps(usage_, res_host_, L, E, adapt_.max_swaps, fits);
    for (float& v : usage_) v *= adapt_.decay;
    size_t staged = 0;
    for (const ExpertSwap& s : swaps) {   // largest gain first: the batch ends when the swap buffer is full
        const int32_t ram = host_ ? host_->slot_of(s.layer, s.in) : -1;
        const size_t need = ram >= 0 ? (pack_.expert(s.layer, s.out).bytes + 255) / 256 * 256 : 0;
        if (staged + need > staging_bytes_) break;
        pending_.push_back(Pending{s.layer, s.in, s.out, res_host_[(size_t) s.layer * E + s.out], ram, staged});
        if (ram >= 0) host_->lock(ram);   // the adaptive RAM tier keeps the slot until `out` is written there
        staged += need;
    }
    if (pending_.empty()) return committed;
    start_phase(1);
    return committed;
}

uint8_t* VramExperts::lend(int n) {
    if (lent_) throw std::logic_error("ds41 vram experts: slots already lent");
    if (n < 0 || n > slots_) throw std::invalid_argument("ds41 vram experts: cannot lend that many slots");
    commit_pending(true);
    const int E = pack_.n_experts();
    lent_owner_.assign(n, {-1, -1});
    for (size_t i = 0; i < res_host_.size(); ++i) {
        const int32_t s = res_host_[i];
        if (s >= slots_ - n) {
            lent_owner_[s - (slots_ - n)] = {(int) (i / E), (int) (i % E)};
            publish_residency(res_host_[i], -1);
        }
    }
    lent_ = n;
    upload_res();
    return arena_ + off_[slots_ - n];
}

uint8_t* VramExperts::lend_bytes(size_t bytes) {
    if (bytes > tail_bytes(slots_)) throw std::invalid_argument("ds41 vram experts: the slots hold fewer bytes");
    int n = 0;
    while (tail_bytes(n) < bytes) ++n;
    return lend(n);
}

void VramExperts::restore() {
    if (!lent_) return;
    const int E = pack_.n_experts();
    const uint8_t* base = pack_.expert_base();
    for (int i = 0; i < lent_; ++i) {
        const auto [l, e] = lent_owner_[i];
        if (l < 0) continue;
        const int s = slots_ - lent_ + i;
        const ExpertSlot& x = pack_.expert(l, e);
        ck(cudaMemcpy(arena_ + off_[s], base + x.offset, x.bytes, cudaMemcpyHostToDevice), "restore lent slot");
        // The descriptors still describe (l, e) at slot s.
        publish_residency(res_host_[(size_t) l * E + e], s);
    }
    unmap_pack_pages(pack_);
    lent_ = 0;
    lent_owner_.clear();
    upload_res();
}

}  // namespace strata::ds41
