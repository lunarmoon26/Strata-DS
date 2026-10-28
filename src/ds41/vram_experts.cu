// src/ds41/vram_experts.cu - see include/strata/ds41/vram_experts.hpp.
#include "strata/ds41/vram_experts.hpp"
#include "strata/ds41/bounded_upload.hpp"
#include "strata/ds41/tensor_shards.hpp"
#include "strata/ds41/ops.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/residency.hpp"
#include "strata/ds41/host_experts.hpp"
#include "strata/ds41/verify.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
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

size_t experimental_decode_cache_bytes(const char* value) {
    if (!value) return 0;
    unsigned mib = 0;
    const char* end = value + std::strlen(value);
    const auto result = std::from_chars(value, end, mib);
    if (result.ec != std::errc{} || result.ptr != end || mib > 8192)
        throw std::invalid_argument("DS41_EXPERIMENTAL_DECODE_CACHE_MIB must be an integer in 0..8192");
    return size_t(mib) << 20;
}

size_t experimental_ram_cache_bytes(const char* value) {
    if (!value) return 0;
    unsigned mib = 0;
    const char* end = value + std::strlen(value);
    const auto result = std::from_chars(value, end, mib);
    if (result.ec != std::errc{} || result.ptr != end || mib > 16384 || mib % 2)
        throw std::invalid_argument("DS41_EXPERIMENTAL_RAM_CACHE_MIB must be an even integer in 0..16384");
    return size_t(mib) << 20;
}

size_t experimental_ram_cache_stage_bytes(const char* split, size_t total_bytes, int rank) {
    if(rank<0 || rank>1 || total_bytes>(16384ull<<20)) throw std::invalid_argument("invalid RAM cache total/rank");
    if(!split) return total_bytes/2;
    const char* comma=std::strchr(split,',');const char* end=split+std::strlen(split);
    unsigned mib[2]={};
    if(!comma) throw std::invalid_argument("DS41_EXPERIMENTAL_RAM_CACHE_SPLIT requires A,B MiB");
    const auto a=std::from_chars(split,comma,mib[0]),b=std::from_chars(comma+1,end,mib[1]);
    if(a.ec!=std::errc{} || a.ptr!=comma || b.ec!=std::errc{} || b.ptr!=end || mib[0]>16384 || mib[1]>16384 ||
       ((size_t(mib[0])+mib[1])<<20)!=total_bytes)
        throw std::invalid_argument("DS41_EXPERIMENTAL_RAM_CACHE_SPLIT must sum to the total RAM cache MiB budget");
    return size_t(mib[rank])<<20;
}

int experimental_resident_slot_limit(const char* value, int rank) {
    if (rank < 0 || rank > 1) throw std::invalid_argument("invalid experimental logical rank");
    if (!value) return -1;
    const char* comma = std::strchr(value, ',');
    const char* end = value + std::strlen(value);
    int counts[2] = {};
    if (!comma) throw std::invalid_argument("DS41_EXPERIMENTAL_RESIDENT_SLOTS requires two positive counts");
    const auto a = std::from_chars(value, comma, counts[0]);
    const auto b = std::from_chars(comma + 1, end, counts[1]);
    if (a.ec != std::errc{} || a.ptr != comma || b.ec != std::errc{} || b.ptr != end ||
        counts[0] < 1 || counts[1] < 1 || counts[0] > 15360 || counts[1] > 15360)
        throw std::invalid_argument("DS41_EXPERIMENTAL_RESIDENT_SLOTS requires two counts in 1..15360");
    return counts[rank];
}

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
                          Adapt adapt, int first_layer, int end_layer, bool require_all, int tensor_rank, bool bounded_misses,
                          size_t decode_cache_bytes, size_t ram_cache_bytes, bool async_cached_routes)
    : pack_(pack), adapt_(adapt), tensor_rank_(tensor_rank) {
    try {
        const int L = pack.n_layers(), E = pack.n_experts();
        if (bounded_misses && (!require_all || !pack.experimental_29()))
            throw std::invalid_argument("bounded expert misses require the experimental 2.9 layer profile");
        if (decode_cache_bytes && (!bounded_misses || decode_cache_bytes > (8192ull << 20)))
            throw std::invalid_argument("decode expert cache requires bounded experimental 2.9 and at most 8192 MiB");
        if (ram_cache_bytes && (!bounded_misses || ram_cache_bytes > (16384ull << 20)))
            throw std::invalid_argument("RAM expert cache requires bounded experimental 2.9 and at most 16384 MiB/stage");
        if (async_cached_routes && (!bounded_misses || tensor_rank >= 0))
            throw std::invalid_argument("async cached routes require bounded experimental layer split");
        const char* exact_slots = std::getenv("DS41_EXPERIMENTAL_RESIDENT_SLOTS");
        if (exact_slots && !bounded_misses)
            throw std::invalid_argument("DS41_EXPERIMENTAL_RESIDENT_SLOTS requires experimental bounded experts");
        const int slot_limit = experimental_resident_slot_limit(exact_slots, tensor_rank >= 0 ? tensor_rank : first_layer == 0 ? 0 : 1);
        if (end_layer < 0) end_layer = L;
        first_layer_=first_layer;end_layer_=end_layer;
        if (first_layer < 0 || end_layer > L || first_layer >= end_layer)
            throw std::invalid_argument("ds41 vram: invalid stage layer range");
        if (require_all && (adapt.every != 0 || n_slots != -1))
            throw std::invalid_argument("ds41 vram: resident stage requires all slots and static residency");
        if (tensor_rank != -1 && (tensor_rank < 0 || tensor_rank > 1 || !require_all || !pack.has_exl3_dense() || pack.has_mcg()))
            throw std::invalid_argument("TP2 expert residency requires rank 0/1, all slots and full mul1 EXL3");
        ck(cudaGetDevice(&device_), "cudaGetDevice");
        if (async_cached_routes)
            ck(cudaHostAlloc((void**)&async_routes_, kVerifyMaxResidentTokens*kTopK*sizeof(int32_t), cudaHostAllocPortable),
               "async cached route buffer");
        res_host_.assign((size_t) L * E, -1);
        if(const char* mode=std::getenv("DS41_EXPERIMENTAL_CACHE_FREQUENCY")) {
            if(std::strcmp(mode,"0") && std::strcmp(mode,"1"))
                throw std::invalid_argument("DS41_EXPERIMENTAL_CACHE_FREQUENCY must be 0 or 1");
            frequency_cache_=!std::strcmp(mode,"1");
        }
        if(frequency_cache_ && (!bounded_misses || tensor_rank>=0))
            throw std::invalid_argument("frequency cache requires bounded layer split");
        if(frequency_cache_) route_frequency_.assign((size_t)L*E,0);
        usage_.assign((size_t) L * E, 0.0f);
        ck(cudaMalloc(&res_dev_, res_host_.size() * sizeof(int32_t)), "residency table");
        ck(cudaMalloc(&ws_, kWorkspaceBytes), "K10 workspace");
        ck(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "copy stream");
        auto slot_size = [&](int l, int e) {
            return tensor_rank >= 0 ? tensor_expert_bytes(pack.expert(l,e)) : (size_t)(pack.expert(l,e).bytes+255)/256*256;
        };
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
                        if (require_all && !bounded_misses) throw std::runtime_error("ds41 vram: all stage experts do not fit with the VRAM reserve; refusing CPU fallback");
                        break;
                    }
                    used += slot_size(l, e);
                    ++n_slots;
                }
            }
            if (slot_limit >= 0) {
                if (slot_limit > n_slots || size_t(slot_limit) > ranked.size())
                    throw std::runtime_error("fixed experimental resident set does not fit; refusing to silently shrink it");
                n_slots = slot_limit;
            }
            slots_ = (int) std::min<int64_t>(n_slots, (int64_t) ranked.size());
            for (int s = 0; s < slots_; ++s) {
                off_.push_back(off_.back() + slot_size(ranked[s].first, ranked[s].second));
                max_slot_bytes_ = std::max(max_slot_bytes_, off_.back() - off_[s]);
            }
            if (slots_ > 0) {
                ck(cudaMalloc(&arena_, off_.back()), "expert slots");
                desc_host_.resize(slots_ + (bounded_misses ? kTopK : 0));
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
                    if (tensor_rank>=0) {
                        desc_host_[s]=upload_tensor_expert(*upload,x,arena_+off_[s],tensor_rank);
                        publish_residency(res_host_[size_t(l)*E+e],s);
                        continue;
                    }
                    if (upload) upload->copy(arena_ + off_[s], x.offset, x.bytes);
                    else ck(cudaMemcpy(arena_ + off_[s], base + x.offset, x.bytes, cudaMemcpyHostToDevice), "expert copy");
                    desc_host_[s] = describe(l, e, s);
                    publish_residency(res_host_[(size_t) l * E + e], s);
                }
                if (!upload) unmap_pack_pages(pack);
            }
        }
        if (bounded_misses) {
            for (int l = first_layer; l < end_layer; ++l)
                for (int e = 0; e < E; ++e) max_slot_bytes_ = std::max(max_slot_bytes_, slot_size(l, e));
            if (!slots_) {
                desc_host_.resize(kTopK);
            }
            ck(cudaMalloc(&miss_arena_, kTopK * max_slot_bytes_), "bounded selected-expert staging");
            miss_upload_ = std::make_unique<BoundedUpload>(pack.dir() + "/experts.bin");
            if (ram_cache_bytes) {
                size_t cold_count=0;
                for (int l = first_layer; l < end_layer; ++l)
                    for (int e = 0; e < E; ++e) {
                        ram_slot_bytes_ = std::max(ram_slot_bytes_, size_t((pack.expert(l, e).bytes + 4095) / 4096 * 4096));
                        cold_count += res_host_[size_t(l)*E+e]<0;
                    }
                const size_t count = std::min(ram_cache_bytes / ram_slot_bytes_, cold_count);
                if (!count && cold_count) throw std::invalid_argument("RAM cache budget cannot hold one complete expert");
                if (count) {
                ck(cudaHostAlloc(&ram_arena_, count * ram_slot_bytes_, cudaHostAllocPortable), "bounded pinned RAM expert cache");
                ram_cache_.resize(count);
                ram_lookup_.assign(size_t(L) * E, -1);
                std::fprintf(stderr, "ds41 experimental RAM cache: %zu slots, %.3f GiB/stage, GPU computation only\n",
                             count, count * ram_slot_bytes_ / 1073741824.);
                }
            }
            if (decode_cache_bytes) {
                const size_t count = std::min(decode_cache_bytes / max_slot_bytes_, size_t(L) * E);
                if (!count) throw std::invalid_argument("decode cache budget cannot hold one complete expert");
                const size_t bytes = count * max_slot_bytes_;
                size_t free_bytes = 0, total_bytes = 0;
                ck(cudaMemGetInfo(&free_bytes, &total_bytes), "decode cache free memory");
                if (bytes > free_bytes || free_bytes - bytes < (512ull << 20))
                    throw std::runtime_error("decode cache budget leaves less than 512 MiB free; reduce it");
                ck(cudaMalloc(&miss_cache_arena_, bytes), "bounded decode expert cache");
                miss_cache_.resize(count);
                miss_cache_lookup_.assign(size_t(L) * E, -1);
                desc_host_.resize(slots_ + kTopK + count);
                std::fprintf(stderr, "ds41 experimental decode cache: %zu slots, %.3f GiB/device, immutable weights only\n",
                             count, bytes / 1073741824.);
            }
            std::fprintf(stderr, "ds41 experimental expert stream: %d/%zu resident, %.3f GiB selected-expert staging; no CPU experts\n",
                         slots_, size_t(end_layer - first_layer) * E, kTopK * max_slot_bytes_ / 1073741824.);
        }
        if (!desc_host_.empty()) {
            ck(cudaMalloc(&experts_dev_, desc_host_.size() * sizeof(desc_host_[0])), "expert descriptors");
            ck(cudaMemcpy(experts_dev_, desc_host_.data(), desc_host_.size() * sizeof(desc_host_[0]),
                          cudaMemcpyHostToDevice), "expert descriptors");
        }
        full_layers_.assign(L, 0);
        for (int l = first_layer; l < end_layer; ++l)
            full_layers_[l] = std::all_of(res_host_.begin() + size_t(l)*E, res_host_.begin() + size_t(l+1)*E,
                                           [](int32_t slot) { return slot >= 0; });
        upload_res();
    } catch (...) {
        cleanup();
        throw;
    }
}

VramExperts::~VramExperts() { cleanup(); }

void VramExperts::cleanup() noexcept {
    cancel_cached_rows();
    if (copier_.joinable()) copier_.join();
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    if (staging_) cudaFreeHost(staging_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    cudaFree(arena_);
    cudaFree(res_dev_);
    cudaFree(experts_dev_);
    cudaFree(ws_);
    cudaFree(miss_arena_);
    cudaFree(miss_cache_arena_);
    if (!miss_cache_.empty()) std::fprintf(stderr,
        "ds41 experimental decode cache totals: hits=%lld uploads=%lld evictions=%lld slots=%zu stage=%d..%d\n",
        (long long)cache_hits_, (long long)streamed_decode_, (long long)cache_evictions_, miss_cache_.size(),first_layer_,end_layer_-1);
    miss_upload_.reset();
    if (!ram_cache_.empty()) std::fprintf(stderr,
        "ds41 experimental RAM cache totals: hits=%lld fills=%lld read_bytes=%llu slots=%zu stage=%d..%d\n",
        (long long)ram_hits_, (long long)ram_fills_, (unsigned long long)ram_read_bytes_, ram_cache_.size(),first_layer_,end_layer_-1);
    if (ram_arena_) cudaFreeHost(ram_arena_);
    // If CUDA cannot drain the copy, retain its pinned target until process
    // exit rather than freeing memory that might still be in flight.
    if (async_routes_ && !async_routes_failed_) cudaFreeHost(async_routes_);
}

const uint8_t* VramExperts::ram_payload(int layer, int expert) {
    if (ram_cache_.empty()) return nullptr;
    const int owner = layer * pack_.n_experts() + expert;
    int slot = ram_lookup_[owner];
    if (slot >= 0) {
        ram_cache_[slot].age = ++ram_clock_;
        ++ram_hits_;
        return ram_arena_ + size_t(slot) * ram_slot_bytes_;
    }
    slot = 0;
    for (size_t j = 1; j < ram_cache_.size(); ++j)
        if (ram_cache_[j].age < ram_cache_[slot].age) slot = int(j);
    auto& entry = ram_cache_[slot];
    if (entry.owner >= 0) ram_lookup_[entry.owner] = -1;
    entry = CacheEntry{};
    const auto& payload = pack_.expert(layer, expert);
    auto* dst = ram_arena_ + size_t(slot) * ram_slot_bytes_;
    // A failed or partial read leaves the victim invalid, never a RAM hit.
    miss_upload_->read_host(dst, payload.offset, payload.bytes);
    entry = CacheEntry{owner, ++ram_clock_};
    ram_lookup_[owner] = slot;
    ++ram_fills_;
    ram_read_bytes_ += payload.bytes;
    return dst;
}

namespace {
struct CachedSelection { int32_t slots[kVerifyMaxResidentTokens*kTopK]; int n; };
__global__ void publish_cached_selection(CachedSelection selection,int32_t* selected) {
    const int i=threadIdx.x;if(i<selection.n) selected[i]=selection.slots[i];
}
}

int VramExperts::prefill_gpu_slot(int layer,int expert) const {
    if(layer<first_layer_ || layer>=end_layer_ || expert<0 || expert>=pack_.n_experts())
        throw std::invalid_argument("prefill cache lookup outside owned experts");
    const int owner=layer*pack_.n_experts()+expert;
    if(res_host_[owner]>=0) return res_host_[owner];
    const int cached=miss_cache_.empty()?-1:miss_cache_lookup_[owner];
    return cached<0?-1:slots_+kTopK+cached;
}

const uint8_t* VramExperts::prefill_ram_source(int layer,int expert) const {
    if(layer<first_layer_ || layer>=end_layer_ || expert<0 || expert>=pack_.n_experts())
        throw std::invalid_argument("prefill RAM lookup outside owned experts");
    if(ram_cache_.empty()) return nullptr;
    const int slot=ram_lookup_[layer*pack_.n_experts()+expert];
    return slot<0?nullptr:ram_arena_+size_t(slot)*ram_slot_bytes_;
}

VramExperts::PrefillPrime VramExperts::prime_prefill_cache() {
    if(!miss_upload_ || lent_ || tensor_rank_>=0 || pending_rows_.active || async_routes_failed_)
        throw std::invalid_argument("prefill cache priming requires idle bounded layer split");
    ck(cudaDeviceSynchronize(),"prefill cache idle fence");
    PrefillPrime stats;
    int empty=0;
    try {
        for(int l=first_layer_;l<end_layer_;++l) for(int e=0;e<pack_.n_experts();++e) {
            if(prefill_gpu_slot(l,e)>=0) continue;
            while(empty<int(miss_cache_.size()) && miss_cache_[empty].owner>=0) ++empty;
            if(empty==int(miss_cache_.size())) break;
            const auto& payload=pack_.expert(l,e);
            const int slot=slots_+kTopK+empty;uint8_t* dst=miss_cache_arena_+size_t(empty)*max_slot_bytes_;
            if(const auto* source=prefill_ram_source(l,e)) miss_upload_->set_source(source,payload.offset,payload.bytes);
            try { miss_upload_->copy(dst,payload.offset,payload.bytes); }
            catch(...) { miss_upload_->clear_source();throw; }
            miss_upload_->clear_source();
            desc_host_[slot]=describe_at(pack_,l,e,dst);
            ck(cudaMemcpy(experts_dev_+slot,desc_host_.data()+slot,sizeof(kernels::Exl3Expert),cudaMemcpyHostToDevice),"prefill cache descriptor");
            ck(cudaStreamSynchronize(nullptr),"prefill cache descriptor fence");
            const int owner=l*pack_.n_experts()+e;
            miss_cache_[empty]=CacheEntry{owner,++cache_clock_};miss_cache_lookup_[owner]=empty;
            ++empty;++stats.gpu_fills;stats.bytes+=payload.bytes;
        }
        int available=0;for(const auto& entry:ram_cache_) available+=entry.owner<0;
        for(int l=first_layer_;l<end_layer_ && available;++l) for(int e=0;e<pack_.n_experts() && available;++e) {
            if(prefill_gpu_slot(l,e)>=0 || prefill_ram_source(l,e)) continue;
            ram_payload(l,e);--available;++stats.ram_fills;stats.bytes+=pack_.expert(l,e).bytes;
        }
    } catch(...) { miss_upload_->clear_source();throw; }
    return stats;
}

bool VramExperts::try_cached_rows(int layer,const int32_t* ids,int rows,int topk,int32_t* selected,cudaStream_t stream) {
    if(!miss_upload_ || lent_ || pending_rows_.active || async_routes_failed_ || layer<first_layer_ || layer>=end_layer_ || !ids || !selected ||
       rows<1 || rows>kVerifyMaxResidentTokens || topk<1 || topk>kTopK)
        throw std::invalid_argument("invalid cache-hit row batch");
    cudaStreamCaptureStatus capture;
    ck(cudaStreamIsCapturing(stream,&capture),"cached rows capture status");
    if(capture!=cudaStreamCaptureStatusNone) throw std::invalid_argument("cached rows require eager route admission");
    const int n=rows*topk;int32_t routes[kVerifyMaxResidentTokens*kTopK];
    ck(cudaMemcpyAsync(routes,ids,n*sizeof(int32_t),cudaMemcpyDeviceToHost,stream),"cached rows route read");
    ck(cudaStreamSynchronize(stream),"cached rows prior-consumer fence");
    return resolve_cached_rows(layer,routes,n,selected,stream);
}

void VramExperts::begin_cached_rows(int layer,const int32_t* ids,int rows,int topk,int32_t* selected,cudaStream_t stream) {
    if(!async_routes_ || async_routes_failed_ || pending_rows_.active || !miss_upload_ || lent_ || layer<first_layer_ || layer>=end_layer_ ||
       !ids || !selected || rows<1 || rows>kVerifyMaxResidentTokens || topk<1 || topk>kTopK)
        throw std::invalid_argument("invalid asynchronous cached-row read");
    cudaStreamCaptureStatus capture;
    ck(cudaStreamIsCapturing(stream,&capture),"async cached rows capture status");
    if(capture!=cudaStreamCaptureStatusNone) throw std::invalid_argument("async cached rows require eager route admission");
    const int n=rows*topk;
    const auto status=cudaMemcpyAsync(async_routes_,ids,n*sizeof(int32_t),cudaMemcpyDeviceToHost,stream);
    if(status!=cudaSuccess) {
        if(cudaStreamSynchronize(stream)!=cudaSuccess) async_routes_failed_=true;
        ck(status,"async cached rows route read");
    }
    pending_rows_={layer,n,selected,stream,true};
    ++async_cached_row_reads_;
}

bool VramExperts::finish_cached_rows() {
    if(!pending_rows_.active) throw std::logic_error("no asynchronous cached-row read");
    const auto read=pending_rows_;
    ck(cudaStreamSynchronize(read.stream),"async cached rows prior-consumer fence");
    pending_rows_=PendingRows{};
    return resolve_cached_rows(read.layer,async_routes_,read.count,read.selected,read.stream);
}

void VramExperts::cancel_cached_rows() noexcept {
    if (!pending_rows_.active) return;
    if(cudaStreamSynchronize(pending_rows_.stream)!=cudaSuccess) async_routes_failed_=true;
    pending_rows_=PendingRows{};
}

bool VramExperts::resolve_cached_rows(int layer,const int32_t* routes,int n,int32_t* selected,cudaStream_t stream) {
    // Validate every route before returning a miss or touching cache ages.
    for(int i=0;i<n;++i) if(routes[i]<0 || routes[i]>=pack_.n_experts())
        throw std::runtime_error("invalid cached-row expert id");
    CachedSelection selection{};selection.n=n;
    for(int i=0;i<n;++i) {
        const int owner=layer*pack_.n_experts()+routes[i];
        const int resident=res_host_[owner];
        if(resident>=0) { selection.slots[i]=resident;continue; }
        const int cached=miss_cache_.empty()?-1:miss_cache_lookup_[owner];
        if(cached<0) return false;
        selection.slots[i]=slots_+kTopK+cached;
    }
    // Every descriptor already exists; no allocation, eviction or payload copy.
    // A later miss admission fences this stream before reusing a selected slot.
    publish_cached_selection<<<1,64,0,stream>>>(selection,selected);
    ck(cudaGetLastError(),"cached rows selection publication");
    for(int i=0;i<n;++i) if(selection.slots[i]>=slots_+kTopK) {
        miss_cache_[selection.slots[i]-slots_-kTopK].age=++cache_clock_;++cache_hits_;
    }
    ++cached_row_batches_;
    if(frequency_cache_) for(int i=0;i<n;++i) ++route_frequency_[layer*pack_.n_experts()+routes[i]];
    return true;
}

void VramExperts::prepare_decode(int layer, const int32_t* ids, int topk, int32_t* selected, cudaStream_t stream,
                                bool resident_fastpath) {
    if (!miss_upload_ || pending_rows_.active || async_routes_failed_ || layer < first_layer_ || layer >= end_layer_ || topk < 1 || topk > kTopK)
        throw std::invalid_argument("invalid bounded expert decode staging");
    if (resident_fastpath && topk == kTopK && fully_resident_layer(layer)) {
        // Static descriptors are immutable and the consumer is on this stream.
        // No upload, eviction, host route copy or publication fence is needed.
        ops::resident_routes(ids, res_dev_ + size_t(layer) * pack_.n_experts(), selected, stream);
        return;
    }
    int32_t routes[kTopK], slots[kTopK];
    ck(cudaMemcpyAsync(routes, ids, topk * sizeof(int32_t), cudaMemcpyDeviceToHost, stream), "stream selected routes");
    ck(cudaStreamSynchronize(stream), "stream prior-consumer fence");
    // Validate the whole group before an eviction or payload access. Cached
    // selections later in route order must survive earlier misses in this call.
    std::vector<uint8_t> protected_slots(miss_cache_.size(), 0);
    for (int i = 0; i < topk; ++i) {
        if (routes[i] < 0 || routes[i] >= pack_.n_experts()) throw std::runtime_error("invalid streamed expert id");
        if (!miss_cache_.empty()) {
            const int cached = miss_cache_lookup_[size_t(layer) * pack_.n_experts() + routes[i]];
            if (cached >= 0) protected_slots[cached] = 1;
        }
    }
    std::vector<int> publish;
    if(frequency_cache_) for(int i=0;i<topk;++i) ++route_frequency_[layer*pack_.n_experts()+routes[i]];
    try {
    for (int i = 0; i < topk; ++i) {
        const int e = routes[i];
        if (e < 0 || e >= pack_.n_experts()) throw std::runtime_error("invalid streamed expert id");
        slots[i] = res_host_[size_t(layer) * pack_.n_experts() + e];
        if (slots[i] >= 0) continue;
        const int owner = layer * pack_.n_experts() + e;
        if (!miss_cache_.empty()) {
            const int cached = miss_cache_lookup_[owner];
            if (cached >= 0) {
                slots[i] = slots_ + kTopK + cached;
                miss_cache_[cached].age = ++cache_clock_;
                protected_slots[cached] = 1;
                ++cache_hits_;
                continue;
            }
        }
        const auto& x = pack_.expert(layer, e);
        uint8_t* target = miss_arena_ + size_t(i) * max_slot_bytes_;
        slots[i] = slots_ + i;
        int victim = -1;
        for (size_t j = 0; j < miss_cache_.size(); ++j) {
            if(protected_slots[j]) continue;
            auto rank=[&](int slot) {
                const int o=miss_cache_[slot].owner;
                return frequency_cache_ && o>=0 ? route_frequency_[o] : uint64_t(0);
            };
            if(victim<0 || rank(int(j))<rank(victim) ||
               (rank(int(j))==rank(victim) && miss_cache_[j].age<miss_cache_[victim].age)) victim=int(j);
        }
        // A cold one-off route uses its staging slot instead of evicting a
        // repeatedly useful expert. The consumer arithmetic is identical.
        if(frequency_cache_ && victim>=0 && miss_cache_[victim].owner>=0 &&
           route_frequency_[owner]<=route_frequency_[miss_cache_[victim].owner]) victim=-1;
        if (victim >= 0) {
            auto& entry = miss_cache_[victim];
            if (entry.owner >= 0) { miss_cache_lookup_[entry.owner] = -1; ++cache_evictions_; }
            entry = CacheEntry{};
            target = miss_cache_arena_ + size_t(victim) * max_slot_bytes_;
            slots[i] = slots_ + kTopK + victim;
        }
        if (const auto* source = ram_payload(layer, e)) miss_upload_->set_source(source, x.offset, x.bytes);
        try {
            if (tensor_rank_ >= 0)
                desc_host_[slots[i]] = upload_tensor_expert(*miss_upload_, x, target, tensor_rank_);
            else {
                miss_upload_->copy(target, x.offset, x.bytes);
                desc_host_[slots[i]] = describe_at(pack_, layer, e, target);
            }
        } catch (...) { miss_upload_->clear_source(); throw; }
        miss_upload_->clear_source();
        if (victim >= 0) {
            miss_cache_[victim] = CacheEntry{owner, ++cache_clock_};
            miss_cache_lookup_[owner] = victim;
            protected_slots[victim] = 1;
        }
        if (!miss_cache_.empty()) publish.push_back(slots[i]);
        ++streamed_decode_;
    }
    // Pageable-host cudaMemcpy on the default stream does not order a
    // nonblocking consumer stream. Publish on that consumer's stream and fence
    // before returning the stack-owned selection or reusing host descriptors.
    if (miss_cache_.empty())
        ck(cudaMemcpyAsync(experts_dev_ + slots_, desc_host_.data() + slots_, kTopK * sizeof(kernels::Exl3Expert), cudaMemcpyHostToDevice, stream),
           "stream descriptor publication");
    else for (int slot : publish)
        ck(cudaMemcpyAsync(experts_dev_ + slot, desc_host_.data() + slot, sizeof(kernels::Exl3Expert), cudaMemcpyHostToDevice, stream),
           "cached descriptor publication");
    ck(cudaMemcpyAsync(selected, slots, topk * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "stream selected publication");
    ck(cudaStreamSynchronize(stream), "stream publication fence");
    } catch (...) {
        // Drain possible stack-owned selection copies before unwind. Earlier
        // uploads may not have published descriptors; never retain their lookups.
        cudaStreamSynchronize(stream);
        std::fill(miss_cache_lookup_.begin(), miss_cache_lookup_.end(), -1);
        std::fill(miss_cache_.begin(), miss_cache_.end(), CacheEntry{});
        throw;
    }
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
    if (pending_rows_.active || async_routes_failed_) throw std::logic_error("ds41 vram experts: cached-row read is pending or failed");
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
    if (pending_rows_.active || async_routes_failed_) throw std::logic_error("ds41 vram experts: cached-row read is pending or failed");
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
    if (pending_rows_.active || async_routes_failed_) throw std::logic_error("ds41 vram experts: cached-row read is pending or failed");
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
