// src/ds41/doorbell.cu - see include/strata/ds41/doorbell.hpp.
#include "strata/ds41/doorbell.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::ds41 {

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 doorbell: ") + what + ": " + cudaGetErrorString(e));
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

constexpr size_t kAlign = 256;   // every buffer, and the two words on separate lines
size_t up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

__device__ void rebase(kernels::Exl3Proj& p, const uint8_t* src, uint8_t* dst) {
    p.trellis = reinterpret_cast<const uint16_t*>(dst + (reinterpret_cast<const uint8_t*>(p.trellis) - src));
    p.suh = reinterpret_cast<const __half*>(dst + (reinterpret_cast<const uint8_t*>(p.suh) - src));
    p.svh = reinterpret_cast<const __half*>(dst + (reinterpret_cast<const uint8_t*>(p.svh) - src));
}

__global__ void publish_k(const uint4* __restrict__ x, int n_x16, const int32_t* __restrict__ ids,
                          const float* __restrict__ w, int m, int topk, const int32_t* __restrict__ res,
                          int32_t* __restrict__ gpu_sel, uint4* mx, int32_t* mids, float* mw, volatile uint32_t* seq,
                          uint32_t round, const kernels::Exl3Expert* vram, const kernels::Exl3Expert* ram,
                          const int* quota, kernels::Exl3Expert* call, ExpertDoorbell::Counts* counts,
                          const ExpertBlob* blobs, uint8_t* staging, size_t stride,
                          ExpertCopy* jobs, int* copy_count, const int32_t* pf_ids,
                          const kernels::Exl3Expert* pf_desc, int pf_n, int tail_rank, int ssd_rank) {
    for (int i = threadIdx.x; i < n_x16; i += blockDim.x) mx[i] = x[i];
    // At most 48 route uses in decode/verify. Warp 0 takes one route use per lane, whole tokens per pass (5 tokens
    // of 6 uses), so the dependent loads (ids, res, descriptors) of all uses overlap; one lane for all of them took
    // 4.8-6.0 us, one lane each 3.0-3.2 us (RTX 5090 Laptop). The zero-copy prefix rule (the first q eligible misses
    // of a token, in route order) and the copy job order come from ballots in route order.
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        ExpertDoorbell::Counts c{};
        const bool descriptors = vram != nullptr || ram != nullptr;
        const int q = quota ? max(0, min(*quota, topk)) : 0;
        const int per_pass = 32 / topk;
        const int t_of = lane / topk, j_of = lane % topk;
        for (int t0 = 0; t0 < m; t0 += per_pass) {
            const int t = t0 + t_of;
            const bool live = t_of < per_pass && t < m;
            const int i = t * topk + j_of;
            const int32_t id = live ? ids[i] : -1;
            const int32_t slot = (live && res && id >= 0) ? res[id] : -1;
            const bool hit = slot >= 0;
            // upstream's route tail skip: the best rank any token of the call gives this expert (lanes of the same id)
            // ssd_rank: the same for experts in neither VRAM nor the RAM tier (their weights come from the SSD)
            bool skip = false;
            if ((tail_rank > 0 || ssd_rank > 0) && m <= per_pass) {   // every use in this one pass
                const unsigned same = __match_any_sync(0xffffffffu, (live && id >= 0) ? id : -1 - lane);
                int best = live ? j_of : 99;
                for (int s = 0; s < 32; ++s) {
                    const int r = __shfl_sync(0xffffffffu, best, s);
                    if ((same >> s) & 1u) best = min(best, r);
                }
                const bool in_ram = live && id >= 0 && ram && ram[id].w1.trellis;
                skip = live && id >= 0 && !hit &&
                       ((tail_rank > 0 && best >= tail_rank) || (ssd_rank > 0 && !in_ram && best >= ssd_rank));
            }
            int pf = -1;   // a miss the prefetch already copied: the GPU computes it from the buffer
            if (live && !hit && !skip && id >= 0 && pf_ids)
                for (int k = 0; k < pf_n && pf < 0; ++k)
                    if (pf_ids[k] == id) pf = k;
            const bool eligible = live && !hit && !skip && pf < 0 && id >= 0 && ram && ram[id].w1.trellis;
            const unsigned below = (1u << lane) - 1;
            const unsigned token_lanes = ((1u << topk) - 1) << (t_of * topk);
            const unsigned eligible_lanes = __ballot_sync(0xffffffffu, eligible);   // every lane votes
            const bool zc = eligible && __popc(eligible_lanes & below & token_lanes) < q;
            const unsigned zc_lanes = __ballot_sync(0xffffffffu, zc);
            if (live) {
                mids[i] = (id < 0 || hit || zc || pf >= 0 || skip) ? -1 : id;
                mw[i] = w[i];
                if (gpu_sel) gpu_sel[i] = descriptors ? ((hit || zc || pf >= 0) ? i : -1) : slot;
                if (descriptors) {
                    call[i] = hit ? vram[slot] : pf >= 0 ? pf_desc[pf] : (zc ? ram[id] : kernels::Exl3Expert{});
                    if (zc && staging) {
                        const ExpertBlob blob = blobs[id];
                        const auto* src = reinterpret_cast<const uint8_t*>(call[i].w1.trellis) - blob.first_trellis;
                        auto* dst = staging + size_t(i) * stride;
                        jobs[c.zero_copy + __popc(zc_lanes & below)] = {src, dst, blob.bytes};
                        rebase(call[i].w1, src, dst);
                        rebase(call[i].w3, src, dst);
                        rebase(call[i].w2, src, dst);
                    }
                }
            }
            c.vram += __popc(__ballot_sync(0xffffffffu, hit));
            c.prefetched += __popc(__ballot_sync(0xffffffffu, !hit && pf >= 0));
            c.zero_copy += __popc(zc_lanes);
            c.cpu += __popc(__ballot_sync(0xffffffffu, live && !hit && !skip && pf < 0 && !zc && id >= 0));
            c.skipped += __popc(__ballot_sync(0xffffffffu, skip));
        }
        if (lane == 0) {
            *counts = c;
            if (copy_count) *copy_count = c.zero_copy;
        }
    }
    __threadfence_system();   // Every thread publishes its writes before seq.
    __syncthreads();
    if (threadIdx.x == 0) {
        *seq = round;
        __threadfence_system();
    }
}

__global__ void wait_add_k(const volatile uint32_t* done, uint32_t round, const float4* y, float4* out, int n4) {
    if (threadIdx.x == 0) {
        while (*done < round) __nanosleep(100);
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = __ldcv(y + i);   // uncached: the host wrote it after this kernel may have started
        float4 o = out[i];
        o.x += v.x; o.y += v.y; o.z += v.z; o.w += v.w;
        out[i] = o;
    }
}

}  // namespace

ExpertDoorbell::ExpertDoorbell(int max_m, int topk, int dim) : max_m_(max_m), topk_(topk), dim_(dim) {
    if (max_m < 1 || topk < 1 || topk > 31 || dim < 8 || dim % 8 != 0) throw std::invalid_argument("ExpertDoorbell: bad shape");
    const size_t sel = (size_t) max_m * topk, rows = (size_t) max_m * dim;
    const size_t o_seq = 0, o_done = kAlign, o_x = 2 * kAlign;
    const size_t o_ids = o_x + up(rows * 2), o_w = o_ids + up(sel * 4), o_y = o_w + up(sel * 4);
    const size_t o_counts = o_y + up(rows * 4);
    const size_t total = o_counts + up(sizeof(Counts));
    ck(cudaHostAlloc(&host_, total, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(host_, 0, total);
    void* dev = nullptr;
    if (cudaHostGetDevicePointer(&dev, host_, 0) != cudaSuccess) {   // the constructor throws: no destructor runs
        cudaFreeHost(host_);
        host_ = nullptr;
        ck(cudaGetLastError(), "cudaHostGetDevicePointer");
        throw std::runtime_error("ExpertDoorbell: cudaHostGetDevicePointer failed");
    }
    auto hp = [&](size_t o) { return (char*) host_ + o; };
    auto dp = [&](size_t o) { return (char*) dev + o; };
    h_seq_ = (uint32_t*) hp(o_seq);    d_seq_ = (uint32_t*) dp(o_seq);
    h_done_ = (uint32_t*) hp(o_done);  d_done_ = (uint32_t*) dp(o_done);
    h_x_ = (uint16_t*) hp(o_x);        d_x_ = (uint16_t*) dp(o_x);
    h_ids_ = (int32_t*) hp(o_ids);     d_ids_ = (int32_t*) dp(o_ids);
    h_w_ = (float*) hp(o_w);           d_w_ = (float*) dp(o_w);
    h_y_ = (float*) hp(o_y);           d_y_ = (float*) dp(o_y);
    h_counts_ = (Counts*) hp(o_counts); d_counts_ = (Counts*) dp(o_counts);
    try {
        ck(cudaMalloc(&gpu_experts_, sel * sizeof(*gpu_experts_)), "call descriptors");
    } catch (...) {
        cudaFreeHost(host_);
        throw;
    }
}

ExpertDoorbell::~ExpertDoorbell() {
    cudaFree(gpu_experts_);
    if (host_) cudaFreeHost(host_);
}

void ExpertDoorbell::publish(const uint16_t* x, const int32_t* ids, const float* w, int m, const int32_t* res,
                             int32_t* gpu_sel, uint32_t round, cudaStream_t stream,
                             const kernels::Exl3Expert* vram, const kernels::Exl3Expert* ram, const int* quota,
                             ExpertStaging* stage, const ExpertBlob* blobs, const int32_t* pf_ids,
                             const kernels::Exl3Expert* pf_desc, int pf_n, int tail_rank, int ssd_rank) {
    if (m < 1 || m > max_m_) throw std::invalid_argument("ExpertDoorbell::publish: bad m");
    if (res && ram && !vram) throw std::invalid_argument("ExpertDoorbell::publish: missing VRAM descriptors");
    if (stage && (!blobs || stage->capacity() < m * topk_))
        throw std::invalid_argument("ExpertDoorbell::publish: missing blob metadata or staging slots");
    publish_k<<<1, 512, 0, stream>>>((const uint4*) x, m * dim_ / 8, ids, w, m, topk_, res, gpu_sel, (uint4*) d_x_,
                                     d_ids_, d_w_, d_seq_, round, vram, ram, quota, gpu_experts_, d_counts_,
                                     blobs, stage ? stage->data() : nullptr, stage ? stage->stride() : 0,
                                     stage ? stage->jobs() : nullptr, stage ? stage->count() : nullptr, pf_ids,
                                     pf_desc, pf_n, tail_rank, ssd_rank);
    ck(cudaGetLastError(), "publish");
}

void ExpertDoorbell::wait_add(float* out, int m, uint32_t round, cudaStream_t stream) {
    if (m < 1 || m > max_m_) throw std::invalid_argument("ExpertDoorbell::wait_add: bad m");
    wait_add_k<<<1, 1024, 0, stream>>>(d_done_, round, (const float4*) d_y_, (float4*) out, m * dim_ / 4);
    ck(cudaGetLastError(), "wait_add");
}

void ExpertDoorbell::reset() {
    __atomic_store_n(h_seq_, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(h_done_, 0u, __ATOMIC_RELEASE);
}

bool ExpertDoorbell::wait_published(uint32_t round, const std::atomic<bool>& stop) const {
    while (__atomic_load_n(h_seq_, __ATOMIC_ACQUIRE) < round) {
        if (stop.load(std::memory_order_relaxed)) return false;
        cpu_relax();
    }
    return true;
}

void ExpertDoorbell::mark_done(uint32_t round) { __atomic_store_n(h_done_, round, __ATOMIC_RELEASE); }

SkipRanks skip_ranks(const char* route_env, const char* ssd_env, bool has_vram, double cached_share) {
    auto valid = [](int r) { return r > 0 && r < 6 ? r : 0; };
    const bool on = has_vram && cached_share < 0.90;
    SkipRanks r;
    r.tail = valid(route_env ? std::atoi(route_env) : (on ? 4 : 0));
    r.ssd = valid(ssd_env ? std::atoi(ssd_env) : (r.tail ? 3 : 0));
    return r;
}

}  // namespace strata::ds41
