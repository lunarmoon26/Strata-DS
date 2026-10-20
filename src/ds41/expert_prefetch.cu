// src/ds41/expert_prefetch.cu - see include/strata/ds41/expert_prefetch.hpp
#include "strata/ds41/expert_prefetch.hpp"

#include "strata/ds41/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::ds41 {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 prefetch: ") + what + ": " + cudaGetErrorString(e));
}

__device__ void rebase(kernels::Exl3Proj& p, const uint8_t* src, uint8_t* dst) {
    p.trellis = reinterpret_cast<const uint16_t*>(dst + (reinterpret_cast<const uint8_t*>(p.trellis) - src));
    p.suh = reinterpret_cast<const __half*>(dst + (reinterpret_cast<const uint8_t*>(p.suh) - src));
    p.svh = reinterpret_cast<const __half*>(dst + (reinterpret_cast<const uint8_t*>(p.svh) - src));
}

/// One block of n_experts threads (<= 1024): the router's selection score (kernels::router_topk: s = sqrt(softplus),
/// ranked by s + bias, ties to the lower id), the `g` best in order, then the copy plan.
__global__ void plan_k(const float* __restrict__ logits, const float* __restrict__ bias, int n, int g,
                       const int32_t* __restrict__ res, const kernels::Exl3Expert* __restrict__ ram,
                       const ExpertBlob* __restrict__ blobs, uint8_t* buffer, size_t cap, int32_t* ranked,
                       int32_t* ids, kernels::Exl3Expert* descs, ExpertCopy* jobs, int* count,
                       unsigned long long* tag, const unsigned long long* epoch, int layer) {
    __shared__ float score[1024];
    const int e = threadIdx.x;
    if (e < n) {
        const float v = logits[e];
        const float sp = v > 20.0f ? v : log1pf(expf(v));
        score[e] = sqrtf(sp) + bias[e];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    size_t used = 0;
    int k = 0;
    for (int r = 0; r < g; ++r) {
        int best = -1;
        for (int j = 0; j < n; ++j)
            if (score[j] > -INFINITY && (best < 0 || score[j] > score[best])) best = j;
        ranked[r] = best;
        if (best < 0) continue;
        score[best] = -INFINITY;
        if (res && res[best] >= 0) continue;            // in VRAM: no copy
        const kernels::Exl3Expert d = ram[best];
        if (!d.w1.trellis) continue;                   // not in the RAM tier (or on its way between tiers)
        const ExpertBlob blob = blobs[best];
        const size_t at = (used + 255) / 256 * 256;
        if (at + blob.bytes > cap) continue;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(d.w1.trellis) - blob.first_trellis;
        uint8_t* dst = buffer + at;
        jobs[k] = {src, dst, blob.bytes};
        kernels::Exl3Expert moved = d;
        rebase(moved.w1, src, dst);
        rebase(moved.w3, src, dst);
        rebase(moved.w2, src, dst);
        descs[k] = moved;
        ids[k++] = best;
        used = at + blob.bytes;
    }
    // every entry past the copies: verify windows search all kMaxGuesses (plan_rows_k fills that many)
    for (int j = k; j < ExpertPrefetch::kMaxGuesses; ++j) ids[j] = -1;
    *count = k;
    if (tag) {   // DMA mode: the copy list is in mapped host memory; the tag tells the copy thread, after the jobs
        __threadfence_system();
        *(volatile unsigned long long*) tag = *(const volatile unsigned long long*) epoch * 64 + layer;
        __threadfence_system();
    }
}

/// plan_k for m rows (logits [m][n]): g guesses per row (m * g <= kMaxGuesses), merged in rank order, each expert
/// once; the copy plan as plan_k's. ids[] is filled up to kMaxGuesses.
__global__ void plan_rows_k(const float* __restrict__ logits, int m, const float* __restrict__ bias, int n, int g,
                            const int32_t* __restrict__ res, const kernels::Exl3Expert* __restrict__ ram,
                            const ExpertBlob* __restrict__ blobs, uint8_t* buffer, size_t cap, int32_t* ranked,
                            int32_t* ids, kernels::Exl3Expert* descs, ExpertCopy* jobs, int* count,
                            unsigned long long* tag, const unsigned long long* epoch, int layer) {
    __shared__ float score[4][1024];
    __shared__ int best_of[4][ExpertPrefetch::kMaxGuesses];
    for (int r = 0; r < m; ++r) {
        const int e = threadIdx.x;
        if (e < n) {
            const float v = logits[r * n + e];
            const float sp = v > 20.0f ? v : log1pf(expf(v));
            score[r][e] = sqrtf(sp) + bias[e];
        }
    }
    __syncthreads();
    if (threadIdx.x < m) {   // each row's g best, in order
        const int r = threadIdx.x;
        for (int k = 0; k < g; ++k) {
            int best = -1;
            for (int j = 0; j < n; ++j)
                if (score[r][j] > -INFINITY && (best < 0 || score[r][j] > score[r][best])) best = j;
            best_of[r][k] = best;
            if (best >= 0) score[r][best] = -INFINITY;
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    size_t used = 0;
    int k = 0, nr = 0;
    for (int rank = 0; rank < g; ++rank)
        for (int r = 0; r < m; ++r) {
            const int best = best_of[r][rank];
            if (best < 0) continue;
            bool seen = false;
            for (int j = 0; j < nr; ++j) seen |= ranked[j] == best;
            if (seen) continue;
            ranked[nr++] = best;
            if (res && res[best] >= 0) continue;
            const kernels::Exl3Expert d = ram[best];
            if (!d.w1.trellis) continue;
            const ExpertBlob blob = blobs[best];
            const size_t at = (used + 255) / 256 * 256;
            if (at + blob.bytes > cap) continue;
            const uint8_t* src = reinterpret_cast<const uint8_t*>(d.w1.trellis) - blob.first_trellis;
            uint8_t* dst = buffer + at;
            jobs[k] = {src, dst, blob.bytes};
            kernels::Exl3Expert moved = d;
            rebase(moved.w1, src, dst);
            rebase(moved.w3, src, dst);
            rebase(moved.w2, src, dst);
            descs[k] = moved;
            ids[k++] = best;
            used = at + blob.bytes;
        }
    for (int j = nr; j < ExpertPrefetch::kMaxGuesses; ++j) ranked[j] = -1;
    for (int j = k; j < ExpertPrefetch::kMaxGuesses; ++j) ids[j] = -1;
    *count = k;
    if (tag) {
        __threadfence_system();
        *(volatile unsigned long long*) tag = *(const volatile unsigned long long*) epoch * 64 + layer;
        __threadfence_system();
    }
}

/// DMA mode, on main: wait until the copy thread has copied `layer` of this step
__global__ void wait_copied_k(const volatile unsigned long long* done, const volatile unsigned long long* epoch,
                              int layer) {
    const unsigned long long want = *epoch * 64 + layer;
    while (*done < want) __nanosleep(200);
    __threadfence_system();
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

/// The planned copies, every block over every job (as the doorbell's staging copy)
__global__ void copy_k(const ExpertCopy* jobs, const int* count) {
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t step = size_t(gridDim.x) * blockDim.x;
    const int n = *count;
    for (int j = 0; j < n; ++j) {
        const ExpertCopy job = jobs[j];
        const auto* src = reinterpret_cast<const uint4*>(job.src);
        auto* dst = reinterpret_cast<uint4*>(job.dst);
        for (size_t i = lane; i < job.bytes / 16; i += step) dst[i] = src[i];
        const size_t tail = job.bytes / 16 * 16 + lane;
        if (tail < job.bytes) job.dst[tail] = job.src[tail];
    }
}

}  // namespace

ExpertPrefetch::ExpertPrefetch(int guesses, size_t buffer_bytes, int n_experts, int dim, bool dma)
    : guesses_(guesses), n_experts_(n_experts), dim_(dim), buffer_bytes_(buffer_bytes), dma_(dma) {
    if (guesses < 1 || guesses > kMaxGuesses || n_experts < 1 || n_experts > 1024 || buffer_bytes == 0)
        throw std::invalid_argument("ExpertPrefetch: bad shape");
    try {
        ck(cudaMalloc(&buffer_, 2 * buffer_bytes_), "buffer");
        ck(cudaMalloc(&logits_, 4 * n_experts_ * sizeof(float)), "logits");
        ck(cudaMalloc(&ids_, 2 * kMaxGuesses * sizeof(int32_t)), "ids");
        ck(cudaMalloc(&ranked_, 2 * kMaxGuesses * sizeof(int32_t)), "ranked");
        ck(cudaMalloc(&descs_, 2 * kMaxGuesses * sizeof(kernels::Exl3Expert)), "descriptors");
        ck(cudaMalloc(&jobs_, 2 * kMaxGuesses * sizeof(ExpertCopy)), "jobs");
        ck(cudaMalloc(&count_, 2 * sizeof(int)), "count");
        ck(cudaMemset(ids_, 0xff, 2 * kMaxGuesses * sizeof(int32_t)), "ids");
        ck(cudaMemset(count_, 0, 2 * sizeof(int)), "count");
        ck(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking), "stream");
        for (int p = 0; p < 2; ++p) {
            ck(cudaEventCreateWithFlags(&ready_[p], cudaEventDisableTiming), "event");
            ck(cudaEventCreateWithFlags(&planned_[p], cudaEventDisableTiming), "event");
            ck(cudaEventCreateWithFlags(&done_[p], cudaEventDisableTiming), "event");
        }
        if (dma_) {
            ck(cudaGetDevice(&device_), "device");
            ck(cudaHostAlloc((void**) &shared_, sizeof(Shared), cudaHostAllocMapped), "shared");
            std::memset(shared_, 0, sizeof(Shared));
            ck(cudaHostGetDevicePointer((void**) &shared_dev_, shared_, 0), "shared alias");
            ck(cudaStreamCreateWithFlags(&dma_stream_, cudaStreamNonBlocking), "DMA stream");
            thread_ = std::thread([this] { copier(); });
        }
    } catch (...) {
        this->~ExpertPrefetch();
        throw;
    }
}

ExpertPrefetch::~ExpertPrefetch() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
    // a wait still queued on main (a step that threw before its sync) would never see its copy: release every wait,
    // so the device can idle before the shared memory is freed
    if (shared_) __atomic_store_n(&shared_->done, ~0ull, __ATOMIC_RELEASE);
    if (dma_stream_) {
        cudaStreamSynchronize(dma_stream_);
        cudaStreamDestroy(dma_stream_);
    }
    dma_stream_ = nullptr;
    if (shared_) cudaFreeHost(shared_);
    shared_ = shared_dev_ = nullptr;
    if (copy_) cudaStreamSynchronize(copy_);
    for (int p = 0; p < 2; ++p) {
        if (ready_[p]) cudaEventDestroy(ready_[p]);
        if (planned_[p]) cudaEventDestroy(planned_[p]);
        if (done_[p]) cudaEventDestroy(done_[p]);
        ready_[p] = planned_[p] = done_[p] = nullptr;
    }
    if (copy_) cudaStreamDestroy(copy_);
    copy_ = nullptr;
    for (void* p : {(void*) buffer_, (void*) logits_, (void*) ids_, (void*) ranked_, (void*) descs_, (void*) jobs_,
                    (void*) count_})
        if (p) cudaFree(p);
    buffer_ = nullptr;
    logits_ = nullptr;
    ids_ = ranked_ = nullptr;
    descs_ = nullptr;
    jobs_ = nullptr;
    count_ = nullptr;
}

void ExpertPrefetch::plan(int layer, const __nv_bfloat16* x, const __nv_bfloat16* w, const float* bias,
                          const int32_t* res, const kernels::Exl3Expert* ram, const ExpertBlob* blobs,
                          cudaStream_t main) {
    const int p = layer & 1;
    ck(cudaEventRecord(ready_[p], main), "record");
    ck(cudaStreamWaitEvent(copy_, ready_[p], 0), "wait");
    ops::bf16_linear(x, nullptr, w, dim_, n_experts_, nullptr, logits_, copy_);
    plan_k<<<1, n_experts_, 0, copy_>>>(logits_, bias, n_experts_, guesses_, res, ram, blobs,
                                        buffer_ + (size_t) p * buffer_bytes_, buffer_bytes_,
                                        ranked_ + p * kMaxGuesses, ids_ + p * kMaxGuesses,
                                        descs_ + p * kMaxGuesses,
                                        dma_ ? shared_dev_->jobs[p] : jobs_ + p * kMaxGuesses,
                                        dma_ ? &shared_dev_->count[p] : count_ + p,
                                        dma_ ? &shared_dev_->tag[p] : nullptr, dma_ ? &shared_dev_->epoch : nullptr,
                                        layer);
    ck(cudaGetLastError(), "plan launch");
    ck(cudaEventRecord(planned_[p], copy_), "record plan");
    if (dma_) return;   // the copy thread copies; join() waits for its flag
    copy_k<<<68, 256, 0, copy_>>>(jobs_ + p * kMaxGuesses, count_ + p);
    ck(cudaGetLastError(), "copy launch");
    ck(cudaEventRecord(done_[p], copy_), "record copy");
}

void ExpertPrefetch::plan_rows(int layer, const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias,
                               const int32_t* res, const kernels::Exl3Expert* ram, const ExpertBlob* blobs,
                               cudaStream_t main) {
    if (m == 1) { plan(layer, x, w, bias, res, ram, blobs, main); return; }
    if (m < 1 || m > 4) throw std::invalid_argument("ExpertPrefetch::plan_rows: m outside 1..4");
    const int p = layer & 1;
    const int g = std::min(guesses_, kMaxGuesses / m);
    ck(cudaEventRecord(ready_[p], main), "record");
    ck(cudaStreamWaitEvent(copy_, ready_[p], 0), "wait");
    ops::bf16_linear_rows(x, m, w, dim_, n_experts_, logits_, copy_);
    plan_rows_k<<<1, n_experts_, 0, copy_>>>(logits_, m, bias, n_experts_, g, res, ram, blobs,
                                             buffer_ + (size_t) p * buffer_bytes_, buffer_bytes_,
                                             ranked_ + p * kMaxGuesses, ids_ + p * kMaxGuesses,
                                             descs_ + p * kMaxGuesses,
                                             dma_ ? shared_dev_->jobs[p] : jobs_ + p * kMaxGuesses,
                                             dma_ ? &shared_dev_->count[p] : count_ + p,
                                             dma_ ? &shared_dev_->tag[p] : nullptr,
                                             dma_ ? &shared_dev_->epoch : nullptr, layer);
    ck(cudaGetLastError(), "plan rows launch");
    ck(cudaEventRecord(planned_[p], copy_), "record plan");
    if (dma_) return;
    copy_k<<<68, 256, 0, copy_>>>(jobs_ + p * kMaxGuesses, count_ + p);
    ck(cudaGetLastError(), "copy launch");
    ck(cudaEventRecord(done_[p], copy_), "record copy");
}

void ExpertPrefetch::begin_step() {
    if (!dma_) return;
    __atomic_store_n(&shared_->epoch, shared_->epoch + 1, __ATOMIC_RELEASE);
}

void ExpertPrefetch::copier() {
    cudaSetDevice(device_);
    unsigned long long epoch = ~0ull;
    int last = 0;
    long idle = 0;
    bool warned = false;
    while (!stop_.load(std::memory_order_relaxed)) {
        const unsigned long long e = __atomic_load_n(&shared_->epoch, __ATOMIC_ACQUIRE);
        if (e != epoch) {   // a new step: its layers start again
            epoch = e;
            last = 0;
        }
        const int layer = detail::next_copy_layer(   // the lowest layer planned in this step and not copied yet
            [&](int p) { return __atomic_load_n(&shared_->tag[p], __ATOMIC_ACQUIRE); }, epoch, last);
        if (layer < 0) {   // spin during a step (a plan comes about every 1.3 ms); nap when idle for long
            if (++idle < 400000) cpu_relax();
            else std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        idle = 0;
        const int p = layer & 1;
        const int n = __atomic_load_n(&shared_->count[p], __ATOMIC_ACQUIRE);
        bool ok = true;
        for (int j = 0; j < n; ++j) {
            const ExpertCopy c = shared_->jobs[p][j];
            ok &= cudaMemcpyAsync(c.dst, c.src, c.bytes, cudaMemcpyDefault, dma_stream_) == cudaSuccess;
        }
        if (n > 0) ok &= cudaStreamSynchronize(dma_stream_) == cudaSuccess;
        if (!ok) {
            failed_ = true;
            if (!warned)
                std::fprintf(stderr, "ds41 prefetch: a DMA copy failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            warned = true;
        }
        // the flag even after a failure: the GPU must not wait forever (the step then reports failed())
        __atomic_store_n(&shared_->done, epoch * 64 + layer, __ATOMIC_RELEASE);
        last = layer;
    }
}

void ExpertPrefetch::ready(int layer, cudaStream_t main) {
    ck(cudaStreamWaitEvent(main, planned_[layer & 1], 0), "ready");
}

void ExpertPrefetch::join(int layer, cudaStream_t main) {
    if (dma_) {
        wait_copied_k<<<1, 1, 0, main>>>(&shared_dev_->done, &shared_dev_->epoch, layer);
        ck(cudaGetLastError(), "wait launch");
        return;
    }
    ck(cudaStreamWaitEvent(main, done_[layer & 1], 0), "join");
}

}  // namespace strata::ds41
