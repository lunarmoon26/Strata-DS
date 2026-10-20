// src/ds41/expert_stream.cpp - prefill's expert stream. See expert_stream.hpp.
#include "strata/ds41/expert_stream.hpp"

#include "strata/ds41/host_experts.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <stdexcept>

namespace strata::ds41 {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 expert stream: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

ExpertStream::ExpertStream(const Pack& pack, const HostExperts* host, uint8_t* ring, int slots, size_t slot_bytes,
                           int readers, int host_buffers, bool unbuffered)
    : pack_(pack), host_(host), ring_(ring), slots_(slots), n_host_(host_buffers), slot_bytes_(slot_bytes) {
    try {
        if (slots < 1 || readers < 1 || host_buffers < 1)
            throw std::invalid_argument("ds41 expert stream: needs a slot, a reader and a host buffer");
        ck(cudaGetDevice(&device_), "cudaGetDevice");
        const std::string path = pack.dir() + "/experts.bin";
        if (unbuffered) {
            fd_ = open(path.c_str(), O_RDONLY | O_DIRECT);
            direct_ = fd_ >= 0;
            if (!direct_) std::fprintf(stderr, "ds41 expert stream: %s refuses O_DIRECT; plain reads\n", path.c_str());
        }
        if (fd_ < 0) fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("ds41 expert stream: cannot open " + path);
        ck(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking), "copy stream");
        auto events = [&](std::vector<cudaEvent_t>& v, size_t n) {
            v.resize(n);
            for (auto& e : v) ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
        };
        events(dma_done_, (size_t) n_host_);
        events(copied_, (size_t) slots_);
        events(used_, (size_t) slots_);
        host_buf_.assign((size_t) n_host_, nullptr);
        // an O_DIRECT window starts up to 4 KiB before the expert and ends up to 4 KiB after it
        for (auto& b : host_buf_) ck(cudaHostAlloc((void**) &b, slot_bytes_ + 8192, cudaHostAllocDefault), "pinned staging");
        for (int i = 0; i < readers; ++i) threads_.emplace_back([this] { reader(); });
        threads_.emplace_back([this] { issuer(); });
    } catch (...) {
        cleanup();
        throw;
    }
}

ExpertStream::~ExpertStream() { cleanup(); }

void ExpertStream::cleanup() noexcept {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_)
        if (t.joinable()) t.join();
    if (copy_) cudaStreamSynchronize(copy_);
    for (auto* v : {&dma_done_, &copied_, &used_})
        for (auto& e : *v)
            if (e) cudaEventDestroy(e);
    for (auto* b : host_buf_)
        if (b) cudaFreeHost(b);
    if (copy_) cudaStreamDestroy(copy_);
    if (fd_ >= 0) close(fd_);
}

void ExpertStream::fail(const std::string& what) {
    std::lock_guard<std::mutex> lk(mu_);
    if (error_.empty()) error_ = what;
    cv_.notify_all();
}

int64_t ExpertStream::push(const std::vector<std::pair<int, int>>& jobs) {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t first = (int64_t) jobs_.size();
    jobs_.insert(jobs_.end(), jobs.begin(), jobs.end());
    src_.resize(jobs_.size(), nullptr);
    read_.resize(jobs_.size(), 0);
    cv_.notify_all();
    return first;
}

uint8_t* ExpertStream::wait(int64_t j, cudaStream_t stream) {
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(mu_);
        if (j < 0 || j >= (int64_t) jobs_.size()) throw std::logic_error("ds41 expert stream: wait for an unknown job");
        cv_.wait(lk, [&] { return issued_ > j || !error_.empty(); });
        if (!error_.empty()) throw std::runtime_error(error_);
        wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    // the copy was recorded before issued_ passed j, and job j + slots is not issued before j is released
    ck(cudaStreamWaitEvent(stream, copied_[j % slots_], 0), "wait for the copy");
    return ring_ + (size_t) (j % slots_) * slot_bytes_;
}

void ExpertStream::release(int64_t j, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(mu_);
    if (j != released_) throw std::logic_error("ds41 expert stream: jobs must be released in order");
    ck(cudaEventRecord(used_[j % slots_], stream), "record release");
    ++released_;
    cv_.notify_all();
}

void ExpertStream::drain() {
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return released_ == (int64_t) jobs_.size() || !error_.empty(); });
        if (!error_.empty()) throw std::runtime_error(error_);
    }
    // the next round starts at job 0 in slot 0 and host buffer 0 without waiting: everything must be done
    ck(cudaStreamSynchronize(copy_), "drain copies");
    for (auto& e : used_) ck(cudaEventSynchronize(e), "drain releases");
    std::lock_guard<std::mutex> lk(mu_);
    jobs_.clear();
    src_.clear();
    read_.clear();
    next_read_ = issued_ = released_ = 0;
}

ExpertStream::Stats ExpertStream::take_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s;
    s.from_ram = n_ram_.exchange(0);
    s.from_cache = n_cache_.exchange(0);
    s.from_ssd = n_ssd_.exchange(0);
    s.jobs = s.from_ram + s.from_cache + s.from_ssd;
    s.consumer_wait_ms = wait_ms_;
    wait_ms_ = 0;
    return s;
}

void ExpertStream::reader() {
    std::vector<unsigned char> vec;
    try {
        ck(cudaSetDevice(device_), "cudaSetDevice");
        for (;;) {
            int64_t j;
            std::pair<int, int> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                // host buffer j % n_host is free once job j - n_host's copy is issued (its DMA event then exists)
                cv_.wait(lk, [&] {
                    return stop_ || !error_.empty() ||
                           (next_read_ < (int64_t) jobs_.size() && next_read_ - n_host_ < issued_);
                });
                if (stop_ || !error_.empty()) return;
                j = next_read_++;
                job = jobs_[j];
            }
            const ExpertSlot& x = pack_.expert(job.first, job.second);
            const int32_t ram = host_ ? host_->slot_of(job.first, job.second) : -1;
            const uint8_t* src;
            if (ram >= 0) {   // pinned already: copied from its RAM slot, no staging
                src = host_->slot_ptr(ram);
                ++n_ram_;
            } else {
                uint8_t* buf = host_buf_[j % n_host_];
                if (j >= n_host_) ck(cudaEventSynchronize(dma_done_[j % n_host_]), "wait for the host buffer");
                // the window read: the expert itself, or with O_DIRECT its 4 KiB-aligned cover
                const uint64_t a0 = direct_ ? x.offset & ~(uint64_t) 4095 : x.offset;
                const uint64_t a1 = direct_ ? (x.offset + x.bytes + 4095) & ~(uint64_t) 4095 : x.offset + x.bytes;
                bool cached = false;
                if (!direct_) {   // was every page cached? (the mapped file shares the page cache with pread)
                    const uintptr_t a = (uintptr_t) (pack_.expert_base() + x.offset) & ~(uintptr_t) 4095;
                    const size_t len = (uintptr_t) (pack_.expert_base() + x.offset + x.bytes) - a;
                    vec.resize((len + 4095) / 4096);
                    cached = mincore((void*) a, len, vec.data()) == 0;
                    for (size_t i = 0; cached && i < vec.size(); ++i) cached = vec[i] & 1;
                }
                uint64_t got = 0;
                while (got < a1 - a0) {
                    const ssize_t r = pread(fd_, buf + got, a1 - a0 - got, (off_t) (a0 + got));
                    if (r <= 0) throw std::runtime_error("ds41 expert stream: short read of experts.bin");
                    got += (uint64_t) r;
                }
                ++(cached ? n_cache_ : n_ssd_);
                src = buf + (x.offset - a0);
            }
            std::lock_guard<std::mutex> lk(mu_);
            src_[j] = src;
            read_[j] = 1;
            cv_.notify_all();
        }
    } catch (const std::exception& e) {
        fail(e.what());
    }
}

void ExpertStream::issuer() {
    try {
        ck(cudaSetDevice(device_), "cudaSetDevice");
        for (;;) {
            int64_t j;
            const uint8_t* src;
            std::pair<int, int> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                // in order: job j is read, and job j - slots is released (its "used" event then exists)
                cv_.wait(lk, [&] {
                    return stop_ || !error_.empty() ||
                           (issued_ < (int64_t) jobs_.size() && read_[issued_] && issued_ - slots_ < released_);
                });
                if (stop_ || !error_.empty()) return;
                j = issued_;
                src = src_[j];
                job = jobs_[j];
            }
            const int sl = (int) (j % slots_);
            if (j >= slots_) ck(cudaStreamWaitEvent(copy_, used_[sl], 0), "wait for the slot");   // on the GPU
            ck(cudaMemcpyAsync(ring_ + (size_t) sl * slot_bytes_, src, pack_.expert(job.first, job.second).bytes,
                               cudaMemcpyHostToDevice, copy_), "copy");
            ck(cudaEventRecord(copied_[sl], copy_), "record copy");
            ck(cudaEventRecord(dma_done_[j % n_host_], copy_), "record host buffer");
            std::lock_guard<std::mutex> lk(mu_);
            issued_ = j + 1;
            cv_.notify_all();
        }
    } catch (const std::exception& e) {
        fail(e.what());
    }
}

}  // namespace strata::ds41
