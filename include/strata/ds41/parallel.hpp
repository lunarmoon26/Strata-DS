// include/strata/ds41/parallel.hpp - run n parts of a job on n threads (part 0 on the calling thread), safely.
//
// A thread that cannot start (std::system_error, e.g. at the process's thread limit) does not end the program: its part
// and every later part run on the calling thread. Every started thread is joined before run_parallel returns or throws;
// the first exception of a part is rethrown then.
//
// ThreadPool: the same contract, with helper threads started once and reused. Starting threads for every call costs
// time: one expert read from the laptop's SSD (1 MiB parts on 8 threads) took 0.26 ms per MB with new threads and
// 0.216 ms with persistent ones (ds41 adaptive RAM tier, 2026-10-09).
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <vector>

namespace strata::ds41 {
namespace detail {
/// tests: the start of thread number k (0 = the first one started) throws; -1 = never
inline std::atomic<int>& spawn_fault() {
    static std::atomic<int> k{-1};
    return k;
}
/// tests: ThreadPool's start of helper k throws std::bad_alloc; -1 = never
inline std::atomic<int>& spawn_alloc_fault() {
    static std::atomic<int> k{-1};
    return k;
}
}  // namespace detail

template <class Fn>
void run_parallel(size_t n, Fn fn) {
    if (n == 0) return;
    std::exception_ptr error;
    std::mutex mu;
    auto part = [&](size_t i) {
        try {
            fn(i);
        } catch (...) {
            std::lock_guard<std::mutex> lk(mu);
            if (!error) error = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    struct Join {
        std::vector<std::thread>& p;
        ~Join() {
            for (auto& t : p)
                if (t.joinable()) t.join();
        }
    } join{pool};
    size_t i = 1;
    try {
        pool.reserve(n - 1);
        for (; i < n; ++i) {
            if (detail::spawn_fault().load() == (int) (i - 1))
                throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
            pool.emplace_back(part, i);
        }
    } catch (const std::system_error&) {
        for (; i < n; ++i) part(i);   // the parts no thread took
    }
    part(0);
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

/// Persistent helpers for run_parallel's contract: run(n, fn) calls fn(0) .. fn(n - 1), each once; part 0 and the parts
/// no helper takes (n > size()) run on the calling thread; the first exception of a part is rethrown after every part
/// ended. A helper that cannot start (std::system_error) leaves its parts to the caller. One run at a time.
class ThreadPool {
public:
    /// `threads` counts the caller: threads - 1 helpers
    explicit ThreadPool(size_t threads) {
        try {
            helpers_.reserve(threads > 1 ? threads - 1 : 0);
            for (size_t k = 0; k + 1 < threads; ++k) {
                try {
                    if (detail::spawn_fault().load() == (int) k)
                        throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
                    if (detail::spawn_alloc_fault().load() == (int) k) throw std::bad_alloc();
                    helpers_.emplace_back([this, k] { loop(k + 1); });
                } catch (const std::system_error&) {
                    break;
                }
            }
        } catch (...) {   // e.g. bad_alloc: the destructor does not run, so stop and join the helpers that started
            stop_helpers();
            throw;
        }
    }
    ~ThreadPool() { stop_helpers(); }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// the caller and the helpers that started
    size_t size() const { return helpers_.size() + 1; }

    template <class Fn>
    void run(size_t n, Fn fn) {
        if (n == 0) return;
        std::function<void(size_t)> job = [&](size_t i) { fn(i); };
        const size_t h = std::min(n - 1, helpers_.size());
        {
            std::lock_guard<std::mutex> lk(mu_);
            error_ = nullptr;
            job_ = &job;
            n_ = n;
            pending_.store(h, std::memory_order_relaxed);
            ++gen_;
        }
        if (h > 0) cv_.notify_all();
        part(job, 0);
        for (size_t i = h + 1; i < n; ++i) part(job, i);
        for (int spins = 0; pending_.load(std::memory_order_acquire) != 0; ++spins)
            if (spins > 4096) std::this_thread::yield();
        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = nullptr;
            error = error_;
            error_ = nullptr;
        }
        if (error) std::rethrow_exception(error);
    }

private:
    void stop_helpers() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& t : helpers_)
            if (t.joinable()) t.join();
    }
    void part(std::function<void(size_t)>& job, size_t i) {
        try {
            job(i);
        } catch (...) {
            std::lock_guard<std::mutex> lk(mu_);
            if (!error_) error_ = std::current_exception();
        }
    }
    void loop(size_t idx) {
        uint64_t seen = 0;
        while (true) {
            std::function<void(size_t)>* job;
            size_t n;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                job = job_;
                n = n_;
            }
            if (job == nullptr || idx >= n) continue;   // this run has fewer parts
            part(*job, idx);
            pending_.fetch_sub(1, std::memory_order_release);
        }
    }

    std::vector<std::thread> helpers_;
    std::mutex mu_;
    std::condition_variable cv_;
    uint64_t gen_ = 0;
    bool stop_ = false;
    std::function<void(size_t)>* job_ = nullptr;
    size_t n_ = 0;
    std::atomic<size_t> pending_{0};
    std::exception_ptr error_;
};

}  // namespace strata::ds41
