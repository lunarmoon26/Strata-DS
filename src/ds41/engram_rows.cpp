// src/ds41/engram_rows.cpp - see include/strata/ds41/engram_rows.hpp.
#include "strata/ds41/engram_rows.hpp"
#include "strata/ds41/parallel.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::ds41 {

namespace {
constexpr uint64_t kAlign = platform::DirectFile::alignment();
constexpr uint32_t kBlock = 2 * kAlign;   // a row of at most 4 KiB spans at most two aligned blocks
}  // namespace

EngramRows::EngramRows(const std::vector<Table>& tables, int max_rows, int row_bytes, int scale_bytes, int io_threads)
    : tables_(tables), max_rows_(max_rows), row_bytes_(row_bytes), scale_bytes_(scale_bytes) {
    // The upstream reader owns its queue/thread policy. Do not import the old
    // laptop fork's per-reader tuning API into upstream's shared DirectFile.
    (void) io_threads;
    if (max_rows < 1 || row_bytes < 1 || scale_bytes < 1 || (uint64_t) row_bytes > kAlign)
        throw std::invalid_argument("EngramRows: bad shape");
    for (const Table& t : tables_) {
        auto f = std::make_unique<platform::DirectFile>();
        std::string err;
        if (f->open(t.path, err)) {
            files_.push_back(std::move(f));
            fds_.push_back(-1);
            continue;
        }
        const int fd = ::open(t.path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("EngramRows: cannot open " + t.path);
        if (direct_)
            std::fprintf(stderr, "ds41: %s refuses O_DIRECT; engram rows use plain reads (they enter the file cache)\n",
                         t.path.c_str());
        direct_ = false;
        files_.push_back(nullptr);
        fds_.push_back(fd);
    }
    const size_t n_req = tables_.size() * (size_t) max_rows * 2;
    buf_ = (uint8_t*) platform::DirectFile::alloc_aligned(n_req * kBlock);
    if (!buf_) throw std::runtime_error("EngramRows: cannot allocate read buffers");
    reqs_.reserve(n_req);
}

EngramRows::~EngramRows() {
    files_.clear();
    for (int fd : fds_)
        if (fd >= 0) ::close(fd);
    platform::DirectFile::free_aligned(buf_);
}

void EngramRows::read(const std::vector<const int64_t*>& ids, int n_rows, const std::vector<uint8_t*>& w_out,
                      const std::vector<uint8_t*>& s_out) {
    if (failed_) throw std::runtime_error("EngramRows::read: failed state; create a new reader");
    if (ids.size() != tables_.size() || w_out.size() != tables_.size() || s_out.size() != tables_.size() ||
        n_rows < 0 || n_rows > max_rows_)
        throw std::invalid_argument("EngramRows::read: bad arguments");
    std::string err;
    // every request of every table first, then the waits: all reads are in flight together
    reqs_.clear();
    std::vector<size_t> first(tables_.size() + 1, 0);
    for (size_t t = 0; t < tables_.size(); ++t) {
        first[t] = reqs_.size();
        for (int i = 0; i < n_rows; ++i) {
            const int64_t id = ids[t][i];
            if (id < 0) throw std::invalid_argument("EngramRows::read: negative row");
            auto add = [&](uint64_t off, uint32_t bytes, uint8_t* dst) {
                const uint64_t a = off / kAlign * kAlign;
                const uint64_t end = (off + bytes + kAlign - 1) / kAlign * kAlign;
                reqs_.push_back(Req{a, (uint32_t) (end - a), (uint32_t) (off - a), bytes, dst});
            };
            add(tables_[t].weight_offset + (uint64_t) id * row_bytes_, row_bytes_, w_out[t] + (size_t) i * row_bytes_);
            add(tables_[t].scale_offset + (uint64_t) id * scale_bytes_, scale_bytes_,
                s_out[t] + (size_t) i * scale_bytes_);
        }
    }
    first[tables_.size()] = reqs_.size();
    // Any exception after submission poisons the reader. Destruction joins the I/O workers
    // before it frees their buffers. No later read can consume old completions.
    failed_ = true;
    for (size_t t = 0; t < tables_.size(); ++t) {
        if (!files_[t]) continue;
        for (size_t r = first[t]; r < first[t + 1]; ++r)
            if (!files_[t]->submit(reqs_[r].aligned, buf_ + r * kBlock, reqs_[r].length, r, err))
                throw std::runtime_error("EngramRows: " + err);
    }
    for (size_t t = 0; t < tables_.size(); ++t) {
        if (files_[t]) {
            platform::Completion c[64];
            size_t got = 0;
            bool ok = true;
            while (got < first[t + 1] - first[t]) {
                const int n = files_[t]->wait(c, 64, -1);
                for (int k = 0; k < n; ++k) {
                    if (c[k].tag == platform::DirectFile::WAKE_TAG) continue;
                    const Req& q = reqs_[c[k].tag];
                    if (!c[k].ok || c[k].bytes < q.skip + q.bytes) ok = false;
                    else std::memcpy(q.dst, buf_ + c[k].tag * kBlock + q.skip, q.bytes);
                    ++got;
                }
            }
            if (!ok) throw std::runtime_error("EngramRows: a direct read failed: " + tables_[t].path);
        } else {
            // plain reads (through the file cache), in parallel: a row the cache does not hold is a random SSD read
            const size_t n = first[t + 1] - first[t];
            const size_t workers = std::min<size_t>(std::max<size_t>(n / 4, 1), 64);
            std::atomic<bool> bad{false};
            run_parallel(workers, [&](size_t w) {
                for (size_t r = first[t] + w; r < first[t + 1]; r += workers) {
                    const Req& q = reqs_[r];
                    if (pread(fds_[t], q.dst, q.bytes, (off_t) (q.aligned + q.skip)) != (ssize_t) q.bytes) bad = true;
                }
            });
            if (bad) throw std::runtime_error("EngramRows: a read failed: " + tables_[t].path);
        }
    }
    failed_ = false;
}

}  // namespace strata::ds41
