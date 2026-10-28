#pragma once

#include "strata/ds41/device_scope.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::ds41 {

// File -> bounded pinned staging -> device, preserving every byte. No mmap or
// weight-sized host allocation. Used only by resident layer-split/TP2 paths.
class BoundedUpload {
public:
    static constexpr size_t kStagingBytes = 4ull << 20;
    explicit BoundedUpload(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat st{};
        if (fd_ < 0 || fstat(fd_, &st) || st.st_size <= 0) {
            if (fd_ >= 0) close(fd_);
            throw std::runtime_error("ds41: cannot size upload file " + path);
        }
        size_ = static_cast<uint64_t>(st.st_size);
        try { device_check(cudaMallocHost(&host_, kStagingBytes), "upload staging"); }
        catch (...) { close(fd_); throw; }
    }
    ~BoundedUpload() {
        cudaFreeHost(host_);
        close(fd_);
    }
    BoundedUpload(const BoundedUpload&) = delete;
    BoundedUpload& operator=(const BoundedUpload&) = delete;
    void validate(uint64_t offset, uint64_t bytes) const {
        if (offset > size_ || bytes > size_ - offset)
            throw std::runtime_error("ds41: upload range past end of file");
    }
    /// Borrow a complete immutable pinned payload for synchronous uploads only.
    /// The owner keeps it alive until clear_source(); no cached file pointer.
    void set_source(const uint8_t* data, uint64_t offset, size_t bytes) {
        validate(offset, bytes);
        if (!data || !bytes || source_) throw std::invalid_argument("invalid bounded upload RAM source");
        source_ = data; source_offset_ = offset; source_bytes_ = bytes;
    }
    void clear_source() noexcept { source_ = nullptr; source_bytes_ = 0; }
    void read_host(void* destination, uint64_t offset, size_t bytes) {
        validate(offset, bytes);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t n = pread(fd_, static_cast<char*>(destination) + done, bytes - done, offset + done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("ds41: short read during RAM cache fill");
            done += size_t(n);
        }
        posix_fadvise(fd_, offset, bytes, POSIX_FADV_DONTNEED);
    }
    uint32_t scalar_u32(uint64_t offset) const {
        validate(offset, 4);
        if (source_) {
            const auto* p = source_at(offset, 4);
            return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
        }
        unsigned char bytes[4];
        size_t done = 0;
        while (done < sizeof(bytes)) {
            const ssize_t n = pread(fd_, bytes + done, sizeof(bytes) - done, offset + done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("ds41: short read of codebook marker");
            done += static_cast<size_t>(n);
        }
        return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
    }
    void copy(void* destination, uint64_t offset, uint64_t bytes) {
        validate(offset, bytes);
        if (source_) {
            device_check(cudaMemcpy(destination, source_at(offset, bytes), bytes, cudaMemcpyHostToDevice), "RAM expert upload");
            return;
        }
        for (uint64_t done = 0; done < bytes;) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(kStagingBytes, bytes - done));
            size_t got = 0;
            while (got < n) {
                const ssize_t r = pread(fd_, static_cast<char*>(host_) + got, n - got, offset + done + got);
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) throw std::runtime_error("ds41: short read during upload");
                got += static_cast<size_t>(r);
            }
            device_check(cudaMemcpy(static_cast<char*>(destination) + done, host_, n, cudaMemcpyHostToDevice),
                         "bounded upload");
            posix_fadvise(fd_, offset + done, n, POSIX_FADV_DONTNEED);
            done += n;
        }
    }
    /// Read bounded groups of complete source rows, upload only the requested
    /// columns. EXL3 trellis tiles and embedding channels stay byte-identical.
    void copy_columns(void* destination, uint64_t offset, size_t rows,
                      size_t row_bytes, size_t first, size_t width) {
        if (!destination || !rows || !row_bytes || row_bytes > kStagingBytes ||
            first > row_bytes || !width || width > row_bytes-first || rows > SIZE_MAX/row_bytes)
            throw std::invalid_argument("ds41: invalid bounded column upload");
        validate(offset, uint64_t(rows)*row_bytes);
        if (source_) {
            device_check(cudaMemcpy2D(destination, width, source_at(offset, rows * row_bytes) + first,
                                     row_bytes, width, rows, cudaMemcpyHostToDevice), "RAM expert shard upload");
            return;
        }
        const size_t batch = kStagingBytes / row_bytes;
        for (size_t done=0; done<rows;) {
            const size_t count=std::min(batch,rows-done), bytes=count*row_bytes;
            size_t got=0;
            while(got<bytes) {
                const ssize_t n=pread(fd_,static_cast<char*>(host_)+got,bytes-got,offset+done*row_bytes+got);
                if(n<0 && errno==EINTR) continue;
                if(n<=0) throw std::runtime_error("ds41: short read during column upload");
                got+=size_t(n);
            }
            device_check(cudaMemcpy2D(static_cast<char*>(destination)+done*width,width,
                                      static_cast<char*>(host_)+first,row_bytes,width,count,cudaMemcpyHostToDevice),
                         "bounded column upload");
            posix_fadvise(fd_,offset+done*row_bytes,bytes,POSIX_FADV_DONTNEED);
            done+=count;
        }
    }
private:
    const uint8_t* source_at(uint64_t offset, uint64_t bytes) const {
        if (offset < source_offset_ || offset - source_offset_ > source_bytes_ ||
            bytes > source_bytes_ - (offset - source_offset_))
            throw std::runtime_error("ds41: upload range outside pinned expert payload");
        return source_ + (offset - source_offset_);
    }
    int fd_ = -1;
    uint64_t size_ = 0;
    void* host_ = nullptr;
    const uint8_t* source_ = nullptr;
    uint64_t source_offset_ = 0;
    size_t source_bytes_ = 0;
};

}  // namespace strata::ds41
