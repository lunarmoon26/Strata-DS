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
// weight-sized host allocation. Used only by the resident split path.
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
    uint32_t scalar_u32(uint64_t offset) const {
        validate(offset, 4);
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
private:
    int fd_ = -1;
    uint64_t size_ = 0;
    void* host_ = nullptr;
};

}  // namespace strata::ds41
