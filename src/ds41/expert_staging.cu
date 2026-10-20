#include "strata/ds41/expert_staging.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace strata::ds41 {
namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 staging: ") + what + ": " + cudaGetErrorString(e));
}

// All blocks stream each contiguous blob. A fixed grid also handles a device count of zero.
// Pack slots and staging slots are aligned. Copy only complete uint4s, then the exact byte tail.
__global__ void stage_experts_k(const ExpertCopy* jobs, const int* count) {
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
} // namespace

ExpertStaging::ExpertStaging(int max_jobs, size_t max_expert_bytes)
    : capacity_(max_jobs), stride_(0) {
    if (max_jobs < 1 || max_expert_bytes == 0 ||
        max_expert_bytes > std::numeric_limits<size_t>::max() - 255)
        throw std::invalid_argument("ExpertStaging: bad capacity");
    stride_ = (max_expert_bytes + 255) / 256 * 256;
    if (stride_ > std::numeric_limits<size_t>::max() / size_t(max_jobs))
        throw std::invalid_argument("ExpertStaging: size overflow");
    try {
        ck(cudaMalloc(&data_, stride_ * max_jobs), "arena");
        ck(cudaMalloc(&jobs_, sizeof(ExpertCopy) * max_jobs), "jobs");
        ck(cudaMalloc(&count_, sizeof(int)), "count");
        ck(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking), "copy stream");
        ck(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming), "ready event");
        ck(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming), "done event");
    } catch (...) {
        if (done_) cudaEventDestroy(done_);
        if (ready_) cudaEventDestroy(ready_);
        if (copy_) cudaStreamDestroy(copy_);
        cudaFree(count_);
        cudaFree(jobs_);
        cudaFree(data_);
        throw;
    }
}

ExpertStaging::~ExpertStaging() {
    cudaEventDestroy(done_);
    cudaEventDestroy(ready_);
    cudaStreamDestroy(copy_);
    cudaFree(count_);
    cudaFree(jobs_);
    cudaFree(data_);
}

void ExpertStaging::fork_copy(cudaStream_t main) {
    ck(cudaEventRecord(ready_, main), "record publish");
    ck(cudaStreamWaitEvent(copy_, ready_, 0), "wait publish");
    stage_experts_k<<<68, 256, 0, copy_>>>(jobs_, count_);
    ck(cudaGetLastError(), "copy launch");
    ck(cudaEventRecord(done_, copy_), "record copy");
}

void ExpertStaging::join(cudaStream_t main) {
    ck(cudaStreamWaitEvent(main, done_, 0), "join copy");
}
} // namespace strata::ds41
