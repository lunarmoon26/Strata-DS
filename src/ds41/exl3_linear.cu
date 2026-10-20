#include "strata/ds41/exl3_linear.hpp"
#include "../../third_party/exllamav3_gpu/quant/exl3_gemv.cuh"
#include "../../third_party/exllamav3_gpu/util.cuh"
#include "../../third_party/exllamav3_gpu/quant/hadamard_inner.cuh"
#include "../../third_party/exllamav3_gpu/quant/reconstruct.cuh"
#include "kernels/k12/blas.cuh"
#include <stdexcept>

namespace strata::ds41 {
namespace {
using Job = strata_exl3::GemvJob;
constexpr float scale = 0.088388347648f;
constexpr size_t blas_bytes = 4ull << 20;
constexpr size_t job_bytes = (sizeof(Job) + 15) & ~size_t(15);

__global__ void half_input(const __nv_bfloat16* x, half* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(__bfloat162float(x[i]));
}

__global__ void prepare(const __nv_bfloat16* x, kernels::Exl3Proj w,
                         half* a, float* c, Job* job) {
    const int off = blockIdx.y * 128;
    __shared__ __align__(16) half input[128];
    for (int i = threadIdx.x; i < 128; i += 32)
        input[i] = __float2half_rn(__bfloat162float(x[off + i]));
    __syncwarp();
    had_hf_r_128_inner<true, false>(input, a + off, w.suh, scale);
    if (off == 0 && threadIdx.x == 0)
        *job = Job{a, w.trellis, c, w.k, w.n, w.tile_w / 16, w.codebook};
}

__global__ void finish(const float* c, const half* svh, __nv_bfloat16* y, float* logits) {
    const int off = blockIdx.y * 128;
    __shared__ __align__(16) float output[128];
    had_ff_r_128_inner<false, true>(c + off, output, svh, scale);
    __syncwarp();
    for (int i = threadIdx.x; i < 128; i += 32) {
        const auto v = __float2bfloat16_rn(output[i]);
        if (y) y[off + i] = v;
        else logits[off + i] = __bfloat162float(v);
    }
}

__global__ void prepare_rows(const __nv_bfloat16* x, kernels::Exl3Proj w,
                             half* a, int stride, strata_exl3::ReconstructJob* job) {
    const int off = blockIdx.y * 128, row = blockIdx.x;
    __shared__ __align__(16) half input[128];
    for (int i = threadIdx.x; i < 128; i += 32)
        input[i] = __float2half_rn(__bfloat162float(x[size_t(row) * stride + off + i]));
    __syncwarp();
    had_hf_r_128_inner<true, false>(input, a + size_t(row) * w.k + off, w.suh, scale);
    if (row == 0 && off == 0 && threadIdx.x == 0) *job = {w.trellis, w.tile_w / 16};
}

__global__ void finish_rows(const float* c, const half* svh, __nv_bfloat16* y, int n, int stride) {
    const int off = blockIdx.y * 128, row = blockIdx.x;
    __shared__ __align__(16) float output[128];
    had_ff_r_128_inner<false, true>(c + size_t(row) * n + off, output, svh, scale);
    __syncwarp();
    for (int i = threadIdx.x; i < 128; i += 32)
        y[size_t(row) * stride + off + i] = __float2bfloat16_rn(output[i]);
}

__global__ void prepare_small_rows(const __nv_bfloat16* x, kernels::Exl3Proj w,
                                   half* a, float* c, Job* jobs, int stride) {
    const int off = blockIdx.y * 128, row = blockIdx.x;
    __shared__ __align__(16) half input[128];
    for (int i = threadIdx.x; i < 128; i += 32)
        input[i] = __float2half_rn(__bfloat162float(x[size_t(row) * stride + off + i]));
    __syncwarp();
    had_hf_r_128_inner<true, false>(input, a + size_t(row) * w.k + off, w.suh, scale);
    if (off == 0 && threadIdx.x == 0)
        jobs[row] = Job{a + size_t(row) * w.k, w.trellis, c + size_t(row) * w.n,
                        w.k, w.n, w.tile_w / 16, w.codebook};
}

__global__ void finish_small_logits(const float* c, const half* svh, float* out, int n, int stride) {
    const int off = blockIdx.y * 128, row = blockIdx.x;
    __shared__ __align__(16) float output[128];
    had_ff_r_128_inner<false, true>(c + size_t(row) * n + off, output, svh, scale);
    __syncwarp();
    for (int i = threadIdx.x; i < 128; i += 32)
        out[size_t(row) * stride + off + i] = __bfloat162float(__float2bfloat16_rn(output[i]));
}
}  // namespace

size_t exl3_small_rows_workspace_bytes(int rows, int k, int n) {
    exl3_linear_workspace_bytes(k, n);
    if (rows < 1 || rows > 8) throw std::invalid_argument("EXL3 small rows: expected 1..8 rows");
    return ((sizeof(Job) * rows + 15) & ~size_t(15)) + size_t(rows) * (k * sizeof(half) + n * sizeof(float));
}

void exl3_linear_small_rows(const __nv_bfloat16* x, const kernels::Exl3Proj& w,
                            __nv_bfloat16* y, float* logits, int rows, int xs, int ys,
                            void* workspace, size_t bytes, cudaStream_t stream) {
    if (!x || (!y == !logits) || !workspace || reinterpret_cast<uintptr_t>(workspace) % 16 ||
        xs < w.k || ys < w.n || !w.trellis || !w.suh || !w.svh || w.codebook != 2 ||
        w.tile_w < 16 || w.tile_w > 96 || w.tile_w % 16 ||
        bytes < exl3_small_rows_workspace_bytes(rows, w.k, w.n))
        throw std::invalid_argument("EXL3 small rows: invalid arguments/workspace");
    auto* jobs = static_cast<Job*>(workspace);
    const size_t table = (sizeof(Job) * rows + 15) & ~size_t(15);
    auto* a = reinterpret_cast<half*>(static_cast<char*>(workspace) + table);
    auto* c = reinterpret_cast<float*>(a + size_t(rows) * w.k);
    prepare_small_rows<<<dim3(rows, w.k / 128), 32, 0, stream>>>(x, w, a, c, jobs, xs);
    strata_exl3::gemv_mul1_rows(jobs,rows,w.n,w.tile_w/16,stream);
    if (y) finish_rows<<<dim3(rows, w.n / 128), 32, 0, stream>>>(c, w.svh, y, w.n, ys);
    else finish_small_logits<<<dim3(rows, w.n / 128), 32, 0, stream>>>(c, w.svh, logits, w.n, ys);
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("EXL3 small rows launch failed");
}

size_t exl3_linear_workspace_bytes(int k, int n) {
    if (k <= 0 || n <= 0 || k % 128 || n % 128 || k > 32768 || n > 129280)
        throw std::invalid_argument("EXL3 dense: unsupported dimensions");
    return job_bytes + size_t(k) * sizeof(half) + size_t(n) * sizeof(float);
}

void exl3_linear(const __nv_bfloat16* x, const kernels::Exl3Proj& w,
                 __nv_bfloat16* y, float* logits, void* workspace,
                 size_t bytes, cudaStream_t stream) {
    if (!x || (!y == !logits) || !workspace || reinterpret_cast<uintptr_t>(workspace) % 16 ||
        !w.trellis || !w.suh || !w.svh || w.codebook != 2 ||
        w.tile_w % 16 || w.tile_w < 16 || w.tile_w > 96 ||
        bytes < exl3_linear_workspace_bytes(w.k, w.n))
        throw std::invalid_argument("EXL3 dense: invalid projection, output or workspace");
    auto* job = static_cast<Job*>(workspace);
    auto* a = reinterpret_cast<half*>(static_cast<char*>(workspace) + job_bytes);
    auto* c = reinterpret_cast<float*>(a + w.k);
    prepare<<<dim3(1, w.k / 128), 32, 0, stream>>>(x, w, a, c, job);
    strata_exl3::gemv_mul1(job, 1, w.n, stream);
    finish<<<dim3(1, w.n / 128), 32, 0, stream>>>(c, w.svh, y, logits);
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("EXL3 dense launch failed");
}

void exl3_input_half(const __nv_bfloat16* x, uint16_t* y, int n, cudaStream_t stream) {
    half_input<<<(n + 255) / 256, 256, 0, stream>>>(x, reinterpret_cast<half*>(y), n);
}

size_t exl3_rows_workspace_bytes(int rows, int k, int n) {
    exl3_linear_workspace_bytes(k, n);
    if (rows < 1 || rows > 4096 || n > 32768) throw std::invalid_argument("EXL3 rows: invalid shape");
    return 256 + size_t(k) * n * 2 + size_t(rows) * k * 2 + size_t(rows) * n * 4 + blas_bytes;
}

void exl3_linear_rows(const __nv_bfloat16* x, const kernels::Exl3Proj& w,
                      __nv_bfloat16* y, int rows, int xs, int ys,
                      void* workspace, size_t bytes, cudaStream_t stream) {
    if (!x || !y || !workspace || reinterpret_cast<uintptr_t>(workspace) % 256 || xs < w.k || ys < w.n ||
        w.codebook != 2 || w.tile_w < 16 || w.tile_w > 96 || w.tile_w % 16 ||
        !w.trellis || !w.suh || !w.svh || bytes < exl3_rows_workspace_bytes(rows, w.k, w.n))
        throw std::invalid_argument("EXL3 rows: invalid arguments/workspace");
    auto* job = static_cast<strata_exl3::ReconstructJob*>(workspace);
    auto* matrix = reinterpret_cast<half*>(static_cast<char*>(workspace) + 256);
    auto* a = matrix + size_t(w.k) * w.n;
    auto* c = reinterpret_cast<float*>(a + size_t(rows) * w.k);
    auto* blas = c + size_t(rows) * w.n;
    prepare_rows<<<dim3(rows, w.k / 128), 32, 0, stream>>>(x, w, a, xs, job);
    strata_exl3::reconstruct_mul1(matrix, job, w.k, w.n, stream);
    auto& registry = kernels::k12::blas_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    int device = 0; kernels::k12::check_cuda(cudaGetDevice(&device), "rows device");
    const auto handle = registry.get(device);
    kernels::k12::check_blas(cublasSetStream(handle, stream), "rows stream");
    kernels::k12::check_blas(cublasSetWorkspace(handle, blas, blas_bytes), "rows workspace");
    kernels::k12::gemm(handle, a, matrix, c, rows, w.k, w.n);
    finish_rows<<<dim3(rows, w.n / 128), 32, 0, stream>>>(c, w.svh, y, w.n, ys);
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("EXL3 rows launch failed");
}
}  // namespace strata::ds41
