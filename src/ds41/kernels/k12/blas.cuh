#pragma once

#include <cublas_v2.h>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace strata::ds41::kernels::k12 {

inline void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("K12: ") + where + ": " + cudaGetErrorString(status));
}
inline void check_blas(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("K12: ") + where + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
}

// Process-lifetime handles, created once per device outside capture. Retaining
// them avoids a cudaDeviceSynchronize in cublasDestroy on an inference path.
// The mutex protects host-side handle configuration/enqueue, not GPU execution.
// Independent concurrent calls must have independent caller workspaces.
struct BlasRegistry {
    std::mutex mutex;
    std::map<int, cublasHandle_t> handles;

    cublasHandle_t get(int device) {
        const auto it = handles.find(device);
        if (it != handles.end()) return it->second;
        cublasHandle_t handle = nullptr;
        check_blas(cublasCreate(&handle), "cublasCreate (first call per device must precede capture)");
        try {
            check_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
            check_blas(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION),
                       "cublasSetMathMode");
            handles.emplace(device, handle);
        } catch (...) {
            cublasDestroy(handle);
            throw;
        }
        return handle;
    }
};

inline BlasRegistry& blas_registry() {
    static BlasRegistry registry;
    return registry;
}

inline void gemm(cublasHandle_t handle, const half* a, const half* b,
                 float* c, int rows, int k, int n) {
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major C[rows,n] = A[rows,k] B[k,n]. Viewed column-major:
    // C^T[n,rows] = B^T[n,k] A^T[k,rows], with no transposition flags.
    // Upstream hgemm uses FP32 compute as well. K10's FP16 partial folds
    // differ, so the fixed GPU comparison is still required.
    check_blas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, rows, k,
                           &alpha, b, CUDA_R_16F, n, a, CUDA_R_16F, k,
                           &beta, c, CUDA_R_32F, n,
                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
}

}  // namespace strata::ds41::kernels::k12
