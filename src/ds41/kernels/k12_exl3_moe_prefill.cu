// K12: grouped prefill with upstream EXL3 reconstruction and cuBLAS GEMM.
#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"

// CMake intentionally builds only this translation unit.
#include "../../../third_party/exllamav3_gpu/quant/reconstruct.cu"
#include "k12/pipeline.cuh"
#include "k12/blas.cuh"

namespace strata::ds41::kernels {

size_t exl3_moe_prefill_workspace_bytes(int max_rows, int max_groups) {
    if (max_rows < 0 || max_groups < 0)
        throw std::invalid_argument("K12: negative workspace limits");
    if (max_rows == 0 || max_groups == 0) return 0;
    return k12::Layout(max_rows).bytes();
}

void exl3_moe_prefill(const __half* x, const int32_t* tok, const float* w,
                      const int32_t* off, int n_groups, const Exl3Expert* experts,
                      float* out, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (n_groups < 0) throw std::invalid_argument("K12: negative group count");
    if (n_groups == 0) return;
    if (!off || off[0] < 0) throw std::invalid_argument("K12: invalid offsets");
    for (int g = 0; g < n_groups; ++g)
        if (off[g + 1] < off[g]) throw std::invalid_argument("K12: decreasing offsets");
    const int rows = off[n_groups] - off[0];
    if (rows == 0) return;  // Do not even initialize cuBLAS for an empty call.
    if (!x || !tok || !w || !experts || !out || !workspace)
        throw std::invalid_argument("K12: null argument");
    const k12::Layout layout(rows);
    if (workspace_bytes < layout.bytes())
        throw std::invalid_argument("K12: insufficient workspace");
    const k12::Workspace ws(workspace, layout);

    auto& registry = k12::blas_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    int device;
    k12::check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    const cublasHandle_t handle = registry.get(device);
    k12::check_blas(cublasSetStream(handle, stream), "cublasSetStream");
    // SetStream resets the library workspace. Restore caller-owned scratch
    // afterward, preventing cuBLAS allocation nodes in captured graphs.
    k12::check_blas(cublasSetWorkspace(handle, ws.blas, k12::BLAS_BYTES), "cublasSetWorkspace");

    for (int g = 0; g < n_groups; ++g) {
        if (off[g] == off[g + 1]) continue;
        const Exl3Expert* expert = experts + g;
        k12::prepare<<<1, 1, 0, stream>>>(expert, ws.trellis);
        strata_exl3::reconstruct_mul1(ws.matrices, ws.trellis, k12::H, k12::F, stream);
        strata_exl3::reconstruct_mul1(ws.matrices + k12::MATRIX_ELEMENTS, ws.trellis + 1,
                                          k12::H, k12::F, stream);
        strata_exl3::reconstruct_mul1(ws.matrices + 2 * k12::MATRIX_ELEMENTS, ws.trellis + 2,
                                          k12::F, k12::H, stream);
        k12::check_cuda(cudaPeekAtLastError(), "reconstruct launches");

        for (int first = off[g]; first < off[g + 1];) {
            const int remaining = off[g + 1] - first;
            const int count = remaining < layout.rows ? remaining : layout.rows;
            k12::input_had<<<dim3(2 * count, k12::H / 128), 32, 0, stream>>>
                (x, tok + first, expert, count, ws.input);
            k12::check_cuda(cudaPeekAtLastError(), "input Hadamard launch");
            k12::gemm(handle, ws.input, ws.matrices, ws.gu, count, k12::H, k12::F);
            k12::gemm(handle, ws.input + size_t(count) * k12::H,
                       ws.matrices + k12::MATRIX_ELEMENTS, ws.gu + size_t(count) * k12::F,
                       count, k12::H, k12::F);
            k12::activate_down_had<<<dim3(count, k12::F / 128), 32, 0, stream>>>
                (w + first, expert, count, ws.gu, ws.down_input);
            k12::check_cuda(cudaPeekAtLastError(), "activation launch");
            k12::gemm(handle, ws.down_input, ws.matrices + 2 * k12::MATRIX_ELEMENTS,
                       ws.down, count, k12::F, k12::H);
            k12::output_had_add<<<dim3(count, k12::H / 128), 32, 0, stream>>>
                (tok + first, expert, ws.down, out);
            k12::check_cuda(cudaPeekAtLastError(), "scatter launch");
            first += count;
        }
    }
}

}  // namespace strata::ds41::kernels
