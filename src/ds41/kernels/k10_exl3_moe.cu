// K10: EXL3 integer K1..K6 mul1 routed experts, using exllamav3's small-row GEMV.
#include "strata/ds41/kernels/k10_exl3_moe.hpp"

// CMake intentionally builds only this translation unit.
#include "../../../third_party/exllamav3_gpu/quant/exl3_gemv.cu"
#include "k10/pipeline.cuh"

namespace strata::ds41::kernels {

void exl3_moe_decode(const __half* x, int m, const int32_t* sel, const float* w, int topk,
                     const Exl3Expert* experts, float* out, void* workspace,
                     size_t workspace_bytes, cudaStream_t stream) {
    // Guard before multiplication and before any launch. The device jobs use grid.y.
    if (m < 1 || m > 8 || topk < 1 || topk > 32767 / m ||
        !x || !sel || !w || !experts || !out || !workspace)
        throw std::invalid_argument("K10: invalid arguments");
    const int slots = m * topk;
    if (workspace_bytes < k10::Workspace::bytes(slots))
        throw std::invalid_argument("K10: insufficient workspace");
    if (reinterpret_cast<uintptr_t>(workspace) % 16 != 0)
        throw std::invalid_argument("K10: workspace must be 16-byte aligned");
    const k10::Workspace ws(workspace, slots);

    k10::input_had<<<dim3(2 * slots, k10::H / 128), 32, 0, stream>>>
        (x, sel, topk, experts, ws.input, ws.gu, ws.jobs);
    strata_exl3::gemv_mul1(ws.jobs, 2 * slots, k10::F, stream);
    k10::activate_down_had<<<dim3(slots, k10::F / 128), 32, 0, stream>>>
        (sel, w, experts, ws.gu, ws.down_input, ws.down, ws.jobs);
    strata_exl3::gemv_mul1(ws.jobs, slots, k10::H, stream);
    k10::output_had_add<<<dim3(m, k10::H / 128), 32, 0, stream>>>
        (sel, topk, experts, ws.down, out);
}

}  // namespace strata::ds41::kernels
