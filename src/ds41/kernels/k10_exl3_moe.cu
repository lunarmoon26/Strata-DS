// K10: EXL3 integer K1..K6 mul1 routed experts, using exllamav3's small-row GEMV.
#include "strata/ds41/kernels/k10_exl3_moe.hpp"

// CMake intentionally builds only this translation unit.
#include "../../../third_party/exllamav3_gpu/quant/exl3_gemv.cu"
#include "k10/pipeline.cuh"
#include "k10/tensor.cuh"

namespace strata::ds41::kernels {

void exl3_moe_tensor_up(const __half* x,int m,const int32_t* selected,const float* weights,int topk,
                        const Exl3Expert* experts,void* workspace,size_t bytes,int rank,cudaStream_t stream) {
    if(!x || !selected || !weights || !experts || !workspace || m<1 || m>8 || topk<1 || topk>6 ||
       rank<0 || rank>1 || bytes<k10::Workspace::bytes(m*topk))
        throw std::invalid_argument("TP2 expert up arguments");
    const int slots=m*topk;const k10::Workspace ws(workspace,slots);
    k10::tensor_input_had<<<dim3(2*slots,k10::H/128),32,0,stream>>>(x,selected,topk,experts,ws.input,ws.gu,ws.jobs);
    strata_exl3::gemv_mul1(ws.jobs,2*slots,k10::F/2,stream);
    k10::tensor_activate<<<dim3(slots,k10::F/256),32,0,stream>>>(selected,weights,experts,ws.gu,ws.down_input,ws.down,ws.jobs,rank);
}
void* exl3_moe_tensor_hidden(void* workspace,int m,int topk) {
    if(!workspace || m<1 || m>8 || topk<1 || topk>6) throw std::invalid_argument("TP2 expert hidden arguments");
    return k10::Workspace(workspace,m*topk).down_input;
}
void exl3_moe_tensor_down(int m,const int32_t* selected,int topk,const Exl3Expert* experts,
                          float* output,void* workspace,size_t bytes,int rank,cudaStream_t stream) {
    if(!selected || !experts || !output || !workspace || m<1 || m>8 || topk<1 || topk>6 ||
       rank<0 || rank>1 || bytes<k10::Workspace::bytes(m*topk))
        throw std::invalid_argument("TP2 expert down arguments");
    const k10::Workspace ws(workspace,m*topk);
    strata_exl3::gemv_mul1(ws.jobs,m*topk,k10::H/2,stream);
    k10::tensor_output<<<dim3(m,k10::H/256),32,0,stream>>>(selected,topk,experts,ws.down,output,rank);
}

void exl3_moe_decode(const __half* x, int m, const int32_t* sel, const float* w, int topk,
                     const Exl3Expert* experts, float* out, void* workspace,
                     size_t workspace_bytes, cudaStream_t stream, int uniform_bits) {
    // Guard before multiplication and before any launch. The device jobs use grid.y.
    if (m < 1 || m > 8 || topk < 1 || topk > 32767 / m ||
        !x || !sel || !w || !experts || !out || !workspace)
        throw std::invalid_argument("K10: invalid arguments");
    if (uniform_bits < 0 || uniform_bits > 6)
        throw std::invalid_argument("K10: uniform_bits must be 0..6");
    static const bool rate_kernel = [] {
        const char* value = std::getenv("DS41_EXPERIMENTAL_EXPERT_RATE_KERNEL");
        if (!value || !std::strcmp(value, "0")) return false;
        if (std::strcmp(value, "1"))
            throw std::invalid_argument("DS41_EXPERIMENTAL_EXPERT_RATE_KERNEL must be 0 or 1");
        return true;
    }();
    // Single-rate tables take the per-rate kernel: no mixed-rate dispatch, less
    // shared memory and no spill stack. rows=1 keeps the one-row MMA/reduction
    // order of the generic jobs kernel bit for bit.
    const int slots = m * topk;
    if (workspace_bytes < k10::Workspace::bytes(slots))
        throw std::invalid_argument("K10: insufficient workspace");
    if (reinterpret_cast<uintptr_t>(workspace) % 16 != 0)
        throw std::invalid_argument("K10: workspace must be 16-byte aligned");
    const k10::Workspace ws(workspace, slots);

    static const bool grouped = [] {
        const char* value = std::getenv("DS41_EXPERIMENTAL_GROUPED_EXPERTS");
        if (!value || !std::strcmp(value, "0")) return false;
        if (std::strcmp(value, "1"))
            throw std::invalid_argument("DS41_EXPERIMENTAL_GROUPED_EXPERTS must be 0 or 1");
        return true;
    }();
    const bool group_rows = grouped && m > 1 && topk <= 6;
    const bool per_rate = rate_kernel && uniform_bits >= 1 && !group_rows;

    k10::input_had<<<dim3(2 * slots, k10::H / 128), 32, 0, stream>>>
        (x, sel, topk, experts, ws.input, ws.gu, ws.jobs);
    if (per_rate) strata_exl3::gemv_mul1_rows(ws.jobs, 1, k10::F, uniform_bits, stream, 2 * slots);
    else if (group_rows) strata_exl3::gemv_mul1_grouped(ws.jobs, 2 * slots, k10::F, ws.groups, stream);
    else strata_exl3::gemv_mul1(ws.jobs, 2 * slots, k10::F, stream);
    k10::activate_down_had<<<dim3(slots, k10::F / 128), 32, 0, stream>>>
        (sel, w, experts, ws.gu, ws.down_input, ws.down, ws.jobs);
    if (per_rate) strata_exl3::gemv_mul1_rows(ws.jobs, 1, k10::H, uniform_bits, stream, slots);
    else if (group_rows) strata_exl3::gemv_mul1_grouped(ws.jobs, slots, k10::H, ws.groups, stream);
    else strata_exl3::gemv_mul1(ws.jobs, slots, k10::H, stream);
    k10::output_had_add<<<dim3(m, k10::H / 128), 32, 0, stream>>>
        (sel, topk, experts, ws.down, out);
}

}  // namespace strata::ds41::kernels
