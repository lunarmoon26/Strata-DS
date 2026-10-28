#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cassert>
#include "workspace.hpp"
#include "../../../../third_party/exllamav3_gpu/quant/hadamard_inner.cuh"

namespace strata::ds41::kernels::k12 {

struct Workspace {
    strata_exl3::ReconstructJob* trellis;
    half* matrices;
    half* input;
    float* gu;
    half* down_input;
    float* down;
    void* blas;

    Workspace(void* p, const Layout& l) {
        char* base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + 255) & ~uintptr_t(255));
        trellis = reinterpret_cast<strata_exl3::ReconstructJob*>(base + l.trellis);
        matrices = reinterpret_cast<half*>(base + l.matrices);
        input = reinterpret_cast<half*>(base + l.input);
        gu = reinterpret_cast<float*>(base + l.gu);
        down_input = reinterpret_cast<half*>(base + l.down_input);
        down = reinterpret_cast<float*>(base + l.down);
        blas = base + l.blas;
    }
};

__device__ inline void check_proj(const Exl3Proj& p, int k, int n) {
    assert(p.k == k && p.n == n);
    assert(p.tile_w % 16 == 0 && p.tile_w / 16 >= 1 && p.tile_w / 16 <= 6);
    assert(p.trellis && p.suh && p.svh);
}

// Expert descriptors live on the device; never read them back to the host.
__global__ void prepare(const Exl3Expert* expert, strata_exl3::ReconstructJob* trellis) {
    check_proj(expert->w1, H, F);
    check_proj(expert->w3, H, F);
    check_proj(expert->w2, F, H);
    trellis[0] = {expert->w1.trellis, expert->w1.tile_w / 16};
    trellis[1] = {expert->w3.trellis, expert->w3.tile_w / 16};
    trellis[2] = {expert->w2.trellis, expert->w2.tile_w / 16};
}

// tok/weights are offset to the global row number by the host. Scratch uses
// tile-local rows, so off[0] > 0 never becomes a workspace index.
__global__ void input_had(const half* x, const int32_t* tok, const Exl3Expert* expert,
                          int rows, half* input) {
    const int job = blockIdx.x;
    const int row = job % rows;
    const int off = blockIdx.y * 128;
    const Exl3Proj p = job < rows ? expert->w1 : expert->w3;
    had_hf_r_128_inner<true, false>(x + size_t(tok[row]) * H + off,
                                   input + size_t(job) * H + off, p.suh, HAD_SCALE);
}

__device__ inline float round_pow2(float a) {
    const unsigned bits = __float_as_uint(a);
    return __uint_as_float((bits + 0x007fffffu) & 0x7f800000u);
}

// K10's arithmetic and cast sequence, with only the row/group indexing changed.
// blockIdx.y must remain the 128-element chunk: upstream helpers index the
// FULL suh/svh scale vector using y.
__device__ inline void activate_down_had_inner(float weight, const Exl3Expert& e,
                                               const float* gate_input,const float* up_input,half* down_input) {
    const int off = blockIdx.y * 128;
    const int lane = threadIdx.x;
    __shared__ __align__(16) float gate[128];
    __shared__ __align__(16) float up[128];
    __shared__ __align__(16) half hidden[128];
    had_ff_r_128_inner<false, true>(gate_input + off,
                                   gate, e.w1.svh, HAD_SCALE);
    had_ff_r_128_inner<false, true>(up_input + off,
                                   up, e.w3.svh, HAD_SCALE);
    __syncwarp();
    #pragma unroll
    for (int i = 0; i < 128; i += 32) {
        const float g = fminf(gate[i + lane], 10.0f);
        const float u = fminf(fmaxf(up[i + lane], -10.0f), 10.0f);
        const float silu = g / (1.0f + expf(-g));
        const float raw = (silu * u) * weight;
        if (!e.native_qdq) {
            hidden[i + lane] = __float2half_rn(raw);
            continue;
        }
        const float h = __bfloat162float(__float2bfloat16_rn(raw));
        float amax = fabsf(h);
        for (int d = 16; d; d >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
        const float scale = round_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
        const __nv_fp8_e4m3 q(fminf(fmaxf(h / scale, -448.0f), 448.0f));
        // Keep BOTH casts in the golden path: QDQ -> BF16 -> FP16.
        hidden[i + lane] = __float2half_rn(
            __bfloat162float(__float2bfloat16_rn(float(q) * scale)));
    }
    __syncwarp();
    had_hf_r_128_inner<true, false>(hidden, down_input + off,
                                    e.w2.suh, HAD_SCALE);
}

__global__ void activate_down_had(const float* weights, const Exl3Expert* expert,
                                  int rows, const float* gu, half* down_input) {
    const int row=blockIdx.x;
    activate_down_had_inner(weights[row],*expert,gu+size_t(row)*F,gu+size_t(rows+row)*F,
                            down_input+size_t(row)*F);
}

// Only assigned tokens are touched. Atomics also cover repeated tokens within
// one expert group; no uniqueness assumption beyond the interface is needed.
__global__ void output_had_add(const int32_t* tok, const Exl3Expert* expert,
                               const float* down, float* out) {
    const int row = blockIdx.x;
    const int off = blockIdx.y * 128;
    const int lane = threadIdx.x;
    __shared__ __align__(16) float result[128];
    had_ff_r_128_inner<false, true>(down + size_t(row) * H + off,
                                   result, expert->w2.svh, HAD_SCALE);
    __syncwarp();
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        atomicAdd(out + size_t(tok[row]) * H + off + lane + 32 * i, result[lane + 32 * i]);
}

}  // namespace strata::ds41::kernels::k12
