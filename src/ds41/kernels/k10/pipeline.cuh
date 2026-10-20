#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cassert>
#include <stdexcept>

namespace strata::ds41::kernels::k10 {

constexpr int H = 5120;
constexpr int F = 2304;
constexpr float HAD_SCALE = 0.088388347648f;  // upstream 1/sqrt(128)
using Job = strata_exl3::GemvJob;

// All offsets are multiples of 16. Reuse the gate/up job table for down only
// after its GEMV completes on the supplied stream. No persistent state/locks.
struct Workspace {
    Job* jobs;
    half* input;       // [slots * 2][H], per-projection input Hadamards
    float* gu;         // [slots * 2][F], FP32 gate/up before output Hadamards
    half* down_input;  // [slots][F], quantized activation after input Hadamard
    float* down;       // [slots][H], FP32 down before output Hadamard

    static constexpr size_t job_bytes(int slots) {
        return (size_t(2) * slots * sizeof(Job) + 15) & ~size_t(15);
    }
    static constexpr size_t bytes(int slots) {
        return job_bytes(slots) + size_t(slots) *
            (2 * H * sizeof(half) + 2 * F * sizeof(float) +
             F * sizeof(half) + H * sizeof(float));
    }
    Workspace(void* p, int slots) {
        jobs = static_cast<Job*>(p);
        input = reinterpret_cast<half*>(static_cast<char*>(p) + job_bytes(slots));
        gu = reinterpret_cast<float*>(input + size_t(2) * slots * H);
        down_input = reinterpret_cast<half*>(gu + size_t(2) * slots * F);
        down = reinterpret_cast<float*>(down_input + size_t(slots) * F);
    }
};
static_assert(Workspace::bytes(48) <= (64ull << 20), "K10 workspace exceeds contract");

__device__ inline void check_proj(const Exl3Proj& p, int k, int n) {
    assert(p.k == k && p.n == n);
    assert(p.tile_w % 16 == 0 && p.tile_w / 16 >= 1 && p.tile_w / 16 <= 6);
    assert(p.trellis && p.suh && p.svh);
    assert(p.codebook == 2 || (p.codebook == 1 && p.tile_w == 32));
}

// Upstream had_*_r_128_inner uses blockIdx.y for scale indexing. Keep y as the
// 128-element chunk, pass the FULL scale vector, and offset only input/output.
__global__ void input_had(const half* x, const int32_t* sel, int topk,
                          const Exl3Expert* experts, half* input, float* gu, Job* jobs) {
    const int job = blockIdx.x;
    const int slot = job / 2;
    const int id = sel[slot];
    const int off = blockIdx.y * 128;
    if (id < 0) {
        if (off == 0 && threadIdx.x == 0) jobs[job] = Job{};
        return;
    }
    const Exl3Proj p = (job & 1) ? experts[id].w3 : experts[id].w1;
    check_proj(p, H, F);
    half* a = input + size_t(job) * H;
    if (off == 0 && threadIdx.x == 0)
        jobs[job] = Job{a, p.trellis, gu + size_t(job) * F, H, F, p.tile_w / 16, p.codebook};
    had_hf_r_128_inner<true, false>(x + size_t(slot / topk) * H + off,
                                   a + off, p.suh, HAD_SCALE);
}

__device__ inline float round_pow2(float a) {
    // Exact powers stay unchanged, matching ops::act_quant_inplace and TK.
    const unsigned bits = __float_as_uint(a);
    return __uint_as_float((bits + 0x007fffffu) & 0x7f800000u);
}

// One warp owns 128 hidden elements. Output Hadamards remain FP32; shared
// memory only changes their destination, never the helper's arithmetic.
__global__ void activate_down_had(const int32_t* sel, const float* weights,
                                  const Exl3Expert* experts, const float* gu,
                                  half* down_input, float* down, Job* jobs) {
    const int slot = blockIdx.x;
    const int id = sel[slot];
    const int off = blockIdx.y * 128;
    const int lane = threadIdx.x;
    if (id < 0) {
        if (off == 0 && lane == 0) jobs[slot] = Job{};
        return;
    }
    const Exl3Expert e = experts[id];
    check_proj(e.w2, F, H);
    __shared__ __align__(16) float gate[128];
    __shared__ __align__(16) float up[128];
    __shared__ __align__(16) half hidden[128];
    had_ff_r_128_inner<false, true>(gu + size_t(2 * slot) * F + off,
                                   gate, e.w1.svh, HAD_SCALE);
    had_ff_r_128_inner<false, true>(gu + size_t(2 * slot + 1) * F + off,
                                   up, e.w3.svh, HAD_SCALE);
    __syncwarp();  // Hadamard stores are lane*4; QDQ reads lane + 32*i.
    #pragma unroll
    for (int i = 0; i < 128; i += 32) {
        const float g = fminf(gate[i + lane], 10.0f);
        const float u = fminf(fmaxf(up[i + lane], -10.0f), 10.0f);
        const float silu = g / (1.0f + expf(-g));
        const float raw = (silu * u) * weights[slot];
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
    half* a = down_input + size_t(slot) * F;
    had_hf_r_128_inner<true, false>(hidden, a + off, e.w2.suh, HAD_SCALE);
    if (off == 0 && lane == 0)
        jobs[slot] = Job{a, e.w2.trellis, down + size_t(slot) * H, F, H, e.w2.tile_w / 16, e.w2.codebook};
}

// Each token/chunk has one owner. Add in slot order, starting with the caller's
// existing output, exactly as the golden loop does. No atomics or output reset.
__global__ void output_had_add(const int32_t* sel, int topk, const Exl3Expert* experts,
                               const float* down, float* out) {
    const int token = blockIdx.x;
    const int off = blockIdx.y * 128;
    const int lane = threadIdx.x;
    __shared__ __align__(16) float result[128];
    float acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = out[size_t(token) * H + off + lane + 32 * i];
    for (int j = 0; j < topk; ++j) {
        const int slot = token * topk + j;
        const int id = sel[slot];
        if (id < 0) continue;
        had_ff_r_128_inner<false, true>(down + size_t(slot) * H + off,
                                       result, experts[id].w2.svh, HAD_SCALE);
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] += result[lane + 32 * i];
        __syncwarp();  // Every lane finishes reading before the next slot writes.
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[size_t(token) * H + off + lane + 32 * i] = acc[i];
}

}  // namespace strata::ds41::kernels::k10
