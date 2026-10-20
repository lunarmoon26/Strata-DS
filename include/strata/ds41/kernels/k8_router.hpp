// include/strata/ds41/kernels/k8_router.hpp - task K8: MoE routing on the GPU.
// Fixed interface; implementations live in src/ds41/kernels/k8_router.cu. Spec: ds41/tasks/K8.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::ds41::kernels {

/// Allocate maximum scratch for the current device. Call before capture.
/// Idempotent. Scratch lives until process exit. Calls/replays on one device must not overlap.
void router_init();

/// For m tokens (1..8): logits = x . w_e in fp32 (bf16 inputs), s_e = sqrt(softplus(logit_e)) with softplus(v) = v
/// for v > 20 else log1p(exp(v)); pick the 6 experts with the largest s_e + bias_e (ties: lower expert id), in that
/// order; weight_i = s_i / (sum of the 6 s + 1e-20) * 1.5.
///   x [m][5120] bf16, w [384][5120] bf16, bias [384] f32  ->  ids [m][6] int32, weights [m][6] f32
void router_topk(const __nv_bfloat16* x, int m, const __nv_bfloat16* w, const float* bias, int32_t* ids,
                 float* weights, cudaStream_t stream);

}  // namespace strata::ds41::kernels
