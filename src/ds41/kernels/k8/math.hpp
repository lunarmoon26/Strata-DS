#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define K8_HD __host__ __device__ __forceinline__
#else
#define K8_HD inline
#endif

namespace strata::ds41::kernels::k8_detail {

// K8 specifies FP32 logits, then sqrt(softplus) without an intervening FP32
// rounding requirement. The fixed oracle explicitly uses double for this
// nonlinear step, biased comparisons and six-score normalization. Keeping
// that precision avoids creating extra ties by rounding the biased score.
K8_HD double score(float logit) {
    const double z = logit;
    return sqrt(z > 20.0 ? z : log1p(exp(z)));
}

K8_HD bool better(double a, int ai, double b, int bi) {
    return a > b || (a == b && ai < bi);
}

}  // namespace strata::ds41::kernels::k8_detail

#undef K8_HD
