#include "strata/ds41/compat_linear.hpp"
#include <stdexcept>

namespace strata::ds41 {
namespace {
__global__ void linear(const __nv_bfloat16* x, const half* w, int rows, int k, int n,
                        __nv_bfloat16* y, float* yf) {
    const int item = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (item >= rows * n) return;
    const int row = item / n, col = item % n;
    float sum = 0;
    for (int i = lane; i < k; i += 32)
        sum = __fmaf_rn(__bfloat162float(x[size_t(row) * k + i]), __half2float(w[size_t(col) * k + i]), sum);
    for (int offset = 16; offset; offset /= 2)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    if (lane == 0) {
        if (y) y[item] = __float2bfloat16_rn(sum);
        else yf[item] = sum;
    }
}
__global__ void convert(const half* x, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}
void check() {
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("FP16 compatibility linear launch failed");
}
}
void fp16_weight_linear(const __nv_bfloat16* x, const half* w, int rows, int k, int n,
                        __nv_bfloat16* y, float* yf, cudaStream_t stream) {
    if (!x || !w || (!y == !yf) || rows < 1 || rows > 4096 || k < 1 || k > 32768 || n < 1 || n > 129280)
        throw std::invalid_argument("FP16 compatibility linear: invalid shape/output");
    linear<<<(rows * n + 3) / 4, 128, 0, stream>>>(x, w, rows, k, n, y, yf);
    check();
}
void fp16_weights_to_float(const half* x, float* y, int n, cudaStream_t stream) {
    if (!x || !y || n < 1) throw std::invalid_argument("FP16 compatibility conversion: invalid arguments");
    convert<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
    check();
}
}
