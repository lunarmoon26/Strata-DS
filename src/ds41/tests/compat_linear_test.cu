#include "bench_util.hpp"
#include "strata/ds41/compat_linear.hpp"
#include "strata/ds41/ops.hpp"
#include "strata/ds41/config.hpp"
#include <cstring>
#include <stdexcept>
using namespace ds41test;
using namespace strata::ds41;

int main() {
    require_gpu(); Verdict verdict;
    try {
        for (int rows : {1, 3, 17}) {
            constexpr int K = 256, N = 32;
            std::vector<__nv_bfloat16> x(rows * K);
            std::vector<half> w(N * K);
            for (int i = 0; i < rows * K; ++i) x[i] = __float2bfloat16_rn(float(i % 97 - 48) / 97);
            // Include FP16 values not representable in BF16, so a lossy cast fails.
            for (int i = 0; i < N * K; ++i) w[i] = __float2half_rn(float(i % 131 - 65) / 127);
            Dev<__nv_bfloat16> dx(x), dy(rows * N);
            Dev<half> dw(w); Dev<float> df(rows * N), converted(N * K);
            fp16_weight_linear(dx.p, dw.p, rows, K, N, nullptr, df.p);
            fp16_weight_linear(dx.p, dw.p, rows, K, N, dy.p, nullptr);
            fp16_weights_to_float(dw.p, converted.p, N * K);
            ck(cudaDeviceSynchronize(), "compat linear");
            const auto f = df.down(), cv = converted.down(); const auto y = dy.down();
            double error = 0, norm = 0;
            bool round = true, exact = true;
            for (int i = 0; i < N * K; ++i) exact &= cv[i] == __half2float(w[i]);
            for (int r = 0; r < rows; ++r) for (int n = 0; n < N; ++n) {
                double expected = 0;
                for (int k = 0; k < K; ++k)
                    expected += double(__bfloat162float(x[r * K + k])) * __half2float(w[n * K + k]);
                const double delta = f[r * N + n] - expected;
                error += delta * delta; norm += expected * expected;
                round &= __bfloat162float(y[r * N + n]) == __bfloat162float(__float2bfloat16_rn(f[r * N + n]));
            }
            verdict.check(exact && round && std::sqrt(error / norm) < 1e-6, "native FP16 weights: CPU reference, output rounding and exact bias widening");
        }
        constexpr int D = 5120;
        std::vector<__nv_bfloat16> h(4 * D), kv(5 * D);
        std::vector<float> qw(4 * D), kw(4 * D);
        for (int i = 0; i < 4 * D; ++i) {
            h[i] = __float2bfloat16_rn(float(i % 37 - 18) / 19);
            qw[i] = .73f + float(i % 13) / 1001; kw[i] = .81f + float(i % 17) / 1003;
        }
        for (int i = 0; i < 5 * D; ++i) kv[i] = __float2bfloat16_rn(float(i % 29 - 14) / 17);
        auto expected = h;
        for (int c = 0; c < 4; ++c) {
            double hh = 0, kk = 0, dot = 0;
            for (int d = 0; d < D; ++d) {
                const double a = __bfloat162float(h[c * D + d]), b = __bfloat162float(kv[c * D + d]);
                hh += a * a; kk += b * b; dot += a * double(qw[c * D + d]) * kw[c * D + d] * b;
            }
            const double x = dot / std::sqrt((hh / D + 1e-20) * (kk / D + 1e-20)) / std::sqrt(double(D));
            const double gate = 1 / (1 + std::exp(-std::copysign(std::sqrt(std::max(std::abs(x), 1e-6)), x)));
            for (int d = 0; d < D; ++d)
                expected[c * D + d] = __float2bfloat16_rn(__bfloat162float(h[c * D + d]) + gate * __bfloat162float(kv[4 * D + d]));
        }
        Dev<__nv_bfloat16> dh(h), dk(kv); Dev<float> dq(qw), dw(kw);
        ops::engram_apply_f32(dh.p, dk.p, dq.p, dw.p, 1e-20);
        ck(cudaDeviceSynchronize(), "FP32 engram gate");
        const auto result = dh.down(); double error = 0, norm = 0;
        for (int i = 0; i < 4 * D; ++i) {
            const double a = __bfloat162float(result[i]), b = __bfloat162float(expected[i]);
            error += (a-b)*(a-b); norm += b*b;
        }
        verdict.check(std::sqrt(error / norm) < .002, "native FP32 Engram gate versus independent CPU reference");
        return verdict.finish();
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
