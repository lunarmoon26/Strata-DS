// MCG codebook bit parity and K2 GEMV against an independent scalar/FP32 oracle.
// No published model weights, Python framework, dequantized weight cache or downloads.
#include "bench_util.hpp"
#include "../../../third_party/exllamav3_gpu/util.cuh"
#include "../../../third_party/exllamav3_gpu/quant/codebook.cuh"
#include "../../../third_party/exllamav3_gpu/quant/exl3_gemv.cuh"

using namespace ds41test;

__global__ void codebook_bits(uint16_t* scalar, uint16_t* paired) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65536) return;
    scalar[i] = __half_as_ushort(decode_3inst<1>(i));
    const half2 pair = decode_3inst_2<1>(i, (i + 1) & 65535);
    paired[i * 2] = __half_as_ushort(__low2half(pair));
    paired[i * 2 + 1] = __half_as_ushort(__high2half(pair));
}

uint16_t mcg_bits(uint32_t index) {
    const uint32_t bits = ((index * 0xCBAC1FEDu) & 0x8fff8fffu) ^ 0x3b603b60u;
    const uint16_t lo = bits, hi = bits >> 16;
    __half a, b;
    std::memcpy(&a, &lo, 2); std::memcpy(&b, &hi, 2);
    return __half_as_ushort(__float2half_rn(__half2float(a) + __half2float(b)));
}

float mcg_value(uint32_t index) {
    const uint16_t bits = mcg_bits(index);
    __half value; std::memcpy(&value, &bits, 2);
    return __half2float(value);
}

// Invert upstream reconstruct_tile's tensor-core -> row-major shuffle. Extract
// the circular 16-bit trellis window with scalar integer operations, not GPU dq.
float weight(const std::vector<uint32_t>& packed, int k, int n, int row, int col) {
    const int r = row % 16, c = col % 16;
    const int lane = (r % 8) / 2 + ((c % 8) / 2) * 8 + (c % 2) * 4;
    const int element = r % 2 + (r >= 8 ? 2 : 0) + (c >= 8 ? 4 : 0);
    const int position = lane * 8 + element;
    const int b0 = position * 2 + 2 - 16 + 512, b1 = b0 + 16;
    const int i0 = b0 / 32, i1 = (b1 - 1) / 32, shift = (i1 + 1) * 32 - b1;
    const size_t base = ((size_t) (row / 16) * (n / 16) + col / 16) * 16;
    const uint64_t pair = (uint64_t(packed[base + i0 % 16]) << 32) | packed[base + i1 % 16];
    return mcg_value((pair >> shift) & 65535);
}

void gemv_case(Verdict& verdict, int k, int n) {
    std::mt19937 rng(487 + k + n);
    std::vector<uint32_t> packed((size_t) (k / 16) * (n / 16) * 16);
    for (auto& value : packed) value = rng();
    std::vector<__half> x(k);
    for (int i = 0; i < k; ++i) x[i] = __float2half_rn(float(int(rng() % 31) - 15) / 256);
    std::vector<float> reference(n, 0);
    for (int col = 0; col < n; ++col) {
        double sum = 0;
        for (int row = 0; row < k; ++row) sum += double(__half2float(x[row])) * weight(packed, k, n, row, col);
        reference[col] = float(sum);
    }
    Dev<uint32_t> dp(packed);
    Dev<__half> dx(x);
    Dev<float> output(n);
    Dev<strata_exl3::GemvJob> job(std::vector<strata_exl3::GemvJob>{{dx.p, (const uint16_t*) dp.p, output.p, k, n, 2, 1}});
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "MCG stream");
    auto enqueue = [&] { strata_exl3::gemv_mul1(job.p, 1, n, stream); };
    enqueue(); ck(cudaStreamSynchronize(stream), "MCG eager");
    const auto eager = output.down();
    double error = 0, norm = 0;
    bool finite = true, nonzero = false;
    for (int i = 0; i < n; ++i) {
        const double delta = eager[i] - reference[i]; error += delta * delta; norm += double(reference[i]) * reference[i];
        finite &= std::isfinite(eager[i]); nonzero |= eager[i] != 0;
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-30));
    verdict.check(finite && nonzero && relative <= 0.003, "MCG K2 GEMV versus independent FP32 reconstruction, rel L2 <= .003");
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "MCG capture"); enqueue();
    ck(cudaStreamEndCapture(stream, &graph), "MCG end capture");
    ck(cudaGraphInstantiate(&exec, graph, 0), "MCG instantiate");
    ck(cudaMemset(output.p, 0xff, n * sizeof(float)), "poison MCG output");
    ck(cudaGraphLaunch(exec, stream), "MCG replay"); ck(cudaStreamSynchronize(stream), "MCG graph");
    const auto captured = output.down();
    verdict.check(std::memcmp(eager.data(), captured.data(), n * sizeof(float)) == 0, "MCG eager/graph parity");
    std::printf("MCG GEMV k=%d n=%d rel_l2=%.9g graph=bitwise\n", k, n, relative);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
}

int main() {
    require_gpu(); Verdict verdict;
    Dev<uint16_t> scalar(65536), paired(2 * 65536);
    codebook_bits<<<256, 256>>>(scalar.p, paired.p); ck(cudaDeviceSynchronize(), "MCG exhaustive codebook");
    const auto single = scalar.down(), pair = paired.down();
    int bad = 0;
    for (unsigned i = 0; i < 65536; ++i) {
        bad += single[i] != mcg_bits(i);
        bad += pair[i * 2] != mcg_bits(i);
        bad += pair[i * 2 + 1] != mcg_bits((i + 1) & 65535);
    }
    verdict.check(bad == 0, "MCG scalar and paired decode cover all 65536 codebook entries bitwise");
    verdict.metric("codebook_entries", 65536); verdict.metric("bit_mismatches", bad);
    gemv_case(verdict, 128, 128);
    gemv_case(verdict, 5120, 2304);
    gemv_case(verdict, 2304, 5120);
    return verdict.finish();
}
