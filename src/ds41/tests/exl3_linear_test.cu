// Stored EXL3 dense GEMV versus scalar trellis decode + CPU Hadamards.
#include "bench_util.hpp"
#include "strata/ds41/exl3_linear.hpp"
#include "../../../third_party/exllamav3_gpu/quant/exl3_gemv.cuh"
#include <cstring>
#include <stdexcept>

namespace strata::ds41::kernels::k10 {
__global__ void activate_down_had(const int32_t*, const float*, const Exl3Expert*,
                                  const float*, half*, float*, strata_exl3::GemvJob*);
}

using namespace ds41test;
using namespace strata::ds41;
namespace {
float half_round(float x) { return __half2float(__float2half_rn(x)); }
float bf_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
float half_bits(uint16_t bits) {
    __half x; std::memcpy(&x, &bits, 2); return __half2float(x);
}
float weight(const std::vector<uint32_t>& p, int n, int row, int col, int bits) {
    const int r = row % 16, c = col % 16;
    const int lane = (r % 8) / 2 + ((c % 8) / 2) * 8 + (c % 2) * 4;
    const int element = r % 2 + (r >= 8 ? 2 : 0) + (c >= 8 ? 4 : 0);
    const int pos = lane * 8 + element, words = 8 * bits;
    const int start = pos * bits + bits - 16 + 256 * bits, end = start + 16;
    const int i0 = start / 32, i1 = (end - 1) / 32, shift = (i1 + 1) * 32 - end;
    const size_t base = ((size_t)(row / 16) * (n / 16) + col / 16) * words;
    const uint64_t pair = (uint64_t(p[base + i0 % words]) << 32) | p[base + i1 % words];
    const uint32_t state = (pair >> shift) & 65535, x = state * 0x83DCD12Du;
    const int sum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
    return half_round(float((1024.0 + sum) * double(half_bits(0x1eee)) + double(half_bits(0xc931))));
}
void had(std::vector<float>& x) {
    for (size_t off = 0; off < x.size(); off += 128) {
        for (int stride = 1; stride < 128; stride *= 2)
            for (int j = 0; j < 128; j += stride * 2)
                for (int i = 0; i < stride; ++i) {
                    const float a = x[off + j + i], b = x[off + j + i + stride];
                    x[off + j + i] = a + b; x[off + j + i + stride] = a - b;
                }
        for (int j = 0; j < 128; ++j) x[off + j] *= 0.088388347648f;
    }
}
void run(Verdict& verdict, int k, int n, int bits) {
    std::mt19937 rng(k + n + bits);
    std::vector<uint32_t> packed(size_t(k / 16) * (n / 16) * 8 * bits);
    for (auto& v : packed) v = rng();
    std::vector<__nv_bfloat16> input(k);
    std::vector<half> suh(k), svh(n);
    std::vector<float> a(k), reference(n);
    for (int i = 0; i < k; ++i) {
        input[i] = __float2bfloat16_rn(float(int(rng() % 101) - 50) / 97);
        suh[i] = __float2half_rn(float(int(rng() % 31) - 15) / 16);
        a[i] = half_round(half_round(__bfloat162float(input[i])) * __half2float(suh[i]));
    }
    for (int i = 0; i < n; ++i) svh[i] = __float2half_rn(float(int(rng() % 23) - 11) / 64);
    had(a); for (auto& v : a) v = half_round(v);
    for (int j = 0; j < n; ++j) {
        double sum = 0;
        for (int i = 0; i < k; ++i) sum += double(a[i]) * weight(packed, n, i, j, bits);
        reference[j] = float(sum);
    }
    had(reference);
    std::vector<float> raw_reference(n);
    for (int j = 0; j < n; ++j) {
        raw_reference[j] = reference[j] * __half2float(svh[j]);
        reference[j] = bf_round(raw_reference[j]);
    }
    Dev<uint32_t> p(packed); Dev<__nv_bfloat16> x(input), output(n);
    Dev<half> su(suh), sv(svh); Dev<float> logits(n);
    const kernels::Exl3Proj w{reinterpret_cast<uint16_t*>(p.p), su.p, sv.p, k, n, bits * 16, 2};
    const auto bytes = exl3_linear_workspace_bytes(k, n);
    Dev<uint8_t> ws(bytes);
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "dense stream");
    auto enqueue = [&] { exl3_linear(x.p, w, output.p, nullptr, ws.p, bytes, stream); };
    enqueue(); ck(cudaStreamSynchronize(stream), "dense eager");
    const auto got = output.down(); double error = 0, norm = 0; bool finite = true;
    for (int j = 0; j < n; ++j) {
        const float value = __bfloat162float(got[j]); finite &= std::isfinite(value);
        error += double(value - reference[j]) * (value - reference[j]);
        norm += double(reference[j]) * reference[j];
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-30));
    verdict.check(finite && norm > 0 && relative < 0.006, "dense EXL3 versus independent CPU reference rel L2 < .006");
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "dense capture"); enqueue();
    ck(cudaStreamEndCapture(stream, &graph), "dense end capture");
    ck(cudaGraphInstantiate(&exec, graph, 0), "dense instantiate");
    ck(cudaMemset(output.p, 0xff, n * 2), "poison dense output");
    ck(cudaGraphLaunch(exec, stream), "dense graph"); ck(cudaStreamSynchronize(stream), "dense replay");
    const auto replay = output.down();
    verdict.check(std::memcmp(got.data(), replay.data(), n * 2) == 0, "dense eager/graph bit parity");
    exl3_linear(x.p, w, nullptr, logits.p, ws.p, bytes, stream);
    ck(cudaStreamSynchronize(stream), "dense logits"); const auto lg = logits.down();
    bool same = true; for (int i = 0; i < n; ++i) same &= lg[i] == __bfloat162float(got[i]);
    verdict.check(same, "EXL3 head returns BF16-rounded FP32");
    exl3_linear(x.p, w, nullptr, logits.p, ws.p, bytes, stream, true);
    ck(cudaStreamSynchronize(stream), "dense raw float");
    const auto raw = logits.down(); double raw_error = 0, raw_norm = 0;
    for (int j = 0; j < n; ++j) {
        raw_error += double(raw[j] - raw_reference[j]) * (raw[j] - raw_reference[j]);
        raw_norm += double(raw_reference[j]) * raw_reference[j];
    }
    verdict.check(raw_norm > 0 && std::sqrt(raw_error / raw_norm) < .006, "raw FP32 EXL3 compressor versus CPU reference");
    for(int rows:{1,4,8}) {
        const int xs=k+128,ys=n+128;
        std::vector<__nv_bfloat16> inputs(size_t(rows)*xs);
        for(int r=0;r<rows;++r) for(int i=0;i<k;++i)
            inputs[size_t(r)*xs+i]=__float2bfloat16_rn(__bfloat162float(input[i])*(r%2?-.5f:1.f+float(r)/8));
        Dev<__nv_bfloat16> sx(inputs),sy(size_t(rows)*ys);
        Dev<__nv_bfloat16> sequential(size_t(rows)*ys);
        Dev<float> sl(size_t(rows)*ys),sequential_logits(size_t(rows)*ys);
        const auto bytes=exl3_small_rows_workspace_bytes(rows,k,n);Dev<uint8_t> small_ws(bytes);
        for(int r=0;r<rows;++r) {
            exl3_linear(sx.p+size_t(r)*xs,w,sequential.p+size_t(r)*ys,nullptr,ws.p,ws.n,stream);
            exl3_linear(sx.p+size_t(r)*xs,w,nullptr,sequential_logits.p+size_t(r)*ys,ws.p,ws.n,stream);
        }
        exl3_linear_small_rows(sx.p,w,sy.p,nullptr,rows,xs,ys,small_ws.p,bytes,stream);
        exl3_linear_small_rows(sx.p,w,nullptr,sl.p,rows,xs,ys,small_ws.p,bytes,stream);
        ck(cudaStreamSynchronize(stream),"small dense rows");const auto ry=sy.down();const auto rl=sl.down();
        const auto want=sequential.down();const auto want_logits=sequential_logits.down();
        bool same=true;
        for(int r=0;r<rows;++r) for(int j=0;j<n;++j)
            same &= !std::memcmp(&ry[size_t(r)*ys+j],&want[size_t(r)*ys+j],2) && rl[size_t(r)*ys+j]==want_logits[size_t(r)*ys+j];
        verdict.check(same,"small strided rows/head bit-identical to sequential GEMV");
    }
    if (n <= 4096) for (int count : {1,3,16}) for (int rows : {1,4,8}) {
        const int xs=count*k+128, ys=count*n+128;
        std::vector<__nv_bfloat16> inputs(size_t(rows)*xs);
        for (size_t i=0;i<inputs.size();++i)
            inputs[i]=__float2bfloat16_rn(float(int(i%29)-14)/64);
        std::vector<half> other_scale=svh;
        for (auto& value:other_scale) value=__float2half(-__half2float(value));
        Dev<half> alt(other_scale);
        std::vector<kernels::Exl3Proj> projections(count,w);
        for (int g=1;g<count;g+=2) projections[g].svh=alt.p;
        Dev<__nv_bfloat16> bx(inputs), by(size_t(rows)*ys), expected(size_t(rows)*ys);
        Dev<uint8_t> bws(exl3_projection_batch_workspace_bytes(count,rows,k,n));
        Dev<uint8_t> rws(exl3_small_rows_workspace_bytes(rows,k,n));
        // Dev(vector) uploads on the default stream; pageable H2D may return
        // after staging, before DMA finishes. Our compute stream is nonblocking.
        ck(cudaStreamSynchronize(nullptr),"projection fixture uploads complete");
        ck(cudaMemsetAsync(by.p,0x5a,by.n*sizeof(__nv_bfloat16),stream),"batch sentinel");
        ck(cudaMemsetAsync(expected.p,0x5a,expected.n*sizeof(__nv_bfloat16),stream),"batch reference sentinel");
        for(int g=0;g<count;++g)
            exl3_linear_small_rows(bx.p+g*k,projections[g],expected.p+g*n,nullptr,
                                  rows,xs,ys,rws.p,rws.n,stream);
        ck(cudaStreamSynchronize(stream),"projection reference complete before capture");
        cudaGraph_t batch_graph; cudaGraphExec_t batch_exec;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed),"projection batch capture");
        exl3_projection_batch(bx.p,projections.data(),by.p,count,rows,xs,ys,bws.p,bws.n,stream);
        ck(cudaStreamEndCapture(stream,&batch_graph),"projection batch capture end");
        ck(cudaGraphInstantiate(&batch_exec,batch_graph,0),"projection batch instantiate");
        ck(cudaGraphLaunch(batch_exec,stream),"projection batch replay");
        ck(cudaStreamSynchronize(stream),"projection batch compare");
        const auto actual=by.down(), reference=expected.down();
        if (std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(__nv_bfloat16))) {
            size_t differences=0;
            for(size_t i=0;i<actual.size();++i) if(std::memcmp(&actual[i],&reference[i],2)) {
                if(differences++<4) std::printf("BATCH mismatch k=%d n=%d bits=%d count=%d rows=%d offset=%zu got=%g ref=%g\n",
                    k,n,bits,count,rows,i,double(__bfloat162float(actual[i])),double(__bfloat162float(reference[i])));
            }
            std::printf("BATCH differences=%zu\n",differences);
        }
        verdict.check(!std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(__nv_bfloat16)),
                      "batched projections match separate calls including stride padding");
        bool rejected=false;
        try { exl3_projection_batch(bx.p,projections.data(),by.p,count,rows,xs,ys,bws.p,bws.n-1,stream); }
        catch(const std::invalid_argument&) { rejected=true; }
        verdict.check(rejected,"projection batch short workspace rejected");
        cudaGraphExecDestroy(batch_exec); cudaGraphDestroy(batch_graph);
    }
    bool bad = false;
    try { exl3_linear(x.p, w, output.p, nullptr, ws.p, bytes - 1, stream); }
    catch (const std::invalid_argument&) { bad = true; }
    verdict.check(bad, "short workspace rejected before launch");
    if (n <= 32768 && !std::getenv("DS41_TEST_NATIVE_ONLY")) {
        constexpr int rows = 3;
        std::vector<__nv_bfloat16> inputs(size_t(rows) * (k + 128));
        for (int row = 0; row < rows; ++row)
            std::copy(input.begin(), input.end(), inputs.begin() + size_t(row) * (k + 128));
        Dev<__nv_bfloat16> dx(inputs), dy(size_t(rows) * (n + 128));
        const auto row_bytes = exl3_rows_workspace_bytes(rows, k, n);
        Dev<uint8_t> rws(row_bytes);
        exl3_linear_rows(dx.p, w, dy.p, rows, k + 128, n + 128, rws.p, row_bytes, stream);
        ck(cudaStreamSynchronize(stream), "batched dense rows"); const auto ry = dy.down();
        double re = 0, rn = 0;
        for (int row = 0; row < rows; ++row) for (int j = 0; j < n; ++j) {
            const double value = __bfloat162float(ry[size_t(row) * (n + 128) + j]);
            re += (value - reference[j]) * (value - reference[j]); rn += double(reference[j]) * reference[j];
        }
        const double relative_rows = std::sqrt(re / std::max(rn, 1e-30));
        verdict.check(relative_rows < .006, "batched strided dense GEMM versus CPU reference");
        Dev<float> raw_rows(size_t(rows) * (n + 128));
        exl3_linear_rows(dx.p, w, nullptr, rows, k + 128, n + 128, rws.p, row_bytes, stream, raw_rows.p);
        ck(cudaStreamSynchronize(stream), "batched raw compressor");
        const auto rf = raw_rows.down(); double raw_error = 0, raw_norm = 0;
        for (int row = 0; row < rows; ++row) for (int j = 0; j < n; ++j) {
            const double d = rf[size_t(row) * (n + 128) + j] - raw_reference[j];
            raw_error += d*d; raw_norm += double(raw_reference[j])*raw_reference[j];
        }
        verdict.check(raw_norm > 0 && std::sqrt(raw_error / raw_norm) < .006, "batched raw FP32 compressor versus CPU reference");
        std::printf("EXL3 rows k=%d n=%d bits=%d rel_l2=%.9g\n", k, n, bits, relative_rows);
    }
    std::printf("EXL3 dense k=%d n=%d bits=%d rel_l2=%.9g graph=bitwise\n", k, n, bits, relative);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
}

void activation(Verdict& verdict) {
    constexpr int F = 2304, H = 5120;
    std::vector<float> gu(2 * F), gate(F), up(F), reference(F);
    std::vector<half> sv1(F), sv3(F), su2(F);
    for (int i = 0; i < F; ++i) {
        gu[i] = float(i % 97 - 48) / 3; gu[F + i] = float(i % 83 - 41) / 7;
        sv1[i] = __float2half_rn((i % 2 ? 1.f : -1.f) / 8);
        sv3[i] = __float2half_rn((i % 3 ? 1.f : -1.f) / 4);
        su2[i] = __float2half_rn((i % 5 ? 1.f : -1.f) / 2);
        gate[i] = gu[i]; up[i] = gu[F + i];
    }
    had(gate); had(up);
    constexpr float route = .173f;
    for (int i = 0; i < F; ++i) {
        const float g = std::min(gate[i] * __half2float(sv1[i]), 10.f);
        const float u = std::clamp(up[i] * __half2float(sv3[i]), -10.f, 10.f);
        reference[i] = half_round(half_round((g / (1.f + std::exp(-g)) * u) * route) * __half2float(su2[i]));
    }
    had(reference); for (auto& v : reference) v = half_round(v);
    Dev<float> dgu(gu), down(H), weights(std::vector<float>{route});
    Dev<half> a(sv1), b(sv3), c(su2), hidden(F), dummy(H);
    Dev<int32_t> sel(std::vector<int32_t>{0});
    kernels::Exl3Expert expert{};
    expert.w1.svh = a.p; expert.w3.svh = b.p;
    expert.w2 = {reinterpret_cast<uint16_t*>(dummy.p), c.p, dummy.p, F, H, 64, 2};
    expert.native_qdq = false;
    Dev<kernels::Exl3Expert> experts(std::vector<kernels::Exl3Expert>{expert});
    Dev<uint8_t> jobs(128);
    kernels::k10::activate_down_had<<<dim3(1, F / 128), 32>>>(sel.p, weights.p, experts.p,
        dgu.p, hidden.p, down.p, reinterpret_cast<strata_exl3::GemvJob*>(jobs.p));
    ck(cudaDeviceSynchronize(), "published EXL3 activation");
    const auto got = hidden.down(); double error = 0, norm = 0;
    for (int i = 0; i < F; ++i) {
        const double delta = __half2float(got[i]) - reference[i]; error += delta * delta;
        norm += double(reference[i]) * reference[i];
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-30));
    verdict.check(norm > 0 && relative < .002, "published FP16 expert activation versus CPU oracle (no BF16/FP8 QDQ)");
    std::printf("EXL3 FP16 routed activation rel_l2=%.9g\n", relative);
}
}  // namespace
int main() {
    require_gpu(); Verdict v;
    for (int bits : {1, 2, 3, 4, 5, 6, 7, 8}) run(v, 128, 128, bits);
    run(v, 512, 128, 8);  // LibertAIDAI indexer wk
    run(v, 5120, 128, 4); run(v, 128, 129280, 4);
    run(v, 1280, 32768, 4); run(v, 4096, 1024, 5);
    activation(v);
    return v.finish();
}
