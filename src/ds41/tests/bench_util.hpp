// src/ds41/tests/bench_util.hpp - shared helpers for the ds41 task tests: device buffers, seeded random data,
// relative error, CUDA-event timing, and the RESULT line the CI runner reads.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace ds41test {

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("RESULT fail cuda-error=%s(%s)\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// exit 77 (CTest skip) when there is no GPU
inline void require_gpu() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        std::exit(77);
    }
}

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { ck(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)), "cudaMalloc"); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { up(h); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    void up(const std::vector<T>& h) { ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return h;
    }
};

inline uint16_t bf16_bits(float v) {
    __nv_bfloat16 b = __float2bfloat16_rn(v);
    uint16_t u;
    std::memcpy(&u, &b, 2);
    return u;
}
inline float bf16_val(uint16_t u) {
    uint32_t x = (uint32_t) u << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

inline std::vector<__nv_bfloat16> rand_bf16(size_t n, float scale, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) x = __float2bfloat16_rn(d(g));
    return v;
}
inline std::vector<float> rand_f32(size_t n, float scale, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

inline double rel_l2(const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
        num += (x - y) * (x - y);
        den += y * y;
    }
    return std::sqrt(num / std::max(den, 1e-300));
}
inline double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

/// median microseconds of `reps` timed calls after 3 warm-up calls; `fn` enqueues on the default stream
template <typename F>
double median_us(F fn, int reps = 25) {
    for (int i = 0; i < 3; ++i) fn();
    ck(cudaDeviceSynchronize(), "warm-up");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        fn();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        t.push_back(ms * 1000.0f);
    }
    ck(cudaGetLastError(), "timed run");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

/// collects pass/fail and metrics; prints one RESULT line at the end (the CI verdict)
struct Verdict;
template <typename Call, typename Read, typename Poison>
void graph_check(Verdict& v, const std::string& what, Call call, Read read, Poison poison);

/// collects pass/fail and metrics; prints one RESULT line at the end (the CI verdict)
struct Verdict {
    bool pass = true;
    std::string metrics;
    void check(bool ok, const std::string& what) {
        if (!ok) { pass = false; std::printf("FAIL: %s\n", what.c_str()); }
    }
    void metric(const std::string& k, double v) {
        char buf[96];
        std::snprintf(buf, sizeof buf, " %s=%.4g", k.c_str(), v);
        metrics += buf;
    }
    int finish() {
        std::printf("RESULT %s%s\n", pass ? "pass" : "fail", metrics.c_str());
        return pass ? 0 : 1;
    }
};

/// bf16 / fp16 / int device outputs as doubles, for graph_check's read()
/// fill a device buffer with 0xFF bytes (NaN for floats, -1 for ints): graph_check's poison()
template <typename T>
void poison_dev(Dev<T>& d) { ck(cudaMemset(d.p, 0xFF, d.n * sizeof(T)), "poison"); }

template <typename T>
std::vector<double> as_doubles(const std::vector<T>& v) {
    std::vector<double> d(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        if constexpr (std::is_same<T, __nv_bfloat16>::value) d[i] = __bfloat162float(v[i]);
        else d[i] = (double) v[i];
    }
    return d;
}

/// Rule 7 (ds41/tasks/README.md): the engine captures each decode step as one CUDA graph. One call captured on a
/// non-default stream in global capture mode, then replayed twice, must give the eager call's outputs (relative L2
/// at most 1e-6; non-finite values must match exactly). A host sync or a legacy allocation inside the call makes the
/// capture fail. call(stream) enqueues one call (and any output reset) on the stream; read() returns the outputs;
/// poison() fills every output with garbage before each replay, so a graph that misses work (for example a call that
/// ignored the stream and ran eagerly during the capture) shows up as a difference.
template <typename Call, typename Read, typename Poison>
void graph_check(Verdict& v, const std::string& what, Call call, Read read, Poison poison) {
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "graph stream");
    std::string why;
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ex = nullptr;
    try {
        call(s);
        ck(cudaStreamSynchronize(s), "eager call on a non-default stream");
        const std::vector<double> eager = read();
        cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        if (e == cudaSuccess) {
            call(s);
            e = cudaStreamEndCapture(s, &g);
        }
        if (e == cudaSuccess) e = cudaGraphInstantiate(&ex, g, 0);
        if (e != cudaSuccess) why = std::string("capture failed: ") + cudaGetErrorString(e);
        for (int r = 0; r < 2 && why.empty(); ++r) {
            poison();
            ck(cudaDeviceSynchronize(), "poison outputs");
            e = cudaGraphLaunch(ex, s);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s);
            if (e != cudaSuccess) { why = std::string("replay failed: ") + cudaGetErrorString(e); break; }
            const std::vector<double> got = read();
            double num = 0, den = 0;
            bool special = got.size() != eager.size();
            for (size_t i = 0; !special && i < got.size(); ++i) {
                if (std::isfinite(got[i]) && std::isfinite(eager[i])) {
                    num += (got[i] - eager[i]) * (got[i] - eager[i]);
                    den += eager[i] * eager[i];
                } else if (!(got[i] == eager[i])) {
                    special = true;
                }
            }
            if (special || std::sqrt(num / std::max(den, 1e-300)) > 1e-6) why = "a replay differs from the eager call";
        }
    } catch (const std::exception& x) {
        why = std::string("exception: ") + x.what();
    }
    if (ex) cudaGraphExecDestroy(ex);
    if (g) cudaGraphDestroy(g);
    cudaGetLastError();
    cudaStreamDestroy(s);
    std::printf("graph check (%s): %s\n", what.c_str(), why.empty() ? "pass" : why.c_str());
    v.check(why.empty(), "rule 7 graph capture, " + what + ": " + why);
}

}  // namespace ds41test
