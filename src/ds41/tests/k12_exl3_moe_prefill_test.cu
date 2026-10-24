// src/ds41/tests/k12_exl3_moe_prefill_test.cu - task K12 acceptance: prefill experts (rows grouped by expert)
// against K10 (exl3_moe_decode, validated against exllamav3) run token by token, then speed. Fixed by
// ds41/tasks/K12.md. The experts are synthetic (random trellis, random signed scales): no pack, no golden files.
#include "bench_util.hpp"

#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"

#include <numeric>
#include <stdexcept>

using namespace ds41test;
namespace kk = strata::ds41::kernels;

namespace {

constexpr int H = 5120, F = 2304, TOPK = 6, E = 384, TILE_W = 48;  // 3 bits per weight: tile_w = 16 * 3
constexpr int CALL_EXPERTS = 64;                                    // experts per call, as the engine streams them

__device__ uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
__global__ void fill_u16(uint16_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        p[i] = (uint16_t) (mix((uint32_t) i * 2654435761U ^ seed) & 0xFFFF);
}
// random sign times scale * (0.5 + U[0, 1))
__global__ void fill_scale(__half* p, size_t n, float scale, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint32_t r = mix((uint32_t) i * 2246822519U ^ seed);
        const float v = scale * (0.5f + (float) (r >> 8) * (1.0f / 16777216.0f));
        p[i] = __float2half((r & 1) ? -v : v);
    }
}

size_t align256(size_t b) { return (b + 255) & ~(size_t) 255; }

/// E synthetic experts in one device buffer, laid out per expert as w1, w3, w2 (trellis, suh, svh each)
struct Experts {
    void* base = nullptr;
    std::vector<kk::Exl3Expert> host;
    Experts(int count = E, bool mixed = false) {
        const size_t tb = (size_t) (H / 16) * (F / 16) * (mixed ? 96 : TILE_W) * 2;  // trellis bytes, same for all three
        const size_t per = 3 * align256(tb) + 2 * align256(H * 2) + 2 * align256(F * 2) + align256(F * 2) +
                           align256(H * 2);
        ck(cudaMalloc(&base, per * count), "experts");
        host.resize(count);
        for (int e = 0; e < count; ++e) {
            char* p = (char*) base + per * e;
            auto proj = [&](int k, int n, float s_in, float s_out, uint32_t seed, int bits) {
                kk::Exl3Proj q;
                q.trellis = (const uint16_t*) p;
                const size_t packed_bytes = (size_t) (k / 16) * (n / 16) * 16 * bits * 2;
                fill_u16<<<256, 256>>>((uint16_t*) p, packed_bytes / 2, seed);
                p += align256(tb);
                q.suh = (const __half*) p;
                fill_scale<<<32, 256>>>((__half*) p, k, s_in, seed ^ 0x1111U);
                p += align256(k * 2);
                q.svh = (const __half*) p;
                fill_scale<<<32, 256>>>((__half*) p, n, s_out, seed ^ 0x2222U);
                p += align256(n * 2);
                q.k = k;
                q.n = n;
                q.tile_w = 16 * bits;
                return q;
            };
            const uint32_t s = 0x9e3779b9U * (uint32_t) (e + 1);
            host[e].w1 = proj(H, F, 1.0f, 0.02f, s ^ 1, mixed ? e % 6 + 1 : 3);
            host[e].w3 = proj(H, F, 1.0f, 0.02f, s ^ 3, mixed ? (e + 1) % 6 + 1 : 3);
            host[e].w2 = proj(F, H, 1.0f, 0.05f, s ^ 2, mixed ? (e + 5) % 6 + 1 : 3);
        }
        ck(cudaDeviceSynchronize(), "fill experts");
    }
    ~Experts() { cudaFree(base); }
};

/// One chunk: T tokens with their 6 experts each (some tokens can have none), sorted by expert
struct Chunk {
    int T = 0;
    std::vector<int32_t> sel;        // [T][6] expert ids or -1
    std::vector<float> wt;           // [T][6]
    std::vector<int32_t> tok;        // rows sorted by expert
    std::vector<float> w;            // rows sorted by expert
    std::vector<int32_t> start;      // E + 1: rows of expert e are [start[e], start[e + 1])
};

/// skew = 0: uniform over the first n_exp experts; skew > 0: expert e drawn with weight 1 / (e + 8)^skew
Chunk make_chunk(int T, int n_exp, double skew, uint32_t seed, const std::vector<int>& idle_tokens = {}) {
    Chunk c;
    c.T = T;
    c.sel.assign((size_t) T * TOPK, -1);
    c.wt.assign((size_t) T * TOPK, 0.0f);
    std::mt19937 g(seed);
    std::vector<double> p(n_exp);
    for (int e = 0; e < n_exp; ++e) p[e] = skew > 0 ? 1.0 / std::pow(e + 8.0, skew) : 1.0;
    std::discrete_distribution<int> pick(p.begin(), p.end());
    std::uniform_real_distribution<float> uw(0.05f, 0.4f);
    for (int t = 0; t < T; ++t) {
        if (std::find(idle_tokens.begin(), idle_tokens.end(), t) != idle_tokens.end()) continue;
        for (int j = 0; j < TOPK; ++j) {
            int e;
            do e = pick(g);
            while (std::find(&c.sel[(size_t) t * TOPK], &c.sel[(size_t) t * TOPK] + j, e) != &c.sel[(size_t) t * TOPK] + j);
            c.sel[(size_t) t * TOPK + j] = e;
            c.wt[(size_t) t * TOPK + j] = uw(g);
        }
    }
    c.start.assign(E + 1, 0);
    for (int32_t e : c.sel)
        if (e >= 0) c.start[e + 1]++;
    std::partial_sum(c.start.begin(), c.start.end(), c.start.begin());
    c.tok.resize(c.start[E]);
    c.w.resize(c.start[E]);
    std::vector<int32_t> fill(c.start.begin(), c.start.end() - 1);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < TOPK; ++j) {
            const int e = c.sel[(size_t) t * TOPK + j];
            if (e < 0) continue;
            c.tok[fill[e]] = t;
            c.w[fill[e]] = c.wt[(size_t) t * TOPK + j];
            fill[e]++;
        }
    return c;
}

/// Device data of a chunk and the calls that cover it: call i uses experts [first_i, first_i + n_i)
struct Run {
    const Chunk& c;
    Dev<__half> x;
    Dev<int32_t> tok;
    Dev<float> w;
    std::vector<std::pair<int, int>> calls;  // (first expert, n_groups)
    std::vector<std::vector<int32_t>> offs;  // host off[] per call
    size_t ws_bytes = 0;
    Run(const Chunk& ch, int call_experts, uint32_t seed)
        : c(ch), x(to_half(rand_f32((size_t) ch.T * H, 1.0f, seed))), tok(ch.tok.empty() ? std::vector<int32_t>{0} : ch.tok),
          w(ch.w.empty() ? std::vector<float>{0} : ch.w) {
        int max_rows = 1;
        for (int first = 0; first < E; first += call_experts) {
            const int n = std::min(call_experts, E - first);
            calls.push_back({first, n});
            offs.emplace_back(c.start.begin() + first, c.start.begin() + first + n + 1);
            max_rows = std::max(max_rows, offs.back().back() - offs.back().front());
        }
        ws_bytes = kk::exl3_moe_prefill_workspace_bytes(max_rows, call_experts);
    }
    static std::vector<__half> to_half(const std::vector<float>& f) {
        std::vector<__half> h(f.size());
        for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half(f[i]);
        return h;
    }
    void enqueue(const kk::Exl3Expert* experts, float* out, void* ws, cudaStream_t st) const {
        for (size_t i = 0; i < calls.size(); ++i)
            kk::exl3_moe_prefill(x.p, tok.p, w.p, offs[i].data(), calls[i].second, experts + calls[i].first, out,
                                 ws, ws_bytes, st);
    }
};

/// K10, 8 tokens per call: the reference
void reference(const Chunk& c, const Run& r, const kk::Exl3Expert* experts, float* out) {
    Dev<int32_t> sel(c.sel.empty() ? std::vector<int32_t>{-1} : c.sel);
    Dev<float> wt(c.wt.empty() ? std::vector<float>{0} : c.wt);
    const size_t ws10 = 64ull << 20;
    Dev<uint8_t> ws(ws10);
    for (int t0 = 0; t0 < c.T; t0 += 8) {
        const int m = std::min(8, c.T - t0);
        kk::exl3_moe_decode(r.x.p + (size_t) t0 * H, m, sel.p + (size_t) t0 * TOPK, wt.p + (size_t) t0 * TOPK, TOPK,
                            experts, out + (size_t) t0 * H, ws.p, ws10, 0);
    }
    ck(cudaDeviceSynchronize(), "reference");
}

/// relative L2 of (got - init) against (want - init)
double delta_err(const std::vector<float>& got, const std::vector<float>& want, const std::vector<float>& init) {
    std::vector<float> a(got.size()), b(got.size());
    for (size_t i = 0; i < got.size(); ++i) {
        a[i] = got[i] - init[i];
        b[i] = want[i] - init[i];
    }
    return rel_l2(a, b);
}

void mixed_k(Verdict& v, bool native_qdq = true) {
    Experts ex(6, true);
    for (auto& expert : ex.host) expert.native_qdq = native_qdq;
    Dev<kk::Exl3Expert> experts(ex.host);
    // Six unique choices from six experts select every K in every projection.
    const Chunk c = make_chunk(37, 6, 0.0, 159, {5, 20});
    const Run r(c, 3, 159);  // Second nonempty call has off[0] > 0.
    Dev<uint8_t> ws(r.ws_bytes);
    const auto init = rand_f32(size_t(c.T) * H, 0.5f, 159);
    Dev<float> want(init), got(init);
    reference(c, r, experts.p, want.p);
    // Restrict empty trailing calls to the allocated descriptor array.
    auto enqueue = [&](cudaStream_t st) {
        for (size_t i = 0; i < 2; ++i)
            kk::exl3_moe_prefill(r.x.p, r.tok.p, r.w.p, r.offs[i].data(), 3,
                                 experts.p + i * 3, got.p, ws.p, r.ws_bytes, st);
    };
    enqueue(0);
    ck(cudaDeviceSynchronize(), "mixed prefill");
    const auto result = got.down();
    const double err = delta_err(result, want.down(), init);
    std::printf("mixed K1..K6 prefill T=37 native_qdq=%d rel_l2=%.6g\n", native_qdq, err);
    v.check(err <= 1e-2, "mixed K prefill: relative L2 above 1e-2 against K10");
    bool idle_same = true;
    for (int t : {5, 20}) for (int i = 0; i < H; ++i)
        idle_same &= result[size_t(t) * H + i] == init[size_t(t) * H + i];
    v.check(idle_same, "mixed K prefill: idle row changed");
    graph_check(v, "mixed K prefill T=37",
                [&](cudaStream_t st) {
                    ck(cudaMemsetAsync(got.p, 0, got.n * sizeof(float), st), "mixed reset");
                    enqueue(st);
                }, [&] { return as_doubles(got.down()); }, [&] { poison_dev(got); });
}

void grouped_k(Verdict& v,bool native_qdq,int T,int n_exp=6) {
    Experts ex(n_exp,true);
    for(auto& expert:ex.host) expert.native_qdq=native_qdq;
    // Reverse descriptor slots to exercise the residency mapping.
    std::reverse(ex.host.begin(),ex.host.end());
    Dev<kk::Exl3Expert> experts(ex.host);
    std::vector<int32_t> mapping(n_exp);std::iota(mapping.rbegin(),mapping.rend(),0);Dev<int32_t> residency(mapping);
    const Chunk c=make_chunk(T,n_exp,n_exp>6?1.:0.,713,{0,T-1});
    const Run r(c,3,713);
    Dev<int32_t> ids(c.sel),active(std::vector<int32_t>{0});Dev<float> weights(c.wt);
    const auto init=rand_f32(size_t(T)*H,.5f,714);
    Dev<float> want(init),got(init);
    Dev<uint8_t> old_ws(r.ws_bytes);
    // Baseline descriptors indexed by expert id, without the reverse mapping.
    std::reverse(ex.host.begin(),ex.host.end());Dev<kk::Exl3Expert> ordered(ex.host);
    for(int i=0;i<n_exp/3;++i)
        kk::exl3_moe_prefill(r.x.p,r.tok.p,r.w.p,r.offs[i].data(),3,ordered.p+i*3,want.p,old_ws.p,r.ws_bytes,0);
    const size_t bytes=kk::exl3_moe_grouped_workspace_bytes(T,TOPK,n_exp);Dev<uint8_t> ws(bytes);
    auto enqueue=[&](cudaStream_t st) {
        kk::exl3_moe_prefill_grouped(r.x.p,ids.p,weights.p,T,TOPK,experts.p,residency.p,n_exp,got.p,ws.p,bytes,active.p,st);
    };
    enqueue(0);ck(cudaDeviceSynchronize(),"grouped prefill");
    const auto result=got.down();const double err=delta_err(result,want.down(),init);
    std::printf("grouped K1..K6 T=%d experts=%d native_qdq=%d rel_l2=%.9g workspace=%.1f MiB\n",T,n_exp,native_qdq,err,bytes/1048576.);
    v.check(std::isfinite(err) && err<1e-2,"grouped K: relative L2 above 1e-2 against K12");
    int active_expected=0;for(int e=0;e<n_exp;++e) active_expected+=c.start[e+1]>c.start[e];
    v.check(active.down()[0]==active_expected,"grouped K: active group count");
    bool idle_same=true;for(int t:{0,T-1}) for(int d=0;d<H;++d) idle_same &= result[size_t(t)*H+d]==init[size_t(t)*H+d];
    v.check(idle_same,"grouped K: idle row changed");
    graph_check(v,"grouped K graph",
        [&](cudaStream_t st) { ck(cudaMemsetAsync(got.p,0,got.n*sizeof(float),st),"grouped reset");enqueue(st); },
        [&] { return as_doubles(got.down()); },[&] { poison_dev(got);poison_dev(ws); });
    if(T==129) {
        const double grouped_us=median_us([&] { enqueue(0); },5);
        const double old_us=median_us([&] {
            for(int i=0;i<2;++i) kk::exl3_moe_prefill(r.x.p,r.tok.p,r.w.p,r.offs[i].data(),3,ordered.p+i*3,want.p,old_ws.p,r.ws_bytes,0);
        },5);
        std::printf("grouped synthetic T=%d K12_us=%.1f grouped_us=%.1f\n",T,old_us,grouped_us);
    }
    bool rejected=false;
    try { kk::exl3_moe_prefill_grouped(r.x.p,ids.p,weights.p,T,TOPK,experts.p,residency.p,n_exp,got.p,ws.p,bytes-1,nullptr,0); }
    catch(const std::invalid_argument&) { rejected=true; }
    v.check(rejected,"grouped K: short workspace accepted");
}

}  // namespace

int main(int argc, char** argv) {
    require_gpu();
    Verdict v;
    for(int T:{1,37,129}) { grouped_k(v,true,T);grouped_k(v,false,T); }
    grouped_k(v,false,1024,12);
    mixed_k(v);
    mixed_k(v, false);
    if (argc == 2 && std::string(argv[1]) == "--component-only") return v.finish();
    Experts ex;
    Dev<kk::Exl3Expert> experts(ex.host);

    struct Case {
        const char* name;
        int T, n_exp;
        double skew;
        std::vector<int> idle;
        int call_experts;
    };
    const std::vector<Case> cases = {
        {"T=1, 384 groups (most empty)", 1, E, 0.0, {}, E},
        {"T=37, 64 experts, tokens 5 and 20 idle", 37, 64, 0.0, {5, 20}, 64},
        {"T=512, skewed over 384, 6 calls", 512, E, 1.0, {}, CALL_EXPERTS},
    };
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case& k = cases[ci];
        const Chunk c = make_chunk(k.T, k.n_exp, k.skew, 100 + (uint32_t) ci, k.idle);
        const Run r(c, k.call_experts, 200 + (uint32_t) ci);
        Dev<uint8_t> ws(r.ws_bytes);
        const std::vector<float> init = rand_f32((size_t) k.T * H, 0.5f, 300 + (uint32_t) ci);
        Dev<float> want(init), got(init);
        reference(c, r, experts.p, want.p);
        r.enqueue(experts.p, got.p, ws.p, 0);
        ck(cudaDeviceSynchronize(), "run");
        const auto g = got.down(), wv = want.down();
        const double err = delta_err(g, wv, init);
        double rms = 0;
        for (size_t i = 0; i < wv.size(); ++i) rms += ((double) wv[i] - init[i]) * ((double) wv[i] - init[i]);
        std::printf("%s: rows %zu, reference output rms %.4g, rel_l2 %.3g\n", k.name, c.tok.size(),
                    std::sqrt(rms / wv.size()), err);
        v.check(err <= 1e-2, std::string(k.name) + ": relative L2 error above 1e-2 against K10");
        bool idle_same = true;
        for (int t : k.idle)
            for (int i = 0; i < H; ++i) idle_same &= g[(size_t) t * H + i] == init[(size_t) t * H + i];
        v.check(idle_same, std::string(k.name) + ": a token without assignments changed");
        if (ci == 1)
            graph_check(v, "exl3_moe_prefill T=37",
                        [&](cudaStream_t st) {
                            ck(cudaMemsetAsync(got.p, 0, got.n * sizeof(float), st), "reset out");
                            r.enqueue(experts.p, got.p, ws.p, st);
                        },
                        [&] { return as_doubles(got.down()); }, [&] { poison_dev(got); });
    }

    // speed: a 4096-token chunk and a 512-token chunk, skewed routing over 384 experts, 6 calls of 64 experts
    double us4k = 0;
    for (int T : {512, 4096}) {
        const Chunk c = make_chunk(T, E, 1.0, 400 + (uint32_t) T);
        const Run r(c, CALL_EXPERTS, 500 + (uint32_t) T);
        Dev<uint8_t> ws(r.ws_bytes);
        Dev<float> out((size_t) T * H);
        ck(cudaMemset(out.p, 0, out.n * sizeof(float)), "zero");
        const double us = median_us([&] { r.enqueue(experts.p, out.p, ws.p, 0); }, T >= 4096 ? 11 : 25);
        std::printf("  time T=%d (%zu rows, workspace %.0f MiB): %.1f us\n", T, c.tok.size(), r.ws_bytes / 1048576.0, us);
        v.metric(T == 512 ? "us_t512" : "us_t4096", us);
        if (T == 4096) {
            us4k = us;
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a);
            reference(c, r, experts.p, out.p);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("  for comparison, K10 8 tokens per call, T=4096: %.1f us\n", ms * 1000.0);
            v.metric("k10_loop_us_t4096", ms * 1000.0);
        }
    }
    v.metric("score_us", us4k);
    return v.finish();
}
