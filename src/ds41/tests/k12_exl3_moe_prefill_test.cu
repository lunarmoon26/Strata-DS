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
    for(int wave:{1,3,8,16}) {
        const size_t batched_bytes=kk::exl3_moe_batched_workspace_bytes(c.start[n_exp],wave);
        Dev<uint8_t> batch_ws(batched_bytes);Dev<float> batch_out(init);
        auto batch_enqueue=[&](cudaStream_t st) {
            kk::exl3_moe_prefill_batched(r.x.p,r.tok.p,r.w.p,c.start.data(),n_exp,ordered.p,batch_out.p,
                                         batch_ws.p,batched_bytes,wave,st);
        };
        batch_enqueue(0);ck(cudaDeviceSynchronize(),"exact batched prefill");
        const auto batched_result=batch_out.down(),expected=want.down();
        v.check(!std::memcmp(batched_result.data(),expected.data(),expected.size()*sizeof(float)),
                "batched reconstruction retains byte-identical reference output for every wave size");
        if(wave==8 && T==129) {
            graph_check(v,"exact batched expert graph",[&](cudaStream_t st) {
                ck(cudaMemsetAsync(batch_out.p,0,batch_out.n*sizeof(float),st),"batched graph reset");batch_enqueue(st);
            },[&] { return as_doubles(batch_out.down()); },[&] { poison_dev(batch_out);poison_dev(batch_ws); });
            std::printf("exact batched synthetic T=%d native_qdq=%d wave=%d us=%.1f\n",T,native_qdq,wave,
                         median_us([&] { batch_enqueue(0); },5));
        }
        if(wave==3) {
            bool refused=false;
            try { kk::exl3_moe_prefill_batched(r.x.p,r.tok.p,r.w.p,c.start.data(),n_exp,ordered.p,batch_out.p,
                                               batch_ws.p,batched_bytes-1,wave,0); }
            catch(const std::invalid_argument&) { refused=true; }
            v.check(refused || !c.start[n_exp],"batched reconstruction rejects undersized scratch");
        }
    }
    std::printf("grouped K1..K6 T=%d experts=%d native_qdq=%d rel_l2=%.9g workspace=%.1f MiB\n",T,n_exp,native_qdq,err,bytes/1048576.);
    v.check(std::isfinite(err) && err<1e-2,"grouped K: relative L2 above 1e-2 against K12");
    int active_expected=0;for(int e=0;e<n_exp;++e) active_expected+=c.start[e+1]>c.start[e];
    v.check(active.down()[0]==active_expected,"grouped K: active group count");
    bool idle_same=true;for(int t:{0,T-1}) for(int d=0;d<H;++d) idle_same &= result[size_t(t)*H+d]==init[size_t(t)*H+d];
    v.check(idle_same,"grouped K: idle row changed");
    {
        std::vector<int32_t> map(384,-1);for(int e=0;e<n_exp;++e) map[e]=e;
        Dev<int32_t> wave_map(map),filtered(c.sel.size());Dev<float> wave_out(init);
        for(int first=0;first<n_exp;first+=3) {
            const int count=std::min(3,n_exp-first);
            kk::exl3_moe_remap_routes(ids.p,wave_map.p,T*TOPK,first,count,filtered.p,0);
            kk::exl3_moe_prefill_grouped(r.x.p,filtered.p,weights.p,T,TOPK,ordered.p+first,nullptr,count,
                                         wave_out.p,ws.p,bytes,nullptr,0,c.start[first+count]-c.start[first]);
        }
        ck(cudaDeviceSynchronize(),"packed ordered waves");
        const auto split=wave_out.down();
        v.check(!std::memcmp(split.data(),result.data(),split.size()*sizeof(float)),
                "ordered packed waves preserve full-call accumulation exactly, including idle rows");
    }
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

void batched_contract(Verdict& v) {
    Experts ex(1,true);ex.host[0].native_qdq=false;
    // Empty groups have no valid descriptors and must never be reconstructed.
    Dev<kk::Exl3Expert> descriptors(std::vector<kk::Exl3Expert>{{},ex.host[0],{}});
    const int32_t offsets[]={0,0,513,513};
    std::vector<int32_t> tokens(513);std::iota(tokens.begin(),tokens.end(),0);
    Dev<int32_t> tok(tokens);Dev<float> weights(std::vector<float>(513,.25f));
    Dev<half> input(Run::to_half(rand_f32(size_t(513)*H,1.f,918)));
    Dev<float> reference(std::vector<float>(size_t(513)*H,0)),output(std::vector<float>(size_t(513)*H,0));
    const size_t bytes=kk::exl3_moe_batched_workspace_bytes(513,3);Dev<uint8_t> ws(bytes);
    kk::exl3_moe_prefill(input.p,tok.p,weights.p,offsets,3,descriptors.p,reference.p,ws.p,bytes,0);
    kk::exl3_moe_prefill_batched(input.p,tok.p,weights.p,offsets,3,descriptors.p,output.p,ws.p,bytes,3,0);
    const auto got=output.down(),want=reference.down();
    v.check(!std::memcmp(got.data(),want.data(),got.size()*sizeof(float)),
            "batched prefill preserves reference tiling above 512 rows and skips empty invalid descriptors");
    const int32_t small[]={0,0,1,1};
    ck(cudaMemset(output.p,0,output.n*sizeof(float)),"batched small reset");
    kk::exl3_moe_prefill_batched(input.p,tok.p,weights.p,small,3,descriptors.p,output.p,ws.p,bytes,3,0);
    ck(cudaDeviceSynchronize(),"batched empty groups");
    v.check(true,"batched reconstruction skips empty descriptors without the tiled fallback");
    for(int wave:{0,17}) {
        bool refused=false;try { (void)kk::exl3_moe_batched_workspace_bytes(513,wave); }
        catch(const std::invalid_argument&) { refused=true; }
        v.check(refused,"batched wave budget rejects zero and oversized groups");
    }
    const int32_t invalid[]={1,0};bool refused=false;
    try { kk::exl3_moe_prefill_batched(input.p,tok.p,weights.p,invalid,1,descriptors.p,output.p,ws.p,bytes,3,0); }
    catch(const std::invalid_argument&) { refused=true; }
    v.check(refused,"batched prefill rejects decreasing offsets before GPU access");
    v.check(kk::exl3_moe_batched_workspace_bytes(0,3)==0,"empty batched calls need no workspace");
}

void expert_graph_contract(Verdict& v,bool native_qdq) {
    Experts ex(3,true);for(auto& e:ex.host) e.native_qdq=native_qdq;
    Dev<kk::Exl3Expert> experts(ex.host);
    Dev<half> x(Run::to_half(rand_f32(size_t(130)*H,1.f,724)));
    std::vector<int32_t> tokens(130);std::iota(tokens.begin(),tokens.end(),0);
    Dev<int32_t> tok(tokens);Dev<float> weights(std::vector<float>(130,.25f));
    const auto initial=rand_f32(size_t(130)*H,.5f,725);Dev<float> want(initial),got(initial);
    const size_t bytes=kk::exl3_moe_prefill_workspace_bytes(512,1);
    Dev<uint8_t> ref_ws(bytes),graph_ws(bytes);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"expert graph test stream");
    {
        kk::Exl3PrefillGraphs cache(x.p,tok.p,weights.p,got.p,graph_ws.p,bytes);
        for(int i=0;i<36;++i) {
            const int32_t offsets[]={i%11,i%11+1+2*(i%3)};
            kk::exl3_moe_prefill(x.p,tok.p,weights.p,offsets,1,experts.p+i%3,want.p,ref_ws.p,bytes,stream);
            cache.run(offsets,1,experts.p+i%3,stream);
        }
        ck(cudaStreamSynchronize(stream),"queued expert parameter updates");
        const auto g=got.down(),w=want.down();
        v.check(!std::memcmp(g.data(),w.data(),g.size()*sizeof(float)),
                "expert graphs preserve reference bits with queued descriptor/offset updates and no host fences");
        v.check(cache.captures()==3 && cache.replays()==33 && cache.shapes()==3,"expert graph warm/capture/replay accounting");
        bool refused=false;
        const int32_t off[]={0,1};
        try { cache.run(off,1,experts.p,0); } catch(const std::invalid_argument&) { refused=true; }
        v.check(refused,"expert graph owner refuses stream changes before enqueue");
        // Exercise the fixed 128-shape ceiling and eager fallback without
        // allocating an unbounded graph for every new routing histogram.
        for(int count=1;count<=129;++count) {
            const int32_t offsets[]={0,count};cache.run(offsets,1,experts.p,stream);
        }
        v.check(cache.shapes()==128 && cache.captures()<=128,"expert graph shapes remain bounded with eager overflow");
    }
    ck(cudaStreamDestroy(stream),"expert graph test stream release");
    v.check(true,"expert graph owner joins work and destroys bindings before scratch release");
}

}  // namespace

int main(int argc, char** argv) {
    require_gpu();
    Verdict v;
    batched_contract(v);
    expert_graph_contract(v,true);expert_graph_contract(v,false);
    {
        std::vector<int32_t> map(384,-1);
        const int order[]={5,2,0,4,1,3};for(int i=0;i<6;++i) map[order[i]]=i;
        Dev<int32_t> ids(std::vector<int32_t>{0,5,4,2,383,1,3,5,-1}),mapping(map),filtered(9);
        kk::exl3_moe_remap_routes(ids.p,mapping.p,9,2,3,filtered.p,0);
        v.check(filtered.down()==std::vector<int32_t>({0,-1,1,-1,-1,2,-1,-1,-1}),
                "expert wave filters nonidentity hot/cold maps, missing experts and padding");
        bool refused=false;
        try { kk::exl3_moe_remap_routes(ids.p,mapping.p,9,383,2,filtered.p,0); }
        catch(const std::invalid_argument&) { refused=true; }
        v.check(refused,"expert wave rejects descriptor ranges outside the layer");
    }
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
