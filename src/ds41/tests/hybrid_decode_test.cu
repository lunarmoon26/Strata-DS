// Synthetic hybrid decode checks. No model pack or golden files are needed.
#include "bench_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/doorbell.hpp"
#include "strata/ds41/kernels/k10_exl3_moe.hpp"
#include "strata/ds41/ops.hpp"

#include <numeric>
#include <memory>

using namespace ds41test;
namespace sd = strata::ds41;
namespace kk = strata::ds41::kernels;

namespace {
constexpr int H = 5120, F = 2304, K = 6, E = 8;

bool same_proj(const kk::Exl3Proj& a, const kk::Exl3Proj& b) {
    return a.trellis == b.trellis && a.suh == b.suh && a.svh == b.svh &&
           a.k == b.k && a.n == b.n && a.tile_w == b.tile_w && a.codebook == b.codebook;
}
bool same_expert(const kk::Exl3Expert& a, const kk::Exl3Expert& b) {
    return same_proj(a.w1, b.w1) && same_proj(a.w3, b.w3) && same_proj(a.w2, b.w2);
}

// Real K10 shapes. Each projection uses a different integer rate in K1..K6.
// The same bytes have both a mapped host address and a VRAM address.
struct Fixture {
    uint16_t* host = nullptr;
    uint16_t* alias = nullptr;
    Dev<uint16_t> vram;
    std::vector<kk::Exl3Expert> hd, vd;
    std::vector<sd::ExpertBlob> blobs;
    bool mcg;
    int uniform_rate = 0;
    static constexpr size_t trellis = size_t(H / 16) * (F / 16) * 96;
    static constexpr size_t per = 128 + 3 * (trellis + H + F);
    explicit Fixture(bool use_mcg = false, int rate = 0) : vram(per * E), hd(E), vd(E), blobs(E), mcg(use_mcg), uniform_rate(rate) {
        ck(cudaHostAlloc(&host, per * E * 2, cudaHostAllocMapped), "mapped experts");
        ck(cudaHostGetDevicePointer(&alias, host, 0), "expert alias");
        std::memset(host, 0x5a, per * E * 2);
        std::mt19937 rng(907);
        size_t at = 0;
        for (int e = 0; e < E; ++e) {
            const size_t start = at;
            at += 128; // Nonzero first-component offset, as allowed by pack metadata.
            auto proj = [&](int k, int n, int bits, kk::Exl3Proj& a, kk::Exl3Proj& b) {
                a = {alias + at, nullptr, nullptr, k, n, bits * 16};
                b = {vram.p + at, nullptr, nullptr, k, n, bits * 16};
                a.codebook = b.codebook = mcg ? 1 : 2;
                const size_t words = size_t(k / 16) * (n / 16) * bits * 16;
                for (size_t i = 0; i < words; ++i) host[at + i] = uint16_t(rng());
                at += words;
                a.suh = (const __half*) (alias + at);
                b.suh = (const __half*) (vram.p + at);
                for (int i = 0; i < k; ++i)
                    host[at++] = __half_as_ushort(__float2half((rng() & 1) ? 1.0f : -1.0f));
                a.svh = (const __half*) (alias + at);
                b.svh = (const __half*) (vram.p + at);
                for (int i = 0; i < n; ++i)
                    host[at++] = __half_as_ushort(__float2half((rng() & 1) ? 0.02f : -0.02f));
            };
            proj(H, F, mcg ? 2 : uniform_rate ? uniform_rate : e % 6 + 1, hd[e].w1, vd[e].w1);
            proj(H, F, mcg ? 2 : uniform_rate ? uniform_rate : (e + 1) % 6 + 1, hd[e].w3, vd[e].w3);
            proj(F, H, mcg ? 2 : uniform_rate ? uniform_rate : (e + 5) % 6 + 1, hd[e].w2, vd[e].w2);
            blobs[e] = {(at - start) * 2, 256};
        }
        ck(cudaMemcpy(vram.p, host, per * E * 2, cudaMemcpyHostToDevice), "expert upload");
    }
    ~Fixture() { cudaFreeHost(host); }
};

// The production resident split bypasses the CPU doorbell. Compare its actual
// route mapping + direct descriptor-table K10 path with the original all-VRAM
// doorbell path, including non-identity slot order and graph replays.
void resident_split(Verdict& v, Fixture& f) {
    sd::ExpertDoorbell db(1, K, H);
    Dev<uint16_t> x(H);
    std::vector<uint16_t> hx(H);
    for (int i = 0; i < H; ++i) hx[i] = __half_as_ushort(__float2half(float(i % 31 - 15) / 32));
    x.up(hx);
    Dev<int32_t> ids(K), sel(K), ref_sel(K), res(sd::kExperts);
    Dev<float> weights(K), got(H), ref(H);
    Dev<kk::Exl3Expert> descriptors(f.vd);
    Dev<uint8_t> workspace(64ull << 20);
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "resident test stream");
    auto enqueue = [&] {
        sd::ops::resident_routes(ids.p, res.p, sel.p, stream);
        ck(cudaMemsetAsync(got.p, 0, H * sizeof(float), stream), "resident output reset");
        kk::exl3_moe_decode((const __half*) x.p, 1, sel.p, weights.p, K, descriptors.p,
                            got.p, workspace.p, workspace.n, stream);
    };
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "resident capture");
    enqueue();
    ck(cudaStreamEndCapture(stream, &graph), "resident end capture");
    ck(cudaGraphInstantiate(&exec, graph, 0), "resident instantiate");
    for (int replay = 0; replay < 12; ++replay) {
        std::vector<int32_t> residency(sd::kExperts, -1), route(K), expected(K);
        std::vector<float> w(K);
        for (int e = 0; e < E; ++e) residency[e] = (e + 3 + replay) % E;
        for (int j = 0; j < K; ++j) {
            route[j] = (j * 3 + replay) % E;
            expected[j] = residency[route[j]];
            w[j] = float(j + 1) / 32;
        }
        ids.up(route); res.up(residency); weights.up(w);
        ck(cudaDeviceSynchronize(), "resident changed inputs");
        db.reset();
        db.publish(x.p, ids.p, weights.p, 1, res.p, ref_sel.p, 1, stream, descriptors.p);
        ck(cudaMemsetAsync(ref.p, 0, H * sizeof(float), stream), "resident reference reset");
        kk::exl3_moe_decode((const __half*) x.p, 1, ref_sel.p, weights.p, K, db.gpu_experts(),
                            ref.p, workspace.p, workspace.n, stream);
        ck(cudaStreamSynchronize(stream), "resident reference");
        const auto want = ref.down();
        for (bool captured : {false, true}) {
            if (captured) ck(cudaGraphLaunch(exec, stream), "resident replay");
            else enqueue();
            ck(cudaStreamSynchronize(stream), "resident output");
            const auto output = got.down();
            v.check(sel.down() == expected, "resident routes preserve exact slot/order");
            v.check(std::memcmp(want.data(), output.data(), H * sizeof(float)) == 0,
                    "resident K10 equals original all-VRAM doorbell path bitwise");
            bool nonzero = false;
            for (float value : output) { v.check(std::isfinite(value), "resident output finite"); nonzero |= value != 0; }
            v.check(nonzero, "resident output is nontrivial");
            v.check(db.counts().cpu == 0 && db.counts().skipped == 0 && db.counts().vram == K,
                    "resident reference computes all six routed experts");
        }
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    v.metric(f.mcg ? "mcg_resident_windows" : "resident_windows", 24);
}

void run(Verdict& v, Fixture& f, int m, bool staged) {
    sd::ExpertDoorbell db(8, K, H), direct_db(8, K, H);
    size_t largest = 0;
    for (const auto& b : f.blobs) largest = std::max(largest, b.bytes);
    std::unique_ptr<sd::ExpertStaging> stage;
    if (staged) stage = std::make_unique<sd::ExpertStaging>(m * K, largest);
    Dev<sd::ExpertBlob> blobs(f.blobs);
    Dev<uint16_t> x(size_t(m) * H);
    std::vector<uint16_t> hx(x.n);
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = __half_as_ushort(__float2half(float(int(i % 17) - 8) / 16));
    x.up(hx);
    Dev<int32_t> ids(m * K), sel(m * K), ref_sel(m * K), quota(1), res(E);
    Dev<float> weights(m * K), out(size_t(m) * H), ref(size_t(m) * H);
    Dev<kk::Exl3Expert> vd(f.vd), ram(E);
    Dev<uint8_t> ws(64ull << 20);
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    auto enqueue = [&] {
        db.publish(x.p, ids.p, weights.p, m, res.p, sel.p, 1, stream, vd.p, ram.p, quota.p,
                   stage.get(), staged ? blobs.p : nullptr);
        if (stage) stage->fork_copy(stream);
        // Independent main-stream work between fork and join.
        ck(cudaMemsetAsync(out.p, 0, out.n * sizeof(float), stream), "out reset");
        if (stage) stage->join(stream);
        kk::exl3_moe_decode((const __half*) x.p, m, sel.p, weights.p, K, db.gpu_experts(),
                            out.p, ws.p, ws.n, stream);
    };
    // Warm up before capture. Every route is inactive.
    ids.up(std::vector<int32_t>(m * K, -1));
    res.up(std::vector<int32_t>(E, -1));
    ram.up(std::vector<kk::Exl3Expert>(E));
    quota.up({0});
    weights.up(std::vector<float>(m * K, 0.125f));
    ck(cudaDeviceSynchronize(), "publish warm-up inputs");
    enqueue();
    ck(cudaStreamSynchronize(stream), "warm up");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
    enqueue();
    ck(cudaStreamEndCapture(stream, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");

    // Replay one graph while routes, weights, residency, RAM entries, and quota change.
    for (int mode = 0; mode < 4; ++mode) {
        std::vector<int32_t> residency(E, -1);
        std::vector<kk::Exl3Expert> rh(E);
        for (int e = 0; e < E; ++e) {
            if (mode == 1 || (mode == 0 && e % 4 == 0)) residency[e] = (e + 3) % E;
            if (mode == 2 || (mode == 0 && e % 4 != 3)) rh[e] = f.hd[e];
        }
        // mode 0: mixed; 1: all VRAM; 2: all RAM; 3: all file.
        res.up(residency);
        ram.up(rh);
        for (int q = 0; q <= K; ++q) {
            std::vector<int32_t> routes(m * K), expected(m * K, -1), cpu(m * K, -1);
            std::vector<float> w(m * K);
            int nv = 0, nz = 0, nc = 0;
            for (int t = 0; t < m; ++t) {
                int used = 0;
                for (int j = 0; j < K; ++j) {
                    const int i = t * K + j;
                    const int e = (j + t + q) % E;
                    routes[i] = (t == m - 1 && j == K - 1 && q == 0) ? -1 : e;
                    w[i] = float(i % 5 + q + 1) / 32;
                    if (routes[i] < 0) continue;
                    if (residency[e] >= 0) { expected[i] = residency[e]; ++nv; }
                    else if (rh[e].w1.trellis && used < q) { expected[i] = e; ++used; ++nz; }
                    else { cpu[i] = e; ++nc; }
                }
                v.check(used <= q, "quota per token");
            }
            ids.up(routes);
            weights.up(w);
            quota.up({q});
            ref_sel.up(expected);
            ck(cudaDeviceSynchronize(), "publish changed inputs");
            ck(cudaMemsetAsync(ref.p, 0, ref.n * sizeof(float), stream), "reference reset");
            if (staged) direct_db.publish(x.p, ids.p, weights.p, m, res.p, ref_sel.p, 1, stream,
                                          vd.p, ram.p, quota.p);
            kk::exl3_moe_decode((const __half*) x.p, m, ref_sel.p, weights.p, K,
                                staged ? direct_db.gpu_experts() : vd.p,
                                ref.p, ws.p, ws.n, stream);
            ck(cudaStreamSynchronize(stream), "reference");
            const auto want = ref.down();
            for (bool replay : {false, true}) {
                db.reset();
                if (stage)
                    ck(cudaMemsetAsync(stage->data(), 0xa5, stage->stride() * m * K, stream), "poison staging");
                if (replay) ck(cudaGraphLaunch(exec, stream), "replay");
                else enqueue();
                ck(cudaStreamSynchronize(stream), "hybrid run");
                const auto got = out.down();
                const auto selection = sel.down();
                std::vector<kk::Exl3Expert> call(m * K);
                ck(cudaMemcpy(call.data(), db.gpu_experts(), call.size() * sizeof(call[0]),
                              cudaMemcpyDeviceToHost), "call descriptors");
                v.check(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0,
                        staged ? "staged and direct K10 are bitwise equal, eager and graph"
                               : "mapped and VRAM K10 are bitwise equal, eager and graph");
                bool nonzero = false;
                for (float value : got) {
                    v.check(std::isfinite(value), "finite K10 output");
                    nonzero |= value != 0;
                }
                if (nv + nz) v.check(nonzero, "nontrivial K10 output");
                for (int i = 0; i < m * K; ++i) {
                    v.check(db.ids()[i] == cpu[i], "CPU keeps exactly the unassigned routes");
                    v.check(db.w()[i] == w[i], "weights keep routing order");
                    v.check(selection[i] == (expected[i] < 0 ? -1 : i), "GPU keeps routing order");
                    if (expected[i] >= 0) {
                        const int e = routes[i];
                        auto desc = residency[e] >= 0 ? f.vd[residency[e]] : f.hd[e];
                        if (staged && residency[e] < 0) {
                            const auto src = reinterpret_cast<uintptr_t>(desc.w1.trellis) - f.blobs[e].first_trellis;
                            const auto dst = reinterpret_cast<uintptr_t>(stage->data()) + size_t(i) * stage->stride();
                            for (auto* p : {&desc.w1, &desc.w3, &desc.w2}) {
                                p->trellis = reinterpret_cast<const uint16_t*>(dst + reinterpret_cast<uintptr_t>(p->trellis) - src);
                                p->suh = reinterpret_cast<const __half*>(dst + reinterpret_cast<uintptr_t>(p->suh) - src);
                                p->svh = reinterpret_cast<const __half*>(dst + reinterpret_cast<uintptr_t>(p->svh) - src);
                            }
                        }
                        v.check(same_expert(call[i], desc), "all descriptor fields match");
                    } else {
                        v.check(call[i].w1.trellis == nullptr, "inactive descriptors are cleared");
                    }
                }
                v.check(std::memcmp(db.x(), hx.data(), hx.size() * 2) == 0, "input published");
                const auto counts = db.counts();
                v.check(counts.vram == nv && counts.zero_copy == nz && counts.cpu == nc,
                        "counts partition active route uses");
                if (stage) {
                    int count = -1;
                    ck(cudaMemcpy(&count, stage->count(), sizeof(count), cudaMemcpyDeviceToHost), "copy count");
                    v.check(count == nz, "copy count resets on each publish, including quota zero");
                    std::vector<sd::ExpertCopy> jobs(m * K);
                    ck(cudaMemcpy(jobs.data(), stage->jobs(), jobs.size() * sizeof(jobs[0]),
                                  cudaMemcpyDeviceToHost), "copy jobs");
                    int at = 0;
                    for (int i = 0; i < m * K; ++i) {
                        const int e = routes[i];
                        if (e < 0 || residency[e] >= 0 || expected[i] < 0) continue;
                        const auto* src = reinterpret_cast<const uint8_t*>(f.hd[e].w1.trellis) - f.blobs[e].first_trellis;
                        v.check(jobs[at].src == src && jobs[at].bytes == f.blobs[e].bytes &&
                                jobs[at].dst == stage->data() + size_t(i) * stage->stride(),
                                "compact copy list contains only eligible RAM experts and exact blob sizes");
                        ++at;
                    }
                }
            }
        }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}
// Null tables cover configurations with no RAM tier or no VRAM tier.
void absent_tiers(Verdict& v, Fixture& f) {
    sd::ExpertDoorbell db(1, K, H);
    Dev<uint16_t> x(std::vector<uint16_t>(H, 0));
    Dev<int32_t> ids(std::vector<int32_t>{0, 1, 2, 3, 4, 5});
    Dev<float> w(std::vector<float>(K, 1));
    Dev<int32_t> res(std::vector<int32_t>{0, -1, -1, -1, -1, -1, -1, -1}), sel(K), q(1);
    Dev<kk::Exl3Expert> vd(f.vd), ram(f.hd);
    for (int quota : {-1, 0, 4, 6, 9}) {
        q.up({quota});
        for (int mode = 0; mode < 3; ++mode) {
            db.reset();
            db.publish(x.p, ids.p, w.p, 1, mode == 1 ? res.p : nullptr, sel.p, 1, nullptr,
                       mode == 1 ? vd.p : nullptr, mode == 2 ? ram.p : nullptr, q.p);
            ck(cudaDeviceSynchronize(), "absent tier publish");
            const auto got = sel.down();
            const int nz = mode == 2 ? std::max(0, std::min(quota, K)) : 0;
            const int nv = mode == 1 ? 1 : 0;
            for (int j = 0; j < K; ++j) {
                const bool gpu = j < nz || (nv && j == 0);
                v.check(got[j] == (gpu ? j : -1) && db.ids()[j] == (gpu ? -1 : j),
                        "null tables preserve CPU fallback and quota bounds");
            }
            const auto c = db.counts();
            v.check(c.vram == nv && c.zero_copy == nz && c.cpu == K - nv - nz, "null table counts");
        }
    }
    // A mapped table alone must not enable zero-copy without a quota pointer.
    db.publish(x.p, ids.p, w.p, 1, nullptr, sel.p, 1, nullptr, nullptr, ram.p, nullptr);
    ck(cudaDeviceSynchronize(), "null quota publish");
    v.check(db.counts().cpu == K && db.counts().zero_copy == 0, "null quota means zero");
}
// Compare multi-row expert execution to independent single-token calls. Covers
// duplicate groups larger than eight, inactive uses and changing graph routes.
void grouped_rows(Verdict& v, Fixture& f) {
    constexpr int m = 8;
    std::vector<uint16_t> hx(m * H);
    for (int i = 0; i < m * H; ++i)
        hx[i] = __half_as_ushort(__float2half(float(i % 37 - 18) / 64));
    Dev<uint16_t> x(hx);
    Dev<int32_t> ids(m * K);
    Dev<float> weights(m * K), got(m * H), ref(m * H);
    Dev<kk::Exl3Expert> descriptors(f.vd);
    Dev<uint8_t> workspace(64ull << 20);
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "grouped stream");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "grouped capture");
    kk::exl3_moe_decode((const half*)x.p,m,ids.p,weights.p,K,descriptors.p,
                       got.p,workspace.p,workspace.n,stream);
    ck(cudaStreamEndCapture(stream,&graph), "grouped capture end");
    ck(cudaGraphInstantiate(&exec,graph,0), "grouped instantiate");
    for (int pass = 0; pass < 10; ++pass) {
        std::vector<int32_t> route(m * K);
        std::vector<float> w(m * K), initial(m * H);
        for (int i = 0; i < m * K; ++i) {
            route[i] = pass < 8 ? pass : (i * 3 + i / K) % E;
            if (pass == 9 && i % 3 == 0) route[i] = -1;
            w[i] = float(i % 11 + 1) / 32;
        }
        for (int i = 0; i < m * H; ++i) initial[i] = float(i % 7 - 3) / 128;
        ids.up(route); weights.up(w); got.up(initial); ref.up(initial);
        ck(cudaGraphLaunch(exec,stream), "grouped replay");
        for (int t = 0; t < m; ++t)
            kk::exl3_moe_decode((const half*)x.p+t*H,1,ids.p+t*K,weights.p+t*K,K,
                               descriptors.p,ref.p+t*H,workspace.p,workspace.n,stream);
        ck(cudaStreamSynchronize(stream), "grouped comparison");
        const auto a = got.down(), b = ref.down();
        v.check(std::memcmp(a.data(),b.data(),a.size()*sizeof(float)) == 0,
                "grouped expert rows equal single-token arithmetic exactly");
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
}
// The opt-in per-rate expert dispatch must equal the generic mixed-rate kernel
// bit for bit on single-rate tables, including inactive routes and CUDA-graph
// replay (production captures the dispatch inside its verifier graphs).
void rate_dispatch(Verdict& v, Fixture& f, int m, bool native_qdq) {
    const int rate = f.uniform_rate;
    std::vector<uint16_t> hx(m * H);
    for (int i = 0; i < m * H; ++i)
        hx[i] = __half_as_ushort(__float2half(float(i % 23 - 11) / 64));
    Dev<uint16_t> x(hx);
    Dev<int32_t> ids(m * K);
    Dev<float> weights(m * K), got(m * H), ref(m * H);
    auto expert_table = f.vd;
    for (auto& expert : expert_table) expert.native_qdq = native_qdq;
    Dev<kk::Exl3Expert> descriptors(expert_table);
    Dev<uint8_t> workspace(64ull << 20);
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "rate stream");
    for (int pass = 0; pass < 6; ++pass) {
        std::vector<int32_t> route(m * K);
        std::vector<float> w(m * K), initial(m * H);
        for (int i = 0; i < m * K; ++i) {
            route[i] = (i * 5 + pass) % E;
            if (pass % 3 == 0 && i % 4 == 0) route[i] = -1;
            w[i] = float(i % 9 + 1) / 16;
        }
        for (int i = 0; i < m * H; ++i) initial[i] = float(i % 5 - 2) / 128;
        ids.up(route); weights.up(w); got.up(initial); ref.up(initial);
        ck(cudaStreamSynchronize(nullptr), "rate uploads complete");
        kk::exl3_moe_decode((const half*)x.p, m, ids.p, weights.p, K, descriptors.p, ref.p,
                            workspace.p, workspace.n, stream, 0);
        kk::exl3_moe_decode((const half*)x.p, m, ids.p, weights.p, K, descriptors.p, got.p,
                            workspace.p, workspace.n, stream, rate);
        ck(cudaStreamSynchronize(stream), "rate comparison");
        const auto a = got.down(), b = ref.down();
        v.check(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                "per-rate expert dispatch equals the generic kernel bit for bit");
    }
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "rate capture");
    kk::exl3_moe_decode((const half*)x.p, m, ids.p, weights.p, K, descriptors.p, got.p,
                        workspace.p, workspace.n, stream, rate);
    ck(cudaStreamEndCapture(stream, &graph), "rate capture end");
    ck(cudaGraphInstantiate(&exec, graph, 0), "rate instantiate");
    for (int replay = 0; replay < 5; ++replay) {
        std::vector<int32_t> route(m * K);
        std::vector<float> w(m * K), initial(m * H);
        for (int i = 0; i < m * K; ++i) {
            route[i] = (i * 7 + replay * 3) % E;
            if (replay % 2 && i % 5 == 0) route[i] = -1;
            w[i] = float(i % 7 + 1) / 32;
        }
        for (int i = 0; i < m * H; ++i) initial[i] = float(i % 9 - 4) / 256;
        ids.up(route); weights.up(w); got.up(initial); ref.up(initial);
        ck(cudaStreamSynchronize(nullptr), "rate replay uploads complete");
        ck(cudaGraphLaunch(exec, stream), "rate replay");
        kk::exl3_moe_decode((const half*)x.p, m, ids.p, weights.p, K, descriptors.p, ref.p,
                            workspace.p, workspace.n, stream, 0);
        ck(cudaStreamSynchronize(stream), "rate replay comparison");
        const auto a = got.down(), b = ref.down();
        v.check(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                "per-rate expert dispatch stays exact under graph replay");
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
}
}  // namespace

int main() {
    require_gpu();
    Verdict v;
    Fixture fixture;
    resident_split(v, fixture);
    grouped_rows(v, fixture);
    absent_tiers(v, fixture);
    for (bool staged : {false, true})
        for (int m = 1; m <= 8; ++m) run(v, fixture, m, staged);
    Fixture mcg(true);
    resident_split(v, mcg);
    grouped_rows(v, mcg);
    // The hint is only honored when the opt-in environment gate is enabled.
    if (const char* enabled = std::getenv("DS41_EXPERIMENTAL_EXPERT_RATE_KERNEL");
        enabled && !std::strcmp(enabled, "1")) {
        for (int rate : {1, 2, 3, 4, 5, 6}) {
            Fixture uniform(false, rate);
            rate_dispatch(v, uniform, 4, false);
            rate_dispatch(v, uniform, 4, true);
            if (rate <= 4) {
                rate_dispatch(v, uniform, 1, false);
                rate_dispatch(v, uniform, 8, true);
            }
        }
        std::printf("RATE_DISPATCH exact=1 rates=1,2,3,4,5,6 rows=1,4,8 qdq=0,1 graph_replay=1\n");
    }
    return v.finish();
}
