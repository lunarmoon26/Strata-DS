// Included by engine.cu after Impl (as verify.cu). DSpark, DeepSeek's MTP drafter (model.py DSparkBlock,
// Transformer.forward_spec): three blocks over a block of 5 rows (the token after the committed ones, then 4 noise
// tokens) draw the next 5 tokens in one pass. Its attention reads a ring of the last 128 committed positions' main_kv,
// projected from the attention input of backbone layers 37-39 (main_hiddens, captured by step, verify and prefill).
// Its routed experts (3 of 128 per stage) stay in RAM and the decode worker computes them; the rest runs on the GPU.
// Drafts only choose what a verify window checks: nothing here changes the output.
namespace dsk = strata::ds41::dspark;

struct Engine::Impl::Dspark {
    static constexpr int B = dsk::kBlock;
    static constexpr int kHid = dsk::kTargets * kDim;   // 15360: the three target layers' means, concatenated
    static constexpr int kMaxPend = kWindow;             // pending main_hiddens rows (a prefill pass leaves 128)
    static constexpr int kPrm = 16;                      // params: [0..4] tokens, [5..9] positions, [10] start_pos
    std::vector<std::shared_ptr<void>> mem;
    template <class T> T* alloc(size_t n) {
        T* p = dalloc<T>(std::max<size_t>(1, n));
        mem.emplace_back(p, [](void* q) { cudaFree(q); });
        return p;
    }
    Layer S[dsk::kStages];
    Fp8 main_proj{};
    const bf16 *main_norm = nullptr, *norm = nullptr, *m_embed = nullptr, *m_head = nullptr, *conf_w = nullptr;
    // main_hiddens: the step's row and the verify window's rows (captured inside their graphs), and the pending rows
    // whose main_kv the next draft writes into the rings first (mh rows 0..pend_n-1, positions pend_pos0 on)
    bf16 *step_mh = nullptr, *verify_mh = nullptr, *mh = nullptr;
    int pend_n = 0, pend_pos0 = 0;
    // ring update scratch: 8 rows at a time (the FP8 GEMV's rows)
    bf16 *mx = nullptr, *mkv = nullptr;
    // the block's rows
    bf16 *h, *h2, *xa, *xf, *qr, *q, *kvv, *o, *oa, *attn_out, *g, *u, *sh_h, *sh_out, *ffn_out, *xfin, *xn, *memb;
    float *act, *mix, *apre, *apost, *acomb, *fpre, *fpost, *fcomb, *routed, *logits, *conf, *weights;
    uint16_t* half = nullptr;
    int32_t *ids = nullptr, *sel = nullptr, *idx = nullptr, *out = nullptr;
    int *prm = nullptr, *prm_host = nullptr;   // [kPrm + kMaxPend]: then the pending rows' positions
    int32_t* out_host = nullptr;
    float* conf_host = nullptr;
    // DS41_DSPARK_PROFILE=<file>: every draft's routes ([stage][B][3], ids_host) are counted and written there as
    // "stage expert uses" lines when the engine ends (the profile of DSpark's VRAM set)
    int32_t* ids_host = nullptr;
    std::string profile_path;
    std::vector<int64_t> uses = std::vector<int64_t>(dsk::kStages * dsk::kExperts, 0);
    std::unique_ptr<ExpertDoorbell> db;
    int64_t moe[dsk::kStages] = {-1, -1, -1};
    // the VRAM set (DS41_DSPARK_VRAM_MB): DSpark's hottest experts in VRAM slots, computed by K10 on the GPU, the
    // others by the CPU. Scores decay per draft; before a draft the hottest expert outside VRAM replaces the coldest
    // inside (copied from the RAM copy, registered for DMA)
    int vslots = 0;
    size_t vslot_bytes = 0;
    uint8_t* varena = nullptr;
    kernels::Exl3Expert* vdesc = nullptr;       // [vslots] device
    int32_t* vres = nullptr;                    // [stage][expert] device: the slot, or -1
    std::vector<int32_t> vres_host;
    std::vector<int> vowner;                    // [slot]: stage * kExperts + expert, or -1
    std::vector<float> score = std::vector<float>(dsk::kStages * dsk::kExperts, 0.0f);
    void* vws = nullptr;                        // K10 workspace
    const void* registered = nullptr;
    int last_hits = 0, last_uses = 0;
    std::map<int, cudaGraphExec_t> graphs;   // by pending rows (0..kVerifyMaxTokens)
    bool warmed = false;
    std::unique_ptr<BoundedUpload> published_experts;
    uint8_t* published_slot = nullptr;
    kernels::Exl3Expert* published_desc = nullptr;
    void* dense = nullptr;
    size_t dense_bytes = 0, cache_bytes = 0, cache_limit = 0;
    uint64_t cache_clock = 0;
    struct Cached { uint8_t* data; size_t bytes; kernels::Exl3Expert desc; uint64_t used; };
    std::map<int,Cached> published_cache;
    int cache_hits = 0;
    uint64_t cache_hit_mask = 0; // one bit per routed draft use, for exact TP intersection
    ~Dspark() {
        for(auto& [id,entry]:published_cache) cudaFree(entry.data);
        for (auto& [k, e] : graphs) cudaGraphExecDestroy(e);
        for (int64_t hnd : moe)
            if (hnd >= 0) exl3_moe_cpu_free_layer(hnd);
        if (!profile_path.empty())
            if (std::FILE* f = std::fopen(profile_path.c_str(), "w")) {
                std::fprintf(f, "# ds41 DSpark routes: stage expert uses\n");
                for (int s = 0; s < dsk::kStages; ++s)
                    for (int e = 0; e < dsk::kExperts; ++e)
                        std::fprintf(f, "%d %d %lld\n", s, e, (long long) uses[s * dsk::kExperts + e]);
                std::fclose(f);
            }
        if (registered) cudaHostUnregister(const_cast<void*>(registered));
        if (ids_host) cudaFreeHost(ids_host);
        if (prm_host) cudaFreeHost(prm_host);
        if (out_host) cudaFreeHost(out_host);
        if (conf_host) cudaFreeHost(conf_host);
    }
};

/// DSpark's weights, buffers and CPU expert layers. Its experts are already in RAM (load_mtp_experts, before the RAM
/// tier sized itself); the buffers are allocated before the VRAM tier takes the free memory.
void Engine::Impl::dspark_init() {
    if (pack.mtp_stages() != dsk::kStages || pack.mtp_experts() != dsk::kExperts)
        throw std::runtime_error("ds41 dspark: the pack has no DSpark experts (tools/ds41/pack.py --mtp-experts)");
    ds = std::make_shared<Dspark>();
    auto& d = *ds;
    for (int s = 0; s < dsk::kStages; ++s) {
        auto& y = d.S[s];
        const std::string p = "mtp." + std::to_string(s) + ".";
        y.wq_a = fp8(p + "attn.wq_a");
        y.wq_b = fp8(p + "attn.wq_b");
        y.wkv = fp8(p + "attn.wkv");
        y.wo_b = fp8(p + "attn.wo_b");
        if (pack.has_dense(p + "attn.wo_a.slice.0.trellis")) {
            for (int g = 0; g < kOGroups; ++g)
                y.wo_a_exl3[g] = pack.exl3_projection(p + "attn.wo_a.slice." + std::to_string(g),
                                                     kHeads * kHeadDim / kOGroups, kOLora);
        } else y.wo_a8 = fp8(p + "attn.wo_a");
        y.q_norm = bf(p + "attn.q_norm.weight");
        y.kv_norm = bf(p + "attn.kv_norm.weight");
        y.sink = f32(p + "attn.attn_sink");
        y.attn_norm = bf(p + "attn_norm.weight");
        y.ffn_norm = bf(p + "ffn_norm.weight");
        y.hc_attn_fn = f32(p + "hc_attn_fn");
        y.hc_attn_base = f32(p + "hc_attn_base");
        y.hc_attn_scale = f32(p + "hc_attn_scale");
        y.hc_ffn_fn = f32(p + "hc_ffn_fn");
        y.hc_ffn_base = f32(p + "hc_ffn_base");
        y.hc_ffn_scale = f32(p + "hc_ffn_scale");
        y.gate_w = bf(p + "ffn.gate.weight");
        y.gate_bias = f32(p + "ffn.gate.bias");
        y.sh_w1 = fp8(p + "ffn.shared_experts.w1");
        y.sh_w2 = fp8(p + "ffn.shared_experts.w2");
        y.sh_w3 = fp8(p + "ffn.shared_experts.w3");
        y.window = d.alloc<bf16>((size_t) kWindow * kHeadDim);
        if (pack.has_exl3_dense()) continue; // ordinary FP16 experts execute on GPU, never the native-QDQ CPU kernel
        std::vector<MoeCpuMatrixDesc> gate(dsk::kExperts), up(dsk::kExperts), down(dsk::kExperts);
        const uint8_t* base = pack.mtp_expert_base();
        for (int e = 0; e < dsk::kExperts; ++e) {
            const auto& x = pack.mtp_expert(s, e);
            auto desc = [&](int c0, int k_tiles, int n_tiles) {
                MoeCpuMatrixDesc m;
                m.trellis = (const uint16_t*) (base + x.offset + x.comp_off[c0]);
                m.suh = (const at::Half*) (base + x.offset + x.comp_off[c0 + 1]);
                m.svh = (const at::Half*) (base + x.offset + x.comp_off[c0 + 2]);
                m.k_tiles = k_tiles;
                m.n_tiles = n_tiles;
                m.tile_w = (int) (x.comp_bytes[c0] / ((uint64_t) k_tiles * n_tiles * 2));
                return m;
            };
            gate[e] = desc(0, kDim / 16, kMoeInter / 16);
            up[e] = desc(4, kDim / 16, kMoeInter / 16);
            down[e] = desc(8, kMoeInter / 16, kDim / 16);
        }
        d.moe[s] = exl3_moe_cpu_make_layer_raw(gate.data(), up.data(), down.data(), dsk::kExperts, 0, kSwigluLimit, 0);
    }
    d.main_proj = fp8("mtp.0.main_proj");
    if (d.main_proj.k != Dspark::kHid || d.main_proj.n != kDim) throw std::runtime_error("ds41 dspark: main_proj shape");
    d.main_norm = bf("mtp.0.main_norm.weight");
    d.norm = bf("mtp.2.norm.weight");
    d.m_embed = bf("mtp.2.markov_head.embed.weight");
    d.m_head = bf("mtp.2.markov_head.head.weight");
    d.conf_w = bf("mtp.2.confidence_head.proj.weight");
    const int B = Dspark::B, M = kVerifyMaxTokens;
    d.step_mh = d.alloc<bf16>(Dspark::kHid);
    d.verify_mh = d.alloc<bf16>((size_t) M * Dspark::kHid);
    d.mh = d.alloc<bf16>((size_t) Dspark::kMaxPend * Dspark::kHid);
    d.mx = d.alloc<bf16>(8 * kDim);
    d.mkv = d.alloc<bf16>(8 * kHeadDim);
    d.h = d.alloc<bf16>(B * kHc * kDim); d.h2 = d.alloc<bf16>(B * kHc * kDim);
    d.xa = d.alloc<bf16>(B * kDim); d.xf = d.alloc<bf16>(B * kDim);
    d.qr = d.alloc<bf16>(B * kQLora); d.q = d.alloc<bf16>((size_t) B * kHeads * kHeadDim);
    d.kvv = d.alloc<bf16>(B * kHeadDim); d.o = d.alloc<bf16>((size_t) B * kHeads * kHeadDim);
    d.oa = d.alloc<bf16>(B * kOGroups * kOLora); d.attn_out = d.alloc<bf16>(B * kDim);
    d.g = d.alloc<bf16>(B * kMoeInter); d.u = d.alloc<bf16>(B * kMoeInter); d.sh_h = d.alloc<bf16>(B * kMoeInter);
    d.sh_out = d.alloc<bf16>(B * kDim); d.ffn_out = d.alloc<bf16>(B * kDim);
    d.xfin = d.alloc<bf16>(B * kDim); d.xn = d.alloc<bf16>(B * kDim); d.memb = d.alloc<bf16>(B * dsk::kMarkovRank);
    d.act = d.alloc<float>((size_t) 8 * Dspark::kHid);
    d.mix = d.alloc<float>(B * kHc);
    d.apre = d.alloc<float>(B * kHc); d.apost = d.alloc<float>(B * kHc); d.acomb = d.alloc<float>(B * kHc * kHc);
    d.fpre = d.alloc<float>(B * kHc); d.fpost = d.alloc<float>(B * kHc); d.fcomb = d.alloc<float>(B * kHc * kHc);
    d.routed = d.alloc<float>(B * kDim);
    d.logits = d.alloc<float>((size_t) B * kVocab);
    d.conf = d.alloc<float>(B);
    d.weights = d.alloc<float>(B * dsk::kTopK);
    d.half = d.alloc<uint16_t>(B * kDim);
    d.ids = d.alloc<int32_t>(dsk::kStages * B * dsk::kTopK);   // per stage: the profile reads all three
    d.sel = d.alloc<int32_t>(B * dsk::kTopK);
    d.idx = d.alloc<int32_t>(B * (kWindow + B));
    d.out = d.alloc<int32_t>(B);
    d.prm = d.alloc<int>(Dspark::kPrm + Dspark::kMaxPend);
    ck(cudaHostAlloc((void**) &d.prm_host, (Dspark::kPrm + Dspark::kMaxPend) * sizeof(int), cudaHostAllocDefault),
       "dspark params");
    ck(cudaHostAlloc((void**) &d.out_host, B * sizeof(int32_t), cudaHostAllocDefault), "dspark drafts");
    ck(cudaHostAlloc((void**) &d.conf_host, B * sizeof(float), cudaHostAllocDefault), "dspark confidence");
    ck(cudaHostAlloc((void**) &d.ids_host, dsk::kStages * B * dsk::kTopK * sizeof(int32_t), cudaHostAllocDefault),
       "dspark routes");
    if (const char* f = std::getenv("DS41_DSPARK_PROFILE")) d.profile_path = f;
    if (pack.has_exl3_dense()) {
        d.dense_bytes=exl3_small_rows_workspace_bytes(8,32768,kVocab);
        d.dense=d.alloc<uint8_t>(d.dense_bytes);
        d.published_experts = std::make_unique<BoundedUpload>(pack.dir() + "/mtp_experts.bin");
        size_t largest = 0;
        for (int s = 0; s < dsk::kStages; ++s)
            for (int e = 0; e < dsk::kExperts; ++e) {
                const auto& x = pack.mtp_expert(s, e);
                d.published_experts->validate(x.offset, x.bytes);
                largest = std::max<size_t>(largest, x.bytes);
            }
        d.published_slot = d.alloc<uint8_t>(largest);
        d.published_desc = d.alloc<kernels::Exl3Expert>(B*dsk::kTopK);
        d.vws = d.alloc<uint8_t>(VramExperts::kWorkspaceBytes);
        ck(cudaMemset(d.sel, 0, sizeof(int32_t)), "draft selected expert");
        std::fprintf(stderr, "ds41: published EXL3 DSpark, bounded single-expert GPU staging (%zu bytes); no CPU/requantization\n", largest);
        return;
    }
    d.db = std::make_unique<ExpertDoorbell>(B, dsk::kTopK, kDim);
    {
        const char* v = std::getenv("DS41_DSPARK_VRAM_MB");
        const size_t mb = v ? (size_t) std::max(0, std::atoi(v)) : kDsparkVramMb;
        for (int s = 0; s < dsk::kStages; ++s)
            for (int e = 0; e < dsk::kExperts; ++e)
                d.vslot_bytes = std::max<size_t>(d.vslot_bytes, pack.mtp_expert(s, e).bytes);
        d.vslot_bytes = (d.vslot_bytes + 255) & ~(size_t) 255;
        d.vslots = (int) std::min<size_t>((mb << 20) / d.vslot_bytes, dsk::kStages * dsk::kExperts);
    }
    d.vres_host.assign(dsk::kStages * dsk::kExperts, -1);
    if (d.vslots > 0) {
        d.varena = d.alloc<uint8_t>((size_t) d.vslots * d.vslot_bytes);
        d.vdesc = d.alloc<kernels::Exl3Expert>(d.vslots);
        d.vres = d.alloc<int32_t>(dsk::kStages * dsk::kExperts);
        ck(cudaMemcpy(d.vres, d.vres_host.data(), d.vres_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
           "dspark vram table");
        d.vws = d.alloc<uint8_t>(VramExperts::kWorkspaceBytes);
        d.vowner.assign(d.vslots, -1);
        const uint64_t bytes = pack.load_mtp_experts();   // already loaded: its size
        if (cudaHostRegister((void*) pack.mtp_expert_base(), bytes, cudaHostRegisterPortable) == cudaSuccess)
            d.registered = pack.mtp_expert_base();
        else
            cudaGetLastError();   // pageable copies still work, slower
        std::fprintf(stderr, "ds41: DSpark VRAM set %d experts (%.2f GiB)%s\n", d.vslots,
                     d.vslots * (double) d.vslot_bytes / (1ull << 30), d.registered ? "" : ", RAM copy not registered");
    }
}

/// Before a draft (the device is idle): the scores decay and take the last draft's routes; then the hottest experts
/// outside VRAM fill free slots (8 per draft) or replace the coldest resident when clearly hotter (2 per draft).
void Engine::Impl::dspark_vram_update() {
    auto& d = *ds;
    const int n = dsk::kStages * dsk::kExperts;
    if (d.vslots == 0) return;
    int free_n = 0;
    for (int o : d.vowner) free_n += o < 0;
    int budget = free_n > 0 ? 8 : 2;
    bool changed = false;
    std::vector<int> slots_changed;
    while (budget-- > 0) {
        int hot = -1;
        for (int i = 0; i < n; ++i)
            if (d.vres_host[i] < 0 && d.score[i] > 0 && (hot < 0 || d.score[i] > d.score[hot])) hot = i;
        if (hot < 0) break;
        int slot = -1;
        for (int k = 0; k < d.vslots && slot < 0; ++k)
            if (d.vowner[k] < 0) slot = k;
        if (slot < 0) {
            for (int k = 0; k < d.vslots; ++k)
                if (slot < 0 || d.score[d.vowner[k]] < d.score[d.vowner[slot]]) slot = k;
            if (d.score[hot] <= 1.25f * d.score[d.vowner[slot]] + 0.5f) break;
            d.vres_host[d.vowner[slot]] = -1;
        }
        const int st_i = hot / dsk::kExperts, e = hot % dsk::kExperts;
        const auto& x = pack.mtp_expert(st_i, e);
        uint8_t* dst = d.varena + (size_t) slot * d.vslot_bytes;
        ck(cudaMemcpy(dst, pack.mtp_expert_base() + x.offset, x.bytes, cudaMemcpyHostToDevice), "dspark vram copy");
        d.vowner[slot] = hot;
        d.vres_host[hot] = slot;
        auto proj = [&](int c0, int k, int nn) {
            kernels::Exl3Proj p;
            p.trellis = (const uint16_t*) (dst + x.comp_off[c0]);
            p.suh = (const __half*) (dst + x.comp_off[c0 + 1]);
            p.svh = (const __half*) (dst + x.comp_off[c0 + 2]);
            p.k = k;
            p.n = nn;
            p.tile_w = (int) (x.comp_bytes[c0] / ((uint64_t) (k / 16) * (nn / 16) * 2));
            return p;
        };
        kernels::Exl3Expert desc;
        desc.w1 = proj(0, kDim, kMoeInter);
        desc.w3 = proj(4, kDim, kMoeInter);
        desc.w2 = proj(8, kMoeInter, kDim);
        ck(cudaMemcpy(d.vdesc + slot, &desc, sizeof desc, cudaMemcpyHostToDevice), "dspark vram descriptor");
        changed = true;
    }
    if (changed)
        ck(cudaMemcpy(d.vres, d.vres_host.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice), "dspark vram table");
}

/// After a step or a commit: rows of `src` (main_hiddens of positions pos0, pos0 + 1, ...) join the pending rows when
/// they follow them (plain steps between drafts), else replace them. A later draft writes their main_kv into the rings
/// first; 128 pending rows are written at once, here. Replaced rows leave older rows in their ring slots, which costs
/// draft quality only.
void Engine::Impl::dspark_pend(const bf16* src, int n, int pos0) {
    auto& d = *ds;
    const bool follows = d.pend_n > 0 && d.pend_pos0 + d.pend_n == pos0;
    if (follows && d.pend_n + n > Dspark::kMaxPend) {   // full: write the rings now (the device is idle)
        for (int i = 0; i < d.pend_n; ++i) d.prm_host[Dspark::kPrm + i] = d.pend_pos0 + i;
        ck(cudaMemcpyAsync(d.prm, d.prm_host, (Dspark::kPrm + d.pend_n) * sizeof(int), cudaMemcpyHostToDevice, st),
           "dspark params");
        dspark_ring(d.pend_n, st);
        ck(cudaStreamSynchronize(st), "dspark ring");
        d.pend_pos0 += d.pend_n;
        d.pend_n = 0;
    }
    if (!follows || d.pend_n == 0) {
        d.pend_n = 0;
        d.pend_pos0 = pos0;
    }
    ck(cudaMemcpyAsync(d.mh + (size_t) d.pend_n * Dspark::kHid, src, (size_t) n * Dspark::kHid * sizeof(bf16),
                       cudaMemcpyDeviceToDevice, st), "dspark pend");
    d.pend_n += n;
}

/// The rings' rows of mh rows 0..n-1, whose positions are prm[kPrm + i] (DSparkAttention, the main_x part):
/// main_x = main_norm(main_proj(main_hiddens)); per stage main_kv = act_quant(rope(kv_norm(wkv(main_x)))) in slot
/// pos % 128. Eight rows at a time (the FP8 GEMV's rows).
void Engine::Impl::dspark_ring(int n, cudaStream_t s) {
    auto& d = *ds;
    for (int c0 = 0; c0 < n; c0 += 8) {
        const int mc = std::min(8, n - c0);
        if(d.main_proj.exl3.trellis)
            projection(d.mh+size_t(c0)*Dspark::kHid,d.main_proj.exl3,d.mx,nullptr,mc,Dspark::kHid,kDim,
                       d.dense,d.dense_bytes,s);
        else for (int r = 0; r < mc; ++r)
            fp8_linear_on(d.mh + size_t(c0 + r) * Dspark::kHid, d.main_proj, d.mx + r * kDim, d.act, s);
        ops::rmsnorm(d.mx, d.main_norm, d.mx, kDim, kNormEps, mc, s);
        for (int st_i = 0; st_i < dsk::kStages; ++st_i) {
            auto& y = d.S[st_i];
            if(y.wkv.exl3.trellis)
                projection(d.mx,y.wkv.exl3,d.mkv,nullptr,mc,kDim,kHeadDim,d.dense,d.dense_bytes,s);
            else for (int r = 0; r < mc; ++r)
                fp8_linear_on(d.mx + r * kDim, y.wkv, d.mkv + r * kHeadDim, d.act, s);
            ops::rmsnorm(d.mkv, y.kv_norm, d.mkv, kHeadDim, kNormEps, mc, s);
            for (int r = 0; r < mc; ++r) {
                const int* p = d.prm + Dspark::kPrm + c0 + r;
                bf16* kv = d.mkv + (size_t) r * kHeadDim;
                ops::rope_device(kv, 1, kHeadDim, rope_plain, p, 0, false, s);
                ops::act_quant_inplace(kv, kHeadDim, s);
                ops::row_copy_device(y.window, kv, kHeadDim * sizeof(bf16), p, 1, kWindow, s);
            }
        }
    }
}

/// One draft pass on st: the pending rows into the rings, then the three blocks over the block's 5 rows (the routed
/// experts on the CPU through ds->db), the head with the Markov bias draft by draft, and the confidence head.
void Engine::Impl::enqueue_draft(int n_pend) {
    auto& d = *ds;
    const int B = Dspark::B;
    ck(cudaMemcpyAsync(d.prm, d.prm_host, (Dspark::kPrm + std::max(n_pend, 0)) * sizeof(int), cudaMemcpyHostToDevice, st),
       "dspark params");
    if (n_pend > 0) dspark_ring(n_pend, st);
    auto lin = [&](const bf16* x, const Fp8& w, bf16* y) {
        if (w.exl3.trellis) {
            projection(x,w.exl3,y,nullptr,B,w.k,w.n,d.dense,d.dense_bytes,st);
            return;
        }
        fp8_quantize_activation_f32((const uint16_t*) x, B, w.k, d.act, st);
        fp8_block_gemv_q(d.act, B, w.k, w.w, w.s, w.n, (uint16_t*) y, st);
    };
    dsk::block_index(d.prm + 10, d.idx, st);
    for (int r = 0; r < B; ++r) {
        if (!partitioned()) embedding(d.prm+r,1,1,d.h+size_t(r)*kHc*kDim,st);
        ck(cudaMemcpyAsync(d.mix + r * kHc, one_hot_dev, kHc * sizeof(float), cudaMemcpyDeviceToDevice, st), "dspark mix");
    }
    for (int s = 0; s < dsk::kStages; ++s) {
        auto& y = d.S[s];
        kernels::hc_mixes_pre(d.h, B, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, d.mix, d.xa, d.apre, d.apost,
                              d.acomb, st);
        ops::rmsnorm(d.xa, y.attn_norm, d.xa, kDim, kNormEps, B, st);
        lin(d.xa, y.wkv, d.kvv);
        ops::rmsnorm(d.kvv, y.kv_norm, d.kvv, kHeadDim, kNormEps, B, st);
        lin(d.xa, y.wq_a, d.qr);
        ops::rmsnorm(d.qr, y.q_norm, d.qr, kQLora, kNormEps, B, st);
        lin(d.qr, y.wq_b, d.q);
        for (int r = 0; r < B; ++r) {   // block row r sits at position start_pos + 1 + r (plain rope: no compression)
            const int* p = d.prm + 5 + r;
            ops::rope_device(d.q + (size_t) r * kHeads * kHeadDim, kHeads, kHeadDim, rope_plain, p, 0, false, st);
            ops::rope_device(d.kvv + (size_t) r * kHeadDim, 1, kHeadDim, rope_plain, p, 0, false, st);
            ops::act_quant_inplace(d.kvv + (size_t) r * kHeadDim, kHeadDim, st);
        }
        // every block row attends to the ring's written slots and to all 5 block rows (no mask inside the block)
        kernels::sparse_attn_decode(d.q, y.window, d.kvv, d.idx, B, kWindow + B, y.sink,
                                    (float) std::pow(kHeadDim, -0.5), d.o, st);
        for (int r = 0; r < B; ++r)
            ops::rope_device(d.o + (size_t) r * kHeads * kHeadDim, kHeads, kHeadDim, rope_plain, d.prm + 5 + r, 0, true, st);
#ifdef STRATA_DS41_1017
        ops::act_quant_inplace(d.o, B * kHeads * kHeadDim, st);
#endif
        if (y.wo_a_exl3[0].trellis) {
            for (int g=0;g<kOGroups;++g)
                projection(d.o+g*(kHeads*kHeadDim/kOGroups),y.wo_a_exl3[g],d.oa+g*kOLora,nullptr,
                           B,kHeads*kHeadDim,kOGroups*kOLora,d.dense,d.dense_bytes,st);
        } else wo_a_grouped_fp8_rows(d.o, B, y.wo_a8.w, y.wo_a8.s, d.oa, st);
        lin(d.oa, y.wo_b, d.attn_out);
        ops::hc_post(d.attn_out, d.h, d.apost, d.acomb, d.h2, B, st);
        kernels::hc_mixes_pre(d.h2, B, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, d.apre, d.xf, d.fpre, d.fpost,
                              d.fcomb, st);
        ops::rmsnorm(d.xf, y.ffn_norm, d.xf, kDim, kNormEps, B, st);
        int32_t* ids = d.ids + s * B * dsk::kTopK;
        dsk::router_small(d.xf, B, y.gate_w, y.gate_bias, dsk::kExperts, dsk::kTopK, ids, d.weights, st);
        if (pack.has_exl3_dense()) exl3_input_half(d.xf, d.half, B * kDim, st);
        else ops::to_half_fp8q(d.xf, d.half, B * kDim, st);
        // VRAM-set experts: K10 on the GPU (descriptor mode); the rest: the CPU
        if (!pack.has_exl3_dense())
            d.db->publish(d.half, ids, d.weights, B, d.vslots ? d.vres + s * dsk::kExperts : nullptr, d.sel,
                          (uint32_t) (s + 1), st, d.vslots ? d.vdesc : nullptr);
        lin(d.xf, y.sh_w1, d.g);
        lin(d.xf, y.sh_w3, d.u);
        ops::swiglu(d.g, d.u, kSwigluLimit, d.sh_h, B * kMoeInter, st);
        lin(d.sh_h, y.sh_w2, d.sh_out);
        ck(cudaMemsetAsync(d.routed, 0, B * kDim * sizeof(float), st), "dspark routed");
        if (pack.has_exl3_dense()) {
            int32_t selected[Dspark::B * dsk::kTopK];
            ck(cudaMemcpyAsync(selected, ids, sizeof selected, cudaMemcpyDeviceToHost, st), "draft routes");
            ck(cudaStreamSynchronize(st), "draft routes ready");
            if(!d.cache_limit && opt.draft_cache_bytes) {
                size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total),"draft cache headroom");
                d.cache_limit=std::min(opt.draft_cache_bytes,free>opt.vram_reserve_bytes?free-opt.vram_reserve_bytes:0);
            }
            std::vector<int> pinned(selected,selected+B*dsk::kTopK);
            std::sort(pinned.begin(),pinned.end());pinned.erase(std::unique(pinned.begin(),pinned.end()),pinned.end());
            size_t needed=0;
            for(int expert:pinned) {
                if(expert<0 || expert>=dsk::kExperts) throw std::runtime_error("invalid DSpark expert");
                const auto& slot=pack.mtp_expert(s,expert);
                needed+=tensor?tensor_expert_bytes(slot):size_t(slot.bytes);
            }
            bool batch=needed<=d.cache_limit;
            if(tensor) batch=tensor->agree(batch);
            kernels::Exl3Expert descriptors[B*dsk::kTopK];
            if(!batch) ck(cudaMemsetAsync(d.sel,0,B*dsk::kTopK*sizeof(int32_t),st),"draft streaming selection");
            for (int r = 0; r < B; ++r) for (int j = 0; j < dsk::kTopK; ++j) {
                const int e = selected[r * dsk::kTopK + j];
                if (e < 0 || e >= dsk::kExperts) throw std::runtime_error("invalid DSpark expert");
                const auto& x = pack.mtp_expert(s, e);
                const int key=s*dsk::kExperts+e;
                auto cached=d.published_cache.find(key);
                uint8_t* base=d.published_slot;
                const size_t bytes=tensor ? tensor_expert_bytes(x) : size_t(x.bytes);
                if(cached!=d.published_cache.end()) {
                    base=cached->second.data;cached->second.used=++d.cache_clock;++d.cache_hits;
                    d.cache_hit_mask |= uint64_t(1) << (s*B*dsk::kTopK+r*dsk::kTopK+j);
                }
                else {
                    if(bytes<=d.cache_limit) {
                        while(d.cache_bytes+bytes>d.cache_limit && !d.published_cache.empty()) {
                            auto victim=d.published_cache.end();
                            for(auto it=d.published_cache.begin();it!=d.published_cache.end();++it) {
                                if(batch && it->first/dsk::kExperts==s &&
                                   std::binary_search(pinned.begin(),pinned.end(),it->first%dsk::kExperts)) continue;
                                if(victim==d.published_cache.end() || it->second.used<victim->second.used) victim=it;
                            }
                            if(victim==d.published_cache.end()) throw std::logic_error("draft cache pinned working set exceeds bound");
                            ck(cudaFree(victim->second.data),"draft cache evict");d.cache_bytes-=victim->second.bytes;
                            d.published_cache.erase(victim);
                        }
                        ck(cudaMalloc(&base,bytes),"draft cache allocate");
                        cached=d.published_cache.emplace(key,Dspark::Cached{base,bytes,{},++d.cache_clock}).first;
                        d.cache_bytes+=bytes;
                    }
                    if(!tensor) d.published_experts->copy(base,x.offset,x.bytes);
                }
                auto projection = [&](int c, int k, int n) {
                    kernels::Exl3Proj p{};
                    p.trellis = reinterpret_cast<uint16_t*>(base + x.comp_off[c]);
                    p.suh = reinterpret_cast<__half*>(base + x.comp_off[c + 1]);
                    p.svh = reinterpret_cast<__half*>(base + x.comp_off[c + 2]);
                    p.k = k; p.n = n;
                    p.tile_w = int(x.comp_bytes[c] / (uint64_t(k / 16) * (n / 16) * 2));
                    return p;
                };
                kernels::Exl3Expert desc{};
                desc.w1 = projection(0, kDim, kMoeInter); desc.w3 = projection(4, kDim, kMoeInter);
                desc.w2 = projection(8, kMoeInter, kDim); desc.native_qdq = false;
                if(tensor) {
                    if(cached!=d.published_cache.end() && cached->second.desc.w1.trellis) desc=cached->second.desc;
                    else desc=upload_tensor_expert(*d.published_experts,x,base,tensor_rank);
                }
                if(cached!=d.published_cache.end()) cached->second.desc=desc;
                descriptors[r*dsk::kTopK+j]=desc;
                if(batch) continue;
                ck(cudaMemcpyAsync(d.published_desc, &desc, sizeof desc, cudaMemcpyHostToDevice, st), "draft expert descriptor");
                if(tensor) {
                    kernels::exl3_moe_tensor_up(reinterpret_cast<const __half*>(d.half)+r*kDim,1,d.sel,
                                               d.weights+r*dsk::kTopK+j,1,d.published_desc,d.vws,
                                               VramExperts::kWorkspaceBytes,tensor_rank,st);
                    tensor->gather(tensor_rank,900+2*s,kernels::exl3_moe_tensor_hidden(d.vws,1,1),1,kMoeInter,2,st);
                    kernels::exl3_moe_tensor_down(1,d.sel,1,d.published_desc,d.routed+r*kDim,d.vws,
                                                 VramExperts::kWorkspaceBytes,tensor_rank,st);
                    tensor->gather(tensor_rank,901+2*s,d.routed+r*kDim,1,kDim,4,st);
                } else kernels::exl3_moe_decode(reinterpret_cast<const __half*>(d.half) + r * kDim, 1, d.sel,
                                         d.weights + r * dsk::kTopK + j, 1, d.published_desc,
                                         d.routed + r * kDim, d.vws, VramExperts::kWorkspaceBytes, st);
                ck(cudaStreamSynchronize(st), "draft expert complete");
            }
            if(batch) {
                int32_t selection[B*dsk::kTopK];
                for(int i=0;i<B*dsk::kTopK;++i) selection[i]=i;
                ck(cudaMemcpyAsync(d.published_desc,descriptors,sizeof descriptors,cudaMemcpyHostToDevice,st),"draft batched descriptors");
                ck(cudaMemcpyAsync(d.sel,selection,sizeof selection,cudaMemcpyHostToDevice,st),"draft batched selection");
                if(tensor) {
                    kernels::exl3_moe_tensor_up(reinterpret_cast<const __half*>(d.half),B,d.sel,d.weights,dsk::kTopK,
                                               d.published_desc,d.vws,VramExperts::kWorkspaceBytes,tensor_rank,st);
                    tensor->gather(tensor_rank,900+2*s,kernels::exl3_moe_tensor_hidden(d.vws,B,dsk::kTopK),B*dsk::kTopK,kMoeInter,2,st);
                    kernels::exl3_moe_tensor_down(B,d.sel,dsk::kTopK,d.published_desc,d.routed,d.vws,VramExperts::kWorkspaceBytes,tensor_rank,st);
                    tensor->gather(tensor_rank,901+2*s,d.routed,B,kDim,4,st);
                } else kernels::exl3_moe_decode(reinterpret_cast<const __half*>(d.half),B,d.sel,d.weights,dsk::kTopK,
                                                d.published_desc,d.routed,d.vws,VramExperts::kWorkspaceBytes,st);
                ck(cudaStreamSynchronize(st),"draft batched experts complete");
            }
        } else if (d.vslots)
            kernels::exl3_moe_decode((const __half*) d.half, B, d.sel, d.weights, dsk::kTopK, d.db->gpu_experts(),
                                     d.routed, d.vws, VramExperts::kWorkspaceBytes, st);
        if (!pack.has_exl3_dense()) d.db->wait_add(d.routed, B, (uint32_t) (s + 1), st);
        ops::add_f32_bf16(d.routed, d.sh_out, d.ffn_out, B * kDim, st);
        ops::hc_post(d.ffn_out, d.h2, d.fpost, d.fcomb, d.h, B, st);
        ck(cudaMemcpyAsync(d.mix, d.fpre, B * kHc * sizeof(float), cudaMemcpyDeviceToDevice, st), "dspark ffn_pre");
    }
    // forward_head: the collapse (its pre-norm rows feed the confidence head), norm, the backbone's head
    ops::hc_pre(d.h, d.mix, d.xfin, B, st);
    ops::rmsnorm(d.xfin, d.norm, d.xn, kDim, kNormEps, B, st);
    if (pack.has_exl3_dense()) {
        projection(d.xn,head_exl3,nullptr,d.logits,B,kDim,kVocab,d.dense,d.dense_bytes,st);
    } else ops::bf16_linear_rows(d.xn, B, head, kDim, kVocab, d.logits, st);
    for (int i = 0; i < B; ++i) {   // draft i: the Markov bias of the token before it, then the greedy pick
        dsk::markov_bias(i == 0 ? d.prm : d.out + i - 1, d.m_embed, d.m_head, kVocab, d.logits + (size_t) i * kVocab,
                         d.memb + i * dsk::kMarkovRank, st);
        ops::argmax_logits(d.logits + (size_t) i * kVocab, d.out + i, st);
    }
    dsk::confidence_rows(d.xfin, d.memb, d.conf_w, B, d.conf, st);
    ck(cudaMemcpyAsync(d.out_host, d.out, B * sizeof(int32_t), cudaMemcpyDeviceToHost, st), "dspark drafts");
    ck(cudaMemcpyAsync(d.conf_host, d.conf, B * sizeof(float), cudaMemcpyDeviceToHost, st), "dspark confidence");
    ck(cudaMemcpyAsync(d.ids_host, d.ids, dsk::kStages * B * dsk::kTopK * sizeof(int32_t), cudaMemcpyDeviceToHost, st),
       "dspark routes");
}

/// The decode worker's DSpark job: per stage, when the GPU publishes it, the block's routed experts from RAM.
void Engine::Impl::run_dspark_job(ExpertDoorbell* d, int m) {
    try {
        exl3_moe_cpu_pool_prime(cpu_threads);
    } catch (...) {
    }
    for (int s = 0; s < dsk::kStages; ++s) {
        if (!d->wait_published((uint32_t) (s + 1), job_cancel)) return;
        c10::Half wh[Dspark::B * dsk::kTopK];
        for (int i = 0; i < m * dsk::kTopK; ++i) {
            const __half hv = __float2half_rn(d->w()[i]);
            wh[i] = c10::Half(__half_as_ushort(hv), c10::Half::from_bits());
        }
        const double t0 = now_ms();
        try {
            if (fault("dspark")) throw std::runtime_error("ds41 test fault: DSpark experts");
            exl3_moe_cpu_forward_raw(ds->moe[s], (const at::Half*) d->x(), d->ids(), wh, d->y(), m, dsk::kTopK,
                                     cpu_threads);
        } catch (...) {
            std::fill_n(d->y(), (size_t) m * kDim, 0.0f);
            std::lock_guard<std::mutex> lk(mu);
            if (!worker_error) worker_error = std::current_exception();
        }
        worker_us += (int64_t) ((now_ms() - t0) * 1000.0);
        d->mark_done((uint32_t) (s + 1));
    }
}

Engine::Draft Engine::Impl::draft(int token) {
    DeviceScope scope(stage.device);
    usable();
    if (!ds) throw std::logic_error("ds41 dspark: the engine was created without EngineOptions::dspark");
    if (verify_pending) throw std::logic_error("commit the pending verify window first");
    check_token(token);
    auto& d = *ds;
    const int pos = (int) history.size();   // token's position; the committed tokens end at pos - 1 (start_pos)
    Draft r;
    if (pos < 1 || pos + Dspark::B > max_seq) return r;   // no committed token yet, or no room for the drafts
    const double begin = now_ms();
    d.cache_hits=0;
    d.cache_hit_mask=0;
    int* p = d.prm_host;
    p[0] = token;
    for (int i = 1; i < Dspark::B; ++i) p[i] = dsk::kNoiseToken;
    for (int i = 0; i < Dspark::B; ++i) p[5 + i] = pos + i;
    p[10] = pos - 1;
    for (int i = 0; i < d.pend_n; ++i) p[Dspark::kPrm + i] = d.pend_pos0 + i;
    int n_pend = d.pend_n;
    if (n_pend > kVerifyMaxTokens) {   // a prefill's rows: written eagerly, the graphs cover a window's rows
        ck(cudaMemcpyAsync(d.prm, d.prm_host, (Dspark::kPrm + n_pend) * sizeof(int), cudaMemcpyHostToDevice, st),
           "dspark params");
        dspark_ring(n_pend, st);
        n_pend = 0;
    }
    d.pend_n = 0;
    dspark_vram_update();
    cudaGraphExec_t exec = nullptr;
    if (use_graph && d.warmed && !pack.has_exl3_dense()) {
        auto it = d.graphs.find(n_pend);
        if (it == d.graphs.end()) {
            cudaGraph_t gr = nullptr;
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "dspark capture");
            try {
                enqueue_draft(n_pend);
                ck(cudaStreamEndCapture(st, &gr), "dspark end capture");
                ck(cudaGraphInstantiate(&exec, gr, 0), "dspark instantiate");
            } catch (...) {
                if (!gr) cudaStreamEndCapture(st, &gr);
                if (gr) cudaGraphDestroy(gr);
                throw;
            }
            cudaGraphDestroy(gr);
            it = d.graphs.emplace(n_pend, exec).first;
        }
        exec = it->second;
    }
    if (d.db) d.db->reset();
    worker_us = 0;
    if (!pack.has_exl3_dense()) {
        std::lock_guard<std::mutex> lk(mu);
        job_db = d.db.get();
        job_m = Dspark::B;
        job_dspark = true;
        job_cancel = false;
        ++go;
    }
    cv.notify_all();
    const uint64_t job = go;
    auto job_end = [&] {
        if (pack.has_exl3_dense()) return;
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return jobs_done >= job; });
    };
    try {
        if (exec) ck(cudaGraphLaunch(exec, st), "dspark launch");
        else enqueue_draft(n_pend);
        ck(cudaStreamSynchronize(st), "dspark draft");
    } catch (...) {
        job_cancel = true;
        if(tensor) tensor->abort();
        job_end();
        if (d.db) d.db->mark_done(dsk::kStages);
        cudaStreamSynchronize(st);
        broken = true;   // the rings may hold half-written rows and the stream state is unknown
        throw;
    }
    job_end();
    {
        std::lock_guard<std::mutex> lk(mu);
        if (worker_error) {
            std::exception_ptr e = worker_error;
            worker_error = nullptr;
            std::rethrow_exception(e);
        }
    }
    d.warmed = true;
    for (float& sc : d.score) sc *= 0.95f;
    d.last_hits = d.last_uses = 0;
    for (int i = 0; i < dsk::kStages * Dspark::B * dsk::kTopK; ++i) {
        const int e = d.ids_host[i];
        if (e < 0 || e >= dsk::kExperts) continue;
        const int k = (i / (Dspark::B * dsk::kTopK)) * dsk::kExperts + e;
        ++d.uses[k];
        d.score[k] += 1.0f;
        ++d.last_uses;
        d.last_hits += !d.vres_host.empty() && d.vres_host[k] >= 0;
    }
    r.n = Dspark::B;
    for (int i = 0; i < Dspark::B; ++i) {
        r.tokens[i] = d.out_host[i];
        r.accept[i] = 1.0f / (1.0f + std::exp(-d.conf_host[i]));
    }
    r.cpu_experts_ms = worker_us.load() / 1000.0;
    r.vram_hits = d.last_hits;
    if(pack.has_exl3_dense()) r.vram_hits=d.cache_hits;
    r.uses = d.last_uses;
    r.ms = now_ms() - begin;
    return r;
}

bf16* Engine::Impl::ds_step_mh() { return ds->step_mh; }
bf16* Engine::Impl::ds_verify_mh() { return ds->verify_mh; }
bf16* Engine::Impl::ds_pend_rows() { return ds->mh; }
void Engine::Impl::ds_set_pending(int n, int pos0) {
    ds->pend_n = n;
    ds->pend_pos0 = pos0;
}

void Engine::Impl::ds_flush_pending() {
    if (!ds || !ds->pend_n) return;
    for (int i = 0; i < ds->pend_n; ++i) ds->prm_host[Dspark::kPrm + i] = ds->pend_pos0 + i;
    ck(cudaMemcpyAsync(ds->prm, ds->prm_host, (Dspark::kPrm + ds->pend_n) * sizeof(int), cudaMemcpyHostToDevice, st), "draft pending parameters");
    dspark_ring(ds->pend_n, st);
    ck(cudaStreamSynchronize(st), "draft flush pending");
    ds->pend_n = 0;
}

void Engine::Impl::ds_slot_regions(std::vector<std::pair<void*, size_t>>& regions) {
    if (!ds) return;
    for (auto& y : ds->S) regions.push_back({y.window, size_t(kWindow) * kHeadDim * sizeof(bf16)});
    regions.push_back({ds->mh, size_t(Dspark::kMaxPend) * Dspark::kHid * sizeof(bf16)});
}

void Engine::Impl::ds_slot_pending(int& n, int& pos0, bool save) {
    if (!ds) return;
    if (save) { n = ds->pend_n; pos0 = ds->pend_pos0; }
    else { ds->pend_n = n; ds->pend_pos0 = pos0; }
}

Engine::Draft Engine::draft(int token) {
    if (tail_) {
        impl_->usable(); tail_->usable(); Impl::check_token(token);
        if (split_verify_pending_) throw std::logic_error("commit the pending split verify window first");
        if (!tail_->ds) throw std::logic_error("DSpark was not enabled");
        try {
            if(tensor_) {
                Draft results[2];
                tensor_both(tensor_,[&](int rank) { results[rank]=(rank?*tail_:*impl_).draft(token); });
                if(results[0].tokens!=results[1].tokens) throw std::runtime_error("TP2 draft ranks disagree");
                if(results[0].n)
                    results[0].vram_hits=__builtin_popcountll(impl_->ds->cache_hit_mask & tail_->ds->cache_hit_mask);
                return results[0];
            }
            // Gather five embedding rows on their owning GPU. No duplicate
            // embedding arena and no peer dereference inside the draft kernels.
            for (int r = 0; r < dspark::kBlock; ++r) {
                DeviceScope first(impl_->stage.device);
                impl_->hp->token = r == 0 ? token : dspark::kNoiseToken;
                ck(cudaMemcpyAsync(impl_->dp, impl_->hp, sizeof(int), cudaMemcpyHostToDevice, impl_->st), "draft embedding token");
                ops::embed_device(impl_->embed, impl_->dp, impl_->h, impl_->st);
                impl_->handoff->copy(tail_->ds->h + size_t(r) * kHc * kDim, impl_->h,
                                     kHc * kDim * sizeof(bf16), impl_->st, tail_->st);
            }
            DeviceScope last(tail_->stage.device);
            return tail_->draft(token);
        } catch (...) { impl_->broken = tail_->broken = true; throw; }
    }
    DeviceScope scope(impl_->stage.device);
    return impl_->draft(token);
}
bool Engine::has_dspark() const { return tail_ ? tail_->ds != nullptr : impl_->ds != nullptr; }
