// Included by engine.cu after Impl. This keeps private engine types in one translation unit.
// No allocation or host wait occurs in enqueue_verify or its kernels.
namespace {
// Save the append-only tail. Rows outside the allocated context are never read.
__global__ void verify_tail_copy(bf16* backup, bf16* cache, const int* params, int width,
                                int ratio, int capacity, int keep, bool restore) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kVerifyMaxTokens * width) return;
    const int row = params[1] / ratio + i / width;
    if (row >= capacity) return;
    const int j = row * width + i % width;
    if (!restore) backup[i] = cache[j];
    else if (row >= (params[1] + keep) / ratio) cache[j] = backup[i];
}
} // namespace

struct Engine::Impl::VerifyWorkspace {
    static constexpr int M = kVerifyMaxTokens;
    std::vector<std::shared_ptr<void>> device, pinned;
    template<class T> T* alloc(size_t n) {
        T* p = dalloc<T>(std::max<size_t>(1, n));
        device.emplace_back(p, [](void* q) { cudaFree(q); }); return p;
    }
    template<class T> T* pin(size_t n) {
        T* p = nullptr;
        ck(cudaHostAlloc((void**)&p, std::max<size_t>(1, n) * sizeof(T), cudaHostAllocDefault), "verify pinned");
        pinned.emplace_back(p, [](void* q) { cudaFreeHost(q); }); return p;
    }
    struct State {
        bf16 *ring = nullptr, *rows = nullptr, *comp_backup = nullptr, *key_backup = nullptr;
        float *kv = nullptr, *score = nullptr, *kv_rows = nullptr, *score_rows = nullptr;
    };
    State state[kLayers];
    bf16 *h, *h2, *xa, *xf, *qr, *q, *kvv, *o, *oa, *attn_out;
    bf16 *g, *u, *sh_h, *sh_out, *ffn_out, *final_x, *eng_vals, *eng_kv;
    float *act, *mix, *apre, *apost, *acomb, *fpre, *fpost, *fcomb, *routed, *logits;
    uint16_t* half;
    int32_t *routes, *routes_host, *selected, *indices;
    float *weights, *logits_host;
    int *params, *params_host, *next, *next_host;
    uint8_t *candidates, *eng, *eng_host;
    size_t eng_bytes;
    std::unique_ptr<ExpertDoorbell> db;
    // the window's engram rows, read by the engine's reader thread beside the first layers (post_engram): per table
    // the rows of all tokens, M * kEngRows weights then M * kEngRows scales (TB bytes), in mapped pinned memory;
    // the GPU copies a table once its flag rises, on st_eng, and computes its wkv GEMV there (veng_kv)
    static constexpr size_t TB = size_t(M) * kEngRows * (256 + 8);
    std::vector<std::unique_ptr<EngramRows>> veng_rows;
    std::vector<std::vector<int64_t>> veng_ids;
    uint8_t *veng = nullptr, *veng_host = nullptr;
    const uint8_t* veng_host_dev = nullptr;
    bf16* veng_kv = nullptr;
    float* act_eng = nullptr;
    float *act_kv = nullptr, *act_sh = nullptr;   // the FP8 activation scratch of the kv branch and the shared expert
    EngJob job;
    std::map<std::array<int64_t, 4>, cudaGraphExec_t> graphs;
    std::vector<int> tokens;
    int pos = 0, m = 0;
    bool warmed = false;
    bool eager = false;   // the window being enqueued runs eagerly (the first one), not as a graph
    void *dense = nullptr, *dense_kv = nullptr, *dense_shared = nullptr, *dense_engram = nullptr;
    size_t dense_bytes = 0;
    bf16* incoming_backup = nullptr;
    VerifyWorkspace(Impl& e) {
        const int S = e.max_seq;
        params = alloc<int>(M * 4 + 1); params_host = pin<int>(M * 4 + 1);   // [M * 4]: the engram epoch
        next = alloc<int>(M); next_host = pin<int>(M);
        h = alloc<bf16>(M*kHc*kDim); h2 = alloc<bf16>(M*kHc*kDim);
        xa = alloc<bf16>(M*kDim); xf = alloc<bf16>(M*kDim);
        qr = alloc<bf16>(M*kQLora); q = alloc<bf16>(M*kHeads*kHeadDim);
        kvv = alloc<bf16>(M*kHeadDim); o = alloc<bf16>(M*kHeads*kHeadDim);
        oa = alloc<bf16>(M*kOGroups*kOLora); attn_out = alloc<bf16>(M*kDim);
        g = alloc<bf16>(M*kMoeInter); u = alloc<bf16>(M*kMoeInter); sh_h = alloc<bf16>(M*kMoeInter);
        sh_out = alloc<bf16>(M*kDim); ffn_out = alloc<bf16>(M*kDim); final_x = alloc<bf16>(M*kDim);
        eng_vals = alloc<bf16>(M*24*256); eng_kv = alloc<bf16>(M*(kHc+1)*kDim);
        act = alloc<float>(M*8192); mix = alloc<float>(M*kHc);
        apre = alloc<float>(M*kHc); apost = alloc<float>(M*kHc); acomb = alloc<float>(M*kHc*kHc);
        fpre = alloc<float>(M*kHc); fpost = alloc<float>(M*kHc); fcomb = alloc<float>(M*kHc*kHc);
        routed = alloc<float>(M*kDim); logits = alloc<float>(M*kVocab); logits_host = pin<float>(M*kVocab);
        half = alloc<uint16_t>(M*kDim);
        routes = alloc<int32_t>(kLayers*M*kTopK); routes_host = pin<int32_t>(kLayers*M*kTopK);
        weights = alloc<float>(kLayers*M*kTopK); selected = alloc<int32_t>(M*kTopK);
        indices = alloc<int32_t>(M*(kWindow+kIndexTopK)); candidates = alloc<uint8_t>(size_t(M)*(S+1));
        eng_bytes = size_t(e.n_eng)*kEngRows*(256+8);
        eng = alloc<uint8_t>(M*eng_bytes); eng_host = pin<uint8_t>(M*eng_bytes);
        if (!e.resident_backend()) db = std::make_unique<ExpertDoorbell>(M, kTopK, kDim);
        if (e.pack.has_exl3_dense()) {
            dense_bytes = exl3_small_rows_workspace_bytes(M, 32768, kVocab);
            dense = alloc<uint8_t>(dense_bytes);
            dense_kv = alloc<uint8_t>(dense_bytes);
            dense_shared = alloc<uint8_t>(dense_bytes);
            dense_engram = alloc<uint8_t>(dense_bytes);
        }
        if (e.stage.begin == 18) incoming_backup = alloc<bf16>(M * kHeadDim);
        act_kv = alloc<float>(M * 8192);
        act_sh = alloc<float>(M * 8192);
        if (e.n_eng) {
            veng = alloc<uint8_t>(e.n_eng * TB);
            ck(cudaHostAlloc((void**)&veng_host, e.n_eng * TB, cudaHostAllocMapped), "verify engram pinned");
            pinned.emplace_back(veng_host, [](void* q) { cudaFreeHost(q); });
            ck(cudaHostGetDevicePointer((void**)&veng_host_dev, veng_host, 0), "verify engram alias");
            veng_kv = alloc<bf16>(size_t(e.n_eng) * M * (kHc + 1) * kDim);
            act_eng = alloc<float>(M * 8192);
            const auto& hs = e.pack.engram_hash();
            job.rows = 0;   // set per window: m * cols
            for (int li = 0; li < e.n_eng; ++li) {
                const auto& t = e.pack.engram_tables()[li];
                veng_rows.push_back(std::make_unique<EngramRows>(
                    std::vector<EngramRows::Table>{{t.path, t.weight_offset, t.scale_offset}},
                    M * ((hs.max_ngram - 1) * hs.n_heads), 256, 8, 128));
                veng_ids.emplace_back(size_t(M) * kEngRows, 0);
                job.reader.push_back(veng_rows.back().get());
                job.ids.push_back(veng_ids.back().data());
                job.w.push_back(veng_host + li * TB);
                job.s.push_back(veng_host + li * TB + size_t(M) * kEngRows * 256);
            }
        }
        for (int l = e.stage.begin; l < e.stage.end; ++l) {
            auto& s = state[l];
            s.ring = alloc<bf16>(kWindow*kHeadDim); s.rows = alloc<bf16>(M*kHeadDim);
            if (is_kv_source(l)) {
                s.comp_backup = alloc<bf16>(M*kHeadDim); s.key_backup = alloc<bf16>(M*kIndexDim);
                if (e.L[l].ratio > 1) {
                    s.kv = alloc<float>(2*kHeadDim); s.score = alloc<float>(2*kHeadDim);
                    s.kv_rows = alloc<float>(M*2*kHeadDim); s.score_rows = alloc<float>(M*2*kHeadDim);
                }
            }
        }
    }
    ~VerifyWorkspace() {
        for (auto& entry : graphs) cudaGraphExecDestroy(entry.second);
    }
};

void Engine::Impl::enqueue_verify(int m) {
    auto& v = *verify_ws;
    auto copy = [&](void* dst, const void* src, size_t bytes) {
        ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, st), "verify copy");
    };
    auto linear_on = [&](const bf16* x, const Fp8& w, bf16* out, float* act, cudaStream_t s,
                         bool communicate=true,bool acknowledge=true) {
        if (w.exl3.trellis) {
            void* ws = s == st_kv ? v.dense_kv : s == st_sh ? v.dense_shared : s == st_eng ? v.dense_engram : v.dense;
            projection(x,w.exl3,out,nullptr,m,w.k,w.n,ws,v.dense_bytes,s,communicate,acknowledge);
            return;
        }
        fp8_quantize_activation_f32((const uint16_t*)x, m, w.k, act, s);
        fp8_block_gemv_q(act, m, w.k, w.w, w.s, w.n, (uint16_t*)out, s);
    };
    auto linear = [&](const bf16* x, const Fp8& w, bf16* out,bool communicate=true,bool acknowledge=true) {
        linear_on(x,w,out,v.act,st,communicate,acknowledge);
    };
    ck(cudaMemcpyAsync(v.params, v.params_host, (VerifyWorkspace::M*4+1)*sizeof(int), cudaMemcpyHostToDevice, st),
       "verify params");
    if (n_eng) {
        // per table on st_eng, as the step's engram_kv: wait for the rows, copy, dequantize (token by token, rows of
        // cols ids), the wkv GEMV of all m rows; the engram layer waits for eng_join[li]
        const int cols = (pack.engram_hash().max_ngram-1)*pack.engram_hash().n_heads;
        ck(cudaEventRecord(eng_fork, st), "verify engram fork");
        ck(cudaStreamWaitEvent(st_eng, eng_fork, 0), "verify engram fork wait");
        for (int li = 0; li < n_eng; ++li) {
            const auto& y = L[pack.engram_tables()[li].layer];
            wait_engram_k<<<1, 1, 0, st_eng>>>(eng_flag_dev + li*kEngFlagStride, v.params + VerifyWorkspace::M*4 - 4);
            ck(cudaGetLastError(), "verify engram wait");
            const int n16 = int(VerifyWorkspace::TB / 16);
            engram_rows_k<<<(n16+255)/256, 256, 0, st_eng>>>((const uint4*)(v.veng_host_dev + li*VerifyWorkspace::TB),
                                                            (uint4*)(v.veng + li*VerifyWorkspace::TB), n16);
            ck(cudaGetLastError(), "verify engram rows");
            const uint8_t* tw = v.veng + li*VerifyWorkspace::TB;
            const uint8_t* ts = tw + size_t(VerifyWorkspace::M)*kEngRows*256;
            for (int t = 0; t < m; ++t)
                ops::engram_dequant(tw + size_t(t)*cols*256, ts + size_t(t)*cols*8, cols, v.eng_vals+t*y.eng_wkv.k,
                                    st_eng);
            linear_on(v.eng_vals, y.eng_wkv,
                      v.veng_kv + size_t(li)*VerifyWorkspace::M*(kHc+1)*kDim, v.act_eng, st_eng);
            ck(cudaEventRecord(eng_join[li], st_eng), "verify engram join");
        }
    }
    if(stage.begin==0) embedding(v.params,m,4,v.h,st);
    for (int t = 0; t < m; ++t) {
        ops::window_index_device(v.params+t*4+1, v.indices+t*(kWindow+kIndexTopK), st);
        if (stage.begin == 0) {
            copy(v.mix+t*kHc, one_hot_dev, kHc*sizeof(float));
        }
    }
    const bf16 *comp = stage.begin == 18 ? incoming_comp : nullptr, *keys = nullptr;
    int eng_i = 0;
    for (int l = stage.begin; l < stage.end; ++l) {
        auto& y = L[l]; auto& s = v.state[l];
        // Formal rings and compressor tails stay unchanged until commit.
        copy(s.ring, y.window, kWindow*kHeadDim*sizeof(bf16));
        if (is_kv_source(l)) {
            const int capacity = max_seq/y.ratio+1;
            if(owns_cache()) verify_tail_copy<<<(VerifyWorkspace::M*kHeadDim+255)/256,256,0,st>>>(
                s.comp_backup, y.comp, v.params, kHeadDim, y.ratio, capacity, 0, false);
            if(owns_cache()) verify_tail_copy<<<(VerifyWorkspace::M*kIndexDim+255)/256,256,0,st>>>(
                s.key_backup, y.idx_keys, v.params, kIndexDim, y.ratio, capacity, 0, false);
            comp = y.comp; keys = y.idx_keys;
            if (y.ratio > 1) {
                copy(s.kv, y.kv_state, 2*kHeadDim*sizeof(float));
                copy(s.score, y.score_state, 2*kHeadDim*sizeof(float));
            }
        }
        if (is_engram_layer(l) && eng_i < n_eng) {
            ck(cudaStreamWaitEvent(st, eng_join[eng_i], 0), "verify engram join wait");
            ops::engram_apply(v.h, v.veng_kv + size_t(eng_i)*VerifyWorkspace::M*(kHc+1)*kDim, y.eng_qw, y.eng_kw,
                              kNormEps, m, st);
            ++eng_i;
        }
        if (ds) dspark_capture(l, v.h, m, ds_verify_mh(), st);
        kernels::hc_mixes_pre(v.h, m, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, v.mix,
                             v.xa, v.apre, v.apost, v.acomb, st);
        ops::rmsnorm(v.xa, y.attn_norm, v.xa, kDim, kNormEps, m, st);
        // kv branch on st_kv beside the q branch, as the step's attention (it reads xa only)
        ck(cudaEventRecord(kv_fork, st), "verify kv fork");
        ck(cudaStreamWaitEvent(st_kv, kv_fork, 0), "verify kv fork wait");
        linear_on(v.xa, y.wkv, v.kvv, v.act_kv, st_kv);
        ops::rmsnorm(v.kvv, y.kv_norm, v.kvv, kHeadDim, kNormEps, m, st_kv);
        ck(cudaEventRecord(kv_join, st_kv), "verify kv join");
        linear(v.xa, y.wq_a, v.qr);
        ops::rmsnorm(v.qr, y.q_norm, v.qr, kQLora, kNormEps, m, st);
        linear(v.qr, y.wq_b, v.q,!tensor);
        ck(cudaStreamWaitEvent(st, kv_join, 0), "verify kv join wait");
        for (int t = 0; t < m; ++t) {
            const int* p = v.params+4*t;
            const bf16* x = v.xa+t*kDim;
            bf16* qt = v.q+t*kHeads*kHeadDim;
            bf16* kt = v.kvv+t*kHeadDim;
            bf16* ot = v.o+t*kHeads*kHeadDim;
            int32_t* idx = v.indices+t*(kWindow+kIndexTopK);
            uint8_t* cand = v.candidates+size_t(t)*(max_seq+1);
            const float* rope = y.ratio ? rope_yarn : rope_plain;
            const int first_head=tensor?tensor_rank*(kHeads/2):0, heads=tensor?kHeads/2:kHeads;
            ops::rope_device(qt+first_head*kHeadDim,heads,kHeadDim,rope,p+1,0,false,st);
            ops::rope_device(kt, 1, kHeadDim, rope, p+1, 0, false, st);
            ops::act_quant_inplace(kt, kHeadDim, st);
            copy(s.rows+t*kHeadDim, kt, kHeadDim*sizeof(bf16));
            ops::row_copy_device(s.ring, kt, kHeadDim*sizeof(bf16), p+1, 1, kWindow, st);
            bool have_latent = false;
            if (is_kv_source(l)) {
                if (y.ratio == 1) {
                    ops::bf16_linear(x, nullptr, y.c_wkv, kDim, kHeadDim, latent, nullptr, st);
                    ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                    have_latent = true;
                } else {
                    const int slot = (parity+t)&1;
                    ops::bf16_linear(x, nullptr, y.c_wkv, kDim, kHeadDim, nullptr, s.kv+slot*kHeadDim, st);
                    ops::bf16_linear(x, nullptr, y.c_wgate, kDim, kHeadDim, nullptr, s.score+slot*kHeadDim, st);
                    if (slot == 1) {
                        ops::compress_pool(s.kv, s.score, 2, latent, 1, st);
                        ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                        have_latent = true;
                    }
                    copy(s.kv_rows+t*2*kHeadDim, s.kv, 2*kHeadDim*sizeof(float));
                    copy(s.score_rows+t*2*kHeadDim, s.score, 2*kHeadDim*sizeof(float));
                }
            }
            if (is_index_source(l)) {
                if (have_latent) {
                    ops::bf16_linear(latent, nullptr, y.idx_wk, kHeadDim, kIndexDim, ik, nullptr, st);
                    ops::rmsnorm(ik, y.idx_knorm, ik, kIndexDim, kNormEps, 1, st);
                    ops::rope_device(ik, 1, kIndexDim, rope_yarn, p+1, 1-y.ratio, false, st);
                    ops::fp4_quant_inplace(ik, kIndexDim, 32, false, st);
                    if(owns_cache()) ops::row_copy_device(y.idx_keys, ik, kIndexDim*sizeof(bf16), p+1, y.ratio, max_seq+1, st);
                }
                if(tensor && is_kv_source(l)) tensor->barrier(tensor_rank,1000+l,st);
                fp8_linear(v.qr+t*kQLora, y.idx_wq_b, iq);
                ops::rope_device(iq, kIndexHeads, kIndexDim, rope_yarn, p+1, 0, false, st);
                ops::fp4_quant_inplace(iq, kIndexHeads*kIndexDim, 32, false, st);
                ops::bf16_linear(x, nullptr, y.idx_wp, kDim, kIndexHeads, iw_raw, nullptr, st);
                ops::scale_bf16(iw_raw, float(std::pow(kIndexDim,-0.5)*std::pow(kIndexHeads,-0.5)), iw, kIndexHeads, st);
                const int64_t cap = y.ratio == 1 ? tcap1 : tcap2;
                kernels::indexer_topk_device(iq, keys, p+1, y.ratio, cap, iw, l > kCandidateLayer ? cand : nullptr,
                                            kIndexTopK, kWindow, scores, idx+kWindow, st);
                if (l == kCandidateLayer)
                    kernels::candidate_blocks_device(scores, p+1, y.ratio, cap, kCandidateBlocks, kCandidateBlock, cand, st);
            }
            if (have_latent) {
                ops::rope_device(latent, 1, kHeadDim, rope_yarn, p+1, 1-y.ratio, false, st);
                ops::fp4_quant_inplace(latent, kHeadDim, 16, true, st);
                // Only append beyond the committed length. Each query uses its own causal length.
                if(owns_cache()) ops::row_copy_device(y.comp, latent, kHeadDim*sizeof(bf16), p+1, y.ratio, max_seq+1, st);
            }
            if(tensor && is_kv_source(l)) tensor->barrier(tensor_rank,960+l,st);
            if (y.ratio)
                kernels::sparse_attn_decode_device(qt, s.ring, comp, idx, p+(y.ratio == 1 ? 2 : 3), y.sink,
                                                   float(std::pow(kHeadDim,-0.5)), ot, st,first_head,heads);
            else
                kernels::sparse_attn_decode(qt, s.ring, nullptr, idx, 1, kWindow, y.sink,
                                            float(std::pow(kHeadDim,-0.5)), ot, st,first_head,heads);
            ops::rope_device(ot+first_head*kHeadDim,heads,kHeadDim,rope,p+1,0,true,st);
#ifdef STRATA_DS41_1017
            ops::act_quant_inplace(ot+first_head*kHeadDim,heads*kHeadDim,st);
#endif
            if (!y.wo_a8.w && !y.wo_a_exl3[tensor?tensor_rank*(kOGroups/2):0].trellis)
                ops::wo_a_grouped(ot, y.wo_a, v.oa+t*kOGroups*kOLora, st);
        }
        if (y.wo_a_exl3[tensor?tensor_rank*(kOGroups/2):0].trellis)
            for (int group = 0; group < kOGroups; ++group)
                if(!tensor || group/(kOGroups/2)==tensor_rank)
                    projection(v.o + group*(kHeads*kHeadDim/kOGroups),y.wo_a_exl3[group],
                           v.oa+group*kOLora,nullptr,m,kHeads*kHeadDim,kOGroups*kOLora,v.dense,v.dense_bytes,st);
        // every row at once, the weight read once (each row as wo_a_grouped_fp8 computes it)
        if (y.wo_a8.w) wo_a_grouped_fp8_rows(v.o, m, y.wo_a8.w, y.wo_a8.s, v.oa, st);
        if(tensor) tensor->gather(tensor_rank,700+l,v.oa,m,kOGroups*kOLora,2,st,0,false);
        linear(v.oa,y.wo_b,v.attn_out,true,!tensor);
        ops::hc_post(v.attn_out, v.h, v.apost, v.acomb, v.h2, m, st);
        kernels::hc_mixes_pre(v.h2, m, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, v.apre,
                             v.xf, v.fpre, v.fpost, v.fcomb, st);
        ops::rmsnorm(v.xf, y.ffn_norm, v.xf, kDim, kNormEps, m, st);
        int32_t* ids = v.routes+l*VerifyWorkspace::M*kTopK;
        float* w = v.weights+l*VerifyWorkspace::M*kTopK;
        kernels::router_topk(v.xf, m, y.gate_w, y.gate_bias, ids, w, st);
        if (pack.has_exl3_dense()) exl3_input_half(v.xf, v.half, m*kDim, st);
        else ops::to_half_fp8q(v.xf, v.half, m*kDim, st);
        const bool tier = vram && vram->slots() > 0;
        const auto* ram = host && host->experts_dev() ? host->experts_dev()+size_t(l)*kExperts : nullptr;
        // prefetch, as the step: layer l's guesses (all rows' union) were planned and copied during layer l - 1
        // not in the eager first window: its thousands of launches fill the launch queue while the GPU waits for the
        // CPU, and the host thread then holds the driver while the DMA copy thread needs it for the copies the GPU
        // waits for (a hang at layer 5, laptop); a graph is one launch
        const bool pf = prefetch && ram && zc_blobs && !v.eager;
        if (resident_backend()) {
            for (int t = 0; t < m; ++t)
                ops::resident_routes(ids+t*kTopK, vram->res_dev()+size_t(l)*kExperts, v.selected+t*kTopK, st);
        } else v.db->publish(v.half, ids, w, m, tier ? vram->res_dev()+size_t(l)*kExperts : nullptr, v.selected,
                      l+1, st, tier ? vram->experts_dev() : nullptr, ram, zc_quota ? zc_quota.get()+l : nullptr,
                      nullptr, nullptr, pf && l > 0 ? prefetch->ids(l) : nullptr,
                      pf && l > 0 ? prefetch->descs(l) : nullptr, pf ? ExpertPrefetch::kMaxGuesses : 0, tail_rank, ssd_rank);
        if (pf && l+1 < kLayers)
            prefetch->plan_rows(l+1, v.xf, m, L[l+1].gate_w, L[l+1].gate_bias,
                                tier ? vram->res_dev()+size_t(l+1)*kExperts : nullptr,
                                host->experts_dev()+size_t(l+1)*kExperts, zc_blobs.get()+size_t(l+1)*kExperts, st);
        // the shared expert reads xf only: on st_sh beside the routed experts until the outputs add up
        ck(cudaEventRecord(sh_fork, st), "verify shared fork");
        ck(cudaStreamWaitEvent(st_sh, sh_fork, 0), "verify shared fork wait");
        linear_on(v.xf, y.sh_w1, v.g, v.act_sh, st_sh);
        linear_on(v.xf, y.sh_w3, v.u, v.act_sh, st_sh);
        ops::swiglu(v.g, v.u, kSwigluLimit, v.sh_h, m*kMoeInter, st_sh);
        linear_on(v.sh_h, y.sh_w2, v.sh_out, v.act_sh, st_sh);
        ck(cudaEventRecord(sh_done, st_sh), "verify shared done");
        ck(cudaMemsetAsync(v.routed, 0, m*kDim*sizeof(float), st), "verify routed");
        if (pf && l > 0) prefetch->join(l, st);   // the GPU computes the prefetched experts below
        if(tensor) {
            kernels::exl3_moe_tensor_up(reinterpret_cast<const __half*>(v.half),m,v.selected,w,kTopK,
                                        vram->experts_dev(),vram->workspace(),VramExperts::kWorkspaceBytes,tensor_rank,st);
            tensor->gather(tensor_rank,768+2*l,kernels::exl3_moe_tensor_hidden(vram->workspace(),m,kTopK),
                           m*kTopK,kMoeInter,2,st,0,false);
            kernels::exl3_moe_tensor_down(m,v.selected,kTopK,vram->experts_dev(),v.routed,vram->workspace(),
                                          VramExperts::kWorkspaceBytes,tensor_rank,st);
            tensor->gather(tensor_rank,769+2*l,v.routed,m,kDim,4,st,0,false);
        } else if (tier || ram)
            kernels::exl3_moe_decode((const __half*)v.half, m, v.selected, w, kTopK,
                                    resident_backend() ? vram->experts_dev() : v.db->gpu_experts(), v.routed,
                                    vram ? vram->workspace() : zc_workspace.get(), VramExperts::kWorkspaceBytes, st);
        if (!resident_backend()) v.db->wait_add(v.routed, m, l+1, st);
        if (pf && l+1 < kLayers) prefetch->ready(l+1, st);   // the next layer overwrites v.xf, which the guesses read
        ck(cudaStreamWaitEvent(st, sh_done, 0), "verify shared join");
        ops::add_f32_bf16(v.routed, v.sh_out, v.ffn_out, m*kDim, st);
        ops::hc_post(v.ffn_out, v.h2, v.fpost, v.fcomb, v.h, m, st);
        copy(v.mix, v.fpre, m*kHc*sizeof(float));
    }
    if (stage.end != kLayers) return;
    ops::hc_pre(v.h, v.mix, v.final_x, m, st);
    ops::rmsnorm(v.final_x, final_norm, v.final_x, kDim, kNormEps, m, st);
    if (head_exl3.trellis)
        projection(v.final_x,head_exl3,nullptr,v.logits,m,kDim,kVocab,v.dense,v.dense_bytes,st,true,!tensor);
    else ops::bf16_linear_rows(v.final_x, m, head, kDim, kVocab, v.logits, st);
    for (int t = 0; t < m; ++t) ops::argmax_logits(v.logits+t*kVocab, v.next+t, st);
    ck(cudaMemcpyAsync(v.next_host, v.next, m*sizeof(int), cudaMemcpyDeviceToHost, st), "verify next");
    ck(cudaMemcpyAsync(v.logits_host, v.logits, size_t(m)*kVocab*sizeof(float), cudaMemcpyDeviceToHost, st), "verify logits");
    ck(cudaMemcpyAsync(v.routes_host, v.routes, kLayers*VerifyWorkspace::M*kTopK*sizeof(int32_t), cudaMemcpyDeviceToHost, st), "verify routes");
}

VerifyResult Engine::Impl::verify(const std::vector<int>& window, int pos, bool want_logits, Timing& tm) {
    DeviceScope scope(stage.device);
    usable();
    if (verify_pending) throw std::logic_error("commit the pending verify window first");
    const int m = int(window.size());
    if (m < 1 || m > kVerifyMaxTokens) throw std::invalid_argument("verify supports 1..4 rows on the CPU path");
    if (pos < 0 || pos != int(history.size()) || pos > max_seq-m)
        throw std::invalid_argument("verify position is not the committed position or exceeds max_seq");
    for (int token : window)
        if (token < 0 || token >= kVocab) throw std::invalid_argument("verify token outside vocabulary");
    if (!verify_ws) verify_ws = std::make_shared<VerifyWorkspace>(*this);
    auto& v = *verify_ws;
    v.tokens = window; v.pos = pos; v.m = m;
    tm = Timing{};
    const double begin = now_ms();
    // Stage hashes from each tentative prefix; the rows of all tokens are read beside the first layers.
    const int cols = n_eng ? (pack.engram_hash().max_ngram-1)*pack.engram_hash().n_heads : 0;
    try {
        for (int t = 0; t < m; ++t) {
            history.push_back(pack.engram_hash().token_map[window[t]]);
            int li = 0;
            for (int l = stage.begin; l < stage.end; ++l)
                if (is_engram_layer(l)) {
                    engram_ids(l, li, pos+t);
                    std::copy_n(eng_ids[li].data(), cols, v.veng_ids[li].data()+size_t(t)*cols);
                    ++li;
                }
            const int params[] = {window[t], pos+t, pos+t+1, (pos+t+1)/2};
            std::copy_n(params, 4, v.params_host+4*t);
        }
    } catch (...) { history.resize(pos); throw; }
    history.resize(pos);
    if (n_eng) {
        v.job.rows = m*cols;
        v.params_host[VerifyWorkspace::M*4] = int(post_engram(v.job));
    }
    if (vram) tm.vram_swaps = vram->between_steps();
    parity = pos&1;
    tcap1 = cap_of(pos+m, max_seq+1);
    tcap2 = cap_of((pos+m)/2, max_seq/2+1);
    const std::array<int64_t, 4> key{m, parity, tcap1, tcap2};
    // The first window is eager. Capture only after every kernel has run once.
    cudaGraphExec_t executable = nullptr;
    bool reused = false;
    if (use_graph && v.warmed) {
        auto it = v.graphs.find(key);
        reused = it != v.graphs.end();
        if (it == v.graphs.end()) {
            cudaGraph_t graph = nullptr;
            v.eager = false;
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "verify capture");
            try {
                enqueue_verify(m);
                ck(cudaStreamEndCapture(st, &graph), "verify end capture");
                ck(cudaGraphInstantiate(&executable, graph, 0), "verify instantiate");
            } catch (...) {
                if (!graph) cudaStreamEndCapture(st, &graph);
                if (graph) cudaGraphDestroy(graph);
                throw;
            }
            cudaGraphDestroy(graph);
            it = v.graphs.emplace(key, executable).first;
        }
        executable = it->second;
    }
    if (v.db) v.db->reset();
    if (prefetch) prefetch->begin_step();   // the device is idle: a new window's layers are copied again
    // the decode worker computes the window's CPU experts: the job is the window's doorbell and its m rows
    worker_us = 0;
    worker_misses = 0;
    worker_ram = worker_file = worker_ssd = 0;
    worker_prefetched = 0;
    worker_skipped = 0;
    if (!resident_backend()) {
        std::lock_guard<std::mutex> lk(mu);
        job_db = v.db.get();
        job_m = m;
        job_dspark = false;
        job_cancel = false;
        ++go;
    }
    cv.notify_all();
    const uint64_t job = go;
    auto job_end = [&] {
        if (resident_backend()) return;
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return jobs_done >= job; });
    };
    verify_pending = true;
    verify_failed = true;
    try {
        if (executable) ck(cudaGraphLaunch(executable, st), "verify launch");
        else {
            v.eager = true;
            enqueue_verify(m);
            v.eager = false;
        }
        ck(cudaStreamSynchronize(st), "verify completion");
    } catch (...) {
        job_cancel = true;
        if(tensor) tensor->abort();  // release a peer before waiting for this rank's queued work
        job_end();
        // Release any enqueued wait before workspace destruction. Never commit this failed window.
        if (v.db) v.db->mark_done(kLayers);
        cudaStreamSynchronize(st);
        throw;
    }
    job_end();
    if (n_eng) finish_engram(tm);
    {
        std::lock_guard<std::mutex> lk(mu);
        if (worker_error) {
            std::exception_ptr e = worker_error;
            worker_error = nullptr;
            std::rethrow_exception(e);
        }
    }
    tm.expert_total = m*(stage.end-stage.begin)*kTopK;
    tm.expert_hits = tm.expert_total - worker_misses.load();
    tm.ram_experts = worker_ram.load();
    tm.file_experts = worker_file.load();
    tm.ssd_experts = worker_ssd.load();
    tm.prefetched = worker_prefetched.load();
    tm.skipped = worker_skipped.load();
    tm.cpu_experts_ms = worker_us.load() / 1000.0;
    tm.worker_lead_ms = worker_wake_at.load() - begin;
    tm.worker_span_ms = worker_end_at.load() - worker_wake_at.load();
    tm.worker_wait_ms = worker_wait.load();
    tm.worker_first_wait_ms = worker_first_wait.load();
    tm.admit_ms = worker_admit.load();
    v.warmed = true;
    VerifyResult result;
    result.graph_reused = reused;
    if (stage.end == kLayers) result.next.assign(v.next_host, v.next_host+m);
    if (want_logits && stage.end == kLayers) {
        result.logits.resize(m);
        for (int t = 0; t < m; ++t)
            result.logits[t].assign(v.logits_host+t*kVocab, v.logits_host+(t+1)*kVocab);
    }
    tm.total_ms = now_ms()-begin;
    tm.gpu_ms = tm.total_ms;   // the engram rows are read beside the first layers
    verify_failed = false;
    return result;
}

void Engine::Impl::commit_verify(int keep) {
    DeviceScope scope(stage.device);
    usable();
    if (!verify_pending || verify_failed) throw std::logic_error("no successful verify window to commit");
    auto& v = *verify_ws;
    if (keep < 1 || keep > v.m) throw std::invalid_argument("commit count outside verify window");
    verify_failed = true; // A partial CUDA failure must prevent reuse of this Engine.
    // All positions come from the unchanged window parameter buffer. These copies are capturable.
    for (int l = stage.begin; l < stage.end; ++l) {
        auto& y = L[l]; auto& s = v.state[l];
        for (int t = 0; t < keep; ++t)
            ops::row_copy_device(y.window, s.rows+t*kHeadDim, kHeadDim*sizeof(bf16), v.params+t*4+1, 1, kWindow, st);
        if (!is_kv_source(l)) continue;
        if (y.ratio > 1) {
            ck(cudaMemcpyAsync(y.kv_state, s.kv_rows+(keep-1)*2*kHeadDim, 2*kHeadDim*sizeof(float),
                               cudaMemcpyDeviceToDevice, st), "commit compressor KV");
            ck(cudaMemcpyAsync(y.score_state, s.score_rows+(keep-1)*2*kHeadDim, 2*kHeadDim*sizeof(float),
                               cudaMemcpyDeviceToDevice, st), "commit compressor scores");
        }
        // Restore the original bytes of every rejected append. Accepted appends are now visible.
        const int capacity = max_seq/y.ratio+1;
        if(owns_cache()) verify_tail_copy<<<(VerifyWorkspace::M*kHeadDim+255)/256,256,0,st>>>(
            s.comp_backup, y.comp, v.params, kHeadDim, y.ratio, capacity, keep, true);
        if(owns_cache()) verify_tail_copy<<<(VerifyWorkspace::M*kIndexDim+255)/256,256,0,st>>>(
            s.key_backup, y.idx_keys, v.params, kIndexDim, y.ratio, capacity, keep, true);
    }
    ck(cudaStreamSynchronize(st), "verify commit");
    for (int t = 0; t < keep; ++t) history.push_back(pack.engram_hash().token_map[v.tokens[t]]);
    if (stage.end == kLayers) lg.assign(v.logits_host+(keep-1)*kVocab, v.logits_host+keep*kVocab);
    if (ds) dspark_pend(ds_verify_mh(), keep, v.pos);   // the kept rows' main_hiddens feed the next draft
    if (!resident_backend() && (vram || (host && host->reserve() > 0))) {
        int32_t row[kLayers*kTopK];
        for (int t = 0; t < keep; ++t) {
            for (int l = 0; l < kLayers; ++l)
                std::copy_n(v.routes_host+(l*VerifyWorkspace::M+t)*kTopK, kTopK, row+l*kTopK);
            if (vram) vram->count(row, kTopK);
            if (host && host->reserve() > 0) host->end_step(row, kTopK);   // publish the window's reads, record uses
        }
    }
    // Step reconstructs its index list and candidate mask. No tentative row uses its scratch.
    verify_pending = false;
    verify_failed = false;
}

VerifyResult Engine::verify(const std::vector<int>& window, int pos, bool logits) {
    if (tail_) {
        impl_->usable(); tail_->usable();
        if (split_verify_pending_) throw std::logic_error("commit the pending split verify window first");
        if (window.empty() || window.size() > kVerifyMaxTokens || pos != position() ||
            pos < 0 || window.size() > size_t(impl_->max_seq - pos))
            throw std::invalid_argument("split verify: invalid window/position");
        for (int token : window) Impl::check_token(token);
        VerifyResult result;
        const double start = now_ms();
        try {
            if(tensor_) {
                Timing times[2];VerifyResult results[2];
                tensor_both(tensor_,[&](int rank) {
                    auto& e=rank?*tail_:*impl_;DeviceScope scope(e.stage.device);
                    e.ds_flush_pending();results[rank]=e.verify(window,pos,logits,times[rank]);
                });
                if(results[0].next!=results[1].next) throw std::runtime_error("TP2 verification ranks disagree");
                split_verify_tokens_=window;split_verify_pending_=true;
                timing_=times[0];timing_.total_ms=timing_.gpu_ms=now_ms()-start;
                return results[0];
            }
            { DeviceScope scope(tail_->stage.device); tail_->ds_flush_pending(); }
            Timing first, last;
            impl_->verify(window, pos, false, first);
            if (!tail_->verify_ws) {
                DeviceScope scope(tail_->stage.device);
                tail_->verify_ws = std::make_shared<Impl::VerifyWorkspace>(*tail_);
            }
            auto& a = *impl_->verify_ws; auto& b = *tail_->verify_ws;
            auto transfer = [&](void* dst, const void* src, size_t bytes) {
                constexpr size_t bound = kHc*kDim*sizeof(bf16);
                for (size_t off=0; off<bytes; off+=bound)
                    impl_->handoff->copy(static_cast<uint8_t*>(dst)+off, static_cast<const uint8_t*>(src)+off,
                                         std::min(bound,bytes-off), impl_->st,tail_->st);
            };
            const int m = int(window.size());
            transfer(b.h,a.h,size_t(m)*kHc*kDim*sizeof(bf16));
            transfer(b.mix,a.mix,size_t(m)*kHc*sizeof(float));
            if (tail_->stage.begin == 18) {
                DeviceScope scope(tail_->stage.device);
                const int params[] = {0,pos};
                ck(cudaMemcpyAsync(b.params,params,sizeof(params),cudaMemcpyHostToDevice,tail_->st),"verify mirror position");
                verify_tail_copy<<<(Impl::VerifyWorkspace::M*kHeadDim+255)/256,256,0,tail_->st>>>(
                    b.incoming_backup,tail_->incoming_comp,b.params,kHeadDim,2,tail_->max_seq/2+1,0,false);
                ck(cudaStreamSynchronize(tail_->st),"verify mirror backup");
                const size_t first=size_t(pos/2)*kHeadDim, rows=size_t((pos+m)/2-pos/2);
                if (rows) transfer(tail_->incoming_comp+first,impl_->L[14].comp+first,rows*kHeadDim*sizeof(bf16));
                for (int t=0;t<m;++t)
                    transfer(b.indices+t*(kWindow+kIndexTopK)+kWindow,a.indices+t*(kWindow+kIndexTopK)+kWindow,
                             kIndexTopK*sizeof(int32_t));
            }
            result = tail_->verify(window,pos,logits,last);
            split_verify_tokens_ = window;
            split_verify_pending_ = true;
            timing_=first; timing_.expert_hits+=last.expert_hits; timing_.expert_total+=last.expert_total;
            timing_.engram_ms+=last.engram_ms; timing_.total_ms=timing_.gpu_ms=now_ms()-start;
            return result;
        } catch (...) {
            impl_->broken = tail_->broken = true;
            throw;
        }
    }
    DeviceScope scope(impl_->stage.device);
    return impl_->verify(window, pos, logits, timing_);
}
void Engine::commit(int n_keep) {
    if (tail_) {
        impl_->usable(); tail_->usable();
        if (!split_verify_pending_) throw std::logic_error("no pending split verify window");
        if (n_keep < 1 || n_keep > int(split_verify_tokens_.size()))
            throw std::invalid_argument("split commit: keep must be within verified inputs");
        auto total = timing_; const double start = now_ms();
        try {
            if(tensor_) tensor_both(tensor_,[&](int rank) { (rank?*tail_:*impl_).commit_verify(n_keep); });
            else { impl_->commit_verify(n_keep);tail_->commit_verify(n_keep); }
            if (!tensor_ && tail_->stage.begin==18) {
                DeviceScope scope(tail_->stage.device);
                auto& v=*tail_->verify_ws;
                verify_tail_copy<<<(Impl::VerifyWorkspace::M*kHeadDim+255)/256,256,0,tail_->st>>>(
                    v.incoming_backup,tail_->incoming_comp,v.params,kHeadDim,2,tail_->max_seq/2+1,n_keep,true);
                ck(cudaStreamSynchronize(tail_->st),"verify mirror commit");
            }
            total.total_ms += now_ms() - start; total.gpu_ms = total.total_ms;
            timing_ = total;
            split_verify_pending_ = false;
            split_verify_tokens_.clear();
            return;
        } catch (...) {
            impl_->broken = tail_->broken = true;
            throw;
        }
    }
    DeviceScope scope(impl_->stage.device);
    impl_->commit_verify(n_keep);
}
