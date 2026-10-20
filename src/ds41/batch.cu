// Included by engine.cu after verify.cu: the batch slots (upstream's --batch). One decode step of several
// conversations: the dense weights and the routed experts are read once for all rows (the decode path's staging and
// zero-copy quota), each row attends over its own slot's state. The rows use a verify workspace for their buffers.

void Engine::Impl::alloc_slots() {
    const int n = std::max(0, std::min(opt.batch_slots, kVerifyMaxTokens));
    if (n == 0) return;
    slot_states.resize(n);
    if (partitioned()) {
        slot_states.resize(n + 1); // one private save area preserves the main session during serial scheduling
        std::fprintf(stderr, "ds41: %d isolated split session slots; serial GPU scheduling, actual-size host KV state\n", n);
        return;
    }
    for (auto& s : slot_states) {
        s.L.resize(kLayers);
        s.history.reserve(max_seq);
        for (int l = 0; l < kLayers; ++l) {
            const auto& y = L[l];
            auto& d = s.L[l];
            d.window = dalloc_own<bf16>((size_t) kWindow * kHeadDim);
            if (is_kv_source(l)) {
                d.comp = dalloc_own<bf16>((size_t) (max_seq / y.ratio + 1) * kHeadDim);
                d.idx_keys = dalloc_own<bf16>((size_t) (max_seq / y.ratio + 1) * kIndexDim);
                if (y.ratio > 1) {
                    d.kv_state = dalloc_own<float>((size_t) y.ratio * kHeadDim);
                    d.score_state = dalloc_own<float>((size_t) y.ratio * kHeadDim);
                }
            }
        }
    }
    if (zc_stage) batch_stage = std::make_unique<ExpertStaging>(kVerifyMaxTokens * kTopK, zc_stage->stride());
    if (n_eng) {
        std::vector<EngramRows::Table> tabs;
        for (const auto& t : pack.engram_tables()) tabs.push_back({t.path, t.weight_offset, t.scale_offset});
        eng_rows_batch = std::make_unique<EngramRows>(tabs, kVerifyMaxTokens * kEngRows, 256, 8,
                                                      2 * kVerifyMaxTokens * kEngRows);
    }
    std::fprintf(stderr, "ds41: %d batch slots\n", n);
}

/// The main session <-> a slot: every layer's window ring, compressed rows and indexer keys up to the position, the
/// compressor state, and the tokens. The device is idle between calls (a step or a prefill ended in a sync).
void Engine::Impl::slot_copy(int slot, bool to_slot) {
    DeviceScope scope(stage.device);
    usable();
    if (slot < 0 || slot >= (int) slot_states.size())
        throw std::invalid_argument("ds41 batch: slot " + std::to_string(slot) + " does not exist");
    if (verify_pending) throw std::logic_error("commit the pending verify window first");
    auto& s = slot_states[slot];
    const int pos = (int) (to_slot ? history : s.history).size();
    if (partitioned()) {
        // Optional session state grows with actual tokens, never with weights.
        // One shared GPU cache is scheduled serially; experts never leave VRAM.
        std::vector<std::pair<void*, size_t>> regions;
        for (int l = stage.begin; l < stage.end; ++l) {
            auto& y = L[l];
            regions.push_back({y.window, size_t(kWindow) * kHeadDim * 2});
            if (!is_kv_source(l)) continue;
            const size_t rows = size_t(pos / y.ratio);
            if (rows) {
                regions.push_back({y.comp, rows * kHeadDim * 2});
                regions.push_back({y.idx_keys, rows * kIndexDim * 2});
            }
            if (y.kv_state) {
                regions.push_back({y.kv_state, size_t(y.ratio) * kHeadDim * 4});
                regions.push_back({y.score_state, size_t(y.ratio) * kHeadDim * 4});
            }
        }
        if (incoming_comp && pos / 2)
            regions.push_back({incoming_comp, size_t(pos / 2) * kHeadDim * 2});
        ds_slot_regions(regions);
        size_t bytes = 0; for (const auto& region : regions) bytes += region.second;
        if (to_slot) {
            size_t total = bytes;
            for (size_t i = 0; i < slot_states.size(); ++i)
                if (int(i) != slot) total += slot_states[i].host_state.size();
            // Per-stage bound, including the private main-session save area.
            // Refuse before allocation instead of swapping under long sessions.
            if (total > (4ull << 30)) throw std::runtime_error("split session cache exceeds 4 GiB per stage");
            s.host_state.resize(bytes);
        }
        else if (s.host_state.empty() && s.history.empty()) {
            if (ds) ds_set_pending(0, 0);
            history.clear(); lg.clear(); return;
        } else if (s.host_state.size() != bytes) throw std::logic_error("split session state size mismatch");
        ds_slot_pending(s.draft_pending, s.draft_pos0, to_slot);
        ck(cudaStreamSynchronize(st), "split session quiesce");
        size_t offset = 0;
        for (const auto& region : regions) {
            for (size_t done = 0; done < region.second; done += 4ull << 20) {
                const size_t nbytes = std::min<size_t>(4ull << 20, region.second - done);
                auto* host = s.host_state.data() + offset + done;
                auto* device = static_cast<uint8_t*>(region.first) + done;
                ck(cudaMemcpy(to_slot ? static_cast<void*>(host) : device,
                              to_slot ? device : static_cast<void*>(host), nbytes,
                              to_slot ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice), "split session state copy");
            }
            offset += region.second;
        }
        if (to_slot) { s.history = history; s.logits = lg; }
        else { history = s.history; lg = s.logits; }
        return;
    }
    auto copy = [&](void* main_buf, void* slot_buf, size_t bytes) {
        if (!bytes) return;
        ck(cudaMemcpyAsync(to_slot ? slot_buf : main_buf, to_slot ? main_buf : slot_buf, bytes,
                           cudaMemcpyDeviceToDevice, st), "slot copy");
    };
    for (int l = 0; l < kLayers; ++l) {
        auto& y = L[l];
        auto& d = s.L[l];
        copy(y.window, d.window, (size_t) kWindow * kHeadDim * sizeof(bf16));
        if (!is_kv_source(l)) continue;
        const size_t rows = (size_t) std::min(pos / y.ratio + 1, max_seq / y.ratio + 1);
        copy(y.comp, d.comp, rows * kHeadDim * sizeof(bf16));
        copy(y.idx_keys, d.idx_keys, rows * kIndexDim * sizeof(bf16));
        if (y.kv_state) {
            copy(y.kv_state, d.kv_state, (size_t) y.ratio * kHeadDim * sizeof(float));
            copy(y.score_state, d.score_state, (size_t) y.ratio * kHeadDim * sizeof(float));
        }
    }
    ck(cudaStreamSynchronize(st), "slot copy");
    if (to_slot) s.history = history;
    else history = s.history;
}

void Engine::Impl::enqueue_slots(int m) {
    auto& v = *batch_ws;
    auto copy = [&](void* dst, const void* src, size_t bytes) {
        ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, st), "batch copy");
    };
    auto linear = [&](const bf16* x, const Fp8& w, bf16* out) {
        fp8_quantize_activation_f32((const uint16_t*) x, m, w.k, v.act, st);
        fp8_block_gemv_q(v.act, m, w.k, w.w, w.s, w.n, (uint16_t*) out, st);
    };
    ck(cudaMemcpyAsync(v.params, v.params_host, m * 4 * sizeof(int), cudaMemcpyHostToDevice, st), "batch params");
    if (n_eng) ck(cudaMemcpyAsync(v.eng, v.eng_host, m * v.eng_bytes, cudaMemcpyHostToDevice, st), "batch engram");
    for (int t = 0; t < m; ++t) {
        ops::window_index_device(v.params + t * 4 + 1, v.indices + t * (kWindow + kIndexTopK), st);
        ops::embed_device(embed, v.params + t * 4, v.h + t * kHc * kDim, st);
        copy(v.mix + t * kHc, one_hot_dev, kHc * sizeof(float));
    }
    const bf16 *comp[kVerifyMaxTokens] = {}, *keys[kVerifyMaxTokens] = {};
    int eng_i = 0;
    for (int l = 0; l < kLayers; ++l) {
        auto& y = L[l];
        if (is_engram_layer(l)) {
            const int cols = (pack.engram_hash().max_ngram - 1) * pack.engram_hash().n_heads;
            for (int t = 0; t < m; ++t) {
                const auto* w = v.eng + t * v.eng_bytes + size_t(eng_i) * kEngRows * (256 + 8);
                ops::engram_dequant(w, w + kEngRows * 256, cols, v.eng_vals + t * y.eng_wkv.k, st);
            }
            linear(v.eng_vals, y.eng_wkv, v.eng_kv);
            ops::engram_apply(v.h, v.eng_kv, y.eng_qw, y.eng_kw, kNormEps, m, st);
            ++eng_i;
        }
        kernels::hc_mixes_pre(v.h, m, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, v.mix, v.xa, v.apre, v.apost,
                              v.acomb, st);
        ops::rmsnorm(v.xa, y.attn_norm, v.xa, kDim, kNormEps, m, st);
        linear(v.xa, y.wq_a, v.qr);
        ops::rmsnorm(v.qr, y.q_norm, v.qr, kQLora, kNormEps, m, st);
        linear(v.qr, y.wq_b, v.q);
        linear(v.xa, y.wkv, v.kvv);
        ops::rmsnorm(v.kvv, y.kv_norm, v.kvv, kHeadDim, kNormEps, m, st);
        for (int t = 0; t < m; ++t) {
            auto& d = slot_states[batch_row_slot[t]].L[l];
            const int* p = v.params + 4 * t;
            const bf16* x = v.xa + t * kDim;
            bf16* qt = v.q + t * kHeads * kHeadDim;
            bf16* kt = v.kvv + t * kHeadDim;
            bf16* ot = v.o + t * kHeads * kHeadDim;
            int32_t* idx = v.indices + t * (kWindow + kIndexTopK);
            uint8_t* cand = v.candidates + size_t(t) * (max_seq + 1);
            const float* rope = y.ratio ? rope_yarn : rope_plain;
            ops::rope_device(qt, kHeads, kHeadDim, rope, p + 1, 0, false, st);
            ops::rope_device(kt, 1, kHeadDim, rope, p + 1, 0, false, st);
            ops::act_quant_inplace(kt, kHeadDim, st);
            ops::row_copy_device(d.window, kt, kHeadDim * sizeof(bf16), p + 1, 1, kWindow, st);
            bool have_latent = false;
            if (is_kv_source(l)) {
                if (y.ratio == 1) {
                    ops::bf16_linear(x, nullptr, y.c_wkv, kDim, kHeadDim, latent, nullptr, st);
                    ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                    have_latent = true;
                } else {   // ratio 2: slot pos % 2 of this row's own position; a group completes at odd positions
                    const int slot = batch_row_parity[t];
                    ops::bf16_linear(x, nullptr, y.c_wkv, kDim, kHeadDim, nullptr, d.kv_state + slot * kHeadDim, st);
                    ops::bf16_linear(x, nullptr, y.c_wgate, kDim, kHeadDim, nullptr, d.score_state + slot * kHeadDim,
                                     st);
                    if (slot == 1) {
                        ops::compress_pool(d.kv_state, d.score_state, 2, latent, 1, st);
                        ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                        have_latent = true;
                    }
                }
                comp[t] = d.comp;
                keys[t] = d.idx_keys;
            }
            if (is_index_source(l)) {
                if (have_latent) {
                    ops::bf16_linear(latent, nullptr, y.idx_wk, kHeadDim, kIndexDim, ik, nullptr, st);
                    ops::rmsnorm(ik, y.idx_knorm, ik, kIndexDim, kNormEps, 1, st);
                    ops::rope_device(ik, 1, kIndexDim, rope_yarn, p + 1, 1 - y.ratio, false, st);
                    ops::fp4_quant_inplace(ik, kIndexDim, 32, false, st);
                    ops::row_copy_device(d.idx_keys, ik, kIndexDim * sizeof(bf16), p + 1, y.ratio, max_seq + 1, st);
                }
                fp8_linear(v.qr + t * kQLora, y.idx_wq_b, iq);
                ops::rope_device(iq, kIndexHeads, kIndexDim, rope_yarn, p + 1, 0, false, st);
                ops::fp4_quant_inplace(iq, kIndexHeads * kIndexDim, 32, false, st);
                ops::bf16_linear(x, nullptr, y.idx_wp, kDim, kIndexHeads, iw_raw, nullptr, st);
                ops::scale_bf16(iw_raw, float(std::pow(kIndexDim, -0.5) * std::pow(kIndexHeads, -0.5)), iw,
                                kIndexHeads, st);
                const int64_t cap = y.ratio == 1 ? tcap1 : tcap2;
                kernels::indexer_topk_device(iq, keys[t], p + 1, y.ratio, cap, iw,
                                             l > kCandidateLayer ? cand : nullptr, kIndexTopK, kWindow, scores,
                                             idx + kWindow, st);
                if (l == kCandidateLayer)
                    kernels::candidate_blocks_device(scores, p + 1, y.ratio, cap, kCandidateBlocks, kCandidateBlock,
                                                     cand, st);
            }
            if (have_latent) {
                ops::rope_device(latent, 1, kHeadDim, rope_yarn, p + 1, 1 - y.ratio, false, st);
                ops::fp4_quant_inplace(latent, kHeadDim, 16, true, st);
                ops::row_copy_device(d.comp, latent, kHeadDim * sizeof(bf16), p + 1, y.ratio, max_seq + 1, st);
            }
            if (y.ratio)
                kernels::sparse_attn_decode_device(qt, d.window, comp[t], idx, p + (y.ratio == 1 ? 2 : 3), y.sink,
                                                   float(std::pow(kHeadDim, -0.5)), ot, st);
            else
                kernels::sparse_attn_decode(qt, d.window, nullptr, idx, 1, kWindow, y.sink,
                                            float(std::pow(kHeadDim, -0.5)), ot, st);
            ops::rope_device(ot, kHeads, kHeadDim, rope, p + 1, 0, true, st);
#ifdef STRATA_DS41_1017
            ops::act_quant_inplace(ot, kHeads * kHeadDim, st);
#endif
            if (y.wo_a8.w) wo_a_grouped_fp8(ot, y.wo_a8.w, y.wo_a8.s, v.oa + t * kOGroups * kOLora, st);
            else ops::wo_a_grouped(ot, y.wo_a, v.oa + t * kOGroups * kOLora, st);
        }
        linear(v.oa, y.wo_b, v.attn_out);
        ops::hc_post(v.attn_out, v.h, v.apost, v.acomb, v.h2, m, st);
        kernels::hc_mixes_pre(v.h2, m, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, v.apre, v.xf, v.fpre, v.fpost,
                              v.fcomb, st);
        ops::rmsnorm(v.xf, y.ffn_norm, v.xf, kDim, kNormEps, m, st);
        // the routed experts of all rows: VRAM hits and a quota of RAM misses per row on the GPU (staged), the CPU
        // the rest, as the decode path does for one row
        int32_t* ids = v.routes + l * VerifyWorkspace::M * kTopK;
        float* w = v.weights + l * VerifyWorkspace::M * kTopK;
        kernels::router_topk(v.xf, m, y.gate_w, y.gate_bias, ids, w, st);
        ops::to_half_fp8q(v.xf, v.half, m * kDim, st);
        const bool tier = vram && vram->slots() > 0;
        const auto* ram = host && host->experts_dev() ? host->experts_dev() + size_t(l) * kExperts : nullptr;
        const bool staged = ram && batch_stage && zc_blobs;
        v.db->publish(v.half, ids, w, m, tier ? vram->res_dev() + size_t(l) * kExperts : nullptr, v.selected, l + 1,
                      st, tier ? vram->experts_dev() : nullptr, ram, zc_quota ? zc_quota.get() + l : nullptr,
                      staged ? batch_stage.get() : nullptr, staged ? zc_blobs.get() + size_t(l) * kExperts : nullptr);
        if (staged) batch_stage->fork_copy(st);
        ck(cudaMemsetAsync(v.routed, 0, m * kDim * sizeof(float), st), "batch routed");
        if (!staged && (tier || ram))
            kernels::exl3_moe_decode((const __half*) v.half, m, v.selected, w, kTopK, v.db->gpu_experts(), v.routed,
                                     vram ? vram->workspace() : zc_workspace.get(), VramExperts::kWorkspaceBytes, st);
        linear(v.xf, y.sh_w1, v.g);
        linear(v.xf, y.sh_w3, v.u);
        ops::swiglu(v.g, v.u, kSwigluLimit, v.sh_h, m * kMoeInter, st);
        linear(v.sh_h, y.sh_w2, v.sh_out);
        if (staged) {
            batch_stage->join(st);
            kernels::exl3_moe_decode((const __half*) v.half, m, v.selected, w, kTopK, v.db->gpu_experts(), v.routed,
                                     vram ? vram->workspace() : zc_workspace.get(), VramExperts::kWorkspaceBytes, st);
        }
        v.db->wait_add(v.routed, m, l + 1, st);
        ops::add_f32_bf16(v.routed, v.sh_out, v.ffn_out, m * kDim, st);
        ops::hc_post(v.ffn_out, v.h2, v.fpost, v.fcomb, v.h, m, st);
        copy(v.mix, v.fpre, m * kHc * sizeof(float));
    }
    ops::hc_pre(v.h, v.mix, v.final_x, m, st);
    ops::rmsnorm(v.final_x, final_norm, v.final_x, kDim, kNormEps, m, st);
    for (int t = 0; t < m; ++t) {
        ops::bf16_linear(v.final_x + t * kDim, nullptr, head, kDim, kVocab, nullptr, v.logits + t * kVocab, st);
        ops::argmax_logits(v.logits + t * kVocab, v.next + t, st);
    }
    ck(cudaMemcpyAsync(v.next_host, v.next, m * sizeof(int), cudaMemcpyDeviceToHost, st), "batch next");
    ck(cudaMemcpyAsync(v.logits_host, v.logits, size_t(m) * kVocab * sizeof(float), cudaMemcpyDeviceToHost, st),
       "batch logits");
    ck(cudaMemcpyAsync(v.routes_host, v.routes, kLayers * VerifyWorkspace::M * kTopK * sizeof(int32_t),
                       cudaMemcpyDeviceToHost, st), "batch routes");
}

std::vector<int> Engine::Impl::step_slots(const std::vector<int>& rows, const std::vector<int>& tokens, Timing& tm) {
    usable();
    const int m = (int) rows.size();
    if (slot_states.empty()) throw std::logic_error("ds41 batch: the engine has no batch slots");
    if (m < 1 || m > (int) slot_states.size() || tokens.size() != rows.size())
        throw std::invalid_argument("ds41 batch: 1 .. batch_slots rows, one token each");
    if (verify_pending) throw std::logic_error("commit the pending verify window first");
    for (int t = 0; t < m; ++t) {
        if (rows[t] < 0 || rows[t] >= (int) slot_states.size())
            throw std::invalid_argument("ds41 batch: slot " + std::to_string(rows[t]) + " does not exist");
        for (int u = 0; u < t; ++u)
            if (rows[u] == rows[t]) throw std::invalid_argument("ds41 batch: a slot appears twice in one step");
        check_token(tokens[t]);
        if ((int) slot_states[rows[t]].history.size() >= max_seq)
            throw std::runtime_error("ds41 batch: slot " + std::to_string(rows[t]) + " is at max_seq");
    }
    if (!batch_ws) batch_ws = std::make_shared<VerifyWorkspace>(*this);
    auto& v = *batch_ws;
    tm = Timing{};
    const double begin = now_ms();
    // every row's engram rows in one read (rows from all tables and all rows in flight together)
    int max_pos = 0, parity_bits = 0;
    int64_t slot_code = 0;
    const int cols = n_eng ? (pack.engram_hash().max_ngram - 1) * pack.engram_hash().n_heads : 0;
    std::vector<std::vector<int64_t>> ids(n_eng, std::vector<int64_t>((size_t) m * kEngRows, 0));
    std::vector<std::vector<uint8_t>> wbuf(n_eng), sbuf(n_eng);
    for (int t = 0; t < m; ++t) {
        auto& s = slot_states[rows[t]];
        const int pos = (int) s.history.size();
        s.history.push_back(pack.engram_hash().token_map[tokens[t]]);
        int li = 0;
        for (int l = 0; l < kLayers; ++l) {
            if (!is_engram_layer(l)) continue;
            engram_ids(l, li, pos, s.history);
            std::copy(eng_ids[li].begin(), eng_ids[li].begin() + cols, ids[li].begin() + (size_t) t * kEngRows);
            ++li;
        }
        const int params[] = {tokens[t], pos, pos + 1, (pos + 1) / 2};
        std::copy_n(params, 4, v.params_host + 4 * t);
        batch_row_slot[t] = rows[t];
        batch_row_parity[t] = pos & 1;
        parity_bits |= (pos & 1) << t;
        slot_code |= (int64_t) rows[t] << (3 * t);
        max_pos = std::max(max_pos, pos);
    }
    try {
        if (n_eng) {
            std::vector<const int64_t*> idp;
            std::vector<uint8_t*> wp, sp;
            for (int li = 0; li < n_eng; ++li) {
                wbuf[li].resize((size_t) m * kEngRows * 256);
                sbuf[li].resize((size_t) m * kEngRows * 8);
                // the rows of row t at t * kEngRows: one read of every row's cols (the unused tail rows are id 0)
                for (int t = 0; t < m; ++t)
                    for (int c = cols; c < kEngRows; ++c) ids[li][(size_t) t * kEngRows + c] = 0;
                idp.push_back(ids[li].data());
                wp.push_back(wbuf[li].data());
                sp.push_back(sbuf[li].data());
            }
            eng_rows_batch->read(idp, m * kEngRows, wp, sp);
            for (int t = 0; t < m; ++t)
                for (int li = 0; li < n_eng; ++li) {
                    uint8_t* dst = v.eng_host + t * v.eng_bytes + (size_t) li * kEngRows * (256 + 8);
                    std::memcpy(dst, wbuf[li].data() + (size_t) t * kEngRows * 256, (size_t) kEngRows * 256);
                    std::memcpy(dst + kEngRows * 256, sbuf[li].data() + (size_t) t * kEngRows * 8, (size_t) kEngRows * 8);
                }
        }
    } catch (...) {
        for (int t = 0; t < m; ++t) slot_states[rows[t]].history.pop_back();   // nothing reached the device
        throw;
    }
    tm.engram_ms = now_ms() - begin;
    if (vram) tm.vram_swaps = vram->between_steps();
    tcap1 = cap_of(max_pos + 1, max_seq + 1);
    tcap2 = cap_of((max_pos + 1) / 2, max_seq / 2 + 1);
    const std::array<int64_t, 5> key{m, slot_code, parity_bits, tcap1, tcap2};
    // the first step of a shape is eager; a graph is captured once every kernel ran once (as verify does)
    cudaGraphExec_t executable = nullptr;
    if (use_graph && batch_warmed) {
        auto it = batch_graphs.find(key);
        if (it == batch_graphs.end()) {
            cudaGraph_t graph = nullptr;
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "batch capture");
            try {
                enqueue_slots(m);
                ck(cudaStreamEndCapture(st, &graph), "batch end capture");
                ck(cudaGraphInstantiate(&executable, graph, 0), "batch instantiate");
            } catch (...) {
                if (!graph) cudaStreamEndCapture(st, &graph);
                if (graph) cudaGraphDestroy(graph);
                broken = true;   // the rows' histories already hold this step's tokens
                throw;
            }
            cudaGraphDestroy(graph);
            it = batch_graphs.emplace(key, executable).first;
        }
        executable = it->second;
    }
    v.db->reset();
    std::atomic<bool> cancel{false};
    std::exception_ptr cpu_error;
    std::atomic<int> cpu_ram{0}, cpu_file{0}, cpu_ssd{0}, hits{0};
    std::atomic<int64_t> cpu_us{0};
    std::thread cpu([&] {
        for (int l = 0; l < kLayers; ++l) {
            if (!v.db->wait_published(l + 1, cancel)) return;
            hits += v.db->counts().vram;
            if (!cpu_error) {
                try {
                    c10::Half weights[kVerifyMaxTokens * kTopK];
                    int misses = 0;
                    for (int i = 0; i < m * kTopK; ++i) {
                        weights[i] = c10::Half(__half_as_ushort(__float2half_rn(v.db->w()[i])), c10::Half::from_bits());
                        const int e = v.db->ids()[i];
                        if (e < 0) continue;
                        ++misses;
                        if (host && host->in_memory(l, e)) ++cpu_ram;
                        else {
                            ++cpu_file;
                            if (file_pages_missing(l, e)) {
                                ++cpu_ssd;
                                if (fetch_now) warm_file_expert(l, e);
                            }
                        }
                    }
                    if (misses) admit_rows(l, v.db->ids(), m * kTopK);   // the adaptive RAM tier keeps them
                    const double start = now_ms();
                    if (misses)
                        exl3_moe_cpu_forward_raw(L[l].moe_handle, (const at::Half*) v.db->x(), v.db->ids(), weights,
                                                 v.db->y(), m, kTopK, cpu_threads);
                    else
                        std::fill_n(v.db->y(), m * kDim, 0.0f);
                    cpu_us += (int64_t) ((now_ms() - start) * 1000.0);
                } catch (...) {
                    cpu_error = std::current_exception();
                }
            }
            if (cpu_error) std::fill_n(v.db->y(), m * kDim, 0.0f);   // keep the protocol; the step rethrows
            v.db->mark_done(l + 1);
        }
    });
    try {
        if (executable) ck(cudaGraphLaunch(executable, st), "batch launch");
        else enqueue_slots(m);
        ck(cudaStreamSynchronize(st), "batch step");
    } catch (...) {
        cancel = true;
        cpu.join();
        v.db->mark_done(kLayers);
        cudaStreamSynchronize(st);
        broken = true;
        throw;
    }
    cpu.join();
    if (cpu_error) {
        broken = true;
        std::rethrow_exception(cpu_error);
    }
    batch_warmed = true;
    if (vram || (host && host->reserve() > 0)) {
        int32_t row[kLayers * kTopK];
        for (int t = 0; t < m; ++t) {
            for (int l = 0; l < kLayers; ++l)
                std::copy_n(v.routes_host + (l * VerifyWorkspace::M + t) * kTopK, kTopK, row + l * kTopK);
            if (vram) vram->count(row, kTopK);
            if (host && host->reserve() > 0) host->end_step(row, kTopK);   // publish the step's reads, record uses
        }
    }
    std::vector<int> next(v.next_host, v.next_host + m);
    tm.expert_total = m * kLayers * kTopK;
    tm.expert_hits = hits.load();
    tm.ram_experts = cpu_ram.load();
    tm.file_experts = cpu_file.load();
    tm.ssd_experts = cpu_ssd.load();
    tm.cpu_experts_ms = cpu_us.load() / 1000.0;
    tm.total_ms = now_ms() - begin;
    tm.gpu_ms = tm.total_ms - tm.engram_ms;
    return next;
}

int Engine::batch_slots() const { return tail_ ? std::max(0, impl_->opt.batch_slots) : (int) impl_->slot_states.size(); }

void Engine::copy_to_slot(int slot) {
    if (tail_) {
        if (split_verify_pending_) throw std::logic_error("commit split verification before slot copy");
        if (slot < 0 || slot >= batch_slots()) throw std::invalid_argument("invalid split session slot");
        try { impl_->slot_copy(slot, true); tail_->slot_copy(slot, true); }
        catch (...) { impl_->broken = tail_->broken = true; throw; }
        return;
    }
    DeviceScope scope(impl_->stage.device);
    impl_->slot_copy(slot, true);
}

void Engine::copy_from_slot(int slot) {
    if (tail_) {
        if (split_verify_pending_) throw std::logic_error("commit split verification before slot copy");
        if (slot < 0 || slot >= batch_slots()) throw std::invalid_argument("invalid split session slot");
        try { impl_->slot_copy(slot, false); tail_->slot_copy(slot, false); }
        catch (...) { impl_->broken = tail_->broken = true; throw; }
        return;
    }
    DeviceScope scope(impl_->stage.device);
    impl_->slot_copy(slot, false);
}

int Engine::slot_position(int slot) const {
    if (tail_ && (slot < 0 || slot >= batch_slots())) throw std::invalid_argument("invalid split session slot");
    if (slot < 0 || slot >= (int) impl_->slot_states.size())
        throw std::invalid_argument("ds41 batch: slot " + std::to_string(slot) + " does not exist");
    return (int) impl_->slot_states[slot].history.size();
}

std::vector<int> Engine::step_slots(const std::vector<int>& slots, const std::vector<int>& tokens) {
    if (tail_) {
        if (split_verify_pending_) throw std::logic_error("commit split verification before batch step");
        if (slots.empty() || slots.size() != tokens.size() || slots.size() > size_t(batch_slots()))
            throw std::invalid_argument("split batch: invalid row count");
        std::vector<int> unique = slots; std::sort(unique.begin(), unique.end());
        if (unique.front() < 0 || unique.back() >= batch_slots() ||
            std::adjacent_find(unique.begin(), unique.end()) != unique.end())
            throw std::invalid_argument("split batch: slots must be distinct valid IDs");
        for (size_t i = 0; i < slots.size(); ++i) {
            Impl::check_token(tokens[i]);
            if (slot_position(slots[i]) >= impl_->max_seq) throw std::invalid_argument("split batch exceeds context");
        }
        const double start = now_ms(); Timing total;
        std::vector<int> next; split_slot_logits_.clear();
        try {
            const int saved = batch_slots();
            impl_->slot_copy(saved, true); tail_->slot_copy(saved, true);
            for (size_t i = 0; i < slots.size(); ++i) {
                copy_from_slot(slots[i]);
                next.push_back(step(tokens[i], position()));
                total.expert_hits += timing_.expert_hits; total.expert_total += timing_.expert_total;
                total.engram_ms += timing_.engram_ms;
                split_slot_logits_.push_back(last_logits()); copy_to_slot(slots[i]);
            }
            impl_->slot_copy(saved, false); tail_->slot_copy(saved, false);
        } catch (...) { impl_->broken = tail_->broken = true; throw; }
        total.total_ms = total.gpu_ms = now_ms() - start; timing_ = total;
        return next;
    }
    DeviceScope scope(impl_->stage.device);
    return impl_->step_slots(slots, tokens, timing_);
}

const float* Engine::slot_logits(int row) const {
    if (tail_) {
        if (row < 0 || row >= int(split_slot_logits_.size())) throw std::invalid_argument("no split batch logits for row");
        return split_slot_logits_[row].data();
    }
    if (!impl_->batch_ws || row < 0 || row >= kVerifyMaxTokens)
        throw std::invalid_argument("ds41 batch: no logits for that row");
    return impl_->batch_ws->logits_host + (size_t) row * kVocab;
}
