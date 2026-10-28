// Bounded staging/publication with tiny synthetic payloads, not a model load.
#include "bench_util.hpp"
#include "strata/ds41/vram_experts.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/bounded_upload.hpp"
#include "strata/ds41/expert_stream.hpp"
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
using namespace ds41test;
using namespace strata::ds41;

__global__ void consume_selection(const int32_t* selected, const kernels::Exl3Expert* descriptors,
                                    const int32_t* routes, int32_t* result, int table_slots,int n=6) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int slot = selected[i];
    if (slot < 0 || slot >= table_slots) { result[i] = 0; return; }
    const auto* bytes = reinterpret_cast<const uint8_t*>(descriptors[slot].w1.trellis);
    bool exact = table_slots == 6 ? slot == i : true;
    const int shift = 64 * (routes[i] % 2);
    for (int k = 0; k < 64; ++k)
        exact &= bytes[k] == uint8_t((shift + k) * 17 + routes[i] % 8);
    result[i] = exact;
}

int main() {
    require_gpu(); Verdict verdict;
    char name[] = "/tmp/opencode/ds41-stream-test-XXXXXX";
    const char* made = mkdtemp(name);
    if (!made) return 1;
    const std::filesystem::path dir(made);
    try {
        {
            std::ofstream(dir / "pack_info.txt") << "profile libertaidai-29\nlayers 40\nexperts 384\nfinished 1\n";
            std::ofstream index(dir / "index.txt");
            index << "embed.weight bf16 2 129280 5120 0 1323827200\n"
                  << "head.trellis i16 3 8 8 64 0 8192\nhead.suh f16 1 5120 0 10240\nhead.svh f16 1 129280 0 258560\n"
                  << "layers.0.attn.wq_b.suh f16 1 1280 0 2560\nlayers.0.attn.wq_b.svh f16 1 32768 0 65536\n"
                  << "layers.0.hc_attn_fn f32 2 24 20480 0 1966080\n";
            for (int g = 0; g < 8; ++g)
                index << "layers.0.attn.wo_a.slice." << g << ".suh f16 1 4096 0 8192\n"
                      << "layers.0.attn.wo_a.slice." << g << ".svh f16 1 1024 0 2048\n";
            std::ofstream experts(dir / "experts.txt");
            const char* projections[] = {"w1", "w3", "w2"};
            const char* components[] = {"trellis", "suh", "svh", "mul1"};
            for (int l = 0; l < 40; ++l) for (int e = 0; e < 384; ++e) {
                experts << l << ' ' << e << ' ' << (e % 8) * 4096 << " 4096 2 2 2";
                for (int c = 0; c < 12; ++c)
                    experts << ' ' << projections[c / 4] << '.' << components[c % 4] << ':' << c * 256 + 64 * (e % 2) << ':'
                            << (c % 4 == 0 ? 64 : c % 4 == 3 ? 4 : 32);
                experts << '\n';
            }
            std::ofstream(dir / "engram.txt") << "# no tables in synthetic fixture\n";
            std::ofstream(dir / "engram_hash.txt") << "# no hashes in synthetic fixture\n";
            std::vector<int32_t> tokenmap(kVocab);
            std::ofstream tm(dir / "engram_tokenmap.bin", std::ios::binary);
            tm.write(reinterpret_cast<const char*>(tokenmap.data()), tokenmap.size() * 4);
        }
        std::vector<uint8_t> payload(8 * 4096);
        for (size_t i = 0; i < payload.size(); ++i) payload[i] = uint8_t(i * 17 + i / 4096);
        {
            std::ofstream weights(dir / "experts.bin", std::ios::binary);
            weights.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        }
        {
            Pack pack(dir.string());
            VramExperts::Adapt adapt; adapt.every = 0;
            // No resident slots even on a large GPU: force the bounded miss path.
            VramExperts tier(pack, "", -1, SIZE_MAX, adapt, 0, 1, true, -1, true);
            verdict.check(tier.slots() == 0 && tier.streaming(), "bounded zero-residency mode constructs without a host expert tier");
            Dev<int32_t> ids(6), selected(6), consumed(6);
            cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream test");
            for (int replay = 0; replay < 128; ++replay) {
                std::vector<int32_t> routes(6);
                for (int i = 0; i < 6; ++i) routes[i] = (i * 17 + replay) % 384;
                ids.up(routes); ck(cudaDeviceSynchronize(), "fixture route upload");
                tier.prepare_decode(0, ids.p, 6, selected.p, stream);
                // Consume immediately on the nonblocking inference stream,
                // before any default-stream D2H copy can hide a publication race.
                consume_selection<<<1,32,0,stream>>>(selected.p, tier.experts_dev(), ids.p, consumed.p, 6);
                ck(cudaStreamSynchronize(stream), "immediate streamed consumer");
                const auto correct = consumed.down();
                verdict.check(std::all_of(correct.begin(), correct.end(), [](int32_t value) { return value == 1; }),
                              "nonblocking consumer observes fresh descriptors with alternating component offsets");
                const auto slots = selected.down();
                for (int i = 0; i < 6; ++i) {
                    kernels::Exl3Expert descriptor;
                    ck(cudaMemcpy(&descriptor, tier.experts_dev() + slots[i], sizeof descriptor, cudaMemcpyDeviceToHost), "staged descriptor");
                    std::vector<uint8_t> got(64);
                    ck(cudaMemcpy(got.data(), descriptor.w1.trellis, got.size(), cudaMemcpyDeviceToHost), "staged payload");
                    verdict.check(slots[i] == i && std::memcmp(got.data(), payload.data() + (routes[i] % 8) * 4096 + 64 * (routes[i] % 2), 64) == 0,
                                  "changed routes publish exact stored payload and ordered descriptor selection");
                }
            }
            verdict.check(tier.streamed_decode_experts() == 768, "all requested misses counted, no skipped experts");
            {
                bool refused=false;
                try { tier.prepare_decode(1,ids.p,6,selected.p,stream); }
                catch(const std::invalid_argument&) { refused=true; }
                verdict.check(refused && tier.streamed_decode_experts()==768,
                              "an unowned layer is refused before using a stage-sized payload arena");
            }
            verdict.check(!tier.fully_resident_layer(0), "zero resident tier never takes the device-only shortcut");
            {
                VramExperts full(pack, "", -1, 0, adapt, 0, 1, true, -1, true);
                verdict.check(full.slots() == 384 && full.fully_resident_layer(0) && !full.fully_resident_layer(1),
                              "fast-path proof covers every expert in only the owned layer");
                ids.up(std::vector<int32_t>{0,33,77,120,256,383});
                ck(cudaDeviceSynchronize(),"fully resident fixture");
                cudaGraph_t graph;cudaGraphExec_t exec;
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"resident lookup capture");
                full.prepare_decode(0,ids.p,6,selected.p,stream,true);
                consume_selection<<<1,32,0,stream>>>(selected.p,full.experts_dev(),ids.p,consumed.p,390);
                ck(cudaStreamEndCapture(stream,&graph),"resident lookup capture end");
                ck(cudaGraphInstantiate(&exec,graph,0),"resident lookup graph");
                for(int replay=0;replay<8;++replay) {
                    ck(cudaGraphLaunch(exec,stream),"resident lookup replay");
                    ck(cudaStreamSynchronize(stream),"resident lookup consumer");
                    const auto correct=consumed.down();
                    verdict.check(std::all_of(correct.begin(),correct.end(),[](int32_t v){return v==1;}),
                                  "fully static routing needs no host copy or synchronization inside capture");
                }
                verdict.check(full.streamed_decode_experts()==0,"fully resident shortcut never uploads an expert");
                ck(cudaGraphExecDestroy(exec),"resident graph release");ck(cudaGraphDestroy(graph),"resident capture release");
            }
            verdict.check(experimental_decode_cache_bytes(nullptr) == 0 && experimental_decode_cache_bytes("0") == 0 &&
                          experimental_decode_cache_bytes("8192") == (8192ull << 20), "cache opt-in parser retains disabled default and bounded maximum");
            for (const char* value : {"", "-1", "+1", "1MiB", "8193", " 2", "2 ", "999999999999999999999999999"}) {
                bool refused = false;
                try { (void)experimental_decode_cache_bytes(value); } catch (const std::invalid_argument&) { refused = true; }
                verdict.check(refused, "malformed or oversized cache budget fails closed");
            }
            verdict.check(experimental_resident_slot_limit(nullptr,0)==-1 &&
                          experimental_resident_slot_limit("12,34",0)==12 && experimental_resident_slot_limit("12,34",1)==34,
                          "exact resident counts select logical stage/rank, automatic remains default");
            for (const char* value : {"", "1", "0,1", "1,-1", "+1,2", "1,2,3", "1, 2", "15361,1"}) {
                bool refused=false;
                try { (void)experimental_resident_slot_limit(value,0); } catch(const std::invalid_argument&) { refused=true; }
                verdict.check(refused,"malformed fixed resident set refused");
            }
            {
                const char* previous=std::getenv("DS41_EXPERIMENTAL_RESIDENT_SLOTS");
                const std::string saved=previous ? previous : "";
                setenv("DS41_EXPERIMENTAL_RESIDENT_SLOTS","1,1",1);
                bool refused=false;
                try { VramExperts impossible(pack,"",-1,SIZE_MAX,adapt,0,1,true,-1,true); }
                catch(const std::runtime_error&) { refused=true; }
                if(previous) setenv("DS41_EXPERIMENTAL_RESIDENT_SLOTS",saved.c_str(),1);
                else unsetenv("DS41_EXPERIMENTAL_RESIDENT_SLOTS");
                verdict.check(refused,"fixed resident set fails instead of silently shrinking under insufficient budget");
            }
            for (int capacity : {2, 8}) {
                // Existing cases below exercise the unchanged LRU default.
                VramExperts cached(pack, "", -1, SIZE_MAX, adapt, 0, 2, true, -1, true, size_t(capacity)*4096);
                verdict.check(cached.decode_cache_slots() == capacity, "cache uses only the requested complete-expert budget");
                auto run = [&](int layer, const std::vector<int32_t>& routes) {
                    ids.up(routes); ck(cudaDeviceSynchronize(), "cached routes upload");
                    cached.prepare_decode(layer, ids.p, 6, selected.p, stream);
                    consume_selection<<<1,32,0,stream>>>(selected.p, cached.experts_dev(), ids.p, consumed.p, 6+capacity);
                    ck(cudaStreamSynchronize(stream), "cached immediate consumer");
                    const auto correct = consumed.down();
                    verdict.check(std::all_of(correct.begin(), correct.end(), [](int32_t value) { return value == 1; }),
                                  "cache hits, evictions and staging fallback publish exact immutable payloads");
                };
                run(0, {0,1,2,3,4,5});
                const int64_t before = cached.streamed_decode_experts();
                run(0, {0,1,2,3,4,5});
                verdict.check(cached.streamed_decode_experts()-before == (capacity==2 ? 4 : 0),
                              "repeated experts reuse GPU bytes without another file upload");
                // A miss ordered before a hit must not evict that selected hit.
                run(0, {2,3,4,5,0,1});
                run(0, {7,7,7,0,0,1});
                for (int replay=0; replay<64; ++replay) {
                    std::vector<int32_t> routes(6);
                    for (int i=0;i<6;++i) routes[i]=(replay*3+i)%192;
                    run(replay%2,routes);
                }
                verdict.check(cached.decode_cache_hits()>0 && cached.decode_cache_evictions()>0,
                              "actual cache hits and LRU evictions exercised across layers");
                // A partial group can upload one entry before the next read
                // fails. Recovery must not keep an unpublished descriptor hit.
                std::filesystem::resize_file(dir / "experts.bin", 4096);
                ids.up(std::vector<int32_t>{376,383,376,383,376,383});
                bool short_read = false;
                try { cached.prepare_decode(0,ids.p,6,selected.p,stream); }
                catch (const std::runtime_error&) { short_read = true; }
                verdict.check(short_read, "cached partial upload fails closed on a real short read");
                { std::ofstream weights(dir / "experts.bin",std::ios::binary); weights.write(reinterpret_cast<const char*>(payload.data()),payload.size()); }
                run(0,{376,383,376,383,376,383});
                if(capacity==8) {
                    std::vector<int32_t> routes(36);for(int i=0;i<36;++i) routes[i]=i%2?383:376;
                    Dev<int32_t> many_ids(routes),many_selected(std::vector<int32_t>(36,-77)),many_result(36);
                    ck(cudaDeviceSynchronize(),"batch fixture inputs");
                    const auto uploads=cached.streamed_decode_experts(),hits=cached.decode_cache_hits();
                    verdict.check(cached.try_cached_rows(0,many_ids.p,6,6,many_selected.p,stream),"six-row GPU-cache hit batch admitted");
                    consume_selection<<<1,64,0,stream>>>(many_selected.p,cached.experts_dev(),many_ids.p,many_result.p,14,36);
                    // Reuse/evict cache slots before any explicit host test fence:
                    // prepare_decode must join the queued prior consumer first.
                    run(0,{3,4,5,6,7,8});
                    const auto got=many_result.down();
                    verdict.check(std::all_of(got.begin(),got.end(),[](int v){return v==1;}),"by-value batch selection and prior-consumer lifetime preserve all 36 payloads");
                    verdict.check(cached.cached_row_batches()==1 && cached.decode_cache_hits()>=hits+36,
                                  "actual batched admissions and per-use cache hits counted");
                    // Cache miss is transactional: no selection, LRU age/hit or
                    // upload changes. Sentinel survives a failed admission.
                    many_selected.up(std::vector<int32_t>(36,-77));routes.assign(36,200);many_ids.up(routes);
                    ck(cudaDeviceSynchronize(),"batch miss fixture");
                    const auto before_hits=cached.decode_cache_hits(),before_uploads=cached.streamed_decode_experts();
                    verdict.check(!cached.try_cached_rows(0,many_ids.p,6,6,many_selected.p,stream),"uncached row batch falls back without staging");
                    const auto sentinel=many_selected.down();
                    verdict.check(cached.decode_cache_hits()==before_hits && cached.streamed_decode_experts()==before_uploads &&
                        std::all_of(sentinel.begin(),sentinel.end(),[](int v){return v==-77;}),"failed admission leaves selection and cache accounting unchanged");
                    routes.back()=384;many_ids.up(routes);ck(cudaDeviceSynchronize(),"batch invalid fixture");
                    bool invalid=false;try { cached.try_cached_rows(0,many_ids.p,6,6,many_selected.p,stream); }
                    catch(const std::runtime_error&) { invalid=true; }
                    verdict.check(invalid,"late invalid route is rejected even when an earlier route misses");
                    verdict.check(before_uploads>=uploads,"batch lifetime test exercised cache replacement");
                }
            }
            {
                // Two independent tiers receive the same routes. Their hit/miss
                // decisions, ordered selections and eventual LRU victims must
                // remain identical, including duplicate and mixed-hit rows.
                VramExperts sync(pack,"",-1,SIZE_MAX,adapt,0,2,true,-1,true,8*4096);
                VramExperts async(pack,"",-1,SIZE_MAX,adapt,0,2,true,-1,true,8*4096,0,true);
                constexpr int rows=4, n=rows*6;
                Dev<int32_t> input(n), a(n), b(n), ca(n), cb(n);
                int hits=0,misses=0;
                for(int round=0;round<96;++round) {
                    std::vector<int32_t> routes(n);
                    for(int i=0;i<n;++i) routes[i]=(round/4+i%4)%16;
                    const int layer=(round/16)%2;
                    input.up(routes);ck(cudaDeviceSynchronize(),"async route fixture");
                    const bool hit_a=sync.try_cached_rows(layer,input.p,rows,6,a.p,stream);
                    async.begin_cached_rows(layer,input.p,rows,6,b.p,stream);
                    const bool hit_b=async.finish_cached_rows();
                    verdict.check(hit_a==hit_b,"async and synchronous batch admission agree");
                    if(hit_a) {
                        ++hits;
                        consume_selection<<<1,64,0,stream>>>(a.p,sync.experts_dev(),input.p,ca.p,14,n);
                        consume_selection<<<1,64,0,stream>>>(b.p,async.experts_dev(),input.p,cb.p,14,n);
                    } else {
                        ++misses;
                        for(int row=0;row<rows;++row) {
                            sync.prepare_decode(layer,input.p+row*6,6,a.p+row*6,stream);
                            consume_selection<<<1,32,0,stream>>>(a.p+row*6,sync.experts_dev(),input.p+row*6,ca.p+row*6,14);
                            async.prepare_decode(layer,input.p+row*6,6,b.p+row*6,stream);
                            consume_selection<<<1,32,0,stream>>>(b.p+row*6,async.experts_dev(),input.p+row*6,cb.p+row*6,14);
                        }
                    }
                    ck(cudaStreamSynchronize(stream),"async route consumers");
                    const auto sa=a.down(),sb=b.down(),va=ca.down(),vb=cb.down();
                    bool same=sa==sb && std::all_of(va.begin(),va.end(),[](int v){return v==1;}) &&
                              std::all_of(vb.begin(),vb.end(),[](int v){return v==1;});
                    for(int l=0;l<2;++l) for(int e=0;e<16;++e)
                        same &= sync.prefill_gpu_slot(l,e)==async.prefill_gpu_slot(l,e);
                    same &= sync.decode_cache_hits()==async.decode_cache_hits() &&
                            sync.decode_cache_evictions()==async.decode_cache_evictions() &&
                            sync.streamed_decode_experts()==async.streamed_decode_experts();
                    verdict.check(same,"async admission retains exact selection, payload, LRU and accounting");
                }
                verdict.check(hits>0 && misses>0 && async.decode_cache_evictions()>0,
                              "async comparison actually covers hits, misses and evictions");
                std::vector<int32_t> routes(n,200),sentinel(n,-77);routes.back()=384;
                input.up(routes);b.up(sentinel);ck(cudaDeviceSynchronize(),"async invalid fixture");
                const auto old_hits=async.decode_cache_hits(),old_uploads=async.streamed_decode_experts();
                async.begin_cached_rows(0,input.p,rows,6,b.p,stream);
                bool double_begin=false,mutation=false;
                try { async.begin_cached_rows(0,input.p,rows,6,b.p,stream); }
                catch(const std::invalid_argument&) { double_begin=true; }
                try { async.prepare_decode(0,input.p,6,b.p,stream); }
                catch(const std::invalid_argument&) { mutation=true; }
                verdict.check(double_begin && mutation,"pending read rejects overlap and cache mutation");
                async.cancel_cached_rows();
                verdict.check(b.down()==sentinel && old_hits==async.decode_cache_hits() &&
                              old_uploads==async.streamed_decode_experts(),"cancel drains without selection or cache changes");
                async.begin_cached_rows(0,input.p,rows,6,b.p,stream);
                bool invalid=false;
                try { async.finish_cached_rows(); } catch(const std::runtime_error&) { invalid=true; }
                verdict.check(invalid && b.down()==sentinel && old_hits==async.decode_cache_hits() &&
                              old_uploads==async.streamed_decode_experts(),"late invalid route fails transactionally after async read");
                routes.back()=200;input.up(routes);ck(cudaDeviceSynchronize(),"async miss fixture");
                async.begin_cached_rows(0,input.p,rows,6,b.p,stream);
                verdict.check(!async.finish_cached_rows() && b.down()==sentinel && old_hits==async.decode_cache_hits() &&
                              old_uploads==async.streamed_decode_experts(),"uncached batch leaves state and selection unchanged");
                verdict.check(async.async_cached_row_reads()==99,"async reads include completed, cancelled and missed attempts");
                cudaGraph_t invalid_graph=nullptr;
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"async route capture fixture");
                bool capture_refused=false;
                try { async.begin_cached_rows(0,input.p,rows,6,b.p,stream); }
                catch(const std::invalid_argument&) { capture_refused=true; }
                ck(cudaStreamEndCapture(stream,&invalid_graph),"async route capture end");
                ck(cudaGraphDestroy(invalid_graph),"async route capture release");
                verdict.check(capture_refused,"async cached route read is refused inside capture");
                {
                    VramExperts pending(pack,"",-1,SIZE_MAX,adapt,0,2,true,-1,true,8*4096,0,true);
                    pending.begin_cached_rows(0,input.p,rows,6,b.p,stream);
                    // Destructor must drain the read before freeing its pinned target.
                }
            }
            {
                VramExperts prime(pack,"",-1,SIZE_MAX,adapt,0,2,true,-1,true,8*4096,8*4096);
                const auto first=prime.prime_prefill_cache();
                verdict.check(first.gpu_fills==8 && first.ram_fills==8 && first.bytes==16*4096,
                              "prefill priming fills only budgeted empty GPU/RAM slots");
                const auto second=prime.prime_prefill_cache();
                verdict.check(!second.gpu_fills && !second.ram_fills && !second.bytes,
                              "prefill priming retains valid cache entries across reset reuse");
                for(int e=0;e<8;++e) verdict.check(prime.prefill_gpu_slot(0,e)>=0,"primed expert has immutable GPU selection");
                for(int e=8;e<16;++e) verdict.check(prime.prefill_ram_source(0,e)!=nullptr,"remaining primed experts have frozen pinned source");
                Dev<uint8_t> ring(3*4096);
                {
                    ExpertStream source(pack,nullptr,ring.p,3,4096,2,2,true,
                        [&](int l,int e){return prime.prefill_ram_source(l,e);});
                    const std::vector<std::pair<int,int>> jobs{{0,8},{0,27},{0,9},{0,8}};
                    const auto first_job=source.push(jobs);
                    for(size_t j=0;j<jobs.size();++j) {
                        const auto* ptr=source.wait(first_job+j,stream);std::vector<uint8_t> got(4096);
                        ck(cudaMemcpyAsync(got.data(),ptr,got.size(),cudaMemcpyDeviceToHost,stream),"frozen stream payload");
                        ck(cudaStreamSynchronize(stream),"frozen stream consumer");
                        const int e=jobs[j].second;
                        verdict.check(!std::memcmp(got.data(),payload.data()+(e%8)*4096,4096),"RAM-source and SSD fallback copy exact complete expert bytes");
                        source.release(first_job+j,stream);
                    }
                    source.drain();const auto stats=source.take_stats();
                    verdict.check(stats.jobs==4 && stats.from_ram==3 && stats.from_ssd==1,"prefill source counters distinguish frozen RAM hits from SSD");
                }
            }
            {
                setenv("DS41_EXPERIMENTAL_CACHE_FREQUENCY","1",1);
                VramExperts hot(pack,"",-1,SIZE_MAX,adapt,0,2,true,-1,true,2*4096);
                unsetenv("DS41_EXPERIMENTAL_CACHE_FREQUENCY");
                auto use=[&](std::vector<int32_t> routes) {
                    ids.up(routes);ck(cudaDeviceSynchronize(),"frequency routes");
                    hot.prepare_decode(0,ids.p,6,selected.p,stream);
                    consume_selection<<<1,32,0,stream>>>(selected.p,hot.experts_dev(),ids.p,consumed.p,8);
                    ck(cudaStreamSynchronize(stream),"frequency consumer");
                    const auto got=consumed.down();
                    verdict.check(std::all_of(got.begin(),got.end(),[](int v){return v==1;}),"frequency admission preserves payloads including bypass staging");
                };
                for(int i=0;i<4;++i) use({0,1,0,1,0,1});
                use({2,3,4,5,6,7});
                const auto uploads=hot.streamed_decode_experts();
                use({0,1,0,1,0,1});
                verdict.check(hot.streamed_decode_experts()==uploads,"one-off scan does not evict frequent experts");
                for(int i=0;i<8;++i) use({2,3,2,3,2,3});
                const auto warmed=hot.streamed_decode_experts();use({2,3,2,3,2,3});
                verdict.check(hot.streamed_decode_experts()==warmed,"new repeated routes eventually displace historical hot entries");
            }
            verdict.check(experimental_ram_cache_bytes(nullptr) == 0 && experimental_ram_cache_bytes("0") == 0 &&
                          experimental_ram_cache_bytes("16384") == (16384ull << 20), "RAM total budget defaults off and has a strict maximum");
            for (const char* value : {"", "-2", "+2", "1", "16386", "2MiB", " 2", "2 ", "9999999999999999999999"}) {
                bool refused = false;
                try { (void)experimental_ram_cache_bytes(value); } catch (const std::invalid_argument&) { refused = true; }
                verdict.check(refused, "malformed, odd and oversized RAM budgets fail closed");
            }
            verdict.check(experimental_ram_cache_stage_bytes(nullptr,8192ull<<20,0)==(4096ull<<20) &&
                          experimental_ram_cache_stage_bytes("8192,0",8192ull<<20,0)==(8192ull<<20) &&
                          experimental_ram_cache_stage_bytes("8192,0",8192ull<<20,1)==0,
                          "RAM split conserves the total budget and can target the stage with actual misses");
            for(const char* value:{"", "8192", "8192,1", "+8192,0", "8192,-1", "8192,0,0", "8192, 0"}) {
                bool refused=false;
                try { (void)experimental_ram_cache_stage_bytes(value,8192ull<<20,0); }
                catch(const std::invalid_argument&) { refused=true; }
                verdict.check(refused,"malformed or over-budget RAM split fails closed");
            }
            for (int capacity : {2, 8}) {
                VramExperts ram(pack, "", -1, SIZE_MAX, adapt, 0, 2, true, -1, true, 2*4096, size_t(capacity)*4096);
                verdict.check(ram.ram_cache_slots() == capacity && ram.ram_cache_bytes() == size_t(capacity)*4096,
                              "pinned RAM payload allocation stays within the exact budget");
                auto run = [&](int layer, const std::vector<int32_t>& routes) {
                    ids.up(routes); ck(cudaDeviceSynchronize(), "RAM routes upload");
                    ram.prepare_decode(layer, ids.p, 6, selected.p, stream);
                    consume_selection<<<1,32,0,stream>>>(selected.p, ram.experts_dev(), ids.p, consumed.p, 8);
                    ck(cudaStreamSynchronize(stream), "RAM immediate consumer");
                    const auto correct = consumed.down();
                    verdict.check(std::all_of(correct.begin(), correct.end(), [](int32_t v) { return v == 1; }),
                                  "RAM hits, GPU hits and both LRUs retain exact stored payloads");
                };
                run(0, {0,1,2,3,4,5});
                const auto fills = ram.ram_cache_fills();
                run(0, {0,1,2,3,4,5});
                verdict.check(capacity != 8 || (ram.ram_cache_hits() == 4 && ram.ram_cache_fills() == fills),
                              "GPU misses reuse pinned bytes without new file reads");
                run(0, {7,7,7,0,0,1});
                for (int replay = 0; replay < 32; ++replay) {
                    std::vector<int32_t> routes(6);
                    for (int i=0; i<6; ++i) routes[i]=(replay*3+i)%192;
                    run(replay%2, routes);
                }
                const auto before_error = ram.ram_cache_fills();
                std::filesystem::resize_file(dir / "experts.bin", 4096);
                ids.up(std::vector<int32_t>(6, 383));
                bool refused = false;
                try { ram.prepare_decode(0, ids.p, 6, selected.p, stream); } catch (const std::runtime_error&) { refused = true; }
                verdict.check(refused && ram.ram_cache_fills() == before_error, "short RAM fill never publishes a valid entry");
                { std::ofstream weights(dir / "experts.bin",std::ios::binary); weights.write(reinterpret_cast<const char*>(payload.data()),payload.size()); }
                run(0, {383,383,383,383,383,383});
                verdict.check(ram.ram_cache_read_bytes() == uint64_t(ram.ram_cache_fills())*4096, "complete RAM fill bytes accounted");
            }
            {
                BoundedUpload upload((dir / "experts.bin").string());
                uint8_t* pinned = nullptr;
                ck(cudaMallocHost(&pinned,4096), "RAM source fixture");
                upload.read_host(pinned,4096,4096);
                upload.set_source(pinned,4096,4096);
                Dev<uint8_t> columns(128);
                upload.copy_columns(columns.p,4096,8,512,64,16);
                const auto got=columns.down();
                bool exact=true;
                for (size_t r=0;r<8;++r) for(size_t c=0;c<16;++c)
                    exact &= got[r*16+c] == payload[4096+r*512+64+c];
                verdict.check(exact, "TP-shaped column extraction uses unchanged pinned source bytes");
                bool refused=false;
                try { upload.copy(columns.p,0,64); } catch(const std::runtime_error&) { refused=true; }
                verdict.check(refused,"RAM source refuses out-of-payload reads instead of silently falling back");
                upload.clear_source();
                ck(cudaFreeHost(pinned),"RAM source release");
            }
            bool rejected = false;
            ids.up(std::vector<int32_t>(6, 384));
            try { tier.prepare_decode(0, ids.p, 6, selected.p, stream); }
            catch (const std::runtime_error&) { rejected = true; }
            verdict.check(rejected, "invalid route rejected before payload access");
            // A real short read propagates instead of reusing a previous payload.
            std::filesystem::resize_file(dir / "experts.bin", 4096);
            ids.up(std::vector<int32_t>(6, 7)); rejected = false;
            try { tier.prepare_decode(0, ids.p, 6, selected.p, stream); }
            catch (const std::runtime_error&) { rejected = true; }
            verdict.check(rejected, "truncated streamed expert fails closed");
            ck(cudaStreamDestroy(stream), "stream test end");
        }
        std::filesystem::remove_all(dir);
        return verdict.finish();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        std::filesystem::remove_all(dir);
        return 1;
    }
}
