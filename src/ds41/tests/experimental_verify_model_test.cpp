// Explicit real-weight verifier qualification; never registered with CTest.
#include "strata/ds41/engine.hpp"
#include "strata/ds41/config.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

using namespace strata::ds41;
namespace {
void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
void exact(const std::vector<float>& got,const std::vector<float>& want,int row) {
    if(got.size()==want.size() && !std::memcmp(got.data(),want.data(),want.size()*sizeof(float))) return;
    double error=0,norm=0;size_t changed=0;
    for(size_t i=0;i<std::min(got.size(),want.size());++i) {
        const double d=double(got[i])-want[i];error+=d*d;norm+=double(want[i])*want[i];changed+=got[i]!=want[i];
    }
    std::fprintf(stderr,"VERIFY mismatch row=%d changed=%zu relative_squared_error=%.9g\n",row,changed,error/std::max(norm,1e-30));
    throw std::runtime_error("verify logits differ from ordinary decode");
}
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        std::string pack,ids; int max_seq = 524288;
        for(int i=1;i<argc;i+=2) {
            require(i+1<argc,"missing argument value");const std::string key=argv[i];
            if(key=="--pack") pack=argv[i+1];
            else if(key=="--ids-file") ids=argv[i+1];
            else if(key=="--max-seq") max_seq=std::stoi(argv[i+1]);
            else throw std::invalid_argument("unknown option "+key);
        }
        std::ifstream input(ids);require(bool(input),"missing frozen prompt");
        std::string value;std::vector<int> prompt;
        while(std::getline(input,value,',')) if(!value.empty()) prompt.push_back(std::stoi(value));
        require(prompt.size()>=2 && prompt.size()<512,"bounded prompt required");
        require(max_seq >= int(prompt.size()) + kVerifyMaxResidentTokens + 2,"capacity too small for verifier fixture");
        EngineOptions opt;opt.devices={0,1};opt.split_layer=18;opt.max_seq=max_seq;opt.cpu_threads=2;
        opt.ram_budget_gib=0;opt.adapt_every=0;opt.prefill_chunk=512;opt.prefill_batch=256;opt.snapshots=1;
        opt.vram_reserve_bytes=12ull<<30;
        Engine e(pack,opt);
        const int capacity=e.verify_capacity();
        int checks=0,replays=0;
        for(int parity=0;parity<2;++parity) {
            e.reset();int next=0;
            const int prefix=int(prompt.size())-parity;
            for(int i=0;i<prefix;++i) next=e.step(prompt[i],e.position());
            e.save_snapshot(0);
            for(const auto& invalid:std::vector<std::vector<int>>{{},std::vector<int>(capacity+1,next)}) {
                bool refused=false;try { e.verify(invalid,e.position()); } catch(const std::invalid_argument&) { refused=true; }
                require(refused && e.position()==prefix,"invalid window changed committed state");
            }
            std::vector<int> seeds,following;std::vector<std::vector<float>> golden;
            for(int i=0;i<capacity+2;++i) {
                seeds.push_back(next);next=e.step(next,e.position());following.push_back(next);golden.push_back(e.last_logits());
            }
            for(int rows=1;rows<=capacity;++rows) for(int keep=1;keep<=rows;++keep) {
                e.restore_snapshot(0);
                const std::vector<int> window(seeds.begin(),seeds.begin()+rows);
                auto result=e.verify(window,e.position(),true);
                require(e.position()==prefix && result.logits.size()==size_t(rows),"verify must not commit history");
                for(int row=0;row<rows;++row) {
                    require(result.next[row]==following[row],"verify greedy token differs");exact(result.logits[row],golden[row],row);
                }
                const auto timing=e.last_timing();
                if (const char* bulk=std::getenv("DS41_EXPERIMENTAL_VERIFY_BULK_HANDOFF");
                    bulk && !std::strcmp(bulk,"1"))
                    require(timing.verify_handoff_groups==1,"bulk verify did not use one handoff group");
                if (const char* async=std::getenv("DS41_EXPERIMENTAL_ASYNC_CACHED_ROUTES");
                    async && !std::strcmp(async,"1"))
                    require(timing.async_cached_row_reads>0,"async cached-route admission was not exercised");
                replays+=timing.static_graph_replays;
                require(timing.expert_total==rows*240 && timing.expert_hits+timing.streamed_experts==rows*240 && !timing.cpu_experts() && !timing.skipped,
                        "verification did not execute every selected expert on GPU");
                bool refused=false;try { e.commit(0); } catch(const std::invalid_argument&) { refused=true; }
                require(refused,"invalid commit must fail before changing state");
                e.commit(keep);require(e.position()==prefix+keep,"partial commit position");exact(e.last_logits(),golden[keep-1],keep-1);
                require(e.step(seeds[keep],e.position())==following[keep],"rejected tentative suffix corrupted continuation");
                exact(e.last_logits(),golden[keep],keep);++checks;
                std::printf("VERIFY rows=%d keep=%d prefix=%d exact_logits=1 rollback=1 streamed=%d graph_replays=%d\n",rows,keep,prefix,timing.streamed_experts,timing.static_graph_replays);
            }
            // Non-greedy tentative inputs must also match their own sequential
            // target distributions; only matching inputs may be committed later.
            e.restore_snapshot(0);std::vector<int> wrong(seeds.begin(),seeds.begin()+capacity);
            wrong[1]=(wrong[1]+1)%kVocab;
            std::vector<std::vector<float>> wrong_logits;
            for(int token:wrong) { e.step(token,e.position());wrong_logits.push_back(e.last_logits()); }
            const int sentinel=seeds[capacity];e.step(sentinel,e.position());wrong_logits.push_back(e.last_logits());
            for(int keep=1;keep<=capacity;++keep) {
                e.restore_snapshot(0);auto result=e.verify(wrong,e.position(),true);
                for(size_t row=0;row<wrong.size();++row) exact(result.logits[row],wrong_logits[row],int(row));
                e.commit(keep);exact(e.last_logits(),wrong_logits[keep-1],keep-1);
                e.step(keep<capacity?wrong[keep]:sentinel,e.position());exact(e.last_logits(),wrong_logits[keep],keep);
                ++checks;
                std::printf("VERIFY wrong_drafts=1 rows=%d keep=%d prefix=%d exact_logits=1 rollback=1\n",capacity,keep,prefix);
            }
        }
        require(replays>0,"explicit verifier qualification must exercise prefix graph replay");
        std::printf("RESULT pass experimental_verify exact_cases=%d parities=2 partial_commit=1 wrong_drafts=1 graph_replays=%d verify_capacity=%d max_seq=%d\n",checks,replays,capacity,max_seq);
    } catch(const std::exception& error) { std::fprintf(stderr,"RESULT fail %s\n",error.what());return 1; }
}
