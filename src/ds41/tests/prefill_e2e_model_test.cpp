// Explicit serialized real-weight correctness/attribution matrix, not a CTest.
#include "strata/ds41/engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace strata::ds41;
namespace {
void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
void flag(const char* name,bool value) {
#ifdef _WIN32
    _putenv_s(name,value?"1":"0");
#else
    setenv(name,value?"1":"0",1);
#endif
}
void equal(const std::vector<float>& a,const std::vector<float>& b,const char* why) {
    require(a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),why);
}
void finite(const Engine& e) {
    require(e.last_logits().size()==kVocab,"logit shape");
    for(float f:e.last_logits()) require(std::isfinite(f),"nonfinite logit");
}
void save(const std::string& path,const std::vector<float>& values) {
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
    require(bool(f),"cannot save full logits");
}
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        std::string pack,ids,out;int experimental=0,host=0;
        for(int i=1;i<argc;i+=2) {
            require(i+1<argc,"missing value");const std::string key=argv[i];
            if(key=="--pack") pack=argv[i+1];else if(key=="--ids-file") ids=argv[i+1];
            else if(key=="--out") out=argv[i+1];else if(key=="--experimental-29") experimental=std::stoi(argv[i+1]);
            else if(key=="--force-host") host=std::stoi(argv[i+1]);else throw std::invalid_argument("unknown option "+key);
        }
        require(!out.empty() && (experimental==0 || experimental==1) && (host==0 || host==1),"invalid test mode");
        std::ifstream input(ids);require(bool(input),"missing frozen input");
        std::vector<int> prompt;std::string value;
        while(std::getline(input,value,',')) if(!value.empty()) prompt.push_back(std::stoi(value));
        require(prompt.size()>=513 && prompt.size()<=16384,"use 513..16384 frozen input tokens");
        EngineOptions o;o.devices={0,1};o.split_layer=18;o.max_seq=524288;o.cpu_threads=2;
        o.ram_budget_gib=0;o.adapt_every=0;o.prefill_chunk=512;o.prefill_batch=256;o.snapshots=3;
        o.force_host_handoff=host;o.dspark=!experimental;o.vram_reserve_bytes=experimental?12ull<<30:1ull<<30;
        Engine e(pack,o);
        std::vector<float> reference,warm_reference;PrefillTiming baseline;
        for(int mode=0;mode<4;++mode) {
            const bool bulk=mode&1,hc=mode&2;
            flag("DS41_PF_BULK_HANDOFF",bulk);flag("DS41_PF_HC_BATCHED",hc);
            e.reset();const int next=e.prefill(prompt,0);finite(e);const auto t=e.last_prefill();
            require(t.attributed && t.bulk_handoff==bulk && t.batched_hc==hc,"candidate/profile dispatch not exercised");
            require(e.position()==int(prompt.size()) && t.hc_calls>0,"prefill coverage");
            if(mode==0) { reference=e.last_logits();baseline=t; }
            else {
                equal(reference,e.last_logits(),"candidate full logits differ from default");
                require(t.stage_chunk==baseline.stage_chunk && t.stage_batch==baseline.stage_batch,"selected shapes changed");
                require(t.handoff_bytes==baseline.handoff_bytes && t.hc_calls==baseline.hc_calls,"candidate work coverage changed");
            }
            if(hc) require(t.hc_launches==2*t.hc_calls && t.hc_launches<baseline.hc_launches,"HC dispatch was not batched");
            if(bulk && !host) require(t.handoff_copies<=4*uint64_t(t.chunks) && t.handoff_host_waits==uint64_t(t.chunks),"peer bulk copy/wait counts");
            if(!experimental) require(t.streamed==0 && e.vram_expert_slots()==kLayers*kExperts,"resident baseline was offloaded");
            save(out+"/mode-"+std::to_string(mode)+".f32",e.last_logits());
            e.save_snapshot(0);const int pos=e.position(),after=e.step(next,pos);finite(e);const auto continuation=e.last_logits();
            require(e.restore_snapshot(0)==pos && e.step(next,pos)==after,"snapshot token differs");
            equal(continuation,e.last_logits(),"snapshot full logits differ");
            if(!experimental) {
                const int token=int(std::max_element(continuation.begin(),continuation.end())-continuation.begin());
                e.save_snapshot(1);e.step(token,e.position());const auto expected=e.last_logits();e.restore_snapshot(1);
                const auto draft=e.draft(token);require(draft.n>=2,"no actual DSpark draft");
                const auto checked=e.verify({token,draft.tokens[0],draft.tokens[1]},e.position(),true);
                equal(expected,checked.logits[0],"warm verify differs");e.commit(1);equal(expected,e.last_logits(),"warm commit differs");
            }
            // Deliberately start the follow-up at an odd position so a compressed
            // row straddles the stage-transfer boundary.
            if(e.position()%2==0) e.step(prompt.front(),e.position());
            const int warm_pos=e.position();require(warm_pos%2==1,"odd follow-up not exercised");
            e.save_snapshot(2);std::vector<int> follow;
            for(int i=0;i<751;++i) follow.push_back(prompt[(153+i)%prompt.size()]);
            const int warm=e.prefill(follow,warm_pos);finite(e);const auto warm_logits=e.last_logits();
            if(mode==0) warm_reference=warm_logits;else equal(warm_reference,warm_logits,"warm candidate differs from baseline");
            require(e.restore_snapshot(2)==warm_pos && e.prefill(follow,warm_pos)==warm,"warm snapshot token differs");
            equal(warm_logits,e.last_logits(),"warm full-logit replay differs");
            e.reset();e.set_prefill_progress([](int,int){return false;});bool cancelled=false;
            try { e.prefill(prompt,0); } catch(const PrefillCancelled&) { cancelled=true; }
            require(cancelled && e.position()==0,"cancel did not reset");e.set_prefill_progress(nullptr);
            require(e.prefill(prompt,0)==next,"cancel recovery token differs");equal(reference,e.last_logits(),"cancel recovery full logits differ");
            std::printf("E2E mode=%d bulk=%d batched_hc=%d forced_host=%d tokens=%zu shapes=%d/%d,%d/%d "
                "bytes=%llu copies=%llu host_waits=%llu event_waits=%llu hc_calls=%llu hc_launches=%llu "
                "wall_ms=%.3f setup_ms=%.3f stage0_ms=%.3f handoff_ms=%.3f stage1_ms=%.3f cleanup_ms=%.3f streamed=%lld "
                "exact_default=1 snapshot=1 odd_followup=1 warm=1 cancel=1\n",
                mode,bulk,hc,host,prompt.size(),t.stage_chunk[0],t.stage_batch[0],t.stage_chunk[1],t.stage_batch[1],
                (unsigned long long)t.handoff_bytes,(unsigned long long)t.handoff_copies,(unsigned long long)t.handoff_host_waits,
                (unsigned long long)t.handoff_event_waits,(unsigned long long)t.hc_calls,(unsigned long long)t.hc_launches,
                t.total_ms,t.setup_ms,t.stage_ms[0],t.handoff_ms,t.stage_ms[1],t.cleanup_ms,(long long)t.streamed);
        }
        std::puts("RESULT pass four-way exact prefill dispatch E2E at 524288 capacity; no Maya benchmark performed");return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"RESULT fail %s\n",e.what());return 1; }
}
