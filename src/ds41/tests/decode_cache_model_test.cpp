// Explicit 2.9-only native decode timing/exactness. Never an automatic CTest.
#include "strata/ds41/engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
using namespace strata::ds41;
namespace {
void require(bool ok,const char* reason) { if(!ok) throw std::runtime_error(reason); }
double clock_ms() { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void same(const std::vector<float>& a,const std::vector<float>& b,const char* reason) {
    require(a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),reason);
}
void write(const std::string& path,const std::vector<float>& values) {
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
    require(bool(f),"cannot save logits");
}
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        std::string pack,ids,out,reference;int generated=96,repeats=3;
        for(int i=1;i<argc;i+=2) {
            require(i+1<argc,"missing argument");const std::string key=argv[i];
            if(key=="--pack") pack=argv[i+1];else if(key=="--ids-file") ids=argv[i+1];
            else if(key=="--out") out=argv[i+1];else if(key=="--reference-in") reference=argv[i+1];
            else if(key=="--gen") generated=std::stoi(argv[i+1]);else if(key=="--repeat") repeats=std::stoi(argv[i+1]);
            else throw std::invalid_argument("unknown option "+key);
        }
        require(!pack.empty() && !out.empty() && generated>=16 && generated<=256 && repeats>=1 && repeats<=5,"invalid bounded test options");
        std::ifstream input(ids);require(bool(input),"missing frozen input IDs");
        std::string text((std::istreambuf_iterator<char>(input)),{}),field;std::istringstream fields(text);std::vector<int> prompt;
        while(std::getline(fields,field,',')) if(!field.empty()) prompt.push_back(std::stoi(field));
        require(!prompt.empty() && prompt.size()<=4097,"use a bounded frozen prompt");
        std::vector<float> external;
        if(!reference.empty()) {
            std::ifstream f(reference,std::ios::binary|std::ios::ate);require(bool(f),"missing full-logit reference");
            require(f.tellg()==std::streamoff(size_t(generated)*kVocab*sizeof(float)),"reference shape differs");
            external.resize(size_t(generated)*kVocab);f.seekg(0);f.read(reinterpret_cast<char*>(external.data()),external.size()*sizeof(float));
            require(bool(f),"short logit reference");
        }
        EngineOptions o;o.devices={0,1};o.split_layer=18;o.max_seq=524288;o.cpu_threads=2;
        o.ram_budget_gib=0;o.adapt_every=0;o.vram_reserve_bytes=12ull<<30;o.snapshots=3;
        o.prefill_chunk=512;o.prefill_batch=256;
        Engine engine(pack,o);
        std::printf("LOADED resident_slots=%d capacity=524288 spec=none requested_shapes=512/256\n",engine.vram_expert_slots());
        // Warm both parity paths before a graph can capture a library's first
        // use. These two unmeasured steps run identically in every mode.
        engine.step(prompt.front(),0);engine.step(prompt.size()>1?prompt[1]:prompt.front(),1);engine.reset();
        std::vector<float> golden, prefix;std::vector<int> golden_tokens;std::vector<double> seconds;
        int next=0;
        for(int round=0;round<=repeats;++round) {
            engine.reset();next=engine.prefill(prompt,0);
            if(round==0) prefix=engine.last_logits();else same(prefix,engine.last_logits(),"reset prefill differs");
            const auto p=engine.last_prefill();
            std::vector<float> logits;logits.reserve(size_t(generated)*kVocab);std::vector<int> tokens;
            int64_t hits=0,streamed=0,ram_hits=0;uint64_t ram_read_bytes=0;double elapsed=0;int captures=0,replays=0;
            for(int i=0;i<generated;++i) {
                tokens.push_back(next);
                const double start=clock_ms();next=engine.step(next,engine.position());elapsed+=clock_ms()-start;
                const auto& t=engine.last_timing();
                require(t.expert_total==240 && t.expert_hits+t.streamed_experts==240 && !t.cpu_experts() && !t.skipped,"incomplete routed-expert execution");
                hits+=t.expert_hits;streamed+=t.streamed_experts;
                ram_hits+=t.ram_cached_uploads;ram_read_bytes+=t.ram_cache_read_bytes;
                captures+=t.static_graph_captures;replays+=t.static_graph_replays;
                for(float f:engine.last_logits()) require(std::isfinite(f),"nonfinite decode logit");
                logits.insert(logits.end(),engine.last_logits().begin(),engine.last_logits().end());
            }
            if(round==0) { golden=logits;golden_tokens=tokens; }
            else { same(golden,logits,"warm-cache reset replay differs");require(tokens==golden_tokens,"greedy tokens differ");seconds.push_back(elapsed/1000.); }
            if(!external.empty()) same(external,logits,"cache-on logits differ from cache-off at some generated position");
            if(const char* graphs=std::getenv("DS41_EXPERIMENTAL_STATIC_GRAPHS"))
                require(std::strcmp(graphs,"1") || replays>0,"static graph flag produced no actual replays");
            std::printf("ROUND warmup=%d tokens=%d decode_seconds=%.6f tok_s=%.3f hits=%lld streamed=%lld selected_shapes=%d/%d,%d/%d ram_hits=%lld ram_read_bytes=%llu graph_captures=%d graph_replays=%d\n",
                round==0,generated,elapsed/1000.,generated*1000./elapsed,(long long)hits,(long long)streamed,
                p.stage_chunk[0],p.stage_batch[0],p.stage_chunk[1],p.stage_batch[1],(long long)ram_hits,(unsigned long long)ram_read_bytes,captures,replays);
        }
        write(out+"/decode-logits.bin",golden);write(out+"/prefill-logits.bin",prefix);
        std::ofstream tokens(out+"/tokens.txt");for(int token:golden_tokens) tokens<<token<<' ';tokens<<'\n';require(bool(tokens),"cannot save tokens");
        const int pos=engine.position();engine.save_snapshot(0);const int after=engine.step(next,pos);const auto expected=engine.last_logits();
        require(engine.restore_snapshot(0)==pos && engine.step(next,pos)==after,"snapshot next differs");same(expected,engine.last_logits(),"snapshot logits differ");
        const std::vector<int> follow(prompt.begin(),prompt.begin()+std::min<size_t>(17,prompt.size()));
        const int warm_pos=engine.position();engine.save_snapshot(1);const int warm=engine.prefill(follow,warm_pos);const auto warm_logits=engine.last_logits();
        require(engine.restore_snapshot(1)==warm_pos && engine.prefill(follow,warm_pos)==warm,"warm prefill next differs");same(warm_logits,engine.last_logits(),"warm prefill logits differ");
        engine.reset();engine.set_prefill_progress([](int,int){return false;});bool cancelled=false;
        try { engine.prefill(prompt,0); } catch(const PrefillCancelled&) { cancelled=true; }
        require(cancelled && engine.position()==0,"cancel did not reset");engine.set_prefill_progress(nullptr);
        engine.prefill(prompt,0);same(prefix,engine.last_logits(),"cancel recovery differs");
        std::sort(seconds.begin(),seconds.end());const double median=seconds[seconds.size()/2];
        std::printf("RESULT pass median_decode_seconds=%.6f median_tok_s=%.3f warmup=1 repeats=%d generated=%d snapshots=1 warm_prefill=1 cancel=1 exact_all_positions=1\n",
                    median,generated/median,repeats,generated);
        return 0;
    } catch(const std::exception& error) { std::fprintf(stderr,"RESULT fail %s\n",error.what());return 1; }
}
