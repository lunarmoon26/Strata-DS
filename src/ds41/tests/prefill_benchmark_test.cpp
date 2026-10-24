// Explicit, serialized model benchmark. Not a CTest; never downloads assets.
#include "strata/ds41/engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#ifdef __linux__
#include <sys/resource.h>
#endif

using namespace strata::ds41;
namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        std::string pack,ids,reference_in,reference_out;
        int capacity=524288,chunk=1024,batch=256,repeat=2,spec=1;
        for(int i=1;i<argc;i+=2) {
            require(i+1<argc,"missing option value");std::string key=argv[i];
            if(key=="--pack") pack=argv[i+1];
            else if(key=="--ids-file") ids=argv[i+1];
            else if(key=="--max-seq") capacity=std::stoi(argv[i+1]);
            else if(key=="--chunk") chunk=std::stoi(argv[i+1]);
            else if(key=="--batch") batch=std::stoi(argv[i+1]);
            else if(key=="--repeat") repeat=std::stoi(argv[i+1]);
            else if(key=="--spec") spec=std::stoi(argv[i+1]);
            else if(key=="--reference-in") reference_in=argv[i+1];
            else if(key=="--reference-out") reference_out=argv[i+1];
            else throw std::invalid_argument("unknown option "+key);
        }
        require(repeat>=1 && repeat<=5 && chunk>=16 && chunk<=1024 && batch>=1 && batch<=512,"benchmark bounds");
        std::ifstream input(ids);require(bool(input),"missing prompt");std::string value;std::vector<int> prompt;
        while(std::getline(input,value,',')) if(!value.empty()) prompt.push_back(std::stoi(value));
        require(!prompt.empty() && prompt.size()+32<=size_t(capacity),"prompt/capacity bounds");
        EngineOptions o;o.devices={0,1};o.split_layer=18;o.max_seq=capacity;o.cpu_threads=2;
        o.ram_budget_gib=0;o.adapt_every=0;o.prefill_chunk=chunk;o.prefill_batch=batch;
        o.vram_reserve_bytes=1ull<<30;o.snapshots=1;o.dspark=spec;
        Engine e(pack,o);std::vector<double> times;std::vector<float> first;
        require(e.vram_expert_slots()==kLayers*kExperts,"not all backbone experts resident");
        int next=-1;
        for(int round=0;round<=repeat;++round) {
            e.reset();require(e.position()==0,"reset position");
            const double start=now();next=e.prefill(prompt,0);const double seconds=now()-start;
            require(e.position()==int(prompt.size()),"prefill position");
            const auto& logits=e.last_logits();require(logits.size()==kVocab,"logit count");
            for(float f:logits) require(std::isfinite(f),"nonfinite logits");
            if(first.empty()) first=logits;
            else require(!std::memcmp(first.data(),logits.data(),kVocab*sizeof(float)),"prefill replay differs");
            if(round) times.push_back(seconds);
            const auto& t=e.last_prefill();
            std::printf("PREFILL warmup=%d tokens=%zu seconds=%.6f tok_s=%.3f chunk=%d batch=%d engram_wait_ms=%.3f unique_engram_rows=%lld resident_groups=%lld streamed=%lld\n",
                round==0,prompt.size(),seconds,prompt.size()/seconds,t.chunk_tokens,t.sub_batch,t.engram_ms,
                (long long)t.engram_unique,(long long)t.vram_experts,(long long)t.streamed);
        }
        if(!reference_out.empty()) {
            std::ofstream f(reference_out,std::ios::binary);f.write(reinterpret_cast<const char*>(first.data()),kVocab*sizeof(float));require(bool(f),"reference output");
        }
        if(!reference_in.empty()) {
            std::vector<float> expected(kVocab);std::ifstream f(reference_in,std::ios::binary);
            f.read(reinterpret_cast<char*>(expected.data()),kVocab*sizeof(float));require(bool(f),"reference input");
            double a=*std::max_element(expected.begin(),expected.end()),b=*std::max_element(first.begin(),first.end()),az=0,bz=0,err=0,norm=0;
            for(int i=0;i<kVocab;++i) { az+=std::exp(expected[i]-a);bz+=std::exp(first[i]-b);err+=std::pow(double(first[i])-expected[i],2);norm+=std::pow(double(expected[i]),2); }
            az=a+std::log(az);bz=b+std::log(bz);double kl=0;
            for(int i=0;i<kVocab;++i) kl+=std::exp(expected[i]-az)*((expected[i]-az)-(first[i]-bz));
            const double rel=std::sqrt(err/std::max(norm,1e-30));
            const int expected_next=int(std::max_element(expected.begin(),expected.end())-expected.begin());
            std::printf("REFERENCE argmax=%d/%d rel_l2=%.9g kl=%.9g byte_identical=%d\n",next,expected_next,rel,kl,!std::memcmp(first.data(),expected.data(),kVocab*sizeof(float)));
            require(next==expected_next && std::isfinite(kl) && kl<.02 && rel<.25,"predictive reference differs");
        }
        e.save_snapshot(0);const int position=e.position(),after=e.step(next,position);const auto continuation=e.last_logits();
        e.restore_snapshot(0);require(e.step(next,position)==after && e.last_logits()==continuation,"snapshot continuation differs");
#ifdef __linux__
        struct rusage usage{};require(getrusage(RUSAGE_SELF,&usage)==0,"getrusage");
        std::printf("MEMORY process_peak_rss_kib=%ld reused_prompt_tokens=0\n",usage.ru_maxrss);
#endif
        std::sort(times.begin(),times.end());
        std::printf("RESULT pass prefill tokens=%zu median_seconds=%.6f median_tok_s=%.3f capacity=%d spec=%d\n",prompt.size(),times[times.size()/2],prompt.size()/times[times.size()/2],capacity,spec);
    } catch(const std::exception& e) { std::fprintf(stderr,"RESULT fail %s\n",e.what());return 1; }
}
