// Opt-in, serialized real-model correctness and matched-workload timing. Never a CTest.
#include "strata/ds41/engine.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

using namespace strata::ds41;
namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
struct CpuStat { double usage=0,throttled=0; };
CpuStat cpu_stat() {
    std::ifstream group("/proc/self/cgroup");std::string line,path;
    while(std::getline(group,line)) if(line.compare(0,3,"0::")==0) path="/sys/fs/cgroup"+line.substr(3)+"/cpu.stat";
    std::ifstream input(path);std::string key;double value;CpuStat result;
    while(input>>key>>value) {
        if(key=="usage_usec") result.usage=value;
        if(key=="throttled_usec") result.throttled=value;
    }
    return result;
}
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        std::string pack,ids,reference_out,reference_in;int tensor=0,spec=0,tokens=96,repeat=3,profile=0,blocking=0;
        size_t cache=1ull<<30;
        for(int i=1;i<argc;i+=2) {
            require(i+1<argc,"missing value");const std::string a=argv[i];
            if(a=="--pack") pack=argv[i+1];
            else if(a=="--ids-file") ids=argv[i+1];
            else if(a=="--tensor") tensor=std::stoi(argv[i+1]);
            else if(a=="--spec") spec=std::stoi(argv[i+1]);
            else if(a=="--tokens") tokens=std::stoi(argv[i+1]);
            else if(a=="--repeat") repeat=std::stoi(argv[i+1]);
            else if(a=="--profile") profile=std::stoi(argv[i+1]);
            else if(a=="--blocking-sync") blocking=std::stoi(argv[i+1]);
            else if(a=="--cache-mib") cache=size_t(std::stoi(argv[i+1]))<<20;
            else if(a=="--reference-out") reference_out=argv[i+1];
            else if(a=="--reference-in") reference_in=argv[i+1];
            else throw std::invalid_argument("unknown option "+a);
        }
        require(tokens>0 && tokens<=1024 && repeat>0 && repeat<=10,"benchmark bounds");
        std::ifstream input(ids);require(bool(input),"prompt missing");std::string value;std::vector<int> prompt;
        while(std::getline(input,value,',')) if(!value.empty()) prompt.push_back(std::stoi(value));
        require(!prompt.empty(),"empty prompt");
        EngineOptions o;o.devices={0,1};o.split_layer=tensor?20:18;o.tensor_parallel=tensor;o.max_seq=int(prompt.size())+tokens+16;
        o.ram_budget_gib=0;o.adapt_every=0;o.cpu_threads=2;o.prefill_chunk=512;o.prefill_batch=256;o.snapshots=1;
        o.dspark=spec;o.draft_cache_bytes=cache;o.vram_reserve_bytes=1ull<<30;
        if(blocking) for(int device:o.devices) {
            require(cudaSetDevice(device)==cudaSuccess,"blocking device");
            require(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)==cudaSuccess,"blocking schedule");
        }
        Engine e(pack,o);int first=0;
        // Match prefill arithmetic: the larger PP GEMM path is not generally
        // bit-identical to the small-row GEMV path. Prefill is outside timing.
        for(int token:prompt) first=e.step(token,e.position());
        e.save_snapshot(0);
        std::vector<int> expected;
        if(!reference_in.empty()) {
            std::ifstream f(reference_in);int t;while(f>>t) expected.push_back(t);
            require(expected.size()==size_t(tokens),"reference output length");
        }
        std::vector<double> rates;
        int reference_windows=-1;
        for(int round=0;round<repeat+1;++round) {
            e.restore_snapshot(0);std::vector<int> output;int next=first,windows=0,cache_hits=0,uses=0;
            if(profile && round==1) cudaProfilerStart();
            const auto cpu_start=cpu_stat();
            double draft_ms=0,verify_ms=0;const double start=now();
            while(output.size()<size_t(tokens)) {
                if(!spec) { output.push_back(next);next=e.step(next,e.position());continue; }
                const auto draft=e.draft(next);draft_ms+=draft.ms;cache_hits+=draft.vram_hits;uses+=draft.uses;
                const int rows=std::min<int>(kVerifyMaxTokens,tokens-int(output.size()));
                std::vector<int> window{next};
                for(int i=1;i<rows && i<=draft.n;++i) window.push_back(draft.tokens[i-1]);
                const auto result=e.verify(window,e.position());int keep=1;
                while(keep<int(window.size()) && window[keep]==result.next[keep-1]) ++keep;
                e.commit(keep);verify_ms+=e.last_timing().total_ms;++windows;
                output.insert(output.end(),window.begin(),window.begin()+keep);next=result.next[keep-1];
            }
            const double elapsed=now()-start,rate=tokens/elapsed;
            const auto cpu_end=cpu_stat();
            if(profile && round==1) cudaProfilerStop();
            if(reference_windows<0) reference_windows=windows;
            require(windows==reference_windows,"snapshot replay changed draft acceptance");
            if(expected.empty()) expected=output;
            if(output!=expected) {
                const auto mismatch=std::mismatch(output.begin(),output.end(),expected.begin());
                std::fprintf(stderr,"greedy mismatch at %zu: actual %d expected %d\n",size_t(mismatch.first-output.begin()),*mismatch.first,*mismatch.second);
                throw std::runtime_error("greedy output differs across speculation mode/runs");
            }
            if(round) rates.push_back(rate);
            std::printf("RUN parallelism=%s spec=%s warmup=%d tokens=%d seconds=%.6f tok_s=%.3f windows=%d accepted_per_window=%.3f draft_ms=%.3f verify_commit_ms=%.3f draft_cache_hits=%d/%d cpu_ms=%.3f throttled_ms=%.3f\n",
                tensor?"tensor":"layer",spec?"dspark":"none",round==0,tokens,elapsed,rate,windows,
                windows?double(tokens)/windows:0,draft_ms,verify_ms,cache_hits,uses,
                (cpu_end.usage-cpu_start.usage)/1000,(cpu_end.throttled-cpu_start.throttled)/1000);
        }
        if(!reference_out.empty()) { std::ofstream f(reference_out);for(int token:expected) f<<token<<'\n';require(bool(f),"write output"); }
        std::sort(rates.begin(),rates.end());
        std::printf("RESULT pass parallelism=%s spec=%s median_tok_s=%.3f greedy_tokens=%d\n",tensor?"tensor":"layer",spec?"dspark":"none",rates[rates.size()/2],tokens);
    } catch(const std::exception& ex) { std::fprintf(stderr,"RESULT fail %s\n",ex.what());return 1; }
}
