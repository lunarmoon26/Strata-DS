// K15 exact-order batching, independent caller scratch and graph replay.
#include "bench_util.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k15_hc_prefill.hpp"
#include <stdexcept>
using namespace ds41test;
using namespace strata::ds41;
namespace kk = strata::ds41::kernels;

struct Outputs {
    Dev<__nv_bfloat16> y;
    Dev<float> pre, post, comb;
    explicit Outputs(int m) : y(size_t(m)*kDim), pre(m*kHc), post(m*kHc), comb(m*kHc*kHc) {}
};
template<class T> bool exact(const Dev<T>& a, const Dev<T>& b) {
    const auto x = a.down(), y = b.down();
    return x.size() == y.size() && !std::memcmp(x.data(), y.data(), x.size()*sizeof(T));
}
bool same(const Outputs& a, const Outputs& b) {
    return exact(a.y,b.y) && exact(a.pre,b.pre) && exact(a.post,b.post) && exact(a.comb,b.comb);
}

int main() {
    require_gpu(); Verdict verdict;
    try {
        std::vector<float> fn(kHcMix*kHc*kDim), scales{.7f,.9f,.6f}, base(kHcMix);
        for(size_t i=0;i<fn.size();++i) fn[i]=float(int((i*17+i/97)%101)-50)/4096;
        for(int i=0;i<kHcMix;++i) base[i]=float(i%7-3)/11;
        Dev<float> dfn(fn), ds(scales), db(base);
        cudaStream_t stream, other;
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"HC stream");
        ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking),"HC other stream");
        for(int m:{1,3,7,8,9,17,63,64,65,255,256,257,513,1024,4097,16384}) {
            std::vector<__nv_bfloat16> x(size_t(m)*kHc*kDim);
            std::vector<float> pre(m*kHc);
            for(size_t i=0;i<x.size();++i) x[i]=__float2bfloat16_rn(float(int((i*13+i/53)%127)-63)/31);
            for(size_t i=0;i<pre.size();++i) pre[i]=float(i%9+1)/37;
            Dev<__nv_bfloat16> dx(x); Dev<float> dp(pre);
            Outputs reference(m), actual(m);
            kk::hc_mixes_pre_rows(dx.p,m,dfn.p,ds.p,db.p,dp.p,reference.y.p,reference.pre.p,reference.post.p,reference.comb.p,nullptr,0,stream);
            ck(cudaStreamSynchronize(stream),"K7 baseline");
            const size_t bytes=kk::hc_mixes_pre_rows_batched_workspace_bytes(m);
            Dev<uint8_t> guarded(bytes+512);
            ck(cudaMemset(guarded.p,0xa5,guarded.n),"HC scratch guards");
            void* scratch=guarded.p+256;
            auto call=[&](Outputs& out,void* ws,cudaStream_t s) {
                kk::hc_mixes_pre_rows_batched(dx.p,m,dfn.p,ds.p,db.p,dp.p,out.y.p,out.pre.p,out.post.p,out.comb.p,ws,bytes,s);
            };
            call(actual,scratch,stream); ck(cudaStreamSynchronize(stream),"batched HC");
            verdict.check(same(reference,actual),"all HC outputs bit-identical to original K7 including partial tiles");
            cudaGraph_t graph; cudaGraphExec_t replay;
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed),"HC capture");
            call(actual,scratch,stream);
            ck(cudaStreamEndCapture(stream,&graph),"HC capture end");
            ck(cudaGraphInstantiate(&replay,graph,0),"HC graph instantiate");
            for(int i=0;i<3;++i) {
                ck(cudaGraphLaunch(replay,stream),"HC graph replay");
                ck(cudaStreamSynchronize(stream),"HC replay fence");
                verdict.check(same(reference,actual),"batched HC exact CUDA-graph replay");
            }
            ck(cudaGraphExecDestroy(replay),"HC graph destroy"); ck(cudaGraphDestroy(graph),"HC graph free");
            const auto guard=guarded.down(); bool intact=true;
            for(size_t i=0;i<256;++i) intact &= guard[i]==0xa5 && guard[bytes+256+i]==0xa5;
            verdict.check(intact,"caller-owned scratch guards unchanged");
            if(m==257) {
                Outputs concurrent(m); Dev<uint8_t> second(bytes);
                call(actual,scratch,stream); call(concurrent,second.p,other);
                ck(cudaStreamSynchronize(stream),"HC concurrent first"); ck(cudaStreamSynchronize(other),"HC concurrent second");
                verdict.check(same(reference,actual) && same(reference,concurrent),"concurrent streams use independent caller scratch");
                int rejected=0;
                for(int mode:{0,1,2,3}) try {
                    kk::hc_mixes_pre_rows_batched(dx.p,mode==0?0:mode==1?16385:m,dfn.p,ds.p,db.p,dp.p,
                        actual.y.p,actual.pre.p,actual.post.p,actual.comb.p,mode==2?guarded.p+1:scratch,mode==3?bytes-1:bytes,stream);
                } catch(const std::invalid_argument&) { ++rejected; }
                verdict.check(rejected==4,"invalid row count, alignment and insufficient scratch fail closed");
            }
            std::printf("HC_SHAPE rows=%d baseline_launches=%d batched_launches=2 workspace_bytes=%zu\n",m,2*((m+7)/8),bytes);
        }
        ck(cudaStreamDestroy(stream),"HC stream destroy"); ck(cudaStreamDestroy(other),"HC other destroy");
        return verdict.finish();
    } catch(const std::exception& e) { std::fprintf(stderr,"RESULT fail %s\n",e.what()); return 1; }
}
