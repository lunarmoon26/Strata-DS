#include "bench_util.hpp"
#include "strata/ds41/kernels/k3_sparse_attn.hpp"

using namespace ds41test;
namespace kk=strata::ds41::kernels;
int main() {
    require_gpu();Verdict verdict;
    for(int rows:{1,4,8}) for(int compressed:{0,300}) {
        constexpr int H=64,D=512,W=128;
        std::mt19937 rng(rows+compressed);
        std::vector<__nv_bfloat16> q(size_t(rows)*H*D),window(W*D),cache(std::max(1,compressed)*D);
        auto fill=[&](auto& values) { for(auto& value:values) value=__float2bfloat16_rn(float(int(rng()%101)-50)/128); };
        fill(q);fill(window);fill(cache);
        const int count=W+compressed;
        std::vector<int32_t> index(rows*count);
        for(int r=0;r<rows;++r) for(int j=0;j<count;++j) index[r*count+j]=j%17==0?-1:j;
        std::vector<float> sink(H);for(int h=0;h<H;++h) sink[h]=float(h%13-6)/8;
        Dev<__nv_bfloat16> x(q),w(window),c(cache),reference(q.size()),result(q.size());
        Dev<int32_t> ids(index),length(std::vector<int32_t>{compressed});Dev<float> s(sink);
        kk::sparse_attn_decode(x.p,w.p,c.p,ids.p,rows,count,s.p,.044194173824f,reference.p,0);
        ck(cudaDeviceSynchronize(),"full attention reference");const auto expected=reference.down();
        for(int rank=0;rank<2;++rank) {
            auto local=q;const auto poison=__float2bfloat16_rn(NAN);
            for(int r=0;r<rows;++r)
                std::fill_n(local.begin()+(size_t(r)*H+(1-rank)*H/2)*D,H/2*D,poison);
            x.up(local);ck(cudaMemset(result.p,0x7f,result.n*2),"attention sentinel");
            kk::sparse_attn_decode(x.p,w.p,c.p,ids.p,rows,count,s.p,.044194173824f,result.p,0,rank*H/2,H/2);
            ck(cudaDeviceSynchronize(),"owned attention heads");auto actual=result.down();bool exact=true,untouched=true;
            for(int r=0;r<rows;++r) {
                const size_t own=(size_t(r)*H+rank*H/2)*D,other=(size_t(r)*H+(1-rank)*H/2)*D;
                exact &= !std::memcmp(expected.data()+own,actual.data()+own,H/2*D*2);
                for(int i=0;i<H/2*D;++i) { uint16_t value;std::memcpy(&value,&actual[other+i],2);untouched &= value==0x7f7f; }
            }
            verdict.check(exact && untouched,"owned heads equal full K3 exactly without reading/writing peer heads");
            if(rows==1) {
                kk::sparse_attn_decode_device(x.p,w.p,c.p,ids.p,length.p,s.p,.044194173824f,result.p,0,rank*H/2,H/2);
                ck(cudaDeviceSynchronize(),"owned dynamic attention");actual=result.down();
                verdict.check(!std::memcmp(expected.data()+rank*H/2*D,actual.data()+rank*H/2*D,H/2*D*2),"dynamic owned-head K3 parity");
            }
        }
    }
    return verdict.finish();
}
