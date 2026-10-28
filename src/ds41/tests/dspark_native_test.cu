// Native FP16 MTP parameters: no BF16 weight conversion is permitted.
#include "bench_util.hpp"
#include "strata/ds41/dspark_ops.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"
#include <cstring>
#include <stdexcept>

using namespace ds41test;
namespace dk=strata::ds41::dspark;
using dk::bf16;
int main() {
    require_gpu();Verdict v;
    const half value=__float2half(1.0009765625f); // representable in FP16, not BF16
    Dev<bf16> embed(std::vector<bf16>(2*dk::kMarkovRank,__float2bfloat16(1.f))),embedding_out(dk::kMarkovRank);
    Dev<half> head(std::vector<half>(8*dk::kMarkovRank,value));
    Dev<int> token(std::vector<int>{1});Dev<float> logits(std::vector<float>(8,0.f));
    dk::markov_bias(token.p,embed.p,head.p,8,logits.p,embedding_out.p);
    for(float got:logits.down()) v.check(got==256.25f,"FP16 Markov head retains bits lost in BF16 conversion");
    for(bf16 got:embedding_out.down()) v.check(__bfloat162float(got)==1.f,"Markov embedding output");
    Dev<bf16> x(std::vector<bf16>(2*strata::ds41::kDim,__float2bfloat16(1.f)));
    Dev<half> weight(std::vector<half>(strata::ds41::kDim+dk::kMarkovRank,value));Dev<float> confidence(2);
    dk::confidence_rows(x.p,embed.p,weight.p,2,confidence.p);
    for(float got:confidence.down()) v.check(got==5381.25f,"FP16 confidence head retains native stored values");
    std::vector<float> bias(128,0);bias[5]=10;bias[4]=8;bias[3]=6;
    Dev<float> biases(bias),scores(std::vector<float>(2*128,0)),weights(6),reference_weights(6);
    Dev<int32_t> ids(6),reference_ids(6);
    Dev<bf16> zero_x(std::vector<bf16>(2*strata::ds41::kDim,__float2bfloat16(0.f)));
    Dev<bf16> zero_w(std::vector<bf16>(128*strata::ds41::kDim,__float2bfloat16(0.f)));
    dk::router_small_scores(scores.p,2,biases.p,128,3,ids.p,weights.p);
    dk::router_small(zero_x.p,2,zero_w.p,biases.p,128,3,reference_ids.p,reference_weights.p);
    v.check(ids.down()==std::vector<int32_t>({5,4,3,5,4,3}) && ids.down()==reference_ids.down(),"native router score selection and tie order");
    v.check(weights.down()==reference_weights.down(),"native score route normalization matches existing router");
    bool refused=false;
    try { dk::router_small_scores(scores.p,2,biases.p,128,129,ids.p,weights.p); }
    catch(const std::invalid_argument&) { refused=true; }
    v.check(refused,"native router rejects unsupported top-k before enqueue");
    {
        namespace ops = strata::ds41::ops;
        constexpr int B=dk::kBlock, D=strata::ds41::kDim, H=strata::ds41::kHc, vocab=8;
        std::vector<bf16> values(size_t(vocab)*D);
        for(size_t i=0;i<values.size();++i) values[i]=__float2bfloat16_rn(float(int(i%97)-48)/64);
        Dev<bf16> table(values), batched(size_t(B)*H*D), sequential(size_t(B)*H*D);
        Dev<int> tokens(B);
        cudaStream_t stream;
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"embedding test stream");
        ck(cudaStreamSynchronize(nullptr),"embedding table upload");
        cudaGraph_t graph; cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed),"embedding capture");
        ops::embed_device_rows(table.p,tokens.p,batched.p,B,stream);
        ck(cudaStreamEndCapture(stream,&graph),"embedding capture end");
        ck(cudaGraphInstantiate(&exec,graph,0),"embedding instantiate");
        for(int replay=0;replay<10;++replay) {
            std::vector<int> input{replay%vocab,(replay+3)%vocab,(replay+3)%vocab,
                                   (replay+3)%vocab,(replay+3)%vocab};
            tokens.up(input);
            ck(cudaStreamSynchronize(nullptr),"embedding tokens upload");
            ck(cudaGraphLaunch(exec,stream),"embedding replay");
            for(int row=0;row<B;++row)
                ops::embed_device(table.p,tokens.p+row,sequential.p+size_t(row)*H*D,stream);
            ck(cudaStreamSynchronize(stream),"embedding compare");
            const auto got=batched.down(),want=sequential.down();
            v.check(!std::memcmp(got.data(),want.data(),got.size()*sizeof(bf16)),
                    "batched draft embeddings equal five separate calls and graph replay");
        }
        bool bad=false;
        try { ops::embed_device_rows(table.p,tokens.p,batched.p,9,stream); }
        catch(const std::invalid_argument&) { bad=true; }
        v.check(bad,"batched draft embedding rejects too many rows");
        cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);cudaStreamDestroy(stream);
    }
    return v.finish();
}
