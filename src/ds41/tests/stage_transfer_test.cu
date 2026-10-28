#include "strata/ds41/stage_transfer.hpp"
#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::ds41;
__global__ void fill(uint32_t* p, int n, const uint32_t* seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (*seed * 2654435761u) ^ (uint32_t(i) * 2246822519u);
}
__global__ void verify(const uint32_t* p, int n, const uint32_t* seed, int* errors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && p[i] != ((*seed * 2654435761u) ^ (uint32_t(i) * 2246822519u))) atomicAdd(errors, 1);
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { std::puts("SKIP: requires two CUDA devices"); return 77; }
    try {
        constexpr size_t capacity = 1 << 20;
        int cases = 0, replays = 0, hc_checks = 0, bulk_checks = 0;
        for (int source : {0, 1}) {
            const int destination = 1 - source;
            uint32_t *src, *dst, *sseed, *dseed;
            int* errors;
            cudaStream_t producer, consumer;
            device_check(cudaSetDevice(source), "test source");
            device_check(cudaMalloc(&src, 4*capacity), "test source allocation");
            device_check(cudaMalloc(&sseed, 4), "test seed");
            device_check(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), "producer stream");
            device_check(cudaSetDevice(destination), "test destination");
            device_check(cudaMalloc(&dst, 4*capacity), "test destination allocation");
            device_check(cudaMalloc(&dseed, 4), "test seed");
            device_check(cudaMalloc(&errors, 4), "test errors");
            device_check(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), "consumer stream");
            for (bool forced : {false, true}) {
                StageTransfer transfer(source, destination, capacity, forced);
                if (forced && transfer.peer()) throw std::runtime_error("forced fallback was ignored");
                for (bool grouped : {false,true}) for (bool graph : {false, true}) for (size_t bytes : {size_t(16), size_t(40960), size_t(327680), capacity}) {
                    cudaGraphExec_t pg = nullptr, cg = nullptr;
                    if (graph) {
                        cudaGraph_t g;
                        DeviceScope scope(source);
                        device_check(cudaStreamBeginCapture(producer, cudaStreamCaptureModeRelaxed), "capture producer");
                        fill<<<(bytes / 4 + 255) / 256, 256, 0, producer>>>(src, bytes / 4, sseed);
                        device_check(cudaStreamEndCapture(producer, &g), "end producer capture");
                        device_check(cudaGraphInstantiate(&pg, g, 0), "producer graph"); cudaGraphDestroy(g);
                    }
                    if (graph) {
                        cudaGraph_t g;
                        DeviceScope scope(destination);
                        device_check(cudaStreamBeginCapture(consumer, cudaStreamCaptureModeRelaxed), "capture consumer");
                        verify<<<(bytes / 4 + 255) / 256, 256, 0, consumer>>>(dst, bytes / 4, dseed, errors);
                        device_check(cudaStreamEndCapture(consumer, &g), "end consumer capture");
                        device_check(cudaGraphInstantiate(&cg, g, 0), "consumer graph"); cudaGraphDestroy(g);
                    }
                    for (uint32_t replay = 1; replay <= 10; ++replay) {
                        const uint32_t seed = replay + source * 101;
                        {
                            DeviceScope scope(source);
                            device_check(cudaMemcpyAsync(sseed, &seed, 4, cudaMemcpyHostToDevice, producer), "producer seed");
                            if (graph) device_check(cudaGraphLaunch(pg, producer), "producer launch");
                            else fill<<<(bytes / 4 + 255) / 256, 256, 0, producer>>>(src, bytes / 4, sseed);
                        }
                        {
                            DeviceScope scope(destination);
                            device_check(cudaMemcpyAsync(dseed, &seed, 4, cudaMemcpyHostToDevice, consumer), "consumer seed");
                            device_check(cudaMemsetAsync(errors, 0, 4, consumer), "clear errors");
                        }
                        if (grouped) transfer.copy_group({{dst,src,bytes}},producer,consumer);
                        else transfer.copy(dst, src, bytes, producer, consumer);
                        DeviceScope scope(destination);
                        if (graph) device_check(cudaGraphLaunch(cg, consumer), "consumer launch");
                        else verify<<<(bytes / 4 + 255) / 256, 256, 0, consumer>>>(dst, bytes / 4, dseed, errors);
                        int bad = -1;
                        device_check(cudaMemcpyAsync(&bad, errors, 4, cudaMemcpyDeviceToHost, consumer), "test result");
                        device_check(cudaStreamSynchronize(consumer), "test result wait");
                        if (bad) throw std::runtime_error("stage transfer corrupted bytes");
                        ++replays;
                    }
                    if (graph) {
                        { DeviceScope scope(source); device_check(cudaGraphExecDestroy(pg), "destroy producer"); }
                        { DeviceScope scope(destination); device_check(cudaGraphExecDestroy(cg), "destroy consumer"); }
                    }
                    ++cases;
                    std::printf("CASE source=%d destination=%d forced_host=%d peer=%d graph=%d bulk=%d bytes=%zu checked=10\n",
                                source, destination, forced, transfer.peer(), graph, grouped, bytes);
                }
                for (size_t bytes : {capacity+16,3*capacity+32}) {
                    transfer.reset_stats();
                    for (uint32_t seed=201;seed<=210;++seed) {
                        { DeviceScope scope(source);
                          device_check(cudaMemcpyAsync(sseed,&seed,4,cudaMemcpyHostToDevice,producer),"bulk producer seed");
                          fill<<<(bytes/4+255)/256,256,0,producer>>>(src,bytes/4,sseed); }
                        { DeviceScope scope(destination);
                          device_check(cudaMemcpyAsync(dseed,&seed,4,cudaMemcpyHostToDevice,consumer),"bulk consumer seed");
                          device_check(cudaMemsetAsync(errors,0,4,consumer),"bulk errors clear");
                          device_check(cudaMemsetAsync(dst,0,bytes,consumer),"bulk prior consumer work"); }
                        auto* sb=reinterpret_cast<uint8_t*>(src); auto* db=reinterpret_cast<uint8_t*>(dst);
                        const size_t first=bytes-1040;
                        const StageTransfer::Buffer group[]={{db,sb,first},{db+first,sb+first,16},{db+first+16,sb+first+16,1024}};
                        transfer.copy_group(group,3,producer,consumer);
                        DeviceScope scope(destination);
                        verify<<<(bytes/4+255)/256,256,0,consumer>>>(dst,bytes/4,dseed,errors);
                        int bad=-1;
                        device_check(cudaMemcpyAsync(&bad,errors,4,cudaMemcpyDeviceToHost,consumer),"bulk result");
                        device_check(cudaStreamSynchronize(consumer),"bulk result fence");
                        if(bad) throw std::runtime_error("large grouped stage transfer corrupted bytes");
                    }
                    const auto before=transfer.stats();
                    if(before.bytes!=bytes*10 || before.groups!=10 ||
                       (transfer.peer() && (before.copy_calls!=30 || before.host_waits!=10 || before.event_waits!=10)))
                        throw std::runtime_error("bulk attribution counts differ");
                    bool rejected=false;
                    try { transfer.copy_group({{dst,src,16},{nullptr,src,16}},producer,consumer); }
                    catch(const std::invalid_argument&) { rejected=true; }
                    if(!rejected || transfer.stats().copy_calls!=before.copy_calls)
                        throw std::runtime_error("malformed bulk group did not fail before side effects");
                    rejected=false;
                    try { transfer.copy_group(nullptr,1,producer,consumer); }
                    catch(const std::invalid_argument&) { rejected=true; }
                    if(!rejected || transfer.stats().copy_calls!=before.copy_calls)
                        throw std::runtime_error("null pointer group did not fail before side effects");
                    ++bulk_checks;
                }
                // Exercise the production handoff, not only the raw copy. The
                // consumer's first HC collapse must receive the four FP32 pre
                // coefficients as well as the BF16 residual (a one-hot restart
                // gives a different answer). Deliberately poison the destination.
                using B = ops::bf16;
                std::vector<B> hidden(kHc * kDim), expected(kDim), received(kDim);
                for (size_t i = 0; i < hidden.size(); ++i)
                    hidden[i] = __float2bfloat16_rn(float(int(i % 79) - 39) / 32);
                const float pre[kHc] = {0.125f, 0.25f, 0.5f, 0.125f};
                auto* src_pre = reinterpret_cast<float*>(reinterpret_cast<char*>(src) + 40960);
                auto* dst_pre = reinterpret_cast<float*>(reinterpret_cast<char*>(dst) + 40960);
                auto* src_out = reinterpret_cast<B*>(src_pre + kHc);
                auto* dst_out = reinterpret_cast<B*>(dst_pre + kHc);
                {
                    DeviceScope scope(source);
                    device_check(cudaMemcpyAsync(src, hidden.data(), hidden.size() * 2, cudaMemcpyHostToDevice, producer), "HC hidden input");
                    device_check(cudaMemcpyAsync(src_pre, pre, sizeof(pre), cudaMemcpyHostToDevice, producer), "HC pre input");
                    ops::hc_pre(reinterpret_cast<B*>(src), src_pre, src_out, 1, producer);
                }
                { DeviceScope scope(destination); device_check(cudaMemsetAsync(dst, 0, capacity, consumer), "poison HC state"); }
                transfer.copy_state(dst, src, dst_pre, src_pre, producer, consumer);
                {
                    DeviceScope scope(source);
                    device_check(cudaMemcpyAsync(expected.data(), src_out, kDim * 2, cudaMemcpyDeviceToHost, producer), "HC reference");
                    device_check(cudaStreamSynchronize(producer), "HC reference wait");
                }
                {
                    DeviceScope scope(destination);
                    ops::hc_pre(reinterpret_cast<B*>(dst), dst_pre, dst_out, 1, consumer);
                    device_check(cudaMemcpyAsync(received.data(), dst_out, kDim * 2, cudaMemcpyDeviceToHost, consumer), "HC result");
                    device_check(cudaStreamSynchronize(consumer), "HC result wait");
                }
                if (std::memcmp(expected.data(), received.data(), kDim * 2)) throw std::runtime_error("HC handoff lost residual or collapse coefficients");
                bool nonzero = false;
                for (const auto& value : received) nonzero |= __bfloat162float(value) != 0;
                if (!nonzero) throw std::runtime_error("HC handoff tested no nonzero output");
                ++hc_checks;
            }
            { DeviceScope scope(source); cudaFree(src); cudaFree(sseed); cudaStreamDestroy(producer); }
            { DeviceScope scope(destination); cudaFree(dst); cudaFree(dseed); cudaFree(errors); cudaStreamDestroy(consumer); }
        }
        if (cases != 64 || replays != 640 || hc_checks != 4 || bulk_checks != 8) throw std::runtime_error("incomplete transfer matrix");
        std::printf("PASS stage transfer: %d cases, %d byte-checked replays, %d nonzero HC handoffs, %d oversized bulk groups checked ten times\n", cases, replays, hc_checks,bulk_checks);
    } catch (const std::exception& ex) { std::fprintf(stderr, "%s\n", ex.what()); return 1; }
}
