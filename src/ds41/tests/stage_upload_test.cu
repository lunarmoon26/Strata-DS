#include "strata/ds41/engine.hpp"
#include "strata/ds41/device_scope.hpp"
#include "strata/ds41/vram_experts.hpp"
#include "fake_pack.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <limits>

using namespace strata::ds41;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Fixture {
    std::string dir;
    Fixture() {
        char name[] = ".ds41-stage-synthetic-XXXXXX";
        const char* path = mkdtemp(name);
        if (!path) throw std::runtime_error("cannot create synthetic fixture");
        dir = path;
        ds41test::write_fake_pack(dir, true);
    }
    ~Fixture() {
        for (const char* name : {"index.txt", "experts.txt", "experts.bin", "engram.txt", "engram_hash.txt",
                                 "engram_tokenmap.bin", "pack_info.txt", "dense.bin", "profile.bin"})
            std::remove((dir + "/" + name).c_str());
        rmdir(dir.c_str());
    }
};
}

int main() {
    // These failures must occur before a pack is opened, even with no CUDA GPU.
    for (int bad = 0; bad < 3; ++bad) {
        EngineOptions opt;
        opt.devices = bad == 0 ? std::vector<int>{0, 0} : std::vector<int>{0, 1};
        if (bad == 2) { opt.ram_budget_gib = 0; opt.adapt_every = 0; opt.prefill_chunk = 0; opt.batch_slots = 5; }
        bool refused = false;
        try { Engine engine("/nonexistent-ds41-synthetic-pack", opt); }
        catch (const std::invalid_argument&) { refused = true; }
        require(refused, "invalid split reached a pack or CUDA initialization");
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { std::puts("SKIP: upload matrix needs two GPUs"); return 77; }
    try {
        Fixture fixture;
        const uint64_t hole = (uint64_t) kVocab * kDim * 2;
        const size_t payload_bytes = (4 << 20) + 19;
        std::vector<uint8_t> payload(payload_bytes);
        for (size_t i = 0; i < payload.size(); ++i) payload[i] = (i * 37 + i / 113) & 255;
        {
            std::ofstream index(fixture.dir + "/index.txt", std::ios::app);
            index << "norm.weight bf16 1 " << kDim << " 0 " << kDim * 2 << "\n";
            for (int layer : {14, 20})
                index << "layers." << layer << ".synthetic u8 1 " << payload_bytes << " "
                      << hole + (layer == 20 ? payload_bytes : 0) << " " << payload_bytes << "\n";
            const int fd = open((fixture.dir + "/dense.bin").c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
            require(fd >= 0, "create synthetic sparse dense file");
            require(ftruncate(fd, hole + payload_bytes * 2) == 0, "size synthetic sparse dense file");
            require(pwrite(fd, payload.data(), payload.size(), hole) == (ssize_t) payload.size(), "write first payload");
            require(pwrite(fd, payload.data(), payload.size(), hole + payload_bytes) == (ssize_t) payload.size(), "write last payload");
            close(fd);
        }
        int cases = 0;
        // Both ranges on each physical GPU: no reliance on device 0 allocation defaults.
        for (int device : {0, 1}) for (const auto& range : split_stages({device, 1 - device})) {
            StageRange stage{device, range.begin, range.end};
            device_check(cudaSetDevice(1 - device), "set unrelated caller device");
            {
                Pack pack(fixture.dir);
                const uint64_t uploaded = pack.upload_dense(stage);
                int current = -1; device_check(cudaGetDevice(&current), "caller device");
                require(current == 1 - device, "dense upload did not restore caller device");
                const auto& t = pack.dense(stage.begin == 0 ? "layers.14.synthetic" : "layers.20.synthetic");
                const auto& foreign = pack.dense(stage.begin == 0 ? "layers.20.synthetic" : "layers.14.synthetic");
                require(t.device && !foreign.device, "dense stage loaded the wrong layers");
                require((pack.dense("embed.weight").device != nullptr) == (stage.begin == 0), "embedding ownership");
                require((pack.dense("head.weight").device != nullptr) == (stage.end == kLayers), "head ownership");
                DeviceScope scope(device);
                cudaPointerAttributes attributes{};
                device_check(cudaPointerGetAttributes(&attributes, t.device), "dense pointer owner");
                require(attributes.device == device, "dense pointer belongs to wrong GPU");
                std::vector<uint8_t> actual(payload.size());
                device_check(cudaMemcpy(actual.data(), t.device, actual.size(), cudaMemcpyDeviceToHost), "dense payload readback");
                require(actual == payload, "dense upload changed bytes across chunk boundary");
                VramExperts::Adapt adapt; adapt.every = 0;
                {
                    bool refused = false;
                    try { VramExperts no_room(pack, "", -1, std::numeric_limits<size_t>::max(), adapt,
                                              stage.begin, stage.end, true); }
                    catch (const std::runtime_error&) { refused = true; }
                    require(refused, "resident stage silently fell back when reserve left no room");
                }
                VramExperts experts(pack, "", -1, 64 << 20, adapt, stage.begin, stage.end, true);
                require(experts.slots() == (stage.end - stage.begin) * kExperts, "incomplete resident stage");
                require(!pack.expert_base(), "resident upload mapped the model-sized expert file");
                for (int l = 0; l < kLayers; ++l) for (int e = 0; e < kExperts; ++e)
                    require((experts.res_host()[l * kExperts + e] >= 0) == stage.owns(l), "foreign or missing resident expert");
                for (int l : {stage.begin, stage.end - 1}) for (int e : {0, 127, kExperts - 1}) {
                    const auto& x = pack.expert(l, e);
                    const auto& desc = experts.desc(experts.res_host()[l * kExperts + e]);
                    const auto* base = reinterpret_cast<const uint8_t*>(desc.w1.trellis) - x.comp_off[0];
                    device_check(cudaPointerGetAttributes(&attributes, base), "expert pointer owner");
                    require(attributes.device == device, "expert belongs to wrong GPU");
                    std::vector<uint8_t> expected(x.bytes), received(x.bytes);
                    std::ifstream input(fixture.dir + "/experts.bin", std::ios::binary);
                    input.seekg(x.offset); input.read((char*) expected.data(), expected.size());
                    require(bool(input), "read synthetic expert reference");
                    device_check(cudaMemcpy(received.data(), base, x.bytes, cudaMemcpyDeviceToHost), "expert payload readback");
                    require(received == expected, "expert upload changed published representation");
                }
                ++cases;
                std::printf("CASE device=%d layers=%d-%d dense_bytes=%llu resident_experts=%d checked_expert_blobs=6\n",
                            device, stage.begin, stage.end - 1, (unsigned long long) uploaded, experts.slots());
            }
            int current = -1; device_check(cudaGetDevice(&current), "caller after destruction");
            require(current == 1 - device, "destruction did not restore caller device");
        }
        require(cases == 4, "incomplete upload matrix");
        // The production split now refuses native-only packs. Give this
        // deliberately incomplete fixture the full-profile spot-check metadata
        // so its first stage still reaches the partial-upload failure below.
        // These aliases are never used for inference or as a valid EXL3 matrix.
        {
            std::ofstream index(fixture.dir + "/index.txt", std::ios::app);
            index << "head.trellis i16 1 1 0 2\n";
            auto vector = [&](const std::string& name, int n) {
                index << name << " f16 1 " << n << " 0 " << n * 2 << "\n";
            };
            vector("head.suh", kDim); vector("head.svh", kVocab);
            vector("layers.0.attn.wq_b.suh", kQLora);
            vector("layers.0.attn.wq_b.svh", kHeads * kHeadDim);
            for (int g = 0; g < kOGroups; ++g) {
                const auto prefix = "layers.0.attn.wo_a.slice." + std::to_string(g);
                vector(prefix + ".suh", kHeads * kHeadDim / kOGroups);
                vector(prefix + ".svh", kOLora);
            }
        }
        // Real Engine constructor/destructor failure path: the intentionally
        // incomplete fixture uploads the first stage, then lacks wq_a. Neither
        // that arena nor the untouched second stage may leak or change the
        // caller's selected device. This is not an engine inference test.
        for (int first : {0, 1}) {
            size_t before[2], total = 0;
            for (int device : {0, 1}) {
                DeviceScope scope(device);
                device_check(cudaMemGetInfo(&before[device], &total), "before failed load");
            }
            device_check(cudaSetDevice(1 - first), "caller before failed load");
            EngineOptions opt;
            opt.devices = {first, 1 - first}; opt.ram_budget_gib = 0; opt.adapt_every = 0; opt.prefill_chunk = 0;
            bool failed_at_tensor = false;
            try { Engine engine(fixture.dir, opt); }
            catch (const std::runtime_error& ex) {
                failed_at_tensor = std::string(ex.what()).find("no dense tensor layers.0.attn.wq_a.weight") != std::string::npos;
            }
            require(failed_at_tensor, "incomplete fixture failed before the intended partial-load boundary");
            int current = -1; device_check(cudaGetDevice(&current), "caller after failed load");
            require(current == 1 - first, "failed Engine load changed the caller's device");
            for (int device : {0, 1}) {
                DeviceScope scope(device);
                size_t after = 0;
                device_check(cudaMemGetInfo(&after, &total), "after failed load");
                // Allow driver/display noise, but not the fixture's >1.3 GiB arena.
                require(after + (128 << 20) >= before[device], "failed Engine load leaked its GPU arena");
            }
        }
        std::puts("PASS stage uploads: 4 cases, 61440 residency checks, 24 byte-exact mixed-size expert blobs");
        std::puts("PASS partial-load cleanup: both stage orders, no arena leak or caller-device change");
    } catch (const std::exception& ex) { std::fprintf(stderr, "%s\n", ex.what()); return 1; }
}
