// Real EXL3 geometry but one tiny source blob shared by every synthetic expert.
// Checks C++ MCG metadata, bounded resident upload and fail-closed loading.
#include "strata/ds41/device_scope.hpp"
#include "strata/ds41/engine.hpp"
#include "strata/ds41/vram_experts.hpp"
#include "fake_pack.hpp"

#include <algorithm>
#include <fstream>
#include <unistd.h>

using namespace strata::ds41;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
size_t align(size_t n, size_t a) { return (n + a - 1) / a * a; }
struct Fixture {
    std::string dir;
    std::vector<uint8_t> blob;
    size_t offset[12]{}, bytes[12]{};
    Fixture() {
        char name[] = ".ds41-stage-synthetic-mcg-XXXXXX";
        const char* path = mkdtemp(name);
        if (!path) throw std::runtime_error("MCG temporary directory");
        dir = path;
        ds41test::write_fake_pack(dir);
        size_t size = 0;
        for (int p = 0; p < 3; ++p) {
            const int k = p == 2 ? kMoeInter : kDim, n = p == 2 ? kDim : kMoeInter;
            const size_t sizes[] = {size_t(k / 16) * (n / 16) * 32 * 2, size_t(k) * 2, size_t(n) * 2, 4};
            for (int c = 0; c < 4; ++c) { offset[p * 4 + c] = size; bytes[p * 4 + c] = sizes[c]; size = align(size + sizes[c], 256); }
        }
        blob.resize(align(size, 4096));
        for (size_t i = 0; i < blob.size(); ++i) blob[i] = (i * 17 + i / 109) & 255;
        for (int p = 0; p < 3; ++p) {
            const uint8_t marker[] = {0xed, 0x1f, 0xac, 0xcb};
            std::copy(marker, marker + 4, blob.begin() + offset[p * 4 + 3]);
        }
        std::ofstream file(dir + "/experts.bin", std::ios::binary | std::ios::trunc);
        file.write((const char*) blob.data(), blob.size());
        write_index(2);
    }
    void write_index(int rate, bool bad_geometry = false) {
        const char* names[] = {"w1.trellis", "w1.suh", "w1.svh", "w1.mcg", "w3.trellis", "w3.suh", "w3.svh", "w3.mcg",
                               "w2.trellis", "w2.suh", "w2.svh", "w2.mcg"};
        std::ofstream file(dir + "/experts.txt", std::ios::trunc);
        for (int l = 0; l < kLayers; ++l) for (int e = 0; e < kExperts; ++e) {
            file << l << ' ' << e << " 0 " << blob.size() << ' ' << rate << ' ' << rate << ' ' << rate;
            for (int c = 0; c < 12; ++c)
                file << ' ' << names[c] << ':' << offset[c] << ':' << bytes[c] + (bad_geometry && c == 0 ? 2 : 0);
            file << '\n';
        }
    }
    ~Fixture() {
        for (const char* name : {"index.txt", "experts.txt", "experts.bin", "mtp_experts.txt", "engram.txt", "engram_hash.txt",
                                 "engram_tokenmap.bin", "pack_info.txt"}) std::remove((dir + "/" + name).c_str());
        rmdir(dir.c_str());
    }
};
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) { std::puts("SKIP: needs a CUDA GPU"); return 77; }
    try {
        Fixture fixture;
        Pack pack(fixture.dir);
        require(pack.has_mcg(), "MCG tags not parsed");
        bool refused = false;
        try { Engine engine(fixture.dir, EngineOptions{}); }
        catch (const std::invalid_argument& error) { refused = std::string(error.what()).find("MCG K2 currently requires") != std::string::npos; }
        require(refused, "MCG entered the single-device CPU fallback path");
        VramExperts::Adapt adapt; adapt.every = 0;
        {
            VramExperts experts(pack, "", -1, 128 << 20, adapt, 7, 8, true);
            require(experts.slots() == kExperts, "not all local MCG experts resident");
            for (int l = 0; l < kLayers; ++l) for (int e = 0; e < kExperts; ++e)
                require((experts.res_host()[l * kExperts + e] >= 0) == (l == 7), "MCG ownership leaked across layers");
            const auto& desc = experts.desc(experts.res_host()[7 * kExperts]);
            for (const auto* p : {&desc.w1, &desc.w3, &desc.w2}) require(p->codebook == 1 && p->tile_w == 32, "MCG descriptor selector lost");
            std::vector<uint8_t> received(fixture.blob.size());
            device_check(cudaMemcpy(received.data(), desc.w1.trellis, received.size(), cudaMemcpyDeviceToHost), "MCG readback");
            require(received == fixture.blob, "MCG blob changed during upload");
            require(!pack.expert_base(), "MCG loader used a model-sized host mapping");
        }
        {
            std::fstream file(fixture.dir + "/experts.bin", std::ios::binary | std::ios::in | std::ios::out);
            const uint32_t invalid = 0; file.seekp(fixture.offset[3]); file.write((const char*) &invalid, 4);
        }
        refused = false;
        try { VramExperts wrong(pack, "", -1, 128 << 20, adapt, 7, 8, true); }
        catch (const std::runtime_error& error) { refused = std::string(error.what()).find("unknown MCG marker") != std::string::npos; }
        require(refused, "unknown MCG marker reached a decoder");
        {
            std::ofstream file(fixture.dir + "/experts.bin", std::ios::binary | std::ios::trunc);
            file.write((const char*) fixture.blob.data(), fixture.blob.size());
        }
        fixture.write_index(2, true);
        Pack bad_geometry(fixture.dir);
        refused = false;
        try { VramExperts wrong(bad_geometry, "", -1, 128 << 20, adapt, 7, 8, true); }
        catch (const std::runtime_error& error) { refused = std::string(error.what()).find("MCG metadata does not describe") != std::string::npos; }
        require(refused, "malformed MCG geometry reached a decoder");
        fixture.write_index(3);
        refused = false;
        try { Pack wrong(fixture.dir); }
        catch (const std::runtime_error& error) { refused = std::string(error.what()).find("MCG requires K2") != std::string::npos; }
        require(refused, "unsupported MCG bitrate reached upload");
        fixture.write_index(2);
        {
            std::ifstream source(fixture.dir + "/experts.txt");
            std::string first; std::getline(source, first);
            std::ofstream target(fixture.dir + "/mtp_experts.txt"); target << first << '\n';
            std::ofstream info(fixture.dir + "/pack_info.txt", std::ios::app); info << "mtp_stages 1\nmtp_experts 1\n";
        }
        refused = false;
        try { Pack wrong(fixture.dir); }
        catch (const std::runtime_error& error) { refused = std::string(error.what()).find("backbone experts only") != std::string::npos; }
        require(refused, "MCG entered an unsupported MTP decoder path");
        std::puts("PASS MCG pack: 384 resident experts, 15360 ownership checks, byte-exact blob; wrong marker/rate/geometry, MTP and CPU fallback rejected");
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
