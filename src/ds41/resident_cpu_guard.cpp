// The resident backend does not link the laptop's optimized CPU expert kernel.
// Keep the core's unused compatibility calls fail-closed, never silently drop an expert.
#include "../../third_party/exllamav3_moe/moe_mul1.h"
#include <stdexcept>

namespace {
[[noreturn]] void forbidden() {
    throw std::runtime_error("resident 1017: CPU expert fallback is not available");
}
}

int64_t exl3_moe_cpu_make_layer_raw(const MoeCpuMatrixDesc*, const MoeCpuMatrixDesc*,
                                   const MoeCpuMatrixDesc*, int, int, float, int) { forbidden(); }
void exl3_moe_cpu_free_layer(int64_t) { forbidden(); }
void exl3_moe_cpu_set_expert_raw(int64_t, int, const MoeCpuMatrixDesc*, const MoeCpuMatrixDesc*,
                               const MoeCpuMatrixDesc*, int) { forbidden(); }
void exl3_moe_cpu_forward_raw(int64_t, const at::Half*, const int32_t*, const at::Half*,
                            float*, int, int, int) { forbidden(); }
void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int) { forbidden(); }
void exl3_moe_cpu_set_prof(bool) { forbidden(); }
void exl3_moe_cpu_pool_prime(int) { forbidden(); }
bool exl3_moe_cpu_has_avx2() { return false; }
bool exl3_moe_cpu_has_avx512_bw() { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }
