#pragma once

#include "strata/ds41/config.hpp"
#include "strata/ds41/pack.hpp"

namespace strata::ds41 {

// A hint must agree with the actual trellis byte geometry used to build the
// device descriptor, not just the human-readable index's nominal bit rate.
inline int expert_slot_integer_rate(const ExpertSlot& slot) {
    const float rate = slot.bits[0];
    if (!(rate >= 1.0f && rate <= 6.0f)) return 0;
    const int bits = static_cast<int>(rate);
    if (rate != static_cast<float>(bits)) return 0;
    constexpr uint64_t tile_bytes = uint64_t(kDim / 16) * (kMoeInter / 16) * 2;
    for (int p = 0; p < 3; ++p) {
        if (slot.bits[p] != rate || slot.codebook[p] != 2 ||
            slot.comp_bytes[4 * p] != tile_bytes * (16 * bits)) return 0;
    }
    return bits;
}

template <typename SlotAt>
int uniform_expert_integer_rate(int count, SlotAt slot_at) {
    if (count <= 0) return 0;
    const int bits = expert_slot_integer_rate(slot_at(0));
    if (!bits) return 0;
    for (int i = 1; i < count; ++i)
        if (expert_slot_integer_rate(slot_at(i)) != bits) return 0;
    return bits;
}

}  // namespace strata::ds41
