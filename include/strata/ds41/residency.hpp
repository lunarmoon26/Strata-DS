// Host synchronization for the lookahead callback and tier residency writes.
#pragma once
#include <cstdint>
#include <mutex>

namespace strata::ds41 {

// The engine callback reads both tables. Keep this lock across that callback.
// Tier writes take it only for publication. GPU table layouts stay unchanged.
inline std::mutex& residency_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline void publish_residency(int32_t& entry, int32_t value) {
    std::lock_guard<std::mutex> lock(residency_mutex());
    entry = value;
}

}  // namespace strata::ds41
