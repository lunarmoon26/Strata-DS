#pragma once

#include "strata/ds41/config.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace strata::ds41 {

struct StageRange {
    int device, begin, end;
    bool owns(int layer) const { return layer >= begin && layer < end; }
    bool owns_dense(const std::string& name) const {
        if (name == "embed.weight") return begin == 0;
        if (name.compare(0, 5, "head.") == 0 || name == "norm.weight") return end == kLayers;
        if (name.compare(0, 7, "layers.") != 0) return false;  // MTP is not part of split decode
        const size_t dot = name.find('.', 7);
        if (dot == std::string::npos || dot == 7) throw std::invalid_argument("bad layer tensor name: " + name);
        int layer = 0;
        for (size_t i = 7; i < dot; ++i) {
            if (name[i] < '0' || name[i] > '9' || layer >= kLayers)
                throw std::invalid_argument("bad layer tensor name: " + name);
            layer = layer * 10 + name[i] - '0';
        }
        if (layer >= kLayers) throw std::invalid_argument("bad layer tensor name: " + name);
        return owns(layer);
    }
};

inline std::vector<StageRange> split_stages(const std::vector<int>& devices, int boundary = 20) {
    if (devices.size() != 2 || devices[0] < 0 || devices[1] < 0 || devices[0] == devices[1])
        throw std::invalid_argument("DeepSeek --devices requires two distinct nonnegative CUDA device IDs");
    if (boundary != 18 && boundary != 20)
        throw std::invalid_argument("DeepSeek split layer must be 18 (mirrored KV/index) or 20");
    return {{devices[0], 0, boundary}, {devices[1], boundary, kLayers}};
}

inline std::vector<int> parse_split_devices(const std::string& text) {
    std::vector<int> ids;
    size_t start = 0;
    do {
        const size_t end = text.find(',', start);
        const std::string part = text.substr(start, end == std::string::npos ? end : end - start);
        if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("--devices expects two CUDA device IDs, e.g. 0,1");
        ids.push_back(std::stoi(part));
        if (end == std::string::npos) break;
        start = end + 1;
    } while (true);
    (void) split_stages(ids);
    return ids;
}

}  // namespace strata::ds41
