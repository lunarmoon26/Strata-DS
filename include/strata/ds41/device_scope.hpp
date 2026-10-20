#pragma once

#include <cuda_runtime.h>
#include <new>
#include <stdexcept>
#include <string>

namespace strata::ds41 {

inline void device_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41: ") + what + ": " + cudaGetErrorString(e));
}

class DeviceScope {
public:
    explicit DeviceScope(int device) {
        device_check(cudaGetDevice(&previous_), "get current device");
        if (device != previous_) device_check(cudaSetDevice(device), "select stage device");
    }
    DeviceScope(int device, const std::nothrow_t&) noexcept {
        cudaGetDevice(&previous_);
        cudaSetDevice(device);
    }
    ~DeviceScope() { cudaSetDevice(previous_); }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
private:
    int previous_ = 0;
};

}  // namespace strata::ds41
