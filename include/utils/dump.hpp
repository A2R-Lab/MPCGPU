#pragma once
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/workspace.cuh"

namespace mpcgpu {
// Debug output must be explicitly directed to an owned, existing directory.
// Never overwrite shared /tmp names or continue after a partial write.
template<class T> void dumpDevice(const char* name, const T* device, size_t count) {
    const char* directory=std::getenv("MPCGPU_DUMP_DIR");
    if (!directory || !*directory) throw std::runtime_error("Set MPCGPU_DUMP_DIR for DUMP_KKT");
    std::vector<T> values(count);
    checkCuda(cudaMemcpy(values.data(),device,count*sizeof(T),cudaMemcpyDeviceToHost));
    std::ofstream output(std::string(directory)+"/"+name,std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),count*sizeof(T));
    output.close();
    if (!output) throw std::runtime_error("Cannot write KKT dump");
}
}
