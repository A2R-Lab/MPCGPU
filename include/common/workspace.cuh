#pragma once
#include <array>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace mpcgpu {
inline void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// A workspace belongs to one synchronous solver instance/device. Reuse across
// calls; do not share concurrently. Slots have fixed sizes and allocation order
// for a given backend. No global/static cache and no process-lifetime resources.
class SqpWorkspace {
    struct Allocation { void* pointer; size_t bytes; };
    std::vector<Allocation> device_, host_;
    size_t device_cursor_ = 0, host_cursor_ = 0;
    unsigned states_, controls_, knots_;
    int device_id_;
    int backend_ = -1;
public:
    std::array<cudaStream_t,8> streams{};
    cublasHandle_t handle = nullptr;
    bool symbolic_ready = false;
    int symbolic_nnz = 0;

    SqpWorkspace(unsigned states, unsigned controls, unsigned knots)
        : states_(states), controls_(controls), knots_(knots) {
        checkCuda(cudaGetDevice(&device_id_));
        try {
            for (auto& stream : streams) checkCuda(cudaStreamCreate(&stream));
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cuBLAS initialization failed");
        } catch (...) { release(); throw; }
    }
    ~SqpWorkspace() { release(); }
    SqpWorkspace(const SqpWorkspace&) = delete;
    SqpWorkspace& operator=(const SqpWorkspace&) = delete;

    void begin(unsigned states, unsigned controls, unsigned knots, int backend = 0) {
        int current; checkCuda(cudaGetDevice(&current));
        if (states != states_ || controls != controls_ || knots != knots_ || current != device_id_)
            throw std::invalid_argument("Workspace dimensions/device do not match the solve");
        if (backend_ != -1 && backend != backend_)
            throw std::invalid_argument("Workspace backend changed");
        backend_ = backend;
        device_cursor_ = host_cursor_ = 0;
    }
    template<class T> T* device(size_t count) {
        const size_t bytes = count*sizeof(T);
        if (device_cursor_ == device_.size()) {
            void* pointer; checkCuda(cudaMalloc(&pointer, bytes));
            try { device_.push_back({pointer,bytes}); }
            catch (...) { cudaFree(pointer); throw; }
        }
        auto slot = device_[device_cursor_++];
        if (slot.bytes != bytes) throw std::invalid_argument("Workspace backend/layout changed");
        return static_cast<T*>(slot.pointer);
    }
    template<class T> T* host(size_t count) {
        const size_t bytes = count*sizeof(T);
        if (host_cursor_ == host_.size()) {
            void* pointer = std::malloc(bytes);
            if (!pointer) throw std::bad_alloc();
            try { host_.push_back({pointer,bytes}); }
            catch (...) { std::free(pointer); throw; }
        }
        auto slot = host_[host_cursor_++];
        if (slot.bytes != bytes) throw std::invalid_argument("Workspace host layout changed");
        return static_cast<T*>(slot.pointer);
    }
    size_t deviceAllocations() const { return device_.size(); }
private:
    void release() noexcept {
        int previous = device_id_;
        cudaGetDevice(&previous);
        cudaSetDevice(device_id_);
        if (handle) { cublasDestroy(handle); handle = nullptr; }
        for (auto& stream : streams) if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        for (auto slot : device_) cudaFree(slot.pointer);
        for (auto slot : host_) std::free(slot.pointer);
        device_.clear(); host_.clear();
        cudaSetDevice(previous);
    }
};
} // namespace mpcgpu
