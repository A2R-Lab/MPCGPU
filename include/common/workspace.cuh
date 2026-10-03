#pragma once
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    struct HostSlot { size_t offset, bytes; };
    std::vector<Allocation> device_, host_;
    // One host arena per workspace for the driver's few result/parameter words. A caller-provided
    // (reused) workspace page-locks and device-maps it once (kernels then store straight into host
    // memory); page-locking costs ~0.4 ms, so a per-call workspace uses pageable memory instead and
    // the driver copies the words back explicitly (`copies_needed()`).
    static constexpr size_t kArenaBytes = 4096, kArenaAlign = 64;
    void* arena_host_ = nullptr; void* arena_device_ = nullptr;
    bool arena_locked_ = false;
    size_t arena_used_ = 0;
    std::vector<HostSlot> pinned_, mapped_;
    std::vector<void*> mapped_device_;        // pageable mode: a device slot per mapped() slot
    std::vector<cudaEvent_t> events_;
    cudaStream_t side_ = nullptr;
    cudaGraphExec_t graphs_[3] = {nullptr, nullptr, nullptr};
    std::vector<uint64_t> graph_key_;      // every by-value launch argument the graphs froze
    bool graphs_ready_ = false;
    size_t device_cursor_ = 0, host_cursor_ = 0, pinned_cursor_ = 0, mapped_cursor_ = 0;
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
            checkCuda(cudaStreamCreate(&side_));
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cuBLAS initialization failed");
            if (cublasSetStream(handle, streams[0]) != CUBLAS_STATUS_SUCCESS)   // cuBLAS follows the main stream
                throw std::runtime_error("cuBLAS stream binding failed");
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
        device_cursor_ = host_cursor_ = pinned_cursor_ = mapped_cursor_ = 0;
    }
    // True when kernels cannot store into the host slots (pageable arena): the driver copies back.
    bool copies_needed() const { return !arena_locked_; }
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
    // Page-locked host slots: the targets of the driver's asynchronous device-to-host copies
    // (merit values, linear-system statistics), so a copy never stages through pageable memory.
    template<class T> T* pinned(size_t count) {
        const size_t bytes = count*sizeof(T);
        if (pinned_cursor_ == pinned_.size()) {
            pinned_.push_back(carve(bytes));
        }
        auto slot = pinned_[pinned_cursor_++];
        if (slot.bytes != bytes) throw std::invalid_argument("Workspace pinned layout changed");
        return reinterpret_cast<T*>(static_cast<char*>(arena_host_) + slot.offset);
    }
    // Streams: streams[0] is the main solver stream (also line-search stream 0), streams[1..7]
    // the other line-search streams, side the initial-merit stream. All are blocking streams,
    // so work the caller enqueues on the legacy stream stays ordered with the solve.
    cudaStream_t main_stream() const { return streams[0]; }
    cudaStream_t side_stream() const { return side_; }

    // Replayable SQP-step segments (pcg backend): captured once per workspace for one set of
    // caller pointers; a different caller pointer set re-captures.
    bool graphsMatch(const std::vector<uint64_t>& key) const { return graphs_ready_ && key == graph_key_; }
    void setGraphKey(const std::vector<uint64_t>& key) { graph_key_ = key; graphs_ready_ = true; }
    template<class Body> void captureGraph(int index, cudaStream_t stream, Body&& body) {
        cudaGraph_t graph = nullptr;
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        body();
        checkCuda(cudaStreamEndCapture(stream, &graph));
        try { checkCuda(cudaGraphInstantiate(&graphs_[index], graph, 0)); }
        catch (...) { cudaGraphDestroy(graph); throw; }
        checkCuda(cudaGraphDestroy(graph));
    }
    void launchGraph(int index, cudaStream_t stream) { checkCuda(cudaGraphLaunch(graphs_[index], stream)); }
    void destroyGraphs() noexcept {
        for (auto& g : graphs_) if (g) { cudaGraphExecDestroy(g); g = nullptr; }
        graphs_ready_ = false;
    }
    // Mapped page-locked slots: kernels store their few result words (merits, PCG statistics)
    // straight into host memory, so no device-to-host copy sits between a launch and the host's
    // read after the stream sync. Returns the host pointer; `*device` receives the device alias.
    template<class T> T* mapped(size_t count, T** device) {
        const size_t bytes = count*sizeof(T);
        if (mapped_cursor_ == mapped_.size()) {
            mapped_.push_back(carve(bytes));
            if (!arena_locked_) { void* d; checkCuda(cudaMalloc(&d, bytes)); mapped_device_.push_back(d); }
        }
        auto slot = mapped_[mapped_cursor_];
        if (slot.bytes != bytes) throw std::invalid_argument("Workspace mapped layout changed");
        *device = arena_locked_ ? reinterpret_cast<T*>(static_cast<char*>(arena_device_) + slot.offset)
                                : static_cast<T*>(mapped_device_[mapped_cursor_]);
        mapped_cursor_++;
        return reinterpret_cast<T*>(static_cast<char*>(arena_host_) + slot.offset);
    }
    bool owned_by_caller = false;   // set by the driver for a caller-provided (reused) workspace

    // Timing events, created on first use and kept for the workspace's lifetime.
    cudaEvent_t event(size_t index) {
        while (events_.size() <= index) {
            cudaEvent_t e; checkCuda(cudaEventCreate(&e));
            events_.push_back(e);
        }
        return events_[index];
    }
    size_t deviceAllocations() const { return device_.size(); }
private:
    HostSlot carve(size_t bytes) {
        if (!arena_host_) {
            if (owned_by_caller) {
                checkCuda(cudaHostAlloc(&arena_host_, kArenaBytes, cudaHostAllocMapped));
                checkCuda(cudaHostGetDevicePointer(&arena_device_, arena_host_, 0));
                arena_locked_ = true;
            } else {
                arena_host_ = std::malloc(kArenaBytes);
                if (!arena_host_) throw std::bad_alloc();
            }
        }
        const size_t offset = (arena_used_ + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
        if (offset + bytes > kArenaBytes) throw std::length_error("Workspace host arena exhausted");
        arena_used_ = offset + bytes;
        return {offset, bytes};
    }
    void release() noexcept {
        int previous = device_id_;
        cudaGetDevice(&previous);
        cudaSetDevice(device_id_);
        destroyGraphs();
        if (handle) { cublasDestroy(handle); handle = nullptr; }
        for (auto& stream : streams) if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        if (side_) { cudaStreamDestroy(side_); side_ = nullptr; }
        for (auto slot : device_) cudaFree(slot.pointer);
        for (auto slot : host_) std::free(slot.pointer);
        if (arena_host_) { if (arena_locked_) cudaFreeHost(arena_host_); else std::free(arena_host_); }
        arena_host_ = arena_device_ = nullptr; arena_used_ = 0; arena_locked_ = false;
        for (auto d : mapped_device_) cudaFree(d);
        mapped_device_.clear();
        for (auto e : events_) cudaEventDestroy(e);
        device_.clear(); host_.clear(); pinned_.clear(); mapped_.clear(); events_.clear();
        cudaSetDevice(previous);
    }
};
} // namespace mpcgpu
