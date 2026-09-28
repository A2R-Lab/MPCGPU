#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>
#include "types.cuh"
#include "pcg.cuh"

namespace gbd_detail {
inline void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Owned CUDA allocation; partial construction and exceptions cannot leak it.
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(&data, count * sizeof(T))); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

template<class T>
void validate(uint32_t states, uint32_t knots, const pcg_config<T>* config) {
    static_assert(STATE_SIZE > 0 && KNOT_POINTS > 0, "PCG dimensions must be positive");
    if (!config || states != STATE_SIZE || knots != KNOT_POINTS)
        throw std::invalid_argument("Runtime PCG dimensions must match STATE_SIZE/KNOT_POINTS");
    if (!std::isfinite(config->pcg_exit_tol) || config->pcg_exit_tol < 0 ||
        !std::isfinite(config->pcg_rel_tol) || config->pcg_rel_tol < 0)
        throw std::invalid_argument("PCG tolerances must be finite and nonnegative");
    if (!config->pcg_block.x || config->pcg_block.x > 1024 ||
        config->pcg_block.y != 1 || config->pcg_block.z != 1)
        throw std::invalid_argument("PCG requires a one-dimensional block of 1..1024 threads");
}
} // namespace gbd_detail

struct PcgResult {
    uint32_t iterations;
    bool iteration_limit;
};

// Synchronous device-pointer API. Inputs and scratch belong to the caller;
// lambda is both the initial guess and the solution. S and Pinv must be SPD.
template<class T>
PcgResult solvePCGChecked(uint32_t states, uint32_t knots,
    T* d_S, T* d_Pinv, T* d_gamma, T* d_lambda, T* d_r, T* d_p,
    T* d_v_temp, T* d_eta_new_temp, const pcg_config<T>* config) {
    gbd_detail::validate(states, knots, config);
    if (!d_S || !d_Pinv || !d_gamma || !d_lambda || !d_r || !d_p ||
        !d_v_temp || !d_eta_new_temp)
        throw std::invalid_argument("PCG device buffers must not be null");
    void* kernel = reinterpret_cast<void*>(pcg<T, STATE_SIZE, KNOT_POINTS>);
    checkPcgOccupancy<T>(kernel, config->pcg_block, states, knots);
    gbd_detail::Buffer<uint32_t> iterations(1);
    gbd_detail::Buffer<bool> capped(1);
    auto max_iter = config->pcg_max_iter;
    auto exit_tol = config->pcg_exit_tol;
    auto rel_tol = config->pcg_rel_tol;
    void* args[] = {&d_S, &d_Pinv, &d_gamma, &d_lambda, &d_r, &d_p,
        &d_v_temp, &d_eta_new_temp, &iterations.data, &capped.data,
        &max_iter, &exit_tol, &rel_tol};
    gbd_detail::check(cudaLaunchCooperativeKernel(kernel, knots, config->pcg_block,
        args, pcgSharedMemSize<T>(states, knots)));
    PcgResult result{};
    gbd_detail::check(cudaMemcpy(&result.iterations, iterations.data, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gbd_detail::check(cudaMemcpy(&result.iteration_limit, capped.data, sizeof(bool), cudaMemcpyDeviceToHost));
    return result;
}

template<class T>
uint32_t solvePCG(uint32_t states, uint32_t knots,
    T* d_S, T* d_Pinv, T* d_gamma, T* d_lambda, T* d_r, T* d_p,
    T* d_v_temp, T* d_eta_new_temp, pcg_config<T>* config) {
    return solvePCGChecked(states, knots, d_S, d_Pinv, d_gamma, d_lambda,
                          d_r, d_p, d_v_temp, d_eta_new_temp, config).iterations;
}

// Host convenience API: ordinary CG (identity preconditioner). Only lambda
// is modified; use the device API to supply a nontrivial preconditioner.
template<class T>
uint32_t solvePCG(const T* h_S, const T* h_gamma, T* h_lambda,
                 uint32_t states, uint32_t knots, pcg_config<T>* config) {
    gbd_detail::validate(states, knots, config);
    if (!h_S || !h_gamma || !h_lambda || !config->empty_pinv)
        throw std::invalid_argument("Host PCG requires non-null arrays and empty_pinv=true");
    const size_t n = size_t(states) * knots, strips = 3 * n * states;
    for (size_t i = 0; i < strips; ++i)
        if (!std::isfinite(h_S[i])) throw std::invalid_argument("Nonfinite matrix");
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(h_gamma[i]) || !std::isfinite(h_lambda[i]))
            throw std::invalid_argument("Nonfinite RHS or initial guess");
    std::vector<T> identity(strips, T(0));
    for (uint32_t k = 0; k < knots; ++k)
        for (uint32_t i = 0; i < states; ++i)
            identity[(3 * size_t(k) + 1) * states * states + i * states + i] = T(1);
    gbd_detail::Buffer<T> S(strips), P(strips), gamma(n), lambda(n), r(n), p(n), v(knots), eta(knots);
    gbd_detail::check(cudaMemcpy(S.data, h_S, strips*sizeof(T), cudaMemcpyHostToDevice));
    gbd_detail::check(cudaMemcpy(P.data, identity.data(), strips*sizeof(T), cudaMemcpyHostToDevice));
    gbd_detail::check(cudaMemcpy(gamma.data, h_gamma, n*sizeof(T), cudaMemcpyHostToDevice));
    gbd_detail::check(cudaMemcpy(lambda.data, h_lambda, n*sizeof(T), cudaMemcpyHostToDevice));
    auto result = solvePCGChecked(states, knots, S.data, P.data, gamma.data, lambda.data,
                                 r.data, p.data, v.data, eta.data, config);
    gbd_detail::check(cudaMemcpy(h_lambda, lambda.data, n*sizeof(T), cudaMemcpyDeviceToHost));
    return result.iterations;
}

template<class T>
uint32_t solvePCG(csr_t<T>*, csr_t<T>*, T*, T*, unsigned, unsigned, pcg_config<T>*) {
    throw std::invalid_argument("CSR input is unsupported; supply column-major [L|D|R] strips");
}
