#pragma once
// QDLDL backend for the shared SQP driver (common/sqp.cuh): the Schur system is formed as a
// CSC matrix on the GPU, copied to the host and factored/solved by QDLDL. The symbolic
// structure is cached in the workspace; the numeric factorization is redone every solve.
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "common/sqp.cuh"
#include "qdldl.h"
#include "qdldl/linsys_setup.cuh"

__host__
void qdldl_solve_schur(const QDLDL_int An,
                       QDLDL_int* h_col_ptr, QDLDL_int* h_row_ind, QDLDL_float* Ax, QDLDL_float* b,
                       QDLDL_float* h_lambda,
                       QDLDL_int* Lp, QDLDL_int* Li, QDLDL_float* Lx, QDLDL_float* D, QDLDL_float* Dinv,
                       QDLDL_int* Lnz, QDLDL_int* etree, QDLDL_bool* bwork, QDLDL_int* iwork, QDLDL_float* fwork) {
    if (QDLDL_factor(An, h_col_ptr, h_row_ind, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork) < 0)
        throw std::runtime_error("QDLDL numerical factorization failed");
    for (QDLDL_int i = 0; i < An; i++) h_lambda[i] = b[i];
    QDLDL_solve(An, Lp, Li, Lx, Dinv, h_lambda);
}

namespace mpcgpu {

template <typename T>
struct QdldlBackend {
    static_assert(std::is_same_v<T, QDLDL_float>, "QDLDL and solver precision must match");
    static constexpr int kind = 1;
    static constexpr bool graphable = false;   // the solve is host work
    QDLDL_int An = 0;
    int nnz = 0;
    T* d_gamma = nullptr;
    QDLDL_float *h_lambda, *h_gamma, *h_val, *D, *Dinv, *fwork, *Lx;
    QDLDL_int *h_col_ptr, *h_row_ind, *etree, *Lnz, *Lp, *iwork, *Li;
    QDLDL_bool* bwork;
    QDLDL_int *d_row_ind, *d_col_ptr;
    QDLDL_float *d_val, *d_lambda_double;

    void setup(SqpWorkspace& workspace, uint32_t state_size, uint32_t, uint32_t knots, T*, T* gamma, T*) {
        d_gamma = gamma;
        nnz = (knots-1)*state_size*state_size + knots*(((state_size+1)*state_size)/2);
        h_lambda = workspace.host<QDLDL_float>(state_size*knots);
        h_gamma = workspace.host<QDLDL_float>(state_size*knots);
        h_col_ptr = workspace.host<QDLDL_int>(state_size*knots+1);
        h_row_ind = workspace.host<QDLDL_int>(nnz);
        h_val = workspace.host<QDLDL_float>(nnz);
        d_col_ptr = workspace.device<QDLDL_int>(state_size*knots+1);
        d_row_ind = workspace.device<QDLDL_int>(nnz);
        d_val = workspace.device<QDLDL_float>(nnz);
        d_lambda_double = workspace.device<QDLDL_float>(state_size*knots);
        if (!workspace.symbolic_ready) {   // column pointers and row indices never change
            prep_csr<<<knots, 64>>>(state_size, knots, d_col_ptr, d_row_ind);
            gpuErrchk(cudaMemcpy(h_col_ptr, d_col_ptr, (state_size*knots+1)*sizeof(QDLDL_int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_row_ind, d_row_ind, nnz*sizeof(QDLDL_int), cudaMemcpyDeviceToHost));
        }
        An = state_size*knots;
        etree = workspace.host<QDLDL_int>(An);
        Lnz = workspace.host<QDLDL_int>(An);
        Lp = workspace.host<QDLDL_int>(An+1);
        D = workspace.host<QDLDL_float>(An);
        Dinv = workspace.host<QDLDL_float>(An);
        iwork = workspace.host<QDLDL_int>(3*An);
        bwork = workspace.host<QDLDL_bool>(An);
        fwork = workspace.host<QDLDL_float>(An);
        if (!workspace.symbolic_ready) {
            const QDLDL_int sumLnz = QDLDL_etree(An, h_col_ptr, h_row_ind, iwork, Lnz, etree);
            if (sumLnz < 0) throw std::runtime_error("QDLDL symbolic factorization failed");
            workspace.symbolic_nnz = sumLnz;
            workspace.symbolic_ready = true;
        }
        Li = workspace.host<QDLDL_int>(workspace.symbolic_nnz);
        Lx = workspace.host<QDLDL_float>(workspace.symbolic_nnz);
    }

    void formSchur(uint32_t state_size, uint32_t control_size, uint32_t knots, T* d_G, T* d_C, T* d_g, T* d_c,
                   T*, T* gamma, T rho, const T*, cudaStream_t stream) {
        form_schur_system_qdldl<T>(state_size, control_size, knots, d_G, d_C, d_g, d_c, d_val, gamma, rho, stream);
    }

    void solve(T* d_lambda, cudaStream_t stream) {   // a direct solve: the host waits for the Schur system
        gpuErrchk(cudaMemcpyAsync(h_val, d_val, nnz*sizeof(T), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(h_gamma, d_gamma, An*sizeof(T), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        qdldl_solve_schur(An, h_col_ptr, h_row_ind, h_val, h_gamma, h_lambda, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);
        gpuErrchk(cudaMemcpyAsync(d_lambda, h_lambda, An*sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    void graphKey(std::vector<uint64_t>&) {}
    void copyStats(cudaStream_t) {}                          // no iterations to report
    void record(std::vector<int>&, std::vector<bool>&) {}

#ifdef DUMP_KKT
    template <class Dump> void dumpSchur(Dump dump) { dump("mpc_S_csc.bin", d_val, nnz); }
#endif
};

} // namespace mpcgpu

template <typename T>
auto sqpSolveQdldl(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T* d_eePos_traj,
                   T* d_lambda, T* d_xu, void* d_dynMem_const, T& rho, T rho_reset, T* d_xs_goal = nullptr,
                   mpcgpu::SqpWorkspace* reuse = nullptr) {
    mpcgpu::QdldlBackend<T> backend;
    return mpcgpu::sqpSolve<T>(state_size, control_size, knot_points, timestep, d_eePos_traj, d_lambda, d_xu,
                               d_dynMem_const, backend, rho, rho_reset, d_xs_goal, reuse);
}
