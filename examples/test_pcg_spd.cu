// test_pcg_spd.cu — SPD block-tridiagonal residual gate for GBD-PCG.
//
// The shipped pcg_solve.cu is degenerate (its hardcoded S is indefinite and the
// solver leaves d_Pinv uninitialized) so it NaNs and validates nothing. This test
// builds a random SYMMETRIC, DIAGONALLY-DOMINANT (hence SPD) block-tridiagonal S in
// the [L|D|R] strip layout, an IDENTITY preconditioner (-> plain CG), runs the
// cooperative GBD-PCG solve, and checks the residual ||gamma - S*lambda|| on host.
// It exercises the bdmv->glass::gemv migration + zero-padded boundaries (block 0 has
// no L, the last block no R) on every iteration.
//
// Build (set the dims to match the kernel template instantiation):
//   nvcc -O3 -I../include -I../GLASS -arch=sm_120 \
//        -DSTATE_SIZE=6 -DKNOT_POINTS=8 test_pcg_spd.cu -o test_spd.exe
//   ./test_spd.exe

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"

// Strip layout helpers: block-row i's [L|D|R] strip starts at i*3*d*d; within a
// slot (0=L,1=D,2=R) the d x d block is COLUMN-MAJOR: element (r,c) at slot*d*d+c*d+r.
static inline int strip_idx(int i, int slot, int r, int c, int d) {
    return i * 3 * d * d + slot * d * d + c * d + r;
}

int main() {
    const int d = STATE_SIZE;
    const int N = KNOT_POINTS;
    srand(12345);

    std::vector<float> h_S(3 * d * d * N, 0.0f);    // [L|D|R] strips
    std::vector<float> h_Pinv(3 * d * d * N, 0.0f); // identity preconditioner strips
    std::vector<float> h_gamma(d * N);
    std::vector<float> h_lambda(d * N, 0.0f);

    auto frand = []() { return (float)rand() / (float)RAND_MAX - 0.5f; }; // [-0.5,0.5]

    // First build the off-diagonal coupling blocks R_i (block (i,i+1)); L_{i+1} = R_i^T
    // keeps S symmetric. Then D_i = 4 I + small symmetric perturbation -> diagonally
    // dominant (||off|| per row <= 0.1*d << 4) -> SPD by Gershgorin.
    std::vector<std::vector<float>> R(N); // R[i] is d*d row-major for convenience
    for (int i = 0; i < N - 1; i++) {
        R[i].resize(d * d);
        for (int k = 0; k < d * d; k++) R[i][k] = 0.1f * frand();
    }
    for (int i = 0; i < N; i++) {
        // D_i
        for (int r = 0; r < d; r++) {
            for (int c = 0; c < d; c++) {
                float off = 0.1f * frand();
                float val = (r == c) ? 4.0f : 0.0f;
                // symmetric perturbation: use the same draw for (r,c) and (c,r)
                if (r <= c) {
                    val += off;
                    h_S[strip_idx(i, 1, r, c, d)] += val;
                    if (r != c) h_S[strip_idx(i, 1, c, r, d)] += off;
                }
            }
        }
        // R_i -> slot 2 of row i (col-major), and its transpose -> slot 0 (L) of row i+1
        if (i < N - 1) {
            for (int r = 0; r < d; r++)
                for (int c = 0; c < d; c++) {
                    float v = R[i][r * d + c];
                    h_S[strip_idx(i,     2, r, c, d)] = v;   // R_i at (i, i+1)
                    h_S[strip_idx(i + 1, 0, c, r, d)] = v;   // L_{i+1} = R_i^T
                }
        }
        // identity Pinv: D = I, L = R = 0
        for (int r = 0; r < d; r++) h_Pinv[strip_idx(i, 1, r, r, d)] = 1.0f;
    }
    for (int k = 0; k < d * N; k++) h_gamma[k] = frand();

    // ---- device solve via the mid-level API (lets us supply our own d_Pinv) ----
    const uint32_t states_sq = d * d;
    float *d_S, *d_Pinv, *d_gamma, *d_lambda, *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_S,            3 * states_sq * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_Pinv,         3 * states_sq * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_gamma,        d * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_lambda,       d * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_r,            d * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_p,            d * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_v_temp,       N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, N * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_S,      h_S.data(),      3*states_sq*N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Pinv,   h_Pinv.data(),   3*states_sq*N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_gamma,  h_gamma.data(),  d*N*sizeof(float),           cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lambda, h_lambda.data(), d*N*sizeof(float),           cudaMemcpyHostToDevice));

    pcg_config<float> config;
    config.pcg_exit_tol = 1e-8f;
    config.pcg_rel_tol = 1e-10f;   // correctness gate: solve fully (not a convergence-rate test)
    config.pcg_max_iter = d * N * 4;

    uint32_t iters = solvePCG<float>(d, N, d_S, d_Pinv, d_gamma, d_lambda,
                                     d_r, d_p, d_v_temp, d_eta_new_temp, &config);

    gpuErrchk(cudaMemcpy(h_lambda.data(), d_lambda, d*N*sizeof(float), cudaMemcpyDeviceToHost));

    // ---- host residual: res = gamma - S*lambda  (block-tridiagonal multiply) ----
    double max_res = 0.0, max_gamma = 0.0;
    for (int i = 0; i < N; i++) {
        for (int r = 0; r < d; r++) {
            double sx = 0.0;
            for (int c = 0; c < d; c++) {
                sx += h_S[strip_idx(i, 1, r, c, d)] * h_lambda[i * d + c];           // D_i x_i
                if (i > 0)     sx += h_S[strip_idx(i, 0, r, c, d)] * h_lambda[(i-1)*d + c]; // L_i x_{i-1}
                if (i < N - 1) sx += h_S[strip_idx(i, 2, r, c, d)] * h_lambda[(i+1)*d + c]; // R_i x_{i+1}
            }
            double res = std::fabs(h_gamma[i * d + r] - sx);
            if (res > max_res) max_res = res;
            if (std::fabs(h_gamma[i * d + r]) > max_gamma) max_gamma = std::fabs(h_gamma[i * d + r]);
        }
    }

    bool finite = true;
    for (float v : h_lambda) if (!std::isfinite(v)) finite = false;

    std::cout << "STATE_SIZE=" << d << " KNOT_POINTS=" << N
              << "  iters=" << iters
              << "  ||res||_inf=" << max_res
              << "  rel=" << (max_res / (max_gamma + 1e-30)) << std::endl;

    bool pass = finite && (max_res / (max_gamma + 1e-30) < 1e-4);
    std::cout << (pass ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_S); cudaFree(d_Pinv); cudaFree(d_gamma); cudaFree(d_lambda);
    cudaFree(d_r); cudaFree(d_p); cudaFree(d_v_temp); cudaFree(d_eta_new_temp);
    return pass ? 0 : 1;
}
