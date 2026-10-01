// Per-kernel GBD-PCG timing on a fixed captured Schur system, for attributing solve-time changes.
// Builds against any GBD-PCG version that defines pcg<T, STATE_SIZE, KNOT_POINTS> and
// pcgSharedMemSize<T>; -DHAS_REL_TOL=1 for versions whose kernel takes the relative tolerance.
// Usage: pcg_kernel_bench <dump_dir> <fixed|tol> <iterations or abs tol> <repeats>
//   fixed: exit tolerance 0, exactly <iterations> iterations (per-iteration cost)
//   tol:   absolute exit tolerance on eta, as the ICRA task uses (iterations to converge)
// Each repeat restarts from lambda = 0 and is timed like MPCGPU's linear-system window:
// synchronize, launch, copy back the iteration count and exit flag, synchronize.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <time.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"

static std::vector<float> load(const std::string& path, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f || fread(v.data(), sizeof(float), n, f) != n) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(2); }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: pcg_kernel_bench <dump_dir> <fixed|tol> <value> <repeats>\n"); return 2; }
    constexpr uint32_t d = STATE_SIZE, N = KNOT_POINTS, strips = 3 * d * d * N;
    const std::string dir = argv[1], mode = argv[2];
    const double value = atof(argv[3]);
    const int repeats = atoi(argv[4]);
    auto S = load(dir + "/mpc_S.bin", strips), P = load(dir + "/mpc_Pinv.bin", strips), g = load(dir + "/mpc_gamma.bin", d * N);
    float *d_S, *d_P, *d_g, *d_l, *d_r, *d_p, *d_v, *d_eta;
    uint32_t* d_iters; bool* d_exit;
    gpuErrchk(cudaMalloc(&d_S, strips * 4)); gpuErrchk(cudaMalloc(&d_P, strips * 4));
    gpuErrchk(cudaMalloc(&d_g, d * N * 4)); gpuErrchk(cudaMalloc(&d_l, d * N * 4));
    gpuErrchk(cudaMalloc(&d_r, d * N * 4)); gpuErrchk(cudaMalloc(&d_p, d * N * 4));
    gpuErrchk(cudaMalloc(&d_v, N * 4)); gpuErrchk(cudaMalloc(&d_eta, N * 4));
    gpuErrchk(cudaMalloc(&d_iters, 4)); gpuErrchk(cudaMalloc(&d_exit, 1));
    gpuErrchk(cudaMemcpy(d_S, S.data(), strips * 4, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_P, P.data(), strips * 4, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_g, g.data(), d * N * 4, cudaMemcpyHostToDevice));
    uint32_t max_iter = mode == "fixed" ? static_cast<uint32_t>(value) : 1000;
    float exit_tol = mode == "fixed" ? 0.0f : static_cast<float>(value);
    float rel_tol = 0.0f;
    void* kernel = (void*)pcg<float, STATE_SIZE, KNOT_POINTS>;
    void* args[] = {&d_S, &d_P, &d_g, &d_l, &d_r, &d_p, &d_v, &d_eta, &d_iters, &d_exit, &max_iter, &exit_tol
#if HAS_REL_TOL
                    , &rel_tol
#endif
    };
    const size_t smem = pcgSharedMemSize<float>(d, N);
    std::vector<double> times;
    uint32_t iters = 0; bool capped = false;
    for (int rep = 0; rep < repeats + 10; ++rep) {          // first 10 are warm-up
        gpuErrchk(cudaMemset(d_l, 0, d * N * 4));
        gpuErrchk(cudaDeviceSynchronize());
        timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        gpuErrchk(cudaLaunchCooperativeKernel(kernel, N, 128, args, smem));
        gpuErrchk(cudaMemcpy(&iters, d_iters, 4, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&capped, d_exit, 1, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (rep >= 10) times.push_back((t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) * 1e-3);
    }
    std::sort(times.begin(), times.end());
    printf("BENCH N=%u mode=%s value=%g iters=%u capped=%d median_us=%.3f p10_us=%.3f p90_us=%.3f repeats=%d\n",
           N, mode.c_str(), value, iters, int(capped), times[times.size() / 2], times[times.size() / 10],
           times[times.size() * 9 / 10], repeats);
    return 0;
}
