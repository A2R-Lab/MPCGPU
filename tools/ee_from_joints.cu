// Recompute end-effector position targets from a joint-space reference with the current model.
// Usage: ee_from_joints <input_traj.csv> <output_eepos.traj>
// Input rows are [q(7) qd(7) u(7)]; output rows are [x y z 0 0 0] for the named EE frame.
// The ICRA pick-and-place circuit keeps its original joint trajectory; its 2024 EE file used the
// pre-correction frame, so the tracked targets are regenerated here (see examples/icra/README.md).
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "dynamics/rbd_plant.cuh"
#include "utils/trajectory.hpp"

template<typename T>
__global__ void k_fk(T* d_ee, const T* d_q, void* d_rm){
    // Static buffers: the FK device function owns the dynamic shared arena from offset zero.
    __shared__ T s_q[grid::NUM_JOINTS], s_ee[6];
    for(int i=threadIdx.x;i<grid::NUM_JOINTS;i+=blockDim.x){ s_q[i]=d_q[i]; }
    __syncthreads();
    grid::end_effector_pose_device_EE<T>(s_ee, s_q, (grid::robotModel<T>*)d_rm);
    __syncthreads();
    for(int i=threadIdx.x;i<6;i+=blockDim.x) d_ee[i]=s_ee[i];
}

int main(int argc, char** argv) try {
    using T = float;
    constexpr int NQ = grid::NUM_JOINTS, WIDTH = 3*NQ;
    if (argc != 3) { fprintf(stderr, "Usage: ee_from_joints <input_traj.csv> <output_eepos.traj>\n"); return 2; }
    const std::vector<T> xu = mpcgpu::readTrajectory<T>(argv[1], WIDTH);
    const size_t rows = xu.size() / WIDTH;
    auto* d_rm = grid::init_robotModel<T>();
    T *d_q, *d_ee;
    gpuErrchk(cudaMalloc(&d_q, NQ*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_ee, 6*sizeof(T)));
    const size_t smem = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT*sizeof(T);
    FILE* out = fopen(argv[2], "w");
    if (!out) throw std::runtime_error(std::string("Cannot open output: ") + argv[2]);
    for (size_t r = 0; r < rows; ++r) {
        gpuErrchk(cudaMemcpy(d_q, &xu[r*WIDTH], NQ*sizeof(T), cudaMemcpyHostToDevice));
        k_fk<T><<<1,32,smem>>>(d_ee, d_q, d_rm);
        gpuErrchk(cudaPeekAtLastError());
        T ee[6];
        gpuErrchk(cudaMemcpy(ee, d_ee, 6*sizeof(T), cudaMemcpyDeviceToHost));
        fprintf(out, "%.9g,%.9g,%.9g,0,0,0\n", ee[0], ee[1], ee[2]);
    }
    fclose(out);
    gpuErrchk(cudaFree(d_q));
    gpuErrchk(cudaFree(d_ee));
    grid::free_robotModel(d_rm);
    printf("wrote %zu EE targets to %s\n", rows, argv[2]);
    return 0;
} catch (const std::exception& error) {
    fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
}
