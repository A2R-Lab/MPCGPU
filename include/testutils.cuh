#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include "gpuassert.cuh"
#include "settings.cuh"


///TODO: is tracking error q & qd, also just current state?
template <typename T>
__global__
void compute_tracking_error_kernel(T *d_tracking_error, uint32_t state_size, T *d_xu_goal, T *d_xs){
    
    T err;
    
    for(int ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){              // just comparing q
        err = abs(d_xs[ind] - d_xu_goal[ind]);
        atomicAdd(d_tracking_error, err);
    }
}



template <typename T>
T compute_tracking_error(uint32_t state_size, T *d_xu_goal, T *d_xs){

    T h_tracking_error = 0.0f;
    T *d_tracking_error;
    gpuErrchk(cudaMalloc(&d_tracking_error, sizeof(T)));
    gpuErrchk(cudaMemcpy(d_tracking_error, &h_tracking_error, sizeof(T), cudaMemcpyHostToDevice));

    compute_tracking_error_kernel<T><<<1,32>>>(d_tracking_error, state_size, d_xu_goal, d_xs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(&h_tracking_error, d_tracking_error, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_tracking_error));
    return h_tracking_error;
}


template <typename T>
void dump_tracking_data(std::vector<int> *pcg_iters, std::vector<bool> *pcg_exits, std::vector<double> *linsys_times, std::vector<double> *sqp_times, std::vector<uint32_t> *sqp_iters, 
                std::vector<bool> *sqp_exits, std::vector<T> *tracking_errors, std::vector<std::vector<T>> *tracking_path, uint32_t timesteps_taken, 
                uint32_t control_updates_taken, uint32_t start_state_ind, uint32_t goal_state_ind, uint32_t test_iter,
                std::string filename_prefix){
    // Helper function to create file names
    auto createFileName = [&](const std::string& data_type) {
        std::string filename = filename_prefix + "_" + std::to_string(test_iter) + "_" + data_type + ".result";
        return filename;
    };
    
    // Helper function to dump single-dimension vector data
    auto dumpVectorData = [&](const auto& data, const std::string& data_type) {
        std::ofstream file(createFileName(data_type));
        if (!file.is_open()) {
            std::cerr << "Failed to open " << data_type << " file.\n";
            return;
        }
        for (const auto& item : *data) {
            file << item << '\n';
        }
        file.close();
    };

    // Dump single-dimension vector data
    dumpVectorData(pcg_iters, "pcg_iters");
    dumpVectorData(linsys_times, "linsys_times");
    dumpVectorData(sqp_times, "sqp_times");
    dumpVectorData(sqp_iters, "sqp_iters");
    dumpVectorData(sqp_exits, "sqp_exits");
    dumpVectorData(tracking_errors, "tracking_errors");
    dumpVectorData(pcg_exits, "pcg_exits");


    // Dump two-dimension vector data (tracking_path)
    std::ofstream file(createFileName("tracking_path"));
    if (!file.is_open()) {
        std::cerr << "Failed to open tracking_path file.\n";
        return;
    }
    for (const auto& outerItem : *tracking_path) {
        for (const auto& innerItem : outerItem) {
            file << innerItem << ',';
        }
        file << '\n';
    }
    file.close();

    std::ofstream statsfile(createFileName("stats"));
    if (!statsfile.is_open()) {
        std::cerr << "Failed to open stats file.\n";
        return;
    }
    statsfile << "timesteps: " << timesteps_taken << "\n";
    statsfile << "control_updates: " << control_updates_taken << "\n";
    // printStatsToFile<double>(&linsys_times, )
    
    statsfile.close();
}


void print_test_config(){
    std::cout << "knot points: " << KNOT_POINTS << "\n";
    std::cout << "datatype: " << (USE_DOUBLES ? "DOUBLE" : "FLOAT") << "\n";
    std::cout << "noise: " << (ADD_NOISE ? "ON" : "OFF") << "\n";
    std::cout << "sqp exits condition: " << (CONST_UPDATE_FREQ ? "CONSTANT TIME" : "CONSTANT ITERS") << "\n";
    std::cout << "QD COST: " << QD_COST << "\n";
    std::cout << "R COST: " << R_COST << "\n";
    std::cout << "rho factor: " << RHO_FACTOR << "\n";
    std::cout << "rho max: " << RHO_MAX << "\n";
    std::cout << "test iters: " << TEST_ITERS << "\n";
#if CONST_UPDATE_FREQ
    std::cout << "max sqp time: " << SQP_MAX_TIME_US << "\n";
#else
    std::cout << "max sqp iter: " << SQP_MAX_ITER << "\n";
#endif
    std::cout << "solver: " << (PCG_SOLVE ? "PCG" : "QDLDL") << "\n";
#if PCG_SOLVE
    std::cout << "max pcg iter: " << PCG_MAX_ITER << "\n";
    // std::cout << "pcg exit tol: " << PCG_EXIT_TOL << "\n";
#endif
    std::cout << "save data: " << (SAVE_DATA ? "ON" : "OFF") << "\n";
    std::cout << "jitters: " << (REMOVE_JITTERS ? "ON" : "OFF") << "\n";

    std::cout << "\n\n";
}
