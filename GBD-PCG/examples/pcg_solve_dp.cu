#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"




int main(){

    const uint32_t state_size = 2;
    const uint32_t knot_points = 3;

    double h_S[36] = {0,0,0,0,
                     -.999, 0, 0, -.999,
                     .999, .0999, -.98, .999,
                     .999, -.98, .0999, .999,
                     -2.008, .8801, .8801, -3.0584,
                     .999, .0999, -.98, .999,
                     .999, -.98, .0999, .999,
                     -1.019, .8801, .8801, -2.0694,
                     0,0,0,0};
    
    double h_gamma[6] = {3.1385, 0, 0, 3.0788, .0031, 3.0788};
    double h_lambda[6] = {0,0,0,0,0,0};


    struct pcg_config<double> config;
    uint32_t res = solvePCG<double>(h_S,
									h_gamma,
									h_lambda,
									state_size,
									knot_points,
									&config);

    std::cout << "GBD-PCG returned in " << res << " iters." << std::endl;
    std::cout << "Lambda: " << std::endl;
    for(int i = 0; i < 6; i++){
        std::cout << h_lambda[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

