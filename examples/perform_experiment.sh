#!/bin/bash

# Define a function to run when the script receives a SIGINT signal
function handle_sigint {
    echo "Received SIGINT, exiting"
    exit 1
}

# Tell the script to use the handle_sigint function to handle SIGINT signals
trap handle_sigint SIGINT

end_compile_command="-o runme.exe runme.cu"

declare -A pcg_max_iters
pcg_max_iters["32"]="173"
pcg_max_iters["64"]="167"
pcg_max_iters["128"]="167"
pcg_max_iters["256"]="118"
pcg_max_iters["512"]="67"


# BEGIN test configuration - shouldn't have to modify anything above this line

# sample test configuration 1 - test linear system solve times for different numbers of knot points on a 3080 GPU
time_linsys="1"
# NOTE: make sure to set -arch value appropiately based on GPU we are using (use 86 for 3080, 89 for 4090)
base_compile_command="nvcc --compiler-options -Wall -arch=sm_86  -O3 -I. -IGPU-PCG/include -IGLASS -IGPU-PCG -I. -Irbdfiles -I./qdldl/include -lcublas -Lqdldl/build/out -lqdldl "
echo $base_compile_command
knot_points=("32")
periods=("2000")
rho_maxs=("10")
rho_factors=("1.2")

# END test configuration - shouldn't have to modify anything below this line


for knot in "${knot_points[@]}"; do
    for per in "${periods[@]}"; do
        for rm in "${rho_maxs[@]}"; do
            for rf in "${rho_factors[@]}"; do
                test qdl
                compile_command=$base_compile_command
                compile_command+="-DPCG_SOLVE=0 "
                compile_command+="-DQD_COST=.0001 "
                compile_command+="-DSQP_MAX_TIME_US=$per "
                compile_command+="-DSIMULATION_PERIOD=$per "
                compile_command+="-DRHO_FACTOR=$rf "
                compile_command+="-DRHO_MAX=$rm "
                compile_command+="-DKNOT_POINTS=$knot "
                compile_command+="-DPCG_EXIT_TOL=-1 "
                compile_command+="-DPCG_MAX_ITER=-1 "
                if [ $knot = "64" ]; then
                    compile_command+="-DR_COST=.001 "
                else
                    compile_command+="-DR_COST=.0001 "
                fi
                compile_command+="-diag-suppress 68 " # question: what is this option? Always seems set in QDLDL tests, do we need it?
                compile_command+="-DTIME_LINSYS=$time_linsys "
                compile_command+=$end_compile_command

                echo $compile_command

                eval $compile_command
                ./runme.exe
                echo "-----------------------------------------------------------"
                
                
                compile_command=$base_compile_command
                compile_command+="-DPCG_SOLVE=1 "
                compile_command+="-DQD_COST=.0001 "
                compile_command+="-DSQP_MAX_TIME_US=$per "
                compile_command+="-DSIMULATION_PERIOD=$per "
                compile_command+="-DRHO_FACTOR=$rf "
                compile_command+="-DRHO_MAX=$rm "
                compile_command+="-DKNOT_POINTS=$knot "
                compile_command+="-DPCG_MAX_ITER=${pcg_max_iters[$knot]} "
                if [ $knot = "64" ]
                then
                    compile_command+="-DR_COST=.001 "
                else
                    compile_command+="-DR_COST=.0001 "
                fi
                compile_command+="-DTIME_LINSYS=$time_linsys "
                compile_command+=$end_compile_command

                echo $compile_command
                eval $compile_command
                ./runme.exe
                echo "-----------------------------------------------------------"
            done
        done
    done
done