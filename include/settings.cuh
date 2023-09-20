#pragma once

// #define KNOT_POINTS 32
#define STATE_SIZE  14


#define ADD_NOISE  0
#define TEST_ITERS 1
// qdldl if 0
// #define PCG_SOLVE       1

// doubles if 1, floats if 0
#define USE_DOUBLES 0

#if USE_DOUBLES
typedef double pcg_t;
#else
typedef float pcg_t;
#endif

// when enabled ABSOLUTE_QD_PENALTY penalizes qd like controls, rather than penalizing relative distance to precomputed traj
#define ABSOLUTE_QD_PENALTY 0
// #define Q_COST          (.10)
// #define R_COST          (0.0001)

#define CONST_UPDATE_FREQ 1

// this constant controls when xu and goal will be shifted, should be a fraction of a timestep
#define SHIFT_THRESHOLD (1 * timestep)

#if TIME_LINSYS
    #define SQP_MAX_ITER    20
    typedef double toplevel_return_type;
#else
    #define SQP_MAX_ITER    40
    typedef uint32_t toplevel_return_type;
#endif


#define PCG_NUM_THREADS     128
// #define PCG_EXIT_TOL        5e-5
// #define PCG_MAX_ITER        200

#define MERIT_THREADS       128
#define SCHUR_THREADS       128
#define DZ_THREADS          128
#define KKT_THREADS         128


#define RHO_MIN 1e-3


// prints state while tracking
#define LIVE_PRINT_PATH 0
#define LIVE_PRINT_STATS 0
#define LIVE_PRINT_STATS 0

// runs sqp a bunch of times before starting to track
#define REMOVE_JITTERS  1

// where to store test results — manually create this directory
#define SAVE_DATA   0
#define DATA_DIRECTORY   "/tmp/testresults"
