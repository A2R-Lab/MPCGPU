#pragma once

// #define KNOT_POINTS 32


#ifdef TIME_LINSYS
#else
#define TIME_LINSYS 1
#endif

// prints state while tracking
#define LIVE_PRINT_PATH 0
#define LIVE_PRINT_STATS 0
#define LIVE_PRINT_STATS 0

#define ADD_NOISE  0
#define TEST_ITERS 1

// where to store test results — manually create this directory
#define SAVE_DATA   0
#define DATA_DIRECTORY   "/tmp/testresults"

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

// runs sqp a bunch of times before starting to track
#define REMOVE_JITTERS  1

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


#ifdef PCG_SOLVE
#else
#define PCG_SOLVE 1 
#endif

// Constants found using experiments
#ifdef PCG_MAX_ITER
#else 
	#if PCG_SOLVE
		#if KNOT_POINTS == 32
#define PCG_MAX_ITER 173 
		#elif KNOT_POINTS == 64
#define PCG_MAX_ITER 167
		#elif KNOT_POINTS == 128
#define PCG_MAX_ITER 167
		#elif KNOT_POINTS == 256
#define PCG_MAX_ITER 118
		#elif KNOT_POINTS == 512
#define PCG_MAX_ITER 67
		#else
#define PCG_MAX_ITER 200	
		#endif	
	#else 
#define PCG_MAX_ITER -1
#define PCG_EXIT_TOL -1 
	#endif

#endif



#define MERIT_THREADS       128
#define SCHUR_THREADS       128
#define DZ_THREADS          128
#define KKT_THREADS         128


#define RHO_MIN 1e-3

//TODO: get rid of rho in defines
#ifdef RHO_FACTOR
#else
#define RHO_FACTOR 1.2 
#endif

#ifdef RHO_MAX
#else
#define RHO_MAX 10 
#endif

//TODO: get rid of sqp in defines
#ifdef SQP_MAX_TIME_US
#else
#define SQP_MAX_TIME_US 2000 
#endif






#ifdef SIMULATION_PERIOD
#else
#define SIMULATION_PERIOD 2000
#endif


#ifdef KNOT_POINTS
#else
#define KNOT_POINTS 32 
#endif

#define STATE_SIZE  14

#ifdef R_COST
#else
	#if KNOT_POINTS == 64
#define R_COST .001 
	#else 
#define R_COST .0001 
	#endif
#endif

#ifdef QD_COST
#else
#define QD_COST .0001 
#endif


