#pragma once



#ifndef KNOT_POINTS
#define KNOT_POINTS 32 
#endif

// default value is for iiwa arm 
#ifndef STATE_SIZE
#define STATE_SIZE  14
#endif


/*******************************************************************************
 *                           Print Settings                               *
 *******************************************************************************/


#ifndef LIVE_PRINT_PATH
#define LIVE_PRINT_PATH 0
#endif 

#ifndef LIVE_PRINT_STATS
#define LIVE_PRINT_STATS 0
#endif

/*******************************************************************************
 *                           Test Settings                               *
 *******************************************************************************/


#ifndef TEST_ITERS
#define TEST_ITERS 1
#endif

#ifndef SAVE_DATA
#define SAVE_DATA   0
#endif 

#ifndef USE_DOUBLES
#define USE_DOUBLES 0
#endif

#if USE_DOUBLES
typedef double linsys_t;
#else
typedef float linsys_t;
#endif

/*******************************************************************************
 *                           MPC Settings                               *
 *******************************************************************************/


// Reference/integration timestep (seconds). The reference trajectory is spaced at TIMESTEP and the
// simulator integrates the real robot with the same TIMESTEP, so the generator (tools/gen_reference.cu)
// and the tracker MUST share this value — both read it from here. Aligned to GATO's iiwa14 fig8 (0.01).
#ifndef TIMESTEP
#define TIMESTEP 0.01
#endif

#ifndef CONST_UPDATE_FREQ
#define CONST_UPDATE_FREQ 1
#endif

// runs sqp a bunch of times before starting to track
#ifndef REMOVE_JITTERS
#define REMOVE_JITTERS  1
#endif

// this constant controls when xu and goal will be shifted, should be a fraction of a timestep
#ifndef SHIFT_THRESHOLD
#define SHIFT_THRESHOLD (1 * timestep)
#endif

#ifndef SIMULATION_PERIOD
#define SIMULATION_PERIOD 2000
#endif

#ifndef MERIT_THREADS
#define MERIT_THREADS       128
#endif 

// when enabled ABSOLUTE_QD_PENALTY penalizes qd like controls, rather than penalizing relative distance to precomputed traj
#ifndef ABSOLUTE_QD_PENALTY
#define ABSOLUTE_QD_PENALTY 0
#endif 


// ---------------------------------------------------------------------------------------------
// Cost weights for grid_plant::tracking_cost (the unified GATO/MPCGPU cost recipe: EE-position +
// quadratic state/input regularization + joint/velocity/torque barriers). Defaults are GATO's
// iiwa14 FIG8 values so MPCGPU, GATO, and the CPU baseline solve the IDENTICAL problem. All are
// -D overridable. EE_COST is the running EE-position weight; N_COST is the (stronger) terminal
// weight — the terminal emphasis the legacy EE-only cost lacked.
#ifndef EE_COST
#define EE_COST 2.0          // q_cost: running EE-position weight
#endif
#ifndef N_COST
#define N_COST 50.0          // terminal EE-position weight
#endif
#ifndef Q_COST
#define Q_COST 0.0           // joint-POSITION cost toward q_nom (0 => EE-only). A small value makes the
                             // state Hessian full-rank PD (the EE-position term alone is rank<=3 → the
                             // Schur complement is only PSD → cooperative PCG can break down on it); a
                             // larger value turns this into a joint-SPACE tracking cost. See iiwa plant.
#endif
#ifndef QD_COST
#define QD_COST 1e-2         // joint-velocity regularization
#endif
#ifndef U_COST
#define U_COST 2e-6          // control regularization
#endif
#ifndef Q_LIM_COST
#define Q_LIM_COST 0.01      // joint-position barrier weight
#endif
#ifndef VEL_LIM_COST
#define VEL_LIM_COST 0.0     // joint-velocity barrier weight
#endif
#ifndef CTRL_LIM_COST
#define CTRL_LIM_COST 0.0    // control barrier weight
#endif

// Legacy alias: the old EE-only cost used R_COST for the control weight. Kept for any external -D.
#ifndef R_COST
#define R_COST U_COST
#endif



/*******************************************************************************
 *                           Linsys Settings                               *
 *******************************************************************************/


/* time_linsys = 1 to record linear system solve times. 
time_linsys = 0 to record number of sqp iterations. 
In both cases, the tracking error will also be recorded. */
    
#ifndef TIME_LINSYS
#define TIME_LINSYS 1
#endif

#ifndef PCG_NUM_THREADS
#define PCG_NUM_THREADS	128
#endif


/* LINSYS_SOLVE = 1 uses pcg as the underlying linear system solver
LINSYS_SOLVE = 0 uses qdldl as the underlying linear system solver */

#ifndef LINSYS_SOLVE
#define LINSYS_SOLVE 1 
#endif

// Values found using experiments
#ifndef PCG_MAX_ITER
	#if LINSYS_SOLVE
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

// Relative tolerance on the preconditioned residual: PCG stops when
// |eta| < pcg_exit_tol + PCG_RES_TOL*|eta_init| (matches glass::pcg's rel_tol).
// A relative test is scale-invariant; an absolute-only threshold over-solves
// large-RHS (moving-reference) systems by thousands of iterations.
#ifndef PCG_RES_TOL
#define PCG_RES_TOL 1e-5
#endif

// Integrator used by the SOLVER's prediction model (KKT linearization + merit defect):
// 0 = explicit Euler (MPCGPU historic), 1 = semi-implicit Euler, 2 = trapezoidal (GATO's
// default). The mpcsim ground-truth simulator is unaffected (fine-substep Euler).
#ifndef MPCGPU_INTEGRATOR
#define MPCGPU_INTEGRATOR 0
#endif

// JOINT_COST_MODE = 1 drives a per-knot STATE goal (d_xs_goal, derived from the reference
// trajectory) into the tracking cost, enabling joint-space tracking (set Q_COST>0, EE_COST=0).
// 0 => EE-only cost (d_xs_goal stays nullptr; the cost falls back to the constant Q_NOM posture).
#ifndef JOINT_COST_MODE
#define JOINT_COST_MODE 0
#endif


/*******************************************************************************
 *                           SQP Settings                               *
 *******************************************************************************/


// SQP_MAX_ITER caps the outer SQP iterations per control step. Default is MPCGPU's native
// time-budget-oriented cap (20 when timing linsys, 40 otherwise); override with -DSQP_MAX_ITER=1
// for the real-time-iteration (one-SQP-step-per-tick) regime GATO and the CPU baseline use, so all
// three do the SAME amount of outer work in the fair 3-way comparison.
#ifndef SQP_MAX_ITER
    #if TIME_LINSYS == 1
        #define SQP_MAX_ITER    20
    #else
        #define SQP_MAX_ITER    40
    #endif
#endif

#if TIME_LINSYS == 1
    typedef double toplevel_return_type;
#else
    typedef uint32_t toplevel_return_type;
#endif


#ifndef SQP_MAX_TIME_US
#define SQP_MAX_TIME_US 2000 
#endif

#ifndef SCHUR_THREADS
#define SCHUR_THREADS       128
#endif 

#ifndef DZ_THREADS
#define DZ_THREADS          128
#endif 

#ifndef KKT_THREADS
#define KKT_THREADS         128
#endif



/*******************************************************************************
 *                           Rho Settings                               *
 *******************************************************************************/



#ifndef RHO_MIN
#define RHO_MIN 1e-3
#endif

// Initial (and reset) SQP regularization rho. Overridable to probe how much regularization the stiff
// Schur system needs for the cooperative PCG to stay well-conditioned.
#ifndef RHO_INIT
#define RHO_INIT 1e-3
#endif

//TODO: get rid of rho in defines
#ifndef RHO_FACTOR
#define RHO_FACTOR 1.2 
#endif

#ifndef RHO_MAX
#define RHO_MAX 10 
#endif



