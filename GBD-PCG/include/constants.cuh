#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#ifndef STATE_SIZE
#define STATE_SIZE  3
#endif

#ifndef KNOT_POINTS
#define KNOT_POINTS  3
#endif


namespace pcg_constants{
    inline constexpr uint32_t DEFAULT_MAX_PCG_ITER = 25;
	template<typename T>
    inline constexpr T DEFAULT_EPSILON = 1e-6;
    inline const dim3 DEFAULT_GRID(128);
    inline const dim3 DEFAULT_BLOCK(64);
}
