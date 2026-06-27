#pragma once
#include <stdio.h>

// Defer to the consumer's gpuErrchk/gpuAssert if one is already defined (e.g. GRiD's
// grid.cuh defines both unconditionally). Guarding on the gpuErrchk macro avoids an
// ODR clash when a TU includes both this header and grid.cuh.
#ifndef gpuErrchk
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif