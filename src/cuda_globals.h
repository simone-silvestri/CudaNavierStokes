
#ifndef CUDA_GLOBALS_H_
#define CUDA_GLOBALS_H_

#include <stdio.h>
#include <assert.h>
#define myprec float


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


// shared memory tiles will be m*-by-*Pencils
// sPencils is used when each thread calculates the derivative at one point
// lPencils is used for coalescing in y and z where each thread has to 
//     calculate the derivative at mutiple points

#if mx==1 || my==1 || mz==1
const int sPencils = 1;
#else
const int sPencils = 2;  // small # pencils
#endif
#if mx==1 || my==1 || mz==1
const int lPencils = 1;  
#else
const int lPencils = 32;  // large # pencils
#endif

__constant__ myprec dcoeff[stencilSize];
__constant__ myprec d_dt, d_dx;

__device__ myprec d_phi[mx*my*mz];

__device__ myprec d_rhs1[mx*my*mz];
__device__ myprec d_rhs2[mx*my*mz];
__device__ myprec d_rhs3[mx*my*mz];
__device__ myprec d_rhs4[mx*my*mz];
__device__ myprec d_temp[mx*my*mz];

#endif
