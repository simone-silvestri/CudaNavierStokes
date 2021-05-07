
#ifndef CUDA_GLOBALS_H_
#define CUDA_GLOBALS_H_

#include "globals.h"
#include <stdio.h>
#include <assert.h>
#define myprec double


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

inline
__device__ cudaError_t checkCudaDev(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// shared memory tiles will be m*-by-*Pencils
// sPencils is used when each thread calculates the derivative at one point
// lPencils is used for coalescing in y and z where each thread has to 
//   				    calculate the derivative at multiple points

#if mx==1 || my==1 
const int sPencils = 1;
#else
#if mx > 255 ||  my>255 || mz>255
const int sPencils = 1;  // small # pencils
#else
const int sPencils = 2;
#endif
#endif
#if mx==1 || my==1 || mz==1
const int lPencils = 1;  
#else
#if mz > 256 || my > 256
const int lPencils = 8;  // large # pencils
#elif mz > 128 || my > 128
const int lPencils = 16;  // large # pencils
#else
const int lPencils = 32;  // large # pencils
#endif
#endif

extern __constant__ myprec dcoeffF[stencilSize];
extern __constant__ myprec dcoeffS[stencilSize+1];
extern __constant__ myprec dcoeffVF[stencilVisc];
extern __constant__ myprec dcoeffVS[stencilVisc+1];
extern __constant__ myprec dcoeffSx[mx*(2*stencilSize+1)];
extern __constant__ myprec dcoeffVSx[mx*(2*stencilVisc+1)];
extern __constant__ myprec d_dx, d_dy, d_dz, d_d2x, d_d2y, d_d2z, d_x[mx], d_xp[mx], d_dxv[mx];

extern __device__ myprec sij[9][mx*my*mz];

extern dim3 d_block[5], grid0;
extern dim3 d_grid[5], block0;


extern myprec *d_r;
extern myprec *d_u;
extern myprec *d_v;
extern myprec *d_w;
extern myprec *d_e;

extern myprec *dtC,*dpdz;

#endif
