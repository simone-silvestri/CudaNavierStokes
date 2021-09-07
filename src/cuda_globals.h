
#ifndef CUDA_GLOBALS_H_
#define CUDA_GLOBALS_H_

#include "globals.h"
#include <stdio.h>
#include <assert.h>


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
// sPencils (small # pencils) is used when each thread calculates the derivative at one point
// sPencils is used for deviceRHSX, deviceRHSY, deviceRHSZ, derVelX

// lPencils (large # pencils) is used for coalescing in y and z where each thread has to
//   				    calculate the derivative at multiple points
// lPencils is used only for derVelY and derVelZ

#if mx==1 || my==1 
const int sPencils = 1;
#else
#if mx+2*stencilSize > 279 ||  my+2*stencilSize>600 || mz/nDivZ+2*stencilSize>600
const int sPencils = 1;
#else
const int sPencils = 2;
#endif
#endif
#if mx<=32 || my<=32 || mz<=32
const int lPencils = 1;  
#else
#if mz/nDivZ > 512 || my > 512
const int lPencils = 4;  // large # pencils
#elif mz/nDivZ > 256 || my > 256
const int lPencils = 8;  // large # pencils
#elif mz/nDivZ > 128 || my > 128
const int lPencils = 16;  // large # pencils
#else
const int lPencils = 32;
#endif
#endif

extern __constant__ myprec dcoeffF[stencilSize];
extern __constant__ myprec dcoeffS[stencilSize+1];
extern __constant__ myprec dcoeffVF[stencilVisc];
extern __constant__ myprec dcoeffVS[stencilVisc+1];

#if mx<=546 //limit on the GPU constant memory usage (655356 bytes)
extern __constant__ myprec dcoeffSx[mx*(2*stencilSize+1)];
extern __constant__ myprec dcoeffVSx[mx*(2*stencilVisc+1)];
#else
extern __device__ myprec dcoeffSx[mx*(2*stencilSize+1)];
extern __device__ myprec dcoeffVSx[mx*(2*stencilVisc+1)];
#endif
extern __constant__ myprec d_dx, d_dy, d_dz, d_d2x, d_d2y, d_d2z, d_x[mx], d_xp[mx], d_dxv[mx];

extern __device__ myprec time_on_GPU;
extern __device__ Communicator rkGPU;

extern dim3 d_block[5], grid0,  gridBC,  gridHalo,  gridHaloY,  gridHaloZ;
extern dim3 d_grid[5], block0, blockBC, blockHalo, blockHaloY, blockHaloZ;

extern cudaStream_t s[8+nDivZ];

extern myprec *d_r;
extern myprec *d_u;
extern myprec *d_v;
extern myprec *d_w;
extern myprec *d_e;

extern myprec *d_rO;
extern myprec *d_eO;
extern myprec *d_uO;
extern myprec *d_vO;
extern myprec *d_wO;

extern myprec *d_h;
extern myprec *d_t;
extern myprec *d_p;
extern myprec *d_m;
extern myprec *d_l;

extern myprec *d_dil;

extern myprec *d_rhsr1;
extern myprec *d_rhsu1;
extern myprec *d_rhsv1;
extern myprec *d_rhsw1;
extern myprec *d_rhse1;

extern myprec *d_rhsr2;
extern myprec *d_rhsu2;
extern myprec *d_rhsv2;
extern myprec *d_rhsw2;
extern myprec *d_rhse2;

extern myprec *d_rhsr3;
extern myprec *d_rhsu3;
extern myprec *d_rhsv3;
extern myprec *d_rhsw3;
extern myprec *d_rhse3;

extern myprec *gij[9];

extern myprec *dtC,*dpdz;

extern myprec *djm, *djp, *dkm, *dkp;
extern myprec *djm5,*djp5,*dkm5,*dkp5;

extern myprec *senYp,*senYm,*senZp,*senZm;
extern myprec *rcvYp,*rcvYm,*rcvZp,*rcvZm;

extern myprec *senYp5,*senYm5,*senZp5,*senZm5;
extern myprec *rcvYp5,*rcvYm5,*rcvZp5,*rcvZm5;

#endif
