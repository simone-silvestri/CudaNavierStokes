
#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

#include "cuda_globals.h"

class Indices {
  public:
    int i,j,k,g,tix,tiy,bix,biy,bdx,bdy;

    __device__ __host__ Indices(int _tix, int _tiy, int _bix, int _biy, int _bdx, int _bdy) {   
       tix = _tix;
       tiy = _tiy;
       bix = _bix;
       biy = _biy;
       bdx = _bdx;
       bdy = _bdy;
    }

    __device__ __host__ void mkidX() {
       i  = tix;
       j  = bix*bdy + tiy;
       k  = biy;
       g = i + j*mx + k*mx*my;
    }

    __device__ __host__ void mkidY() {
       i  = bix*bdx + tix;
       j  = tiy;
       k  = biy;
       g = i + j*mx + k*mx*my;
    }

    __device__ __host__ void mkidZ() {
       i  = bix*bdx + tix;
       j  = biy;
       k  = tiy;
       g = i + j*mx + k*mx*my;
    }

};

void setDerivativeParameters(dim3 &grid, dim3 &block);
void copyInit(int direction, dim3 grid, dim3 block); 

__device__ void derDev1x(myprec *df , myprec *f, Indices id);
__device__ void derDev1y(myprec *df , myprec *f, Indices id);
__device__ void derDev2x(myprec *d2f, myprec *f, Indices id);
__global__ void initDevice(myprec *d_f); 
__device__ void RHSDevice(myprec *var, myprec *rhs, Indices id);
__device__ void rk4Device(Indices id); 
__global__ void runDevice(); 
__global__ void getResults(myprec *d_f); 

#endif
