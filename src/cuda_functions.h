
#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

#include "cuda_globals.h"

class Indices {
  public:
    int i,j,k,g,tix,tiy,bix,biy,bdy;

     __host__ __device__ void mkidx(int _tix, int _tiy, int _bix, int _biy, int _bdy) {

    tix = _tix;
    tiy = _tiy;
    bix = _bix;
    biy = _biy;
    bdy = _bdy;

    i  = tix;
    j  = bix*bdy + tiy;
    k  = biy;

    g = i + j*mx + k*mx*my;
  }

};

void setDerivativeParameters(dim3 &grid, dim3 &block);
void copyInit(int direction, dim3 grid, dim3 block); 

__device__ void derivative_x(myprec *df, myprec *f, Indices id);
__global__ void initDevice(myprec *d_f); 
__device__ void RHSDevice(myprec *var, myprec *rhs, Indices id);
__device__ void rk4Device(Indices id); 
__global__ void runDevice(); 
__global__ void getResults(myprec *d_f); 
#endif
