
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
#if parentGrid == 0
       mkidX();
#elif parentGrid == 1
       mkidY();
#else
       mkidZ();
#endif
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


//global functions
__global__ void RHSDeviceX(myprec *rhs, myprec *var); // using s-pencil derivatives
__global__ void RHSDeviceY(myprec *rhs, myprec *var);
__global__ void RHSDeviceZ(myprec *rhs, myprec *var);
__global__ void RHSDeviceXL(myprec *rhsX, myprec *var);  // using L-pencil derivatives
__global__ void RHSDeviceYL(myprec *rhsY, myprec *var);
__global__ void RHSDeviceZL(myprec *rhsZ, myprec *var);

__global__ void RHSDeviceYSum(myprec *rhs, myprec *var); // sums to previous rhs
__global__ void RHSDeviceZSum(myprec *rhs, myprec *var); // sums to previous rhs
__global__ void runDevice(); 
__global__ void getResults(myprec *d_f); 
__global__ void initDevice(myprec *d_f);

//device functions
__device__ void RHSDevice(myprec *var, myprec *rhs, Indices id);
__device__ void rk4Device(Indices id);

//derivatives
__device__ void derDev1x(myprec *df , myprec *f, Indices id);
__device__ void derDev1y(myprec *df , myprec *f, Indices id);
__device__ void derDev1z(myprec *df , myprec *f, Indices id);
__device__ void derDev1xL(myprec *df, myprec *f, Indices id);
__device__ void derDev1yL(myprec *df, myprec *f, Indices id);
__device__ void derDev1zL(myprec *df, myprec *f, Indices id);
__device__ void derDev2x(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2y(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2z(myprec *d2f, myprec *f, Indices id);

//if dividing the whole rk step in the different directions
__global__ void rkStepZ(myprec *rhs1, myprec *rhs2, myprec *rhs3, myprec *rhs4, myprec *temp, myprec *phi, myprec *dt);


//If running the rk4 on the host side
void runHost();

#endif
