
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
__global__ void RHSDeviceX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX, 
						   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
						   myprec *t,  myprec *p, myprec *mu, myprec *lam);
__global__ void RHSDeviceY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY, 
						   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
						   myprec *t,  myprec *p, myprec *mu, myprec *lam);
__global__ void RHSDeviceZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ, 
						   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
						   myprec *t,  myprec *p, myprec *mu, myprec *lam);
__global__ void RHSDeviceZL(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
							myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
							myprec *t,  myprec *p, myprec *mu, myprec *lam);
__global__ void RHSDeviceYL(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
							myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
							myprec *t,  myprec *p, myprec *mu, myprec *lam);
__global__ void RHSDeviceSharedX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
								 myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
								 myprec *t,  myprec *p,  myprec *mu, myprec *lam);

__global__ void RHSDeviceFlxX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
							  myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
							  myprec *t,  myprec *p,  myprec *mu, myprec *lam);

__global__ void RHSDeviceXL(myprec *rhsX, myprec *var);  // using L-pencil derivatives

__global__ void RHSDeviceYSum(myprec *rhs, myprec *var); // sums to previous rhs
__global__ void RHSDeviceZSum(myprec *rhs, myprec *var); // sums to previous rhs
__global__ void runDevice(); 
__global__ void getResults(myprec *d_fr, myprec *d_fu, myprec *d_fv, myprec *d_fw, myprec *d_fe);
__global__ void initDevice(myprec *d_fr, myprec *d_fu, myprec *d_fv, myprec *d_fw, myprec *d_fe);

//device functions
__device__ void RHSDevice(myprec *var, myprec *rhs, Indices id);
__device__ void rk4Device(Indices id);

//derivatives
__device__ void derDev1x(myprec *df , myprec *f, Indices id);
__device__ void derDev1y(myprec *df , myprec *f, Indices id);
__device__ void derDev1z(myprec *df , myprec *f, Indices id);
__device__ void derDev2x(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2y(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2z(myprec *d2f , myprec *f, Indices id);
__device__ void derDev1xL(myprec *df , myprec *f, Indices id);
__device__ void derDev1yL(myprec *df , myprec *f, Indices id);
__device__ void derDev1zL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2xL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2yL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2zL(myprec *d2f, myprec *f, Indices id);
__device__ void derDevShared1x(myprec *df , myprec *s_f, int si);
__device__ void derDevShared2x(myprec *d2f, myprec *s_f, int si);
__device__ void fluxCubex(myprec *df, myprec *f, myprec *g, myprec *h, Indices id);

#endif
