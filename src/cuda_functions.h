
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
       mkidX();
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

    __device__ __host__ void mkidYFlx() {
        i  = bix*bdy + tiy;
        j  = tix;
        k  = biy;
        g = i + j*mx + k*mx*my;
     }

    __device__ __host__ void mkidZFlx(int kNum) {
        i  = bix*bdy + tiy;
        j  = biy;
        k  = tix + kNum*mz/nDivZ;
        g = i + j*mx + k*mx*my;
     }

    __device__ __host__ void mkidYBC(int dir) {
        i  = bix*bdy + tiy;
        if(dir==1) {
        	j  = my - stencilSize + tix;
        } else {
        	j = tix; }
        k  = biy;
        g = i + j*mx + k*mx*my;
     }

    __device__ __host__ void mkidZBC(int dir) {
        i  = bix*bdy + tiy;
        j  = biy;
        if(dir==1) {
        	k  = mz - stencilSize + tix;
        } else {
        	k = tix; }
        g = i + j*mx + k*mx*my;
     }
};

void calcTimeStep(myprec *dt, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu, Communicator rk);
void calcPressureGrad(myprec *dpdx, myprec *r, myprec *w, Communicator rk);
void calcBulk(myprec *par1, myprec *par2, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, Communicator rk);

#include "boundary_condition_z.h"
#include "cuda_derivs.h"

//global functions
__global__ void deviceRHSX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dudx, myprec *dvdx, myprec *dwdx, myprec *dudy, myprec *dudz,
		myprec *dil, myprec *dpdz, int iNum);
__global__ void deviceRHSY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dvdx, myprec *dudy, myprec *dvdy, myprec *dwdy, myprec *dvdz,
		myprec *dil, myprec *dpdz, int jNum);
__global__ void deviceRHSZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dwdx, myprec *dwdy, myprec *dudz, myprec *dvdz, myprec *dwdz,
		myprec *dil, myprec *dpdz, int kNum);

__global__ void derVelX(myprec *u, myprec *v, myprec *w, myprec *dudx, myprec *dvdx, myprec *dwdx);
__global__ void derVelY(myprec *u, myprec *v, myprec *w, myprec *dudy, myprec *dvdy, myprec *dwdy);
__global__ void derVelZ(myprec *u, myprec *v, myprec *w, myprec *dudz, myprec *dvdz, myprec *dwdz, int kNum);
__global__ void derVelYBC(myprec *u, myprec *v, myprec *w, myprec *dudy, myprec *dvdy, myprec *dwdy, int direction);
__global__ void derVelZBC(myprec *u, myprec *v, myprec *w, myprec *dudz, myprec *dvdz, myprec *dwdz, int direction);
__global__ void calcDil(myprec *dil, myprec *dudx, myprec *dvdy, myprec *dwdz);
__global__ void deviceCalcDt(myprec *wrkArray, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu);
__global__ void calcState(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam, int bc);
__global__ void deviceAdvanceTime(myprec *dt);

//derivatives
__device__ void derDev1x(myprec *df , myprec *f, Indices id);
__device__ void derDev1y(myprec *df , myprec *f, Indices id);
__device__ void derDev1z(myprec *df , myprec *f, Indices id);
__device__ void derDev2x(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2y(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2z(myprec *d2f , myprec *f, Indices id);
__device__ void derDev1xL(myprec *df , myprec *f, Indices id);
__device__ void derDev2xL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2yL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2zL(myprec *d2f, myprec *f, Indices id);

#endif
