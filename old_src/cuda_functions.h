
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

    __device__ __host__ void mkidZFlx() {
        i  = bix*bdy + tiy;
        j  = biy;
        k  = tix;
        g = i + j*mx + k*mx*my;
     }
};

void setDerivativeParameters(dim3 &grid, dim3 &block);
void copyField(int direction);
void checkGpuMem();

//global functions
__global__ void RHSDeviceSharedFlxX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil, myprec dpdz);
__global__ void RHSDeviceSharedFlxY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil, myprec dpdz);
__global__ void RHSDeviceSharedFlxZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil, myprec dpdz);
__global__ void RHSDeviceSharedFlxX_old(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil, myprec dpdz);
__global__ void RHSDeviceSharedFlxY_old(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil, myprec dpdz);
__global__ void RHSDeviceSharedFlxZ_old(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil, myprec dpdz);
__global__ void RHSDeviceFullYL(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil);
__global__ void RHSDeviceFullZL(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
			myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
			myprec *t,  myprec *p,  myprec *mu, myprec *lam,
			myprec *sij[9], myprec *dil);
__global__ void FLXDeviceY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil);
__global__ void FLXDeviceZ(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil);

__global__ void runDevice(myprec *kin, myprec *enst, myprec *time);
__global__ void initDevice(myprec *d_fr, myprec *d_fu, myprec *d_fv, myprec *d_fw, myprec *d_fe, double dpdz);
__global__ void getResults(myprec *d_fr, myprec *d_fu, myprec *d_fv, myprec *d_fw, myprec *d_fe);
__global__ void calcStressX(myprec *u, myprec *v, myprec *w, myprec *stress[9]);
__global__ void calcStressY(myprec *u, myprec *v, myprec *w, myprec *stress[9]);
__global__ void calcStressZ(myprec *u, myprec *v, myprec *w, myprec *stress[9]);
__global__ void calcDil(myprec *stress[9], myprec *dil);
__global__ void deviceCalcDt(myprec *wrkArray, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu);

//device functions
__device__ void RHSDevice(myprec *var, myprec *rhs, Indices id);
__device__ void rk4Device(Indices id);
__device__ void initStress();
__device__ void clearStress();
__device__ void initRHS();
__device__ void clearRHS();
__device__ void threadBlockDeviceSynchronize(void);
__device__ void calcIntegrals(myprec *r, myprec *u, myprec *v, myprec *w, myprec *sij[9], myprec *kin, myprec *enst);
__device__ void calcTimeStep(myprec *dt, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu);
__device__ void calcPressureGrad(myprec *dpdx, myprec *r, myprec *w);

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
__device__ void derDevV1yL(myprec *df , myprec *f, Indices id);
__device__ void derDevV1zL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2xL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2yL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2zL(myprec *d2f, myprec *f, Indices id);
__device__ void derDevShared1x(myprec *df , myprec *s_f, int si);
__device__ void derDevShared2x(myprec *d2f, myprec *s_f, int si);
__device__ void derDevSharedV1x(myprec *df , myprec *s_f, int si);
__device__ void derDevSharedV2x(myprec *d2f, myprec *s_f, int si);
__device__ void derDevShared1y(myprec *df , myprec *s_f, int si);
__device__ void derDevShared2y(myprec *d2f, myprec *s_f, int si);
__device__ void derDevSharedV1y(myprec *df , myprec *s_f, int si);
__device__ void derDevSharedV2y(myprec *d2f, myprec *s_f, int si);
__device__ void derDevShared1z(myprec *df , myprec *s_f, int si);
__device__ void derDevShared2z(myprec *d2f, myprec *s_f, int si);
__device__ void derDevSharedV1z(myprec *df , myprec *s_f, int si);
__device__ void derDevSharedV2z(myprec *d2f, myprec *s_f, int si);
__device__ void fluxQuadSharedx(myprec *df, myprec *s_f, myprec *s_g, int si);
__device__ void fluxCubeSharedx(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si);
__device__ void fluxQuadSharedy(myprec *df, myprec *s_f, myprec *s_g, int si);
__device__ void fluxCubeSharedy(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si);
__device__ void fluxQuadSharedz(myprec *df, myprec *s_f, myprec *s_g, int si);
__device__ void fluxCubeSharedz(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si);
#endif
