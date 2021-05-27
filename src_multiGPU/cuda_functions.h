
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
void calcBulk(myprec *par1, myprec *par2, myprec *r, myprec *w, myprec *e, Communicator rk);

//global functions
__global__ void RHSDeviceSharedFlxX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dil, myprec *dpdz);
__global__ void RHSDeviceSharedFlxY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dil, myprec *dpdz);
__global__ void RHSDeviceSharedFlxZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dil, myprec *dpdz);

__global__ void calcStressX(myprec *u, myprec *v, myprec *w);
__global__ void calcStressY(myprec *u, myprec *v, myprec *w);
__global__ void calcStressZ(myprec *u, myprec *v, myprec *w);
__global__ void calcStressYBC(myprec *u, myprec *v, myprec *w, int direction);
__global__ void calcStressZBC(myprec *u, myprec *v, myprec *w, int direction);
__global__ void calcDil(myprec *dil);
__global__ void deviceCalcDt(myprec *wrkArray, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu);
__global__ void calcState(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam, int bc);

//derivatives
__device__ void derDev1x(myprec *df , myprec *f, Indices id);
__device__ void derDev1y(myprec *df , myprec *f, Indices id);
__device__ void derDev1z(myprec *df , myprec *f, Indices id);
__device__ void derDev1yBC(myprec *df, myprec *f, Indices id, int direction);
__device__ void derDev1zBC(myprec *df, myprec *f, Indices id, int direction);
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
extern __device__ __forceinline__ void derDevShared1x(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derDevShared2x(myprec *d2f, myprec *s_f, int si);
extern __device__ __forceinline__ void derDevSharedV1x(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derDevSharedV2x(myprec *d2f, myprec *s_f, int si);
extern __device__ __forceinline__ void derDevShared1y(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derDevShared2y(myprec *d2f, myprec *s_f, int si);
extern __device__ __forceinline__ void derDevSharedV1y(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derDevSharedV2y(myprec *d2f, myprec *s_f, int si);
extern __device__ __forceinline__ void derDevShared1z(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derDevShared2z(myprec *d2f, myprec *s_f, int si);
extern __device__ __forceinline__ void derDevSharedV1z(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derDevSharedV2z(myprec *d2f, myprec *s_f, int si);
extern __device__ __forceinline__ void fluxQuadSharedx(myprec *df, myprec *s_f, myprec *s_g, int si);
extern __device__ __forceinline__ void fluxCubeSharedx(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si);
extern __device__ __forceinline__ void fluxQuadSharedy(myprec *df, myprec *s_f, myprec *s_g, int si);
extern __device__ __forceinline__ void fluxCubeSharedy(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si);
extern __device__ __forceinline__ void fluxQuadSharedz(myprec *df, myprec *s_f, myprec *s_g, int si);
extern __device__ __forceinline__ void fluxCubeSharedz(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si);

__device__ __forceinline__ __attribute__((always_inline)) void fluxQuadSharedx(myprec *df, myprec *s_f, myprec *s_g, int si)
{


	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;
	__syncthreads();

	for (int lt=1; lt<stencilSize+1; lt++)
		for (int mt=0; mt<lt; mt++) {
			flxp -= dcoeffF[stencilSize-lt]*(s_f[si-mt]+s_f[si-mt+lt])*(s_g[si-mt]+s_g[si-mt+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si-mt-1]+s_f[si-mt+lt-1])*(s_g[si-mt-1]+s_g[si-mt+lt-1]);
		}

	*df = 0.5*d_dx*(flxm - flxp);

#if nonUniformX
	*df = (*df)*d_xp[si-stencilSize];
#endif

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void fluxCubeSharedx(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si)
{

	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;
	__syncthreads();

	for (int lt=1; lt<stencilSize+1; lt++)
		for (int mt=0; mt<lt; mt++) {
			flxp -= dcoeffF[stencilSize-lt]*(s_f[si-mt]+s_f[si-mt+lt])*(s_g[si-mt]+s_g[si-mt+lt])*(s_h[si-mt]+s_h[si-mt+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si-mt-1]+s_f[si-mt+lt-1])*(s_g[si-mt-1]+s_g[si-mt+lt-1])*(s_h[si-mt-1]+s_h[si-mt+lt-1]);
		}

	*df = 0.25*d_dx*(flxm - flxp);

#if nonUniformX
	*df = (*df)*d_xp[si-stencilSize];
#endif

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void fluxQuadSharedy(myprec *df, myprec *s_f, myprec *s_g, int si)
{


	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;
	__syncthreads();

	for (int lt=1; lt<stencilSize+1; lt++)
		for (int mt=0; mt<lt; mt++) {
			flxp -= dcoeffF[stencilSize-lt]*(s_f[si-mt]+s_f[si-mt+lt])*(s_g[si-mt]+s_g[si-mt+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si-mt-1]+s_f[si-mt+lt-1])*(s_g[si-mt-1]+s_g[si-mt+lt-1]);
		}

	*df = 0.5*d_dy*(flxm - flxp);

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void fluxCubeSharedy(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si)
{

	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;
	__syncthreads();

	for (int lt=1; lt<stencilSize+1; lt++)
		for (int mt=0; mt<lt; mt++) {
			flxp -= dcoeffF[stencilSize-lt]*(s_f[si-mt]+s_f[si-mt+lt])*(s_g[si-mt]+s_g[si-mt+lt])*(s_h[si-mt]+s_h[si-mt+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si-mt-1]+s_f[si-mt+lt-1])*(s_g[si-mt-1]+s_g[si-mt+lt-1])*(s_h[si-mt-1]+s_h[si-mt+lt-1]);
		}

	*df = 0.25*d_dy*(flxm - flxp);

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void fluxQuadSharedz(myprec *df, myprec *s_f, myprec *s_g, int si)
{


	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;
	__syncthreads();

	for (int lt=1; lt<stencilSize+1; lt++)
		for (int mt=0; mt<lt; mt++) {
			flxp -= dcoeffF[stencilSize-lt]*(s_f[si-mt]+s_f[si-mt+lt])*(s_g[si-mt]+s_g[si-mt+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si-mt-1]+s_f[si-mt+lt-1])*(s_g[si-mt-1]+s_g[si-mt+lt-1]);
		}

	*df = 0.5*d_dz*(flxm - flxp);

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void fluxCubeSharedz(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si)
{

	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;
	__syncthreads();

	for (int lt=1; lt<stencilSize+1; lt++)
		for (int mt=0; mt<lt; mt++) {
			flxp -= dcoeffF[stencilSize-lt]*(s_f[si-mt]+s_f[si-mt+lt])*(s_g[si-mt]+s_g[si-mt+lt])*(s_h[si-mt]+s_h[si-mt+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si-mt-1]+s_f[si-mt+lt-1])*(s_g[si-mt-1]+s_g[si-mt+lt-1])*(s_h[si-mt-1]+s_h[si-mt+lt-1]);
		}

	*df = 0.25*d_dz*(flxm - flxp);

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void derDevShared1x(myprec *df, myprec *s_f, int si)
{
	*df = 0.0;
	for (int it=0; it<stencilSize; it++)  {
		*df += dcoeffF[it]*(s_f[si+it-stencilSize]-s_f[si+stencilSize-it]);
	}

	*df = *df*d_dx;

#if nonUniformX
	*df = *df*d_xp[si-stencilSize];
#endif

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void derDevShared2x(myprec *d2f, myprec *s_f, int si)
{


#if nonUniformX
	*d2f = 0.0;
	for (int it=0; it<2*stencilSize+1; it++)  {
		*d2f += dcoeffSx[it*mx+(si-stencilSize)]*(s_f[si+it-stencilSize]);
	}
#else
	*d2f = dcoeffS[stencilSize]*s_f[si]*d_d2x;
	for (int it=0; it<stencilSize; it++)  {
		*d2f += dcoeffS[it]*(s_f[si+it-stencilSize]+s_f[si+stencilSize-it])*d_d2x;
	}
#endif

	__syncthreads();

}

__device__ __forceinline__ __attribute__((always_inline)) void derDevSharedV1x(myprec *df, myprec *s_f, int si)
{
	*df = 0.0;
	for (int it=0; it<stencilVisc; it++)  {
		*df += dcoeffVF[it]*(s_f[si+it-stencilVisc]-s_f[si+stencilVisc-it]);
	}

	*df = *df*d_dx;
#if nonUniformX
	*df = *df*d_xp[si-stencilSize];
#endif

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void derDevSharedV2x(myprec *d2f, myprec *s_f, int si)
{

#if nonUniformX
	*d2f = 0.0;
	for (int it=0; it<2*stencilVisc+1; it++)  {
		*d2f += dcoeffVSx[it*mx+(si-stencilSize)]*(s_f[si+it-stencilVisc]);
	}
#else
	*d2f = dcoeffVS[stencilVisc]*s_f[si]*d_d2x;
	for (int it=0; it<stencilVisc; it++)  {
		*d2f += dcoeffVS[it]*(s_f[si+it-stencilVisc]+s_f[si+stencilVisc-it])*d_d2x;
	}
#endif

	__syncthreads();

}

__device__ __forceinline__ __attribute__((always_inline)) void derDevShared1y(myprec *df, myprec *s_f, int si)
{
	*df = 0.0;
	for (int it=0; it<stencilSize; it++)  {
		*df += dcoeffF[it]*(s_f[si+it-stencilSize]-s_f[si+stencilSize-it])*d_dy;
	}

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void derDevShared2y(myprec *d2f, myprec *s_f, int si)
{

	*d2f = dcoeffS[stencilSize]*s_f[si]*d_d2y;
	for (int it=0; it<stencilSize; it++)  {
		*d2f += dcoeffS[it]*(s_f[si+it-stencilSize]+s_f[si+stencilSize-it])*d_d2y;
	}

	__syncthreads();

}

__device__ __forceinline__ __attribute__((always_inline)) void derDevSharedV1y(myprec *df, myprec *s_f, int si)
{
	*df = 0.0;
	for (int it=0; it<stencilVisc; it++)  {
		*df += dcoeffVF[it]*(s_f[si+it-stencilVisc]-s_f[si+stencilVisc-it])*d_dy;
	}

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void derDevSharedV2y(myprec *d2f, myprec *s_f, int si)
{

	*d2f = dcoeffVS[stencilVisc]*s_f[si]*d_d2y;
	for (int it=0; it<stencilVisc; it++)  {
		*d2f += dcoeffVS[it]*(s_f[si+it-stencilVisc]+s_f[si+stencilVisc-it])*d_d2y;
	}

	__syncthreads();

}

__device__ __forceinline__ __attribute__((always_inline)) void derDevShared1z(myprec *df, myprec *s_f, int si)
{
	*df = 0.0;
	for (int it=0; it<stencilSize; it++)  {
		*df += dcoeffF[it]*(s_f[si+it-stencilSize]-s_f[si+stencilSize-it])*d_dz;
	}

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void derDevShared2z(myprec *d2f, myprec *s_f, int si)
{

	*d2f = dcoeffS[stencilSize]*s_f[si]*d_d2z;
	for (int it=0; it<stencilSize; it++)  {
		*d2f += dcoeffS[it]*(s_f[si+it-stencilSize]+s_f[si+stencilSize-it])*d_d2z;
	}

	__syncthreads();

}

__device__ __forceinline__ __attribute__((always_inline)) void derDevSharedV1z(myprec *df, myprec *s_f, int si)
{
	*df = 0.0;
	for (int it=0; it<stencilVisc; it++)  {
		*df += dcoeffVF[it]*(s_f[si+it-stencilVisc]-s_f[si+stencilVisc-it])*d_dz;
	}

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void derDevSharedV2z(myprec *d2f, myprec *s_f, int si)
{

	*d2f = dcoeffVS[stencilVisc]*s_f[si]*d_d2z;
	for (int it=0; it<stencilVisc; it++)  {
		*d2f += dcoeffVS[it]*(s_f[si+it-stencilVisc]+s_f[si+stencilVisc-it])*d_d2z;
	}

	__syncthreads();

}

#endif
