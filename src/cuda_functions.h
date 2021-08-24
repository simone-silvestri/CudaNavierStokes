
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
void calcBulk(myprec *par1, myprec *par2, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, Communicator rk);

//global functions
__global__ void deviceRHSX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dil, myprec *dpdz);
__global__ void deviceRHSY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dil, myprec *dpdz);
__global__ void deviceRHSZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dil, myprec *dpdz);

__global__ void derVelX(myprec *u, myprec *v, myprec *w);
__global__ void derVelY(myprec *u, myprec *v, myprec *w);
__global__ void derVelZ(myprec *u, myprec *v, myprec *w);
__global__ void derVelYBC(myprec *u, myprec *v, myprec *w, int direction);
__global__ void derVelZBC(myprec *u, myprec *v, myprec *w, int direction);
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
__device__ void derDev2xL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2yL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2zL(myprec *d2f, myprec *f, Indices id);
extern __device__ __forceinline__ void derDevV1yL(myprec *df , myprec *f, Indices id);
extern __device__ __forceinline__ void derDevV1zL(myprec *d2f, myprec *f, Indices id);
extern __device__ __forceinline__ void derDev1yL(myprec *df , myprec *f, Indices id);
extern __device__ __forceinline__ void derDev1zL(myprec *d2f, myprec *f, Indices id);
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

__device__ __forceinline__ __attribute__((always_inline)) void derDev1yL(myprec *df, myprec *f, Indices id)
{
  __shared__ myprec s_f[my+stencilSize*2][lPencils];

  int i  = id.bix*id.bdx + id.tix;
  int k  = id.biy;
  int si = id.tix;

  for (int j = id.tiy; j < my; j += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + stencilSize;
    s_f[sj][si] = f[globalIdx];
  }

  __syncthreads();

  int sj = id.tiy + stencilSize;
  if (sj < stencilSize*2) {
	  if(multiGPU) {
		  int j = sj - stencilSize;
		  s_f[sj-stencilSize][si]  = f[mx*my*mz + j + i*stencilSize + k*mx*stencilSize];
		  s_f[sj+my][si]           = f[mx*my*mz + stencilSize*mx*mz + j + i*stencilSize + k*mx*stencilSize];
	  } else {
		  s_f[sj-stencilSize][si]  = s_f[sj+my-stencilSize][si];
		  s_f[sj+my][si] 		   = s_f[sj][si];
	  }
  }

  __syncthreads();

  for (int j = id.tiy; j < my; j += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + stencilSize;
	myprec dftemp = 0.0;
	for (int jt=0; jt<stencilSize; jt++)  {
		dftemp += dcoeffF[jt]*(s_f[sj+jt-stencilSize][si]-s_f[sj+stencilSize-jt][si])*d_dy;
	}
	df[globalIdx] = dftemp;
  }
  __syncthreads();

}


__device__ __forceinline__ __attribute__((always_inline)) void derDev1zL(myprec *df, myprec *f, Indices id)
{
  __shared__ myprec s_f[mz+stencilSize*2][lPencils];

  int i  = id.bix*id.bdx + id.tix;
  int j  = id.biy;
  int si = id.tix;

  for (int k = id.tiy; k < mz; k += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + stencilSize;
    s_f[sk][si] = f[globalIdx];
  }

  __syncthreads();

  int sk = id.tiy + stencilSize;
  if (sk < stencilSize*2) {
	  if(multiGPU) {
		  int k = sk - stencilSize;
		  s_f[sk-stencilSize][si]  = f[mx*my*mz + 2*stencilSize*mx*mz + k + i*stencilSize + j*mx*stencilSize];
		  s_f[sk+mz][si]           = f[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + k + i*stencilSize + j*mx*stencilSize];
	  } else {
		  if(boundaryLayer) {
			  s_f[sk+mz][si]           = 2.0*s_f[mz+stencilSize-1][si] - s_f[mz+2*stencilSize-sk-2][si];
			  s_f[sk-stencilSize][si]  = 2.0*s_f[stencilSize][si]      - s_f[3*stencilSize-sk][si]; //here we assume that the boundary is at stencilSize (at the node not at the face)
		  } else {
			  s_f[sk-stencilSize][si]  = s_f[sk+mz-stencilSize][si];
			  s_f[sk+mz][si]           = s_f[sk][si];
		  }
	  }
  }

  __syncthreads();

  for (int k = id.tiy; k < mz; k += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + stencilSize;
	myprec dftemp = 0.0;
	for (int kt=0; kt<stencilSize; kt++)  {
		dftemp += dcoeffF[kt]*(s_f[sk+kt-stencilSize][si]-s_f[sk+stencilSize-kt][si])*d_dz;
	}
	df[globalIdx] = dftemp;
  }
  __syncthreads();

}

__device__ __forceinline__ __attribute__((always_inline)) void derDevV1yL(myprec *df, myprec *f, Indices id)
{
  __shared__ myprec s_f[my+stencilVisc*2][lPencils];

  int i  = id.bix*id.bdx + id.tix;
  int k  = id.biy;
  int si = id.tix;

  for (int j = id.tiy; j < my; j += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + stencilVisc;
    s_f[sj][si] = f[globalIdx];
  }

  __syncthreads();

  int sj = id.tiy + stencilVisc;
  if (sj < stencilVisc*2) {
	  if(multiGPU) {
		  int j = sj - stencilVisc;
		  s_f[sj-stencilVisc][si]  = f[mx*my*mz + j + i*stencilSize + k*mx*stencilSize];
		  s_f[sj+my][si]           = f[mx*my*mz + stencilSize*mx*mz + j + i*stencilSize + k*mx*stencilSize];
	  } else {
		  s_f[sj-stencilVisc][si]  = s_f[sj+my-stencilVisc][si];
		  s_f[sj+my][si] 		   = s_f[sj][si];
	  }
  }
  __syncthreads();

  for (int j = id.tiy; j < my; j += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + stencilVisc;
	myprec dftemp = 0.0;
	for (int jt=0; jt<stencilVisc; jt++)  {
		dftemp += dcoeffVF[jt]*(s_f[sj+jt-stencilVisc][si]-s_f[sj+stencilVisc-jt][si])*d_dy;
	}
	df[globalIdx] = dftemp;
  }
  __syncthreads();

}

__device__ __forceinline__ __attribute__((always_inline)) void derDevV1zL(myprec *df, myprec *f, Indices id)
{
  __shared__ myprec s_f[mz+stencilVisc*2][lPencils];

  int i  = id.bix*id.bdx + id.tix;
  int j  = id.biy;
  int si = id.tix;

  for (int k = id.tiy; k < mz; k += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + stencilVisc;
    s_f[sk][si] = f[globalIdx];
  }

  __syncthreads();

  int sk = id.tiy + stencilVisc;
  if (sk < stencilVisc*2) {
	  if(multiGPU) {
		  int k = sk - stencilVisc;
		  s_f[sk-stencilVisc][si]  = f[mx*my*mz + 2*stencilSize*mx*mz + k + i*stencilSize + j*mx*stencilSize];
		  s_f[sk+mz][si]           = f[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + k + i*stencilSize + j*mx*stencilSize];
	  } else {
		  if(boundaryLayer) {
			  s_f[sk+mz][si]           = 2.0*s_f[mz+stencilVisc-1][si] - s_f[mz+2*stencilVisc-sk-2][si];
			  s_f[sk-stencilVisc][si]  = 2.0*s_f[stencilVisc][si]      - s_f[3*stencilVisc-sk][si]; //here we assume that the boundary is at stencilSize (at the node not at the face)
		  } else {
			  s_f[sk-stencilVisc][si]  = s_f[sk+mz-stencilVisc][si];
			  s_f[sk+mz][si]           = s_f[sk][si];
		  }
	  }
  }

  __syncthreads();

  for (int k = id.tiy; k < mz; k += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + stencilVisc;
	myprec dftemp = 0.0;
	for (int kt=0; kt<stencilVisc; kt++)  {
		dftemp += dcoeffVF[kt]*(s_f[sk+kt-stencilVisc][si]-s_f[sk+stencilVisc-kt][si])*d_dz;
	}
	df[globalIdx] = dftemp;
  }
  __syncthreads();

}



#endif
