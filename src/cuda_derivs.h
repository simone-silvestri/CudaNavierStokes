
#ifndef CUDA_DERIVS_H_
#define CUDA_DERIVS_H_

extern __device__ __forceinline__ void derDevV1yL(myprec *df , myprec *f, Indices id);
extern __device__ __forceinline__ void derDevV1zL(myprec *d2f, myprec *f, myprec *fref, Indices id, int kNum);
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
extern __device__ __forceinline__ void derDevSharedV1z(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derDevSharedV1z(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derDev1yBC(myprec *df, myprec *f, Indices id, int direction);
extern __device__ __forceinline__ void derDev1zBC(myprec *df, myprec *f, Indices id, int direction);
extern __device__ __forceinline__ void fluxQuadSharedx(myprec *df, myprec *s_f, myprec *s_g, int si);
extern __device__ __forceinline__ void fluxCubeSharedx(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si);
extern __device__ __forceinline__ void fluxQuadSharedy(myprec *df, myprec *s_f, myprec *s_g, int si);
extern __device__ __forceinline__ void fluxCubeSharedy(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si);
extern __device__ __forceinline__ void fluxQuadSharedz(myprec *df, myprec *s_f, myprec *s_g, int si);
extern __device__ __forceinline__ void fluxCubeSharedz(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si);
extern __device__ __forceinline__ void derShared1x_BD(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derShared1x_FD(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derShared1z_BD(myprec *df , myprec *s_f, int si);
extern __device__ __forceinline__ void derShared1z_FD(myprec *df , myprec *s_f, int si);



__device__ __forceinline__ __attribute__((always_inline)) void derShared1x_BD(myprec *df, myprec *s_f, int si)
{
	*df = 1.5*s_f[si]-2*s_f[si-1] + 0.5*s_f[si-2];

	*df = *df*d_dx;

#if nonUniformX
	*df = *df*d_xp[si-stencilSize];
#endif

}
__device__ __forceinline__ __attribute__((always_inline)) void derShared1x_FD(myprec *df, myprec *s_f, int si)
{

	*df = -1.5*s_f[si] + 2*s_f[si+1] - 0.5*s_f[si+2];

	*df = *df*d_dx;

#if nonUniformX
	*df = *df*d_xp[si-stencilSize];
#endif

}

__device__ __forceinline__ __attribute__((always_inline)) void derShared1z_BD(myprec *df, myprec *s_f, int si)
{
	*df = 1.5*s_f[si]-2*s_f[si-1] + 0.5*s_f[si-2];

	*df = *df*d_dz;
}
__device__ __forceinline__ __attribute__((always_inline)) void derShared1z_FD(myprec *df, myprec *s_f, int si)
{

	*df = -1.5*s_f[si] + 2*s_f[si+1] - 0.5*s_f[si+2];

	*df = *df*d_dz;

}



__device__ __forceinline__ __attribute__((always_inline)) void fluxQuadSharedx(myprec *df, myprec *s_f, myprec *s_g, int si)
{
	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;
	for (int lt=1; lt<stencilSize+1; lt++) {
		flxp -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si+lt])*(s_g[si]+s_g[si+lt]);
		flxm -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si-lt])*(s_g[si]+s_g[si-lt]);
	}

	*df = 0.5*d_dx*(flxm - flxp);

#if nonUniformX
*df = (*df)*d_xp[si-stencilSize];
#endif

}

__device__ __forceinline__ __attribute__((always_inline)) void fluxCubeSharedx(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si)
{

	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;

	for (int lt=1; lt<stencilSize+1; lt++) {

			flxp -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si+lt])*(s_g[si]+s_g[si+lt])*(s_h[si]+s_h[si+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si-lt])*(s_g[si]+s_g[si-lt])*(s_h[si]+s_h[si-lt]);
        }
	*df = 0.25*d_dx*(flxm - flxp);

#if nonUniformX
	*df = (*df)*d_xp[si-stencilSize];
#endif

}

__device__ __forceinline__ __attribute__((always_inline)) void fluxQuadSharedy(myprec *df, myprec *s_f, myprec *s_g, int si)
{

	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;
	__syncthreads();

	for (int lt=1; lt<stencilSize+1; lt++) {
                        flxp -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si+lt])*(s_g[si]+s_g[si+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si-lt])*(s_g[si]+s_g[si-lt]);
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

	for (int lt=1; lt<stencilSize+1; lt++) {

			flxp -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si+lt])*(s_g[si]+s_g[si+lt])*(s_h[si]+s_h[si+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si-lt])*(s_g[si]+s_g[si-lt])*(s_h[si]+s_h[si-lt]);
        }
	*df = 0.25*d_dy*(flxm - flxp);

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void fluxQuadSharedz(myprec *df, myprec *s_f, myprec *s_g, int si)
{

	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;

	for (int lt=1; lt<stencilSize+1; lt++) {
		//for (int mt=0; mt<lt; mt++) {
			//flxp -= dcoeffF[stencilSize-lt]*(s_f[si-mt]+s_f[si-mt+lt])*(s_g[si-mt]+s_g[si-mt+lt]);
			//flxm -= dcoeffF[stencilSize-lt]*(s_f[si-mt-1]+s_f[si-mt+lt-1])*(s_g[si-mt-1]+s_g[si-mt+lt-1]);
		flxp -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si+lt])*(s_g[si]+s_g[si+lt]);
		flxm -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si-lt])*(s_g[si]+s_g[si-lt]);

		//}
                 }
	*df = 0.5*d_dz*(flxm - flxp);

}

__device__ __forceinline__ __attribute__((always_inline)) void fluxCubeSharedz(myprec *df, myprec *s_f, myprec *s_g, myprec *s_h, int si)
{

	myprec flxp,flxm;

	flxp = 0.0;
	flxm = 0.0;

	for (int lt=1; lt<stencilSize+1; lt++) {
//		for (int mt=0; mt<lt; mt++) {
//			flxp -= dcoeffF[stencilSize-lt]*(s_f[si-mt]+s_f[si-mt+lt])*(s_g[si-mt]+s_g[si-mt+lt])*(s_h[si-mt]+s_h[si-mt+lt]);
//			flxm -= dcoeffF[stencilSize-lt]*(s_f[si-mt-1]+s_f[si-mt+lt-1])*(s_g[si-mt-1]+s_g[si-mt+lt-1])*(s_h[si-mt-1]+s_h[si-mt+lt-1]);

			flxp -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si+lt])*(s_g[si]+s_g[si+lt])*(s_h[si]+s_h[si+lt]);
			flxm -= dcoeffF[stencilSize-lt]*(s_f[si]+s_f[si-lt])*(s_g[si]+s_g[si-lt])*(s_h[si]+s_h[si-lt]);


//		}
                }  
	*df = 0.25*d_dz*(flxm - flxp);

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
}

__device__ __forceinline__ __attribute__((always_inline)) void derDevSharedV2z(myprec *d2f, myprec *s_f, int si)
{

	*d2f = dcoeffVS[stencilVisc]*s_f[si]*d_d2z;
	for (int it=0; it<stencilVisc; it++)  {
		*d2f += dcoeffVS[it]*(s_f[si+it-stencilVisc]+s_f[si+stencilVisc-it])*d_d2z;
	}

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

/*__device__ __forceinline__ __attribute__((always_inline)) void derDevV1zL(myprec *df, myprec *f, myprec *fref, Indices id, int kNum)
{
  __shared__ myprec s_f[mz/nDivZ+stencilVisc*2][lPencils];

  int i  = id.bix*id.bdx + id.tix;
  int j  = id.biy;
  int si = id.tix;

  for (int k = id.tiy+kNum*mz/nDivZ; k < (kNum+1)*mz/nDivZ; k += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + stencilVisc - kNum*mz/nDivZ;
    s_f[sk][si] = f[globalIdx];
  }

  __syncthreads();

  BCzderVel(s_f,f,fref,id,si,i,j,kNum);
  __syncthreads();

  for (int k = id.tiy+kNum*mz/nDivZ; k < (kNum+1)*mz/nDivZ; k += id.bdy) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + stencilVisc - kNum*mz/nDivZ;
	myprec dftemp = 0.0;
	for (int kt=0; kt<stencilVisc; kt++)  {
		dftemp += dcoeffVF[kt]*(s_f[sk+kt-stencilVisc][si]-s_f[sk+stencilVisc-kt][si])*d_dz;
	}
	df[globalIdx] = dftemp;
  }
  __syncthreads();

}*/

__device__ __forceinline__ __attribute__((always_inline)) void derDev1yBC(myprec *df, myprec *f, Indices id, int direction)
{
	__shared__ myprec s_f[stencilSize*3][sPencils];

	int si = id.tiy;
	int sj = id.tix + stencilSize;

	s_f[sj][si] = f[id.g];

	__syncthreads();

	if(direction==0) {
		s_f[sj-stencilSize][si]  = f[mx*my*mz + id.j + id.i*stencilSize + id.k*mx*stencilSize];
		s_f[sj+stencilSize][si]  = f[id.i + (id.j+stencilSize)*mx + id.k*mx*my];
	} else {
		s_f[sj-stencilSize][si]  = f[id.i + (id.j-stencilSize)*mx + id.k*mx*my];
		s_f[sj+stencilSize][si]  = f[mx*my*mz + stencilSize*mx*mz + id.tix + id.i*stencilSize + id.k*mx*stencilSize];
	}

	__syncthreads();

	myprec dftemp = 0.0;
	for (int jt=0; jt<stencilSize; jt++)  {
		dftemp += dcoeffF[jt]*(s_f[sj+jt-stencilSize][si]-s_f[sj+stencilSize-jt][si])*d_dy;
	}

	df[id.g] = dftemp;
	__syncthreads();

}

__device__ __forceinline__ __attribute__((always_inline)) void derDev1zBC(myprec *df, myprec *f, Indices id, int direction)
{
	__shared__ myprec s_f[stencilSize*3][sPencils];

	int si = id.tiy;
	int sk = id.tix + stencilSize;
	s_f[sk][si] = f[id.g];

	__syncthreads();

	if(direction==0) {
		s_f[sk-stencilSize][si]  = f[mx*my*mz + 2*stencilSize*mx*mz + id.k + id.i*stencilSize + id.j*mx*stencilSize];
		s_f[sk+stencilSize][si]  = f[id.i + id.j*mx + (id.k+stencilSize)*mx*my];
	} else {
		s_f[sk-stencilSize][si]  = f[id.i + id.j*mx + (id.k-stencilSize)*mx*my];
		s_f[sk+stencilSize][si]  = f[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + id.tix + id.i*stencilSize + id.j*mx*stencilSize];
	}

	__syncthreads();

	myprec dftemp = 0.0;
	for (int kt=0; kt<stencilSize; kt++)  {
		dftemp += dcoeffF[kt]*(s_f[sk+kt-stencilSize][si]-s_f[sk+stencilSize-kt][si])*d_dz;
	}

	df[id.g] = dftemp;

	__syncthreads();
}

#endif
