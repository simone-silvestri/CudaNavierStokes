#ifndef BOUNDARY_CONDITION_H_
#define BOUNDARY_CONDITION_H_

#include "boundary.h"
#include "sponge.h"

extern __device__ __forceinline__ void BCzderVel(myprec s_f[mz+stencilSize*2][lPencils], myprec *f, myprec *fref, Indices id, int si, int i, int j);

extern __device__ __forceinline__ void BCzNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_m, myprec *s_dil,
												  myprec *u, myprec *v, myprec *w,
												  myprec *m, myprec *dil,
												  Indices id, int si, int n);

extern __device__ __forceinline__ void BCzNumber2(myprec *s_p, myprec *p, Indices id, int si, int m);

extern __device__ __forceinline__ void BCzNumber3(myprec *s_l, myprec *s_t,
												  myprec *l,   myprec *t,
												  Indices id, int si, int n);

extern __device__ __forceinline__ void BCzNumber4(myprec *s_r, myprec *s_h,
												  myprec *r,   myprec *h,
												  Indices id, int si, int n);

__device__ __forceinline__ __attribute__((always_inline)) void BCzderVel(myprec s_f[mz+stencilSize*2][lPencils], myprec *f, myprec *fref, Indices id, int si, int i, int j) {
	int sk = id.tiy + stencilVisc;
	if (sk < stencilVisc*2) {
		if(multiGPU) {
			int k = sk - stencilVisc;
			s_f[sk-stencilVisc][si]  = f[mx*my*mz + 2*stencilSize*mx*mz + k + i*stencilSize + j*mx*stencilSize];
			s_f[sk+mz][si]           = f[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + k + i*stencilSize + j*mx*stencilSize];
		} else {
			if(boundaryLayer) {
				//extrapolation
				s_f[sk+mz][si]           = 2.0*s_f[mz+stencilVisc-1][si] - s_f[mz+2*stencilVisc-sk-2][si];
				//extrapolation on reference solution
//				s_f[sk-stencilVisc][si]  = 2.0*fref[i]              - s_f[3*stencilVisc-sk-1][si];
				//extrapolation
				s_f[sk-stencilVisc][si]  = 2.0*s_f[stencilVisc][si] - s_f[3*stencilVisc-sk][si];
			} else {
				//periodic boundary condition
				s_f[sk-stencilVisc][si]  = s_f[sk+mz-stencilVisc][si];
				s_f[sk+mz][si]           = s_f[sk][si];
			}
		}
	}

	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCzNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
																		  myprec *s_m, myprec *s_dil,
																		  myprec *u, myprec *v, myprec *w,
																		  myprec *m, myprec *dil,
																		  Indices id, int si, int n) {

	if (id.k < stencilSize) {
		if(multiGPU) {
			haloBCz(s_u,u,si,id); haloBCz(s_v,v,si,id); haloBCz(s_w,w,si,id);
			haloBCz(s_m,m,si,id); haloBCz(s_dil,dil,si,id);
		} else {
			if(boundaryLayer) {
				topBCzExt(s_u,si);
				topBCzExt(s_v,si);
				topBCzExt(s_w,si);
				topBCzExt(s_m,si);
				topBCzExt(s_dil,si);
//				botBCzInit(s_u,uInit,si,id.i);
//				botBCzInit(s_v,vInit,si,id.i);
//				botBCzInit(s_w,wInit,si,id.i);
//				botBCzInit(s_m,mInit,si,id.i);
				botBCzExt(s_u,si);
				botBCzExt(s_v,si);
				botBCzExt(s_w,si);
				botBCzExt(s_m,si);
				botBCzExt(s_dil,si);
			} else {
				perBCz(s_u,si);	perBCz(s_v,si); perBCz(s_w,si);
				perBCz(s_m,si); perBCz(s_dil,si);
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCzNumber2(myprec *s_p, myprec *p, Indices id, int si, int n) {
	if (id.k < stencilSize) {
		if(multiGPU) {
			haloBCz(s_p,p,si,id);
		} else {
			if(boundaryLayer) {
				topBCzExt(s_p,si);
				botBCzExt(s_p,si);
//				botBCzInit(s_p,pInit,si,id.i);
			} else {
				perBCz(s_p,si);
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCzNumber3(myprec *s_l, myprec *s_t,
																		  myprec *l,   myprec *t,
																		  Indices id, int si, int n) {
	if (id.k < stencilSize) {
		if(multiGPU) {
			haloBCz(s_l,l,si,id); haloBCz(s_t,t,si,id);
		} else {
			if(boundaryLayer) {
				topBCzExt(s_l,si);
				topBCzExt(s_t,si);
				botBCzExt(s_l,si);
				botBCzExt(s_t,si);
//				botBCzInit(s_l,lInit,si,id.i);
//				botBCzInit(s_t,tInit,si,id.i);
			} else {
				perBCz(s_l,si); perBCz(s_t,si);
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCzNumber4(myprec *s_r, myprec *s_h,
																		  myprec *r,   myprec *h,
																		  Indices id, int si, int n) {
	if (id.k < stencilSize) {
		if(multiGPU) {
			haloBCz(s_h,h,si,id); haloBCz(s_r,r,si,id);
		} else {
			if(boundaryLayer) {
				topBCzExt(s_r,si);
				topBCzExt(s_h,si);
//				botBCzInit(s_r,rInit,si,id.i);
//				botBCzInit(s_h,hInit,si,id.i);
				botBCzExt(s_r,si);
				botBCzExt(s_h,si);
			} else {
				perBCz(s_r,si); perBCz(s_h,si);
			}
		}
	}
	__syncthreads();
}


#endif /* BOUNDARY_CONDITION_H_ */
