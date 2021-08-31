#ifndef BOUNDARY_CONDITION_X_H_
#define BOUNDARY_CONDITION_X_H_

#include "boundary.h"
#include "sponge.h"
#include "perturbation.h"

extern __device__ __forceinline__ void BCxderVel(myprec *s_u, myprec *s_v, myprec *s_w,
												 Indices id, int si, int m);

extern __device__ __forceinline__ void BCxNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_p, myprec *s_t,
												  myprec *s_m, myprec *s_l,
												  Indices id, int si, int m);

extern __device__ __forceinline__ void BCxNumber2(myprec *s_dil, Indices id, int si, int m);

extern __device__ __forceinline__ void BCxNumber3(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_p, myprec *s_t,
												  myprec *s_r, myprec *s_h,
												  Indices id, int si, int m);

__device__ __forceinline__ __attribute__((always_inline)) void BCxderVel(myprec *s_u, myprec *s_v, myprec *s_w,
																		 Indices id, int si, int m) {
	if(id.i<stencilSize) {
		if(periodicX) {
			perBCx(s_u,si);perBCx(s_v,si);perBCx(s_w,si);
		} else {
			if(boundaryLayer) {
				topBCxExt(s_u,si);
				topBCxExt(s_v,si);
				topBCxExt(s_w,si);
				botBCxExt(s_u,si,0.0);
				botBCxExt(s_v,si,0.0);
				botBCxExt(s_w,si,0.0);
				if(perturbed) PerturbUvel(s_u,id,si);
			} else {
				wallBCxVel(s_u,si);
				wallBCxVel(s_v,si);
				wallBCxVel(s_w,si);
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCxNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
		  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  myprec *s_p, myprec *s_t,
		  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  myprec *s_m, myprec *s_l,
		  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  Indices id, int si, int m) {
	if (id.i < stencilSize) {
		if(periodicX) {
			perBCx(s_u,si);
			perBCx(s_v,si);
			perBCx(s_w,si);
			perBCx(s_t,si);
			perBCx(s_p,si);
			perBCx(s_m,si);
			perBCx(s_l,si);
		}else {
			if(boundaryLayer) {
				topBCxExt(s_u,si);
				topBCxExt(s_v,si);
				topBCxExt(s_w,si);
				topBCxExt(s_p,si);
				topBCxExt(s_t,si);
				botBCxMir(s_p,si);
				botBCxMir(s_t,si);
				botBCxExt(s_u,si,0.0);
				botBCxExt(s_v,si,0.0);
				botBCxExt(s_w,si,0.0);
				if(perturbed) PerturbUvel(s_u,id,si);
				mlBoundPT(s_m, s_l, s_p, s_t, s_u, s_v, s_w, si);
			} else {
				wallBCxMir(s_p,si);
				wallBCxVel(s_u,si);
				wallBCxVel(s_v,si);
				wallBCxVel(s_w,si);
				wallBCxExt(s_t,si,TwallTop,TwallBot);
				mlBoundPT(s_m, s_l, s_p, s_t, s_u, s_v, s_w, si);
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCxNumber2(myprec *s_dil, Indices id, int si, int m) {
	if (id.i < stencilSize) {
		if(periodicX) {
			perBCx(s_dil,si);
		}else {
			if(boundaryLayer) {
				topBCxExt(s_dil,si);
				botBCxMir(s_dil,si);
			} else {
				wallBCxMir(s_dil,si);
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCxNumber3(myprec *s_u, myprec *s_v, myprec *s_w,
																		  myprec *s_p, myprec *s_t,
																		  myprec *s_r, myprec *s_h,
																		  Indices id, int si, int m) {
	if (id.i < stencilSize) {
		if(periodicX) {
			perBCx(s_r,si); perBCx(s_h,si);
		} else {
			rhBoundPT(s_r, s_h, s_p, s_t, s_u, s_v, s_w, si);
		}
	}
	__syncthreads();
}

#endif /* BOUNDARY_CONDITION_H_ */
