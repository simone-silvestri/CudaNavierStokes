#ifndef BOUNDARY_CONDITION_Y_H_
#define BOUNDARY_CONDITION_Y_H_

#include "boundary.h"
#include "sponge.h"

extern __device__ __forceinline__ void BCyNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_m, myprec *s_dil,
												  myprec *u, myprec *v, myprec *w,
												  myprec *m, myprec *dil,
												  Indices id, int si, int n);

extern __device__ __forceinline__ void BCyNumber2(myprec *s_p, myprec *p, Indices id, int si, int m);

extern __device__ __forceinline__ void BCyNumber3(myprec *s_f, myprec *s_g,
												  myprec *f,   myprec *g,
												  Indices id, int si, int n);

__device__ __forceinline__ __attribute__((always_inline)) void BCyNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
																		  myprec *s_m, myprec *s_dil,
																		  myprec *u, myprec *v, myprec *w,
																		  myprec *m, myprec *dil,
																		  Indices id, int si, int n) {
	if (id.j < stencilSize) {
		if(multiGPU) {
			haloBCy(s_u,u,si,id); haloBCy(s_v,v,si,id); haloBCy(s_w,w,si,id);
			haloBCy(s_m,m,si,id); haloBCy(s_dil,dil,si,id);
		} else {
			perBCy(s_u,si);	perBCy(s_v,si); perBCy(s_w,si);
			perBCy(s_m,si); perBCy(s_dil,si);
		}
	}
}

__device__ __forceinline__ __attribute__((always_inline)) void BCyNumber2(myprec *s_p, myprec *p, Indices id, int si, int n) {
	if (id.j < stencilSize) {
		if(multiGPU) {
			haloBCy(s_p,p,si,id);
		} else {
			perBCy(s_p,si);
		}
	}
}

__device__ __forceinline__ __attribute__((always_inline)) void BCyNumber3(myprec *s_f, myprec *s_g,
																		  myprec *f,   myprec *g,
																		  Indices id, int si, int n) {
	if (id.j < stencilSize) {
		if(multiGPU) {
			haloBCy(s_f,f,si,id); haloBCy(s_g,g,si,id);
		} else {
			perBCy(s_f,si); perBCy(s_g,si);
		}
	}
}

#endif /* BOUNDARY_CONDITION_H_ */
