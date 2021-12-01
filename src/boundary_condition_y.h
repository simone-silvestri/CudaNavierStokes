#ifndef BOUNDARY_CONDITION_Y_H_
#define BOUNDARY_CONDITION_Y_H_

#include "boundary.h"
#include "sponge.h"

extern __device__ __forceinline__ void BCyNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_m, myprec *s_dil,
												  myprec *u, myprec *v, myprec *w,
												  myprec *m, myprec *dil,
												  Indices id, int si, int n, int jNum);

extern __device__ __forceinline__ void BCyNumber2(myprec *s_p, myprec *p, Indices id, int si, int m, int jNum);

extern __device__ __forceinline__ void BCyNumber3(myprec *s_f, myprec *s_g,
												  myprec *f,   myprec *g,
												  Indices id, int si, int n, int jNum);

__device__ __forceinline__ __attribute__((always_inline)) void BCyNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
																		  myprec *s_m, myprec *s_dil,
																		  myprec *u, myprec *v, myprec *w,
																		  myprec *m, myprec *dil,
																		  Indices id, int si, int n, int jNum) {
	if (id.tiy < stencilSize) {
	    if(nDivY==1) {
		     if(multiGPU) {
			haloBCy(s_u,u,si,id); haloBCy(s_v,v,si,id); haloBCy(s_w,w,si,id);
			haloBCy(s_m,m,si,id); haloBCy(s_dil,dil,si,id);
		     } else {
			perBCy(s_u,si);	perBCy(s_v,si); perBCy(s_w,si);
			perBCy(s_m,si); perBCy(s_dil,si);
		}
	   } 
           else {
			if(jNum==0) {
				topBCyCpy(s_u,u,si,id); topBCyCpy(s_v,v,si,id); topBCyCpy(s_w,w,si,id);
				topBCyCpy(s_m,m,si,id); topBCyCpy(s_dil,dil,si,id);
				
						perBCyBot(s_u,u,si,id);
						perBCyBot(s_v,v,si,id);
						perBCyBot(s_w,w,si,id);
						perBCyBot(s_m,m,si,id);
						perBCyBot(s_dil,dil,si,id);
				
				
			} else if (jNum==nDivY-1) {
				botBCyCpy(s_u,u,si,id); botBCyCpy(s_v,v,si,id); botBCyCpy(s_w,w,si,id);
				botBCyCpy(s_m,m,si,id); botBCyCpy(s_dil,dil,si,id);
				
						perBCyTop(s_u,u,si,id);
						perBCyTop(s_v,v,si,id);
						perBCyTop(s_w,w,si,id);
						perBCyTop(s_m,m,si,id);
						perBCyTop(s_dil,dil,si,id);
				
			} else {
				topBCyCpy(s_u,u,si,id); topBCyCpy(s_v,v,si,id); topBCyCpy(s_w,w,si,id);
				topBCyCpy(s_m,m,si,id); topBCyCpy(s_dil,dil,si,id);
				botBCyCpy(s_u,u,si,id); botBCyCpy(s_v,v,si,id); botBCyCpy(s_w,w,si,id);
				botBCyCpy(s_m,m,si,id); botBCyCpy(s_dil,dil,si,id);
			}
		}
	}




	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCyNumber2(myprec *s_p, myprec *p, Indices id, int si, int n, int jNum) {
	if (id.tiy < stencilSize) {
	    if(nDivY==1) {
		     if(multiGPU) {
			haloBCy(s_p,p,si,id);
		     } else {
			perBCy(s_p,si);
		}
	   } 
           else {
			if(jNum==0) {
				topBCyCpy(s_p,p,si,id); 
			        perBCyBot(s_p,p,si,id);
				
				
			} else if (jNum==nDivY-1) {
				botBCyCpy(s_p,p,si,id); 
						perBCyTop(s_p,p,si,id);
				
			} else {
				topBCyCpy(s_p,p,si,id); botBCyCpy(s_p,p,si,id);
			}
		}
	}




	__syncthreads();

}

__device__ __forceinline__ __attribute__((always_inline)) void BCyNumber3(myprec *s_f, myprec *s_g,
																		  myprec *f,   myprec *g,
																		  Indices id, int si, int n, int jNum) {
	if (id.tiy < stencilSize) {
	    if(nDivY==1) {
		     if(multiGPU) {
			haloBCy(s_f,f,si,id); haloBCy(s_g,g,si,id); 
		     } else {
			perBCy(s_f,si);	perBCy(s_g,si); 
		}
	   } 
           else {
			if(jNum==0) {
				topBCyCpy(s_f,f,si,id); topBCyCpy(s_g,g,si,id); 						perBCyBot(s_f,f,si,id);
						perBCyBot(s_g,g,si,id);
				
				
			} else if (jNum==nDivY-1) {
				botBCyCpy(s_f,f,si,id); botBCyCpy(s_g,g,si,id); 						perBCyTop(s_f,f,si,id);
						perBCyTop(s_g,g,si,id);
				
			} else {
				topBCyCpy(s_f,f,si,id); topBCyCpy(s_g,g,si,id); 				botBCyCpy(s_f,f,si,id); botBCyCpy(s_g,g,si,id);
			}
		}
	}

	__syncthreads();
}

#endif /* BOUNDARY_CONDITION_H_ */
