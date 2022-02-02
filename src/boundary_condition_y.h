#ifndef BOUNDARY_CONDITION_Y_H_
#define BOUNDARY_CONDITION_Y_H_

#include "boundary.h"
#include "sponge.h"

extern __device__ __forceinline__ void TopBCyNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_m, myprec *s_dil,
												  myprec *u, myprec *v, myprec *w,
												  myprec *m, myprec *dil,
												  Indices id, int si, int n);
extern __device__ __forceinline__ void BotBCyNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_m, myprec *s_dil,
												  myprec *u, myprec *v, myprec *w,
												  myprec *m, myprec *dil,
												  Indices id, int si, int n);

extern __device__ __forceinline__ void TopBCyNumber2(myprec *s_p, myprec *p, Indices id, int si, int m);
extern __device__ __forceinline__ void BotBCyNumber2(myprec *s_p, myprec *p, Indices id, int si, int m);


extern __device__ __forceinline__ void TopBCyNumber3(myprec *s_f, myprec *s_g,
												  myprec *f,   myprec *g,
												  Indices id, int si, int n);
extern __device__ __forceinline__ void BotBCyNumber3(myprec *s_f, myprec *s_g,
												  myprec *f,   myprec *g,
												  Indices id, int si, int n);



__device__ __forceinline__ __attribute__((always_inline)) void TopBCyNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
																		  myprec *s_m, myprec *s_dil,
																		  myprec *u, myprec *v, myprec *w,
																		  myprec *m, myprec *dil,
																		  Indices id, int si, int n) {
	perBCyTop(s_u,u,si,id);
	perBCyTop(s_v,v,si,id);
	perBCyTop(s_w,w,si,id);
	perBCyTop(s_m,m,si,id);
	perBCyTop(s_dil,dil,si,id);
}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCyNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
																		  myprec *s_m, myprec *s_dil,
																		  myprec *u, myprec *v, myprec *w,
																		  myprec *m, myprec *dil,
																		  Indices id, int si, int n) {
	perBCyBot(s_u,u,si,id);
	perBCyBot(s_v,v,si,id);
	perBCyBot(s_w,w,si,id);
	perBCyBot(s_m,m,si,id);
	perBCyBot(s_dil,dil,si,id);
}





__device__ __forceinline__ __attribute__((always_inline)) void TopBCyNumber2(myprec *s_p, myprec *p, Indices id, int si, int n) {

	perBCyTop(s_p,p,si,id);

}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCyNumber2(myprec *s_p, myprec *p, Indices id, int si, int n) {

	perBCyBot(s_p,p,si,id);

}


__device__ __forceinline__ __attribute__((always_inline)) void TopBCyNumber3(myprec *s_f, myprec *s_g,
																		  myprec *f,   myprec *g,
																		  Indices id, int si, int n) {
	perBCyTop(s_f,f,si,id);
	perBCyTop(s_g,g,si,id);
	}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCyNumber3(myprec *s_f, myprec *s_g,
																		  myprec *f,   myprec *g,
																		  Indices id, int si, int n) {
	perBCyBot(s_f,f,si,id);
	perBCyBot(s_g,g,si,id);

}


#endif /* BOUNDARY_CONDITION_H_ */
