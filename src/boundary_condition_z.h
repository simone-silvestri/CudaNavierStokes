#ifndef BOUNDARY_CONDITION_H_
#define BOUNDARY_CONDITION_H_

#include "boundary.h"
#include "sponge.h"

extern __device__ __forceinline__ void TopBCzderVel(myprec *s_u, myprec *s_v, myprec *s_w, myprec *u, myprec *v, myprec *w,
		                                            Indices id, int si, int m, int bctype);
extern __device__ __forceinline__ void BotBCzderVel(myprec *s_u, myprec *s_v, myprec *s_w, myprec *u, myprec *v, myprec *w,
		                                            Indices id, int si, int m, int bctype, recycle rec);


extern __device__ __forceinline__ void TopBCzNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_m, myprec *s_dil,
												  myprec *u, myprec *v, myprec *w,
												  myprec *m, myprec *dil,
												  Indices id, int si, int n, int bctype);
extern __device__ __forceinline__ void BotBCzNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_m, myprec *s_dil,
												  myprec *u, myprec *v, myprec *w,
												  myprec *m, myprec *dil,
												  Indices id, int si, int n, int bctype, recycle rec);


extern __device__ __forceinline__ void TopBCzNumber2(myprec *s_p, myprec *p, Indices id, int si, int n, int bctype);
extern __device__ __forceinline__ void BotBCzNumber2(myprec *s_p, myprec *p, Indices id, int si, int n, int bctype, recycle rec);



extern __device__ __forceinline__ void TopBCzNumber3(myprec *s_l, myprec *s_t,
												  myprec *l,   myprec *t,
												  Indices id, int si, int n, int bctype);
extern __device__ __forceinline__ void BotBCzNumber3(myprec *s_l, myprec *s_t,
												  myprec *l,   myprec *t,
												  Indices id, int si, int n, int bctype, recycle rec);


extern __device__ __forceinline__ void TopBCzNumber4(myprec *s_r, myprec *s_h,
												  myprec *r,   myprec *h,
												  Indices id, int si, int n, int bctype);
extern __device__ __forceinline__ void BotBCzNumber4(myprec *s_r, myprec *s_h,
												  myprec *r,   myprec *h,
												  Indices id, int si, int n, int bctype, recycle rec);



__device__ __forceinline__ __attribute__((always_inline)) void TopBCzderVel(myprec *s_u, myprec *s_v, myprec *s_w, myprec *u, myprec *v, myprec *w,
		Indices id, int si, int m, int bctype) {

	if (bctype == 1) {
		perBCzTop(s_u,u,si,id);
		perBCzTop(s_v,v,si,id);
		perBCzTop(s_w,w,si,id);

	}else if (bctype == 4) {
		BCzExtrapolate(s_u,si);
		BCzExtrapolate(s_v,si);
		BCzExtrapolate(s_w,si);
	}  // add more else ifs here
}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCzderVel(myprec *s_u, myprec *s_v, myprec *s_w, myprec *u, myprec *v, myprec *w,
		Indices id, int si, int m, int bctype, recycle rec) {

	if (bctype == 1) {
		perBCzBot(s_u,u,si,id);
		perBCzBot(s_v,v,si,id);
		perBCzBot(s_w,w,si,id);

	} else if (bctype == 5) {
	
		InflowBCzBot(s_u,rec.RRu,si,id);
		InflowBCzBot(s_v,rec.RRv,si,id);
		InflowBCzBot(s_w,rec.RRw,si,id);
	
	}  // add more else ifs here
}

__device__ __forceinline__ __attribute__((always_inline)) void TopBCzNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
																		  myprec *s_m, myprec *s_dil,
																		  myprec *u, myprec *v, myprec *w,
																		  myprec *m, myprec *dil,
																		  Indices id, int si, int n, int bctype) {
	if (bctype == 1) {

		perBCzTop(s_u,u,si,id);
		perBCzTop(s_v,v,si,id);
		perBCzTop(s_w,w,si,id);
		perBCzTop(s_m,m,si,id);
		perBCzTop(s_dil,dil,si,id);

	} else if (bctype == 4) {

		BCzExtrapolate(s_u,si);
		BCzExtrapolate(s_v,si);
		BCzExtrapolate(s_w,si);
		BCzExtrapolate(s_m,si);
		BCzExtrapolate(s_dil,si);

	}

}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCzNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
																		  myprec *s_m, myprec *s_dil,
																		  myprec *u, myprec *v, myprec *w,
																		  myprec *m, myprec *dil,
																		  Indices id, int si, int n, int bctype, recycle rec) {
	if (bctype == 1) {

		perBCzBot(s_u,u,si,id);
		perBCzBot(s_v,v,si,id);
		perBCzBot(s_w,w,si,id);
		perBCzBot(s_m,m,si,id);
		perBCzBot(s_dil,dil,si,id);

	} else if (bctype == 5) {

		InflowBCzBot(s_u,rec.RRu,si,id);
		InflowBCzBot(s_v,rec.RRv,si,id);
		InflowBCzBot(s_w,rec.RRw,si,id);
		InflowBCzBot(s_m,rec.RRm,si,id);
		BotBCzMir(s_dil,si);
	}

}



__device__ __forceinline__ __attribute__((always_inline)) void TopBCzNumber2(myprec *s_p, myprec *p, Indices id, int si, int n, int bctype) {

	if (bctype == 1) {

		perBCzTop(s_p,p,si,id);

	} else if (bctype == 4) {

		BCzExtrapolate(s_p,si);

	}

}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCzNumber2(myprec *s_p, myprec *p, Indices id, int si, int n, int bctype, recycle rec) {

	if (bctype == 1) {

		perBCzBot(s_p,p,si,id);

	} else if (bctype == 5) {

		InflowBCzBot(s_p,rec.RRp,si,id);
	}

}

__device__ __forceinline__ __attribute__((always_inline)) void TopBCzNumber3(myprec *s_l, myprec *s_t,
																		  myprec *l,   myprec *t,
																		  Indices id, int si, int n, int bctype) {
	if (bctype == 1) {

		perBCzTop(s_l,l,si,id);
		perBCzTop(s_t,t,si,id);

	} else if (bctype == 4) {

		BCzExtrapolate(s_l,si);
		BCzExtrapolate(s_t,si);

	}

}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCzNumber3(myprec *s_l, myprec *s_t,
																		  myprec *l,   myprec *t,
																		  Indices id, int si, int n, int bctype, recycle rec) {
	if (bctype == 1) {

		perBCzBot(s_l,l,si,id);
		perBCzBot(s_t,t,si,id);

	} else if (bctype == 5) {

		InflowBCzBot(s_l,rec.RRl,si,id);
		InflowBCzBot(s_t,rec.RRt,si,id);

	}

}


__device__ __forceinline__ __attribute__((always_inline)) void TopBCzNumber4(myprec *s_r, myprec *s_h,
																		  myprec *r,   myprec *h,
																		  Indices id, int si, int n, int bctype) {
	if (bctype == 1) {

		perBCzTop(s_r,r,si,id);
		perBCzTop(s_h,h,si,id);

	} else if (bctype == 4) {

		BCzExtrapolate(s_r,si);
		BCzExtrapolate(s_h,si);

	}

}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCzNumber4(myprec *s_r, myprec *s_h,
																		  myprec *r,   myprec *h,
																		  Indices id, int si, int n, int bctype, recycle rec) {
	if (bctype == 1) {

		perBCzBot(s_r,r,si,id);
		perBCzBot(s_h,h,si,id);

	} else if (bctype == 5) {

		InflowBCzBot(s_r,rec.RRr,si,id);
		InflowBCzBot(s_h,rec.RRh,si,id);

	}

}


#endif /* BOUNDARY_CONDITION_H_ */
