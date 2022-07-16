#ifndef BOUNDARY_CONDITION_X_H_
#define BOUNDARY_CONDITION_X_H_

#include "boundary.h"
#include "sponge.h"
#include "perturbation.h"

extern __device__ __forceinline__ void TopBCxderVel(myprec *s_u, myprec *s_v, myprec *s_w,
		myprec *u, myprec *v, myprec *w,
		Indices id, int si, int m, int bctype);
extern __device__ __forceinline__ void BotBCxderVel(myprec *s_u, myprec *s_v, myprec *s_w,
		myprec *u, myprec *v, myprec *w,
		Indices id, int si, int m, int bctype);

extern __device__ __forceinline__ void TopBCxNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
		myprec *s_p, myprec *s_t,
		myprec *s_m, myprec *s_l, myprec *u, myprec *v, myprec *w,
		myprec *p, myprec *t, myprec *mu, myprec *lam,
		Indices id, int si, int m, int bctype);
extern __device__ __forceinline__ void BotBCxNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
		myprec *s_p, myprec *s_t,
		myprec *s_m, myprec *s_l, myprec *u, myprec *v, myprec *w,
		myprec *p, myprec *t, myprec *mu, myprec *lam,
		Indices id, int si, int m, int bctype);

extern __device__ __forceinline__ void TopBCxNumber2(myprec *s_dil, myprec *dil, Indices id, int si, int m, int bctype);
extern __device__ __forceinline__ void BotBCxNumber2(myprec *s_dil, myprec *dil, Indices id, int si, int m, int bctype);


extern __device__ __forceinline__ void TopBCxNumber3(myprec *s_u, myprec *s_v, myprec *s_w,
		myprec *s_p, myprec *s_t,
		myprec *s_r, myprec *s_h, myprec *r, myprec *h,
		Indices id, int si, int m, int bctype);

extern __device__ __forceinline__ void BotBCxNumber3(myprec *s_u, myprec *s_v, myprec *s_w,
		myprec *s_p, myprec *s_t,
		myprec *s_r, myprec *s_h, myprec *r, myprec *h,
		Indices id, int si, int m, int bctype);

__device__ __forceinline__ __attribute__((always_inline)) void TopBCxderVel(myprec *s_u, myprec *s_v, myprec *s_w, myprec *u, myprec *v, myprec *w,
		Indices id, int si, int m, int bctype) {

	if (bctype == 1) {
		TopperBCx(s_u,u,si,id);
		TopperBCx(s_v,v,si,id);
		TopperBCx(s_w,w,si,id);

	} else if (bctype == 2) {
		TopwallBCxVel(s_u,si);
		TopwallBCxVel(s_v,si);
		TopwallBCxVel(s_w,si);
	} else if (bctype == 3) {
		TopwallBCxVel_cen(s_u,si);
		TopwallBCxVel_cen(s_v,si);
		TopwallBCxVel_cen(s_w,si);
	} else if (bctype == 4) {
		BCxExtrapolate(s_u,si);
		BCxExtrapolate(s_v,si);
		BCxExtrapolate(s_w,si);
	}  // add more else ifs here
}


__device__ __forceinline__ __attribute__((always_inline)) void BotBCxderVel(myprec *s_u, myprec *s_v, myprec *s_w, myprec *u, myprec *v, myprec *w,
		Indices id, int si, int m, int bctype) {

	if (bctype == 1) {
		BotperBCx(s_u,u,si,id);
		BotperBCx(s_v,v,si,id);
		BotperBCx(s_w,w,si,id);

	} else if (bctype == 2) {
		BotwallBCxVel(s_u,si);
		BotwallBCxVel(s_v,si);
		BotwallBCxVel(s_w,si);

	} else if (bctype == 3) {
		BotwallBCxVel_cen(s_u,si);
		BotwallBCxVel_cen(s_v,si);
		BotwallBCxVel_cen(s_w,si);

	} else if (bctype == 4) {
       // do nothing for bottom
	}
}

__device__ __forceinline__ __attribute__((always_inline)) void TopBCxNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
		myprec *s_p, myprec *s_t,
		myprec *s_m, myprec *s_l, myprec *u, myprec *v, myprec *w,
		myprec *p, myprec *t, myprec *mu, myprec *lam,
		Indices id, int si, int m, int bctype) {
	if (bctype == 1) {
		TopperBCx(s_p,p,si,id);
		TopperBCx(s_u,u,si,id);
		TopperBCx(s_v,v,si,id);
		TopperBCx(s_w,w,si,id);
		TopperBCx(s_t,t,si,id);
		TopperBCx(s_m,mu,si,id);
		TopperBCx(s_l,lam,si,id);

	} else if (bctype == 2) {
		TopwallBCxMir(s_p,si);
		TopwallBCxVel(s_u,si);
		TopwallBCxVel(s_v,si);
		TopwallBCxVel(s_w,si);
		TopwallBCxExt(s_t,si,TwallTop,TwallBot);
		TopmlBoundPT(s_m, s_l, s_p, s_t, s_u, s_v, s_w, si);

	} else if (bctype == 3) {

		TopwallBCxMir_cen(s_p,si);
		TopwallBCxVel_cen(s_u,si);
		TopwallBCxVel_cen(s_v,si);
		TopwallBCxVel_cen(s_w,si);
		TopwallBCxExt_cen(s_t,si,TwallTop,TwallBot);
		TopmlBoundPT(s_m, s_l, s_p, s_t, s_u, s_v, s_w, si);

	} else if (bctype == 4) {

		BCxExtrapolate(s_p,si);
		BCxExtrapolate(s_u,si);
		BCxExtrapolate(s_v,si);
		BCxExtrapolate(s_w,si);
		BCxExtrapolate(s_t,si);
		TopmlBoundPT(s_m, s_l, s_p, s_t, s_u, s_v, s_w, si);
	} // add more else ifs here
}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCxNumber1(myprec *s_u, myprec *s_v, myprec *s_w,\
		myprec *s_p, myprec *s_t,
		myprec *s_m, myprec *s_l, myprec *u, myprec *v, myprec *w,
		myprec *p, myprec *t, myprec *mu, myprec *lam,
		Indices id, int si, int m, int bctype) {
	if (bctype == 1) {
		BotperBCx(s_p,p,si,id);
		BotperBCx(s_u,u,si,id);
		BotperBCx(s_v,v,si,id);
		BotperBCx(s_w,w,si,id);
		BotperBCx(s_t,t,si,id);
		BotperBCx(s_m,mu,si,id);
		BotperBCx(s_l,lam,si,id);

	} else if (bctype == 2) {
		BotwallBCxMir(s_p,si);
		BotwallBCxVel(s_u,si);
		BotwallBCxVel(s_v,si);
		BotwallBCxVel(s_w,si);
		BotwallBCxExt(s_t,si,TwallTop,TwallBot);
		BotmlBoundPT(s_m, s_l, s_p, s_t, s_u, s_v, s_w, si);

	} else if (bctype == 3) {

		BotwallBCxMir_cen(s_p,si);
		BotwallBCxVel_cen(s_u,si);
		BotwallBCxVel_cen(s_v,si);
		BotwallBCxVel_cen(s_w,si);
		BotwallBCxExt_cen(s_t,si,TwallTop,TwallBot);
		BotmlBoundPT(s_m, s_l, s_p, s_t, s_u, s_v, s_w, si);

	} else if (bctype == 4) {
      // Do nothing as extrapolation will not be used at the bottom boundary
	}

}



__device__ __forceinline__ __attribute__((always_inline)) void TopBCxNumber2(myprec *s_dil, myprec *dil, Indices id, int si, int m, int bctype) {

	if (bctype == 1) {
		TopperBCx(s_dil,dil,si,id);

	} else if (bctype == 2) {
		TopwallBCxMir(s_dil,si);

	} else if (bctype == 3) {
		TopwallBCxMir_cen(s_dil,si);
	} else if (bctype == 4) {
		BCxExtrapolate(s_dil,si);
	}

}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCxNumber2(myprec *s_dil, myprec *dil, Indices id, int si, int m, int bctype) {

	if (bctype == 1) {
		BotperBCx(s_dil,dil,si,id);

	} else if (bctype == 2) {
		BotwallBCxMir(s_dil,si);

	} else if (bctype == 3) {
		BotwallBCxMir_cen(s_dil,si);
	} else if (bctype == 4) {
        // no extrapolation for bottom
	}

}



__device__ __forceinline__ __attribute__((always_inline)) void TopBCxNumber3(myprec *s_u, myprec *s_v, myprec *s_w,
		myprec *s_p, myprec *s_t,
		myprec *s_r, myprec *s_h, myprec *r, myprec *h,
		Indices id, int si, int m, int bctype) {
	if (bctype == 1) {
		TopperBCx(s_r,r,si,id); TopperBCx(s_h,h,si,id);
	} else if (bctype == 2) {
		ToprhBoundPT(s_r, s_h, s_p, s_t, s_u, s_v, s_w, si);

	} else if (bctype == 3) {
		ToprhBoundPT(s_r, s_h, s_p, s_t, s_u, s_v, s_w, si);

	} else if (bctype == 4) {
		ToprhBoundPT(s_r, s_h, s_p, s_t, s_u, s_v, s_w, si);
	}
}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCxNumber3(myprec *s_u, myprec *s_v, myprec *s_w,
		myprec *s_p, myprec *s_t,
		myprec *s_r, myprec *s_h, myprec *r, myprec *h,
		Indices id, int si, int m, int bctype) {
	if (bctype == 1) {
		BotperBCx(s_r,r,si,id); BotperBCx(s_h,h,si,id);
	} else if (bctype == 2) {
		BotrhBoundPT(s_r, s_h, s_p, s_t, s_u, s_v, s_w, si);

	} else if (bctype == 3) {
		BotrhBoundPT(s_r, s_h, s_p, s_t, s_u, s_v, s_w, si);

	}  else if (bctype == 4) {
		//no extrapolation at bottom BC
	}
}


#endif /* BOUNDARY_CONDITION_H_ */
