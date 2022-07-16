
#ifndef NRBCX_H_
#define NRBCX_H_

extern __device__ __forceinline__ void IsothnrbcX_top(myprec *rXtmp, myprec *uXtmp, myprec *vXtmp, myprec *wXtmp, myprec *eXtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id); 
extern __device__ __forceinline__ void IsothnrbcX_bot(myprec *rXtmp, myprec *uXtmp, myprec *vXtmp, myprec *wXtmp, myprec *eXtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id); 
extern __device__ __forceinline__ void FreestreamnrbcX_top(myprec *rXtmp, myprec *uXtmp, myprec *vXtmp, myprec *wXtmp, myprec *eXtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id); 


__device__ __forceinline__ __attribute__((always_inline)) void IsothnrbcX_bot(myprec *rXtmp, myprec *uXtmp, myprec *vXtmp, myprec *wXtmp, myprec *eXtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id) {


	myprec dp = 0 ;
	myprec du = 0 ;
	myprec dv = 0 ;
	myprec dw = 0 ;
	myprec dr = 0 ;

	myprec wrk1 = 0;

	derShared1x_FD(&dp, s_p, si) ;
	derShared1x_FD(&du, s_u, si) ;
	derShared1x_FD(&dv, s_v, si) ;
	derShared1x_FD(&dw, s_w, si) ;
	derShared1x_FD(&dr, s_r, si) ;

	myprec sos = pow(gam*s_p[si]/s_r[si],0.5);

	myprec ev[5];

	ev[0] = s_u[si] - sos;
	ev[1] = s_u[si]      ;
	ev[2] = s_u[si]      ;
	ev[3] = s_u[si]      ;
	ev[4] = s_u[si] + sos;


	myprec L[5];

	if (ev[0]<0){
		L[0] = ev[0]*(dp - s_r[si]*sos*du);
	} else {
		L[0] = 0; // either 0 (perfectly non-reflecting) or use LODI relations ( Eq. 24-28 in Poinsot and Lele (1992) )
	}
	if (ev[1]<0){
		L[1] = ev[1]*(sos*sos*dr - dp);
	} else {
		L[1] = 0;
	}
	if (ev[2]<0){
		L[2] = ev[2]*(dv);
	} else {
		L[2] = 0;
	}
	if (ev[3]<0){
		L[3] = ev[3]*(dw);
	} else {
		L[3] = 0;
	}
	if (ev[4]<0){
		L[4] = ev[4]*(dp + s_r[si]*sos*du);
	} else {
		L[4] = L[0]; // used LODI relation eq 26, as du/dt = 0 at the boundary (no-slip)
	}

	myprec d[5];

	d[0] = 1/(sos*sos)*( L[1] + 0.5*(L[4] + L[0]) ) ;
	d[1] = 0.5*(  L[4] + L[0]  ) ;
	d[2] = 1/(2*s_r[si]*sos) * ( L[4] - L[0] ) ;
	d[3] = L[2] ;
	d[4] = L[3] ;

	*rXtmp  = - d[0];
	fluxCubeSharedx(&wrk1,s_r,s_u,s_u,si);
	*uXtmp = *uXtmp + wrk1;
	fluxCubeSharedx(&wrk1,s_r,s_u,s_v,si);
	*vXtmp = *vXtmp + wrk1;
	fluxCubeSharedx(&wrk1,s_r,s_u,s_w,si);
	*wXtmp = *wXtmp + wrk1;
	fluxCubeSharedx(&wrk1,s_r,s_u,s_h,si);
	*eXtmp = *eXtmp + wrk1;
}

__device__ __forceinline__ __attribute__((always_inline)) void IsothnrbcX_top(myprec *rXtmp, myprec *uXtmp, myprec *vXtmp, myprec *wXtmp, myprec *eXtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id) {


	myprec dp = 0 ;
	myprec du = 0 ;
	myprec dv = 0 ;
	myprec dw = 0 ;
	myprec dr = 0 ;

	myprec wrk1 = 0;

	derShared1x_BD(&dp, s_p, si) ;
	derShared1x_BD(&du, s_u, si) ;
	derShared1x_BD(&dv, s_v, si) ;
	derShared1x_BD(&dw, s_w, si) ;
	derShared1x_BD(&dr, s_r, si) ;

	myprec sos = pow(gam*s_p[si]/s_r[si],0.5);

	myprec ev[5];

	ev[0] = s_u[si] - sos;
	ev[1] = s_u[si];
	ev[2] = s_u[si];
	ev[3] = s_u[si];
	ev[4] = s_u[si] + sos;


	myprec L[5];

	if (ev[0]>0){
		L[0] = ev[0]*(dp - s_r[si]*sos*du);
	} else {
		L[0] =  L[4] ; // either 0 (perfectly non-reflecting) or use LODI relations ( Eq. 24-28 in Poinsot and Lele (1992) )
	}
	if (ev[1]>0){
		L[1] = ev[1]*(sos*sos*dr - dp);
	} else {
		L[1] = 0;
	}
	if (ev[2]>0){
		L[2] = ev[2]*(dv);
	} else {
		L[2] = 0;
	}
	if (ev[3]>0){
		L[3] = ev[3]*(dw);
	} else {
		L[3] = 0;
	}
	if (ev[4]>0){
		L[4] = ev[4]*(dp + s_r[si]*sos*du);
	} else {
		L[4] = 0;
	}

	myprec d[5];

	d[0] = 1/(sos*sos) * ( L[1] + 0.5*(L[4] + L[0]) ) ;
	d[1] = 0.5*( L[4] + L[0] ) ;
	d[2] = 1/(2*s_r[si]*sos) * ( L[4] - L[0] ) ;
	d[3] = L[2] ;
	d[4] = L[3] ;

	*rXtmp  = - d[0];
	fluxCubeSharedx(&wrk1,s_r,s_u,s_u,si);
	*uXtmp = *uXtmp + wrk1;
	fluxCubeSharedx(&wrk1,s_r,s_u,s_v,si);
	*vXtmp = *vXtmp + wrk1;
	fluxCubeSharedx(&wrk1,s_r,s_u,s_w,si);
	*wXtmp = *wXtmp + wrk1;
	fluxCubeSharedx(&wrk1,s_r,s_u,s_h,si);
	*eXtmp = *eXtmp + wrk1;


}


__device__ __forceinline__ __attribute__((always_inline)) void FreestreamnrbcX_top(myprec *rXtmp, myprec *uXtmp, myprec *vXtmp, myprec *wXtmp, myprec *eXtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id) {

	myprec dp = 0 ;
	myprec du = 0 ;
	myprec dv = 0 ;
	myprec dw = 0 ;
	myprec dr = 0 ;

	derShared1x_BD(&dp, s_p, si) ;
	derShared1x_BD(&du, s_u, si) ;
	derShared1x_BD(&dv, s_v, si) ;
	derShared1x_BD(&dw, s_w, si) ;
	derShared1x_BD(&dr, s_r, si) ;

	myprec sos = pow(gam*s_p[si]/s_r[si],0.5);

	myprec ev[5];

	ev[0] = s_u[si] - sos;
	ev[1] = s_u[si];
	ev[2] = s_u[si];
	ev[3] = s_u[si];
	ev[4] = s_u[si] + sos;

	myprec sig = 0.25;
	myprec K = sig * (1-Ma*Ma) * sos / Lx; // Ma is the free stream Mach number that we specify. Poinsot and Lele say that maximum Mach number in the flow should be used. So should there be a comparison kernel that computes maximum Mach number? I dont think its needed as the free stream Mach number and Ma in eqn would be very similar in order and equation 40 is not something accurate but a mere model.

	myprec L[5];

	if (ev[0]>0){
		L[0] = ev[0]*(dp - s_r[si]*sos*du);
	} else {
		L[0] =  K*(s_p[si] - pinf) ; // either 0 (perfectly non-reflecting) or use LODI relations ( Eq. 24-28 in Poinsot and Lele (1992) )
	}
	if (ev[1]>0){
		L[1] = ev[1]*(sos*sos*dr - dp);
	} else {
		L[1] = 0;
	}
	if (ev[2]>0){
		L[2] = ev[2]*(dv);
	} else {
		L[2] = 0;
	}
	if (ev[3]>0){
		L[3] = ev[3]*(dw);
	} else {
		L[3] = 0;
	}
	if (ev[4]>0){
		L[4] = ev[4]*(dp + s_r[si]*sos*du);
	} else {
		L[4] = 0;
	}

	myprec d[5];

	d[0] = 1/(sos*sos) * ( L[1] + 0.5*(L[4] + L[0]) ) ;
	d[1] = 0.5*( L[4] + L[0] ) ;
	d[2] = 1/(2*s_r[si]*sos) * ( L[4] - L[0] ) ;
	d[3] = L[2] ;
	d[4] = L[3] ;

	*rXtmp   = - d[0];
	*uXtmp  += - s_u[si]*d[0] - s_r[si]*d[2];
	*vXtmp  += - s_v[si]*d[0] - s_r[si]*d[3];
	*wXtmp  += - s_w[si]*d[0] - s_r[si]*d[4];
	*eXtmp  += - ( 0.5*( s_u[si]*s_u[si] + s_v[si]*s_v[si] + s_w[si]*s_w[si] )*d[0] + d[1]/(gam - 1) + s_r[si]*s_u[si] * d[2] + s_r[si]*s_v[si] * d[3] + s_r[si]*s_w[si] * d[4]  )     ;

}


#endif
