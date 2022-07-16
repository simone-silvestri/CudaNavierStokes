
#ifndef NRBCZ_H_
#define NRBCZ_H_

extern __device__ __forceinline__ void InflownrbcZ_bot(myprec *rZtmp, myprec *uZtmp, myprec *vZtmp, myprec *wZtmp, myprec *eZtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id); 
extern __device__ __forceinline__ void OutflownrbcZ_top(myprec *rZtmp, myprec *uZtmp, myprec *vZtmp, myprec *wZtmp, myprec *eZtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id); 


__device__ __forceinline__ __attribute__((always_inline)) void InflownrbcZ_bot(myprec *rZtmp, myprec *uZtmp, myprec *vZtmp, myprec *wZtmp, myprec *eZtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id) {


	myprec dpf = 0 ;
	myprec duf = 0 ;
	myprec dvf = 0 ;
	myprec dwf = 0 ;
	myprec drf = 0 ;

	myprec dpb = 0 ;
	myprec dub = 0 ;
	myprec dvb = 0 ;
	myprec dwb = 0 ;
	myprec drb = 0 ;


	derShared1z_FD(&dpf, s_p, si) ;
	derShared1z_FD(&duf, s_u, si) ;
	derShared1z_FD(&dvf, s_v, si) ;
	derShared1z_FD(&dwf, s_w, si) ;
	derShared1z_FD(&drf, s_r, si) ;

	derShared1z_BD(&dpb, s_p, si) ;
	derShared1z_BD(&dub, s_u, si) ;
	derShared1z_BD(&dvb, s_v, si) ;
	derShared1z_BD(&dwb, s_w, si) ;
	derShared1z_BD(&drb, s_r, si) ;

	myprec sos = pow(gam*s_p[si]/s_r[si],0.5);

	myprec ev[5];

	ev[0] = s_w[si] - sos;
	ev[1] = s_w[si]      ;
	ev[2] = s_w[si]      ;
	ev[3] = s_w[si]      ;
	ev[4] = s_w[si] + sos;

	myprec L[5];

	if (ev[0]<0){
		L[0] = ev[0]*(dpf - s_r[si]*sos*dwf);
	} else {
		L[0] = ev[0]*(dpb - s_r[si]*sos*dwb);
	}
	if (ev[1]<0){
		L[1] = ev[1]*(sos*sos*drf - dpf);
	} else {
		L[1] = ev[1]*(sos*sos*drb - dpb);
	}
	if (ev[2]<0){
		L[2] = ev[2]*(dvf);
	} else {
		L[2] = ev[2]*(dvb);
	}
	if (ev[3]<0){
		L[3] = ev[3]*(duf);
	} else {
		L[3] = ev[3]*(dub);
	}
	if (ev[4]<0){
		L[4] = ev[4]*(dpf + s_r[si]*sos*dwf);
	} else {
		L[4] = ev[4]*(dpb + s_r[si]*sos*dwb);
	}

	myprec d[5];

	d[0] = 1/(sos*sos)*( L[1] + 0.5*(L[4] + L[0]) ) ;
	d[1] = 0.5*(  L[4] + L[0]  ) ;
	d[2] = 1/(2*s_r[si]*sos) * ( L[4] - L[0] ) ;
	d[3] = L[2] ;
	d[4] = L[3] ;

	*rZtmp   = - d[0];
	*uZtmp  += - s_u[si]*d[0] - s_r[si]*d[4];
	*vZtmp  += - s_v[si]*d[0] - s_r[si]*d[3];
	*wZtmp  += - s_w[si]*d[0] - s_r[si]*d[2];
	*eZtmp  += - ( 0.5*( s_u[si]*s_u[si] + s_v[si]*s_v[si] + s_w[si]*s_w[si] )*d[0] + d[1]/(gam - 1) + s_r[si]*s_u[si] * d[4] + s_r[si]*s_v[si] * d[3] + s_r[si]*s_w[si] * d[2]  )     ;
}


__device__ __forceinline__ __attribute__((always_inline)) void OutflownrbcZ_top(myprec *rZtmp, myprec *uZtmp, myprec *vZtmp, myprec *wZtmp, myprec *eZtmp, myprec *s_r, myprec *s_u, myprec *s_v, myprec *s_w, myprec *s_p, myprec *s_h, int si, Indices id) {

	myprec dp = 0 ;
	myprec du = 0 ;
	myprec dv = 0 ;
	myprec dw = 0 ;
	myprec dr = 0 ;

	derShared1z_BD(&dp, s_p, si) ;
	derShared1z_BD(&du, s_u, si) ;
	derShared1z_BD(&dv, s_v, si) ;
	derShared1z_BD(&dw, s_w, si) ;
	derShared1z_BD(&dr, s_r, si) ;

	myprec sos = pow(gam*s_p[si]/s_r[si],0.5);

	myprec ev[5];

	ev[0] = s_w[si] - sos;
	ev[1] = s_w[si];
	ev[2] = s_w[si];
	ev[3] = s_w[si];
	ev[4] = s_w[si] + sos;

	myprec sig = 0.25;
	myprec K = sig * (1-Ma*Ma) * sos / Lz; // Ma is the free stream Mach number that we specify. Poinsot and Lele say that maximum Mach number in the flow should be used. So should there be a comparison kernel that computes maximum Mach number? I dont think its needed as the free stream Mach number and Ma in eqn would be very similar in order and equation 40 is not something accurate but a mere model.

	myprec L[5];

	if (ev[0]>0){
		L[0] = ev[0]*(dp - s_r[si]*sos*dw);
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
		L[3] = ev[3]*(du);
	} else {
		L[3] = 0;
	}
	if (ev[4]>0){
		L[4] = ev[4]*(dp + s_r[si]*sos*dw);
	} else {
		L[4] = 0;
	}

	myprec d[5];

	d[0] = 1/(sos*sos) * ( L[1] + 0.5*(L[4] + L[0]) ) ;
	d[1] = 0.5*( L[4] + L[0] ) ;
	d[2] = 1/(2*s_r[si]*sos) * ( L[4] - L[0] ) ;
	d[3] = L[2] ;
	d[4] = L[3] ;

	*rZtmp   = - d[0];
	*uZtmp  += - s_u[si]*d[0] - s_r[si]*d[4];
	*vZtmp  += - s_v[si]*d[0] - s_r[si]*d[3];
	*wZtmp  += - s_w[si]*d[0] - s_r[si]*d[2];
	*eZtmp  += - ( 0.5*( s_u[si]*s_u[si] + s_v[si]*s_v[si] + s_w[si]*s_w[si] )*d[0] + d[1]/(gam - 1) + s_r[si]*s_u[si] * d[4] + s_r[si]*s_v[si] * d[3] + s_r[si]*s_w[si] * d[2]  )     ;

}


#endif
