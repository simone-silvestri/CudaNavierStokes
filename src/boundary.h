/*
 * boundary.h
 *
 *  Created on: Apr 20, 2021
 *      Author: simone
 */

#ifndef BOUNDARY_H_
#define BOUNDARY_H_

extern __device__ __forceinline__ void TopperBCx(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void BotperBCx(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void TopwallBCxExt(myprec *s_f, int si, const myprec Bctop, const myprec Bcbot);
extern __device__ __forceinline__ void BotwallBCxExt(myprec *s_f, int si, const myprec Bctop, const myprec Bcbot);
extern __device__ __forceinline__ void TopwallBCxMir(myprec *s_f, int si);
extern __device__ __forceinline__ void BotwallBCxMir(myprec *s_f, int si);
extern __device__ __forceinline__ void TopwallBCxVel(myprec *s_f, int si);
extern __device__ __forceinline__ void BotwallBCxVel(myprec *s_f, int si);
extern __device__ __forceinline__ void TopwallBCxExt_cen(myprec *s_f, int si, const myprec Bctop, const myprec Bcbot);
extern __device__ __forceinline__ void BotwallBCxExt_cen(myprec *s_f, int si, const myprec Bctop, const myprec Bcbot);
extern __device__ __forceinline__ void TopwallBCxMir_cen(myprec *s_f, int si);
extern __device__ __forceinline__ void BotwallBCxMir_cen(myprec *s_f, int si);
extern __device__ __forceinline__ void TopwallBCxVel_cen(myprec *s_f, int si);
extern __device__ __forceinline__ void BotwallBCxVel_cen(myprec *s_f, int si);
extern __device__ __forceinline__ void ToprhBoundPT(myprec *r, myprec *h, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si);
extern __device__ __forceinline__ void BotrhBoundPT(myprec *r, myprec *h, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si);
extern __device__ __forceinline__ void TopmlBoundPT(myprec *m, myprec *l, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si);
extern __device__ __forceinline__ void BotmlBoundPT(myprec *m, myprec *l, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si);
extern __device__ __forceinline__ void BCxExtrapolate(myprec *s_f, int si);


extern __device__ __forceinline__ void perBCy(myprec *s_f, int si);
extern __device__ __forceinline__ void perBCz(myprec *s_f, int si);
extern __device__ __forceinline__ void perBCzBot(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void perBCyBot(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void perBCzTop(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void perBCyTop(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void haloBCy(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void haloBCz(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void haloBCzBot(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void haloBCzTop(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void wallBCxDil(myprec *s_f, myprec *s_u, myprec *s_v, myprec *s_w, int si);
extern __device__ __forceinline__ void stateBoundPT(myprec *r, myprec *t, myprec *u, myprec *v, myprec *w, myprec *h, myprec *p, myprec *m, myprec *l);
extern __device__ __forceinline__ void botBCxMir(myprec *s_f, int si);
extern __device__ __forceinline__ void topBCxExt(myprec *s_f, int si);
extern __device__ __forceinline__ void topBCxVal(myprec *s_f, int si, myprec value);
extern __device__ __forceinline__ void topBCzVal(myprec *s_f, int si, myprec value);
extern __device__ __forceinline__ void botBCzVal(myprec *s_f, int si, myprec value);
extern __device__ __forceinline__ void botBCxExt(myprec *s_f, int si, myprec Bcbot);
extern __device__ __forceinline__ void topBCzExt(myprec *s_f, int si);
extern __device__ __forceinline__ void botBCzExt(myprec *s_f, int si);
extern __device__ __forceinline__ void BotBCzCpy(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void BotBCyCpy(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void TopBCzCpy(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void TopBCyCpy(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void TopBCxCpy(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void BotBCxCpy(myprec *s_f, myprec *f, int si, Indices id);

extern __device__ __forceinline__ void InflowBCzBot(myprec *s_f, myprec *recy_f, int si, Indices id);
extern __device__ __forceinline__ void BotBCzMir(myprec *s_f, int si);

extern __device__ __forceinline__ void haloBCyBot(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void haloBCyTop(myprec *s_f, myprec *f, int si, Indices id);

__device__ __forceinline__ __attribute__((always_inline)) void TopperBCx(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si+mx/nX]           = f[idx(si-stencilSize,id.j, id.k)];
}
__device__ __forceinline__ __attribute__((always_inline)) void BotperBCx(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize]     = f[idx(id.i+mx-stencilSize,id.j,id.k)];
}

__device__ __forceinline__ __attribute__((always_inline)) void TopwallBCxVel(myprec *s_f, int si) {
	s_f[si+mx/nX]           = - s_f[mx/nX+2*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void BotwallBCxVel(myprec *s_f, int si) {
	s_f[si-stencilSize]  = - s_f[3*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void TopwallBCxMir(myprec *s_f, int si) {
	s_f[si+mx/nX]           = s_f[mx/nX+2*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void BotwallBCxMir(myprec *s_f, int si) {
	s_f[si-stencilSize]  = s_f[3*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void TopwallBCxExt(myprec *s_f, int si, myprec Bctop, myprec Bcbot) {
	s_f[si+mx/nX]           = 2.0*Bctop - s_f[mx/nX+2*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void BotwallBCxExt(myprec *s_f, int si, myprec Bctop, myprec Bcbot) {
	s_f[si-stencilSize]  = 2.0*Bcbot - s_f[3*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void TopwallBCxVel_cen(myprec *s_f, int si) {
	s_f[si+mx/nX]           = - s_f[mx/nX+2*stencilSize-si-2];
}

__device__ __forceinline__ __attribute__((always_inline)) void BotwallBCxVel_cen(myprec *s_f, int si) {
	s_f[si-stencilSize]  = - s_f[3*stencilSize-si];
}

__device__ __forceinline__ __attribute__((always_inline)) void TopwallBCxMir_cen(myprec *s_f, int si) {
	s_f[si+mx/nX]           = s_f[mx/nX+2*stencilSize-si-2];
}

__device__ __forceinline__ __attribute__((always_inline)) void BotwallBCxMir_cen(myprec *s_f, int si) {
	s_f[si-stencilSize]  = s_f[3*stencilSize-si];
}

__device__ __forceinline__ __attribute__((always_inline)) void TopwallBCxExt_cen(myprec *s_f, int si, myprec Bctop, myprec Bcbot) {
	s_f[si+mx/nX]           = 2.0*Bctop - s_f[mx/nX+2*stencilSize-si-2];
}

__device__ __forceinline__ __attribute__((always_inline)) void BotwallBCxExt_cen(myprec *s_f, int si, myprec Bctop, myprec Bcbot) {
	s_f[si-stencilSize]  = 2.0*Bcbot - s_f[3*stencilSize-si];
}


__device__ __forceinline__ __attribute__((always_inline)) void ToprhBoundPT(myprec *r, myprec *h, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si)
{


    int idx = si+mx/nX;
    h[idx]  = t[idx]*Rgas*gam/(gam - 1.0)
        		    		  + 0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx]);
    r[idx]  = p[idx]/(Rgas*t[idx]);
}

__device__ __forceinline__ __attribute__((always_inline)) void BotrhBoundPT(myprec *r, myprec *h, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si)
{
	int idx = si-stencilSize;

    h[idx]  = t[idx]*Rgas*gam/(gam - 1.0)
    		    		  + 0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx]);
    r[idx]  = p[idx]/(Rgas*t[idx]);

}


__device__ __forceinline__ __attribute__((always_inline)) void TopmlBoundPT(myprec *m, myprec *l, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si)
{
    int idx = si+mx/nX;

    myprec suth = pow(t[idx],viscexp);
    m[idx]   = suth/Re;
    l[idx]   = suth/Re/Pr/Ec;
}

__device__ __forceinline__ __attribute__((always_inline)) void BotmlBoundPT(myprec *m, myprec *l, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si)
{
	int idx = si-stencilSize;

    myprec suth = pow(t[idx],viscexp);
    m[idx]   = suth/Re;
    l[idx]   = suth/Re/Pr/Ec;
}

__device__ __forceinline__ __attribute__((always_inline)) void BCxExtrapolate(myprec *s_f, int si)
{
	s_f[si+mx/nX]  = s_f[mx/nX-1 + stencilSize];
}

__device__ __forceinline__ __attribute__((always_inline)) void TopBCxCpy(myprec *s_f, myprec *f, int si, Indices id) {
		s_f[si+mx/nX] = f[idx(id.i+mx/nX,id.j,id.k)];

}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCxCpy(myprec *s_f, myprec *f, int si, Indices id) {
		s_f[si-stencilSize] = f[idx(id.i-stencilSize,id.j,id.k)];

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ __attribute__((always_inline)) void perBCyTop(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si+my/nDivY]     = f[idx(id.i,si-stencilSize, id.k)];
}
__device__ __forceinline__ __attribute__((always_inline)) void perBCyBot(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize]  = f[idx(id.i,id.j+my-stencilSize,id.k)];
}

__device__ __forceinline__ __attribute__((always_inline)) void TopBCyCpy(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si+my/nDivY] = f[idx(id.i,id.j+my/nDivY,id.k)];
}
__device__ __forceinline__ __attribute__((always_inline)) void BotBCyCpy(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize] = f[idx(id.i,id.j-stencilSize,id.k)];
}
__device__ __forceinline__ __attribute__((always_inline)) void haloBCyTop(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si+my/nDivY]     = f[mx*my*mz + stencilSize*mx*mz + (si-stencilSize) + id.i*stencilSize + id.k*mx*stencilSize];
}

__device__ __forceinline__ __attribute__((always_inline)) void haloBCyBot(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize]  = f[mx*my*mz + id.j + id.i*stencilSize + id.k*mx*stencilSize];
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ __attribute__((always_inline)) void perBCzTop(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si+mz/nDivZ]     = f[idx(id.i,id.j,si-stencilSize)];
}

__device__ __forceinline__ __attribute__((always_inline)) void perBCzBot(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize]  = f[idx(id.i,id.j,id.k+mz-stencilSize)];
}

__device__ __forceinline__ __attribute__((always_inline)) void TopBCzCpy(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si+mz/nDivZ] = f[idx(id.i,id.j,id.k+mz/nDivZ)];
}

__device__ __forceinline__ __attribute__((always_inline)) void BotBCzCpy(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize] = f[idx(id.i,id.j,id.k-stencilSize)];
}

__device__ __forceinline__ __attribute__((always_inline)) void BCzExtrapolate(myprec *s_f, int si){
	s_f[si+mz/nDivZ]  = s_f[mz/nDivZ-1 + stencilSize];
}

__device__ __forceinline__ __attribute__((always_inline)) void InflowBCzBot(myprec *s_f, myprec *recy_f, int si, Indices id){

	s_f[si-stencilSize] = recy_f[idx(id.i,id.j,si-stencilSize)];

}
__device__ __forceinline__ __attribute__((always_inline)) void BotBCzMir(myprec *s_f, int si) {
	s_f[si-stencilSize]  = s_f[3*stencilSize-si];
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




__device__ __forceinline__ __attribute__((always_inline)) void haloBCy(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize]  = f[mx*my*mz + id.j + id.i*stencilSize + id.k*mx*stencilSize];
	s_f[si+my]           = f[mx*my*mz + stencilSize*mx*mz + id.j + id.i*stencilSize + id.k*mx*stencilSize];
}






__device__ __forceinline__ __attribute__((always_inline)) void haloBCz(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize]  = f[mx*my*mz + 2*stencilSize*mx*mz + id.k + id.i*stencilSize + id.j*mx*stencilSize];
	s_f[si+mz]           = f[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + id.k + id.i*stencilSize + id.j*mx*stencilSize];
}

__device__ __forceinline__ __attribute__((always_inline)) void haloBCzBot(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize]  = f[mx*my*mz + 2*stencilSize*mx*mz + id.k + id.i*stencilSize + id.j*mx*stencilSize];
}

__device__ __forceinline__ __attribute__((always_inline)) void haloBCzTop(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si+mz/nDivZ]     = f[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + (si-stencilSize) + id.i*stencilSize + id.j*mx*stencilSize];
}

__device__ __forceinline__ __attribute__((always_inline)) void wallBCxDil(myprec *s_f, myprec *s_u, myprec *s_v, myprec *s_w, int si) {
	s_f[si-stencilSize]  = s_u[si-stencilSize] +  s_v[si-stencilSize] +  s_w[si-stencilSize];
	s_f[si+mx]           = s_u[si+mx]          +  s_v[si+mx]          +  s_w[si+mx];
}


__device__ __forceinline__ __attribute__((always_inline)) void stateBoundPT(myprec *r, myprec *t, myprec *u, myprec *v, myprec *w, myprec *h, myprec *p, myprec *m, myprec *l, int si)
{
	int idx = si-stencilSize;

    r[idx]  = p[idx]/(Rgas*t[idx]);
    h[idx]  = t[idx]*Rgas*gam/(gam - 1.0)
    		    		  + 0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx]);

    myprec suth = pow(t[idx],viscexp);
    m[idx]   = suth/Re;
    l[idx]   = suth/Re/Pr/Ec;

    idx = si+mx;
    r[idx]  = p[idx]/(Rgas*t[idx]);
    h[idx]  = t[idx]*Rgas*gam/(gam - 1.0)
        		    		  + 0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx]);

    suth = pow(t[idx],viscexp);
    m[idx]   = suth/Re;
    l[idx]   = suth/Re/Pr/Ec;
}


__device__ __forceinline__ __attribute__((always_inline)) void topBCxExt(myprec *s_f, int si) {
	s_f[si+mx]           = 2.0*s_f[mx+stencilSize-1] - s_f[mx+2*stencilSize-si-2];
}

__device__ __forceinline__ __attribute__((always_inline)) void topBCzExt(myprec *s_f, int si) {
	s_f[si+mz/nDivZ]           = 2.0*s_f[mz/nDivZ+stencilSize-1] - s_f[mz/nDivZ+2*stencilSize-si-2];
}

__device__ __forceinline__ __attribute__((always_inline)) void botBCzExt(myprec *s_f, int si) {
	s_f[si-stencilSize]  = 2.0*s_f[stencilSize] - s_f[3*stencilSize-si]; //here we assume that the boundary is at stencilSize (at the node not at the face)
}

__device__ __forceinline__ __attribute__((always_inline)) void botBCxExt(myprec *s_f, int si, myprec Bcbot) {
	s_f[si-stencilSize]  = 2.0*Bcbot - s_f[3*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void topBCxVal(myprec *s_f, int si, myprec value) {
	s_f[si+mx]  = value;
}

__device__ __forceinline__ __attribute__((always_inline)) void topBCzVal(myprec *s_f, int si, myprec value) {
	s_f[si+mz]  = value;
}

__device__ __forceinline__ __attribute__((always_inline)) void botBCzVal(myprec *s_f, int si, myprec value) {
	s_f[si-stencilSize]  = value;
}

__device__ __forceinline__ __attribute__((always_inline)) void botBCxMir(myprec *s_f, int si) {
	s_f[si-stencilSize]  = s_f[3*stencilSize-si-1];
}









#endif /* BOUNDARY_H_ */
