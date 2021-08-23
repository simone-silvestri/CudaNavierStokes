/*
 * boundary.h
 *
 *  Created on: Apr 20, 2021
 *      Author: simone
 */

#ifndef BOUNDARY_H_
#define BOUNDARY_H_

extern __device__ __forceinline__ void perBCx(myprec *s_f, int si);
extern __device__ __forceinline__ void perBCy(myprec *s_f, int si);
extern __device__ __forceinline__ void perBCz(myprec *s_f, int si);
extern __device__ __forceinline__ void haloBCy(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void haloBCz(myprec *s_f, myprec *f, int si, Indices id);
extern __device__ __forceinline__ void wallBCxExt(myprec *s_f, int si, const myprec Bctop, const myprec Bcbot);
extern __device__ __forceinline__ void wallBCxMir(myprec *s_f, int si);
extern __device__ __forceinline__ void wallBCxVel(myprec *s_f, int si);
extern __device__ __forceinline__ void wallBCxDil(myprec *s_f, myprec *s_u, myprec *s_v, myprec *s_w, int si);
extern __device__ __forceinline__ void stateBoundPT(myprec *r, myprec *t, myprec *u, myprec *v, myprec *w, myprec *h, myprec *p, myprec *m, myprec *l);
extern __device__ __forceinline__ void rhBoundPT(myprec *r, myprec *h, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si);
extern __device__ __forceinline__ void mlBoundPT(myprec *m, myprec *l, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si);
extern __device__ __forceinline__ void botBCxMir(myprec *s_f, int si);
extern __device__ __forceinline__ void topBCxExt(myprec *s_f, int si);
extern __device__ __forceinline__ void botBCxExt(myprec *s_f, int si, myprec Bcbot);
extern __device__ __forceinline__ void topBCzExt(myprec *s_f, int si);
extern __device__ __forceinline__ void botBCzExt(myprec *s_f, int si);

__device__ __forceinline__ __attribute__((always_inline)) void perBCx(myprec *s_f, int si) {
	s_f[si-stencilSize]  = s_f[si+mx-stencilSize];
	s_f[si+mx]           = s_f[si];
}

__device__ __forceinline__ __attribute__((always_inline)) void perBCy(myprec *s_f, int si) {
	s_f[si-stencilSize]  = s_f[si+my-stencilSize];
	s_f[si+my]           = s_f[si];
}

__device__ __forceinline__ __attribute__((always_inline)) void perBCz(myprec *s_f, int si) {
	s_f[si-stencilSize]  = s_f[si+mz-stencilSize];
	s_f[si+mz]           = s_f[si];
}

__device__ __forceinline__ __attribute__((always_inline)) void haloBCy(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize]  = f[mx*my*mz + id.j + id.i*stencilSize + id.k*mx*stencilSize];
	s_f[si+my]           = f[mx*my*mz + stencilSize*mx*mz + id.j + id.i*stencilSize + id.k*mx*stencilSize];
}

__device__ __forceinline__ __attribute__((always_inline)) void haloBCz(myprec *s_f, myprec *f, int si, Indices id) {
	s_f[si-stencilSize]  = f[mx*my*mz + 2*stencilSize*mx*mz + id.k + id.i*stencilSize + id.j*mx*stencilSize];
	s_f[si+mz]           = f[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + id.k + id.i*stencilSize + id.j*mx*stencilSize];
}

__device__ __forceinline__ __attribute__((always_inline)) void wallBCxExt(myprec *s_f, int si, myprec Bctop, myprec Bcbot) {
	s_f[si-stencilSize]  = 2.0*Bcbot - s_f[3*stencilSize-si-1];
	s_f[si+mx]           = 2.0*Bctop - s_f[mx+2*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void wallBCxDil(myprec *s_f, myprec *s_u, myprec *s_v, myprec *s_w, int si) {
	s_f[si-stencilSize]  = s_u[si-stencilSize] +  s_v[si-stencilSize] +  s_w[si-stencilSize];
	s_f[si+mx]           = s_u[si+mx]          +  s_v[si+mx]          +  s_w[si+mx];
}

__device__ __forceinline__ __attribute__((always_inline)) void wallBCxVel(myprec *s_f, int si) {
	s_f[si-stencilSize]  = - s_f[3*stencilSize-si-1];
	s_f[si+mx]           = - s_f[mx+2*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void wallBCxMir(myprec *s_f, int si) {
	s_f[si-stencilSize]  = s_f[3*stencilSize-si-1];
	s_f[si+mx]           = s_f[mx+2*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void stateBoundPT(myprec *r, myprec *t, myprec *u, myprec *v, myprec *w, myprec *h, myprec *p, myprec *m, myprec *l, int si)
{
	int idx = si-stencilSize;

    r[idx]  = p[idx]/(Rgas*t[idx]);
    h[idx]  = t[idx]*Rgas*gamma/(gamma - 1.0)
    		    		  + 0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx]);

    myprec suth = pow(t[idx],viscexp);
    m[idx]   = suth/Re;
    l[idx]   = suth/Re/Pr/Ec;

    idx = si+mx;
    r[idx]  = p[idx]/(Rgas*t[idx]);
    h[idx]  = t[idx]*Rgas*gamma/(gamma - 1.0)
        		    		  + 0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx]);

    suth = pow(t[idx],viscexp);
    m[idx]   = suth/Re;
    l[idx]   = suth/Re/Pr/Ec;
}

__device__ __forceinline__ __attribute__((always_inline)) void rhBoundPT(myprec *r, myprec *h, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si)
{
	int idx = si-stencilSize;

    h[idx]  = t[idx]*Rgas*gamma/(gamma - 1.0)
    		    		  + 0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx]);
    r[idx]  = p[idx]/(Rgas*t[idx]);

    idx = si+mx;
    h[idx]  = t[idx]*Rgas*gamma/(gamma - 1.0)
        		    		  + 0.5*(u[idx]*u[idx]+v[idx]*v[idx]+w[idx]*w[idx]);
    r[idx]  = p[idx]/(Rgas*t[idx]);
}

__device__ __forceinline__ __attribute__((always_inline)) void mlBoundPT(myprec *m, myprec *l, myprec *p, myprec *t, myprec *u, myprec *v, myprec *w, int si)
{
	int idx = si-stencilSize;

    myprec suth = pow(t[idx],viscexp);
    m[idx]   = suth/Re;
    l[idx]   = suth/Re/Pr/Ec;

    idx = si+mx;

    suth = pow(t[idx],viscexp);
    m[idx]   = suth/Re;
    l[idx]   = suth/Re/Pr/Ec;
}

__device__ __forceinline__ __attribute__((always_inline)) void topBCxExt(myprec *s_f, int si) {
	s_f[si+mx]           = 2.0*s_f[mx+stencilSize-1] - s_f[mx+2*stencilSize-si-2];  //here we assume that the boundary is at mx+stencilSize-1 (at the node not at the face)
}

__device__ __forceinline__ __attribute__((always_inline)) void topBCzExt(myprec *s_f, int si) {
	s_f[si+mz]           = 2.0*s_f[mz+stencilSize-1] - s_f[mz+2*stencilSize-si-2];  //here we assume that the boundary is at mx+stencilSize-1 (at the node not at the face)
}

__device__ __forceinline__ __attribute__((always_inline)) void botBCzExt(myprec *s_f, int si) {
	s_f[si-stencilSize]  = 2.0*s_f[stencilSize] - s_f[3*stencilSize-si]; //here we assume that the boundary is at stencilSize (at the node not at the face)
}

__device__ __forceinline__ __attribute__((always_inline)) void botBCxExt(myprec *s_f, int si, myprec Bcbot) {
	s_f[si-stencilSize]  = 2.0*Bcbot - s_f[3*stencilSize-si-1];
}

__device__ __forceinline__ __attribute__((always_inline)) void botBCxMir(myprec *s_f, int si) {
	s_f[si-stencilSize]  = s_f[3*stencilSize-si-1];
}


#endif /* BOUNDARY_H_ */
