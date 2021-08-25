#ifndef SPONGE_H_
#define SPONGE_H_

extern __device__ myprec spongeX[mx];
extern __device__ myprec spongeZ[mz];
extern __device__ myprec rref[mx*mz];
extern __device__ myprec uref[mx*mz];
extern __device__ myprec vref[mx*mz];
extern __device__ myprec wref[mx*mz];
extern __device__ myprec eref[mx*mz];
extern __device__ myprec href[mx*mz];
extern __device__ myprec tref[mx*mz];
extern __device__ myprec pref[mx*mz];
extern __device__ myprec mref[mx*mz];
extern __device__ myprec lref[mx*mz];

#define idx2(i,k) \
		({ ( k )*mx + ( i ); })

__global__ void addSponge(myprec *rhsr, myprec *rhsu, myprec *rhsv, myprec *rhsw, myprec *rhse,
						  myprec *r, myprec *u, myprec *v, myprec *w, myprec *e);

extern __device__ __forceinline__ void topBCxRef(myprec *s_f, myprec *fref, int si, int k);
extern __device__ __forceinline__ void topBCzRef(myprec *s_f, myprec *fref, int si, int i);
extern __device__ __forceinline__ void botBCzRef(myprec *s_f, myprec *fref, int si, int i);
extern __device__ __forceinline__ void    refBCz(myprec *s_f, myprec *fref, int si, int i);

__device__ __forceinline__ __attribute__((always_inline)) void topBCxRef(myprec *s_f, myprec *fref, int si, int k) {
	s_f[si+mx]           = fref[idx2(mx-1,k)];
}

__device__ __forceinline__ __attribute__((always_inline)) void refBCz(myprec *s_f, myprec *fref, int si, int i) {
	s_f[si+mz]           = fref[idx2(i,mz-1)];
	s_f[si-stencilSize]  = fref[idx2(i,0) ];
}

__device__ __forceinline__ __attribute__((always_inline)) void topBCzRef(myprec *s_f, myprec *fref, int si, int i) {
	s_f[si+mz]           = fref[idx2(i,mz-1)];
}

__device__ __forceinline__ __attribute__((always_inline)) void botBCzRef(myprec *s_f, myprec *fref, int si, int i) {
	s_f[si-stencilSize]  = fref[idx2(i,0)];
}

#endif /* SPONGE_H_ */
