#ifndef SPONGE_H_
#define SPONGE_H_

//x = Lx
const myprec spTopStr = 1.0;
const myprec spTopLen = 1.0;
const myprec spTopExp = 2.0;

//z = 0
const myprec spInlStr = 0.5;
const myprec spInlLen = 20.0;
const myprec spInlExp = 2.0;

//z = Lz
const myprec spOutStr = 0.5;
const myprec spOutLen = 20.0;
const myprec spOutExp = 2.0;

extern __device__ myprec spongeX[mx];
extern __device__ myprec spongeZ[mz];
extern __device__ myprec rref[mx*mz];
extern __device__ myprec uref[mx*mz];
extern __device__ myprec vref[mx*mz];
extern __device__ myprec wref[mx*mz];
extern __device__ myprec eref[mx*mz];

extern __device__ myprec rInit[mx];
extern __device__ myprec uInit[mx];
extern __device__ myprec vInit[mx];
extern __device__ myprec wInit[mx];
extern __device__ myprec hInit[mx];
extern __device__ myprec tInit[mx];
extern __device__ myprec pInit[mx];
extern __device__ myprec mInit[mx];
extern __device__ myprec lInit[mx];

#define idx2(i,k) \
		({ ( i ) + ( k )*mx; })

__global__ void addSponge(myprec *rhsr, myprec *rhsu, myprec *rhsv, myprec *rhsw, myprec *rhse,
						  myprec *r, myprec *u, myprec *v, myprec *w, myprec *e);

extern __device__ __forceinline__ void  topBCxRef(myprec *s_f, myprec *fref, int si, int k);
extern __device__ __forceinline__ void  topBCzRef(myprec *s_f, myprec *fref, int si, int i);
extern __device__ __forceinline__ void  botBCzRef(myprec *s_f, myprec *fref, int si, int i);
extern __device__ __forceinline__ void botBCzInit(myprec *s_f, myprec *fIn, int si, int i);
extern __device__ __forceinline__ void     refBCz(myprec *s_f, myprec *fref, int si, int i);

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

__device__ __forceinline__ __attribute__((always_inline)) void botBCzInit(myprec *s_f, myprec *finit, int si, int i) {
	s_f[si-stencilSize]  = 2.0*finit[i] - s_f[3*stencilSize-si-1];
}

#endif /* SPONGE_H_ */
