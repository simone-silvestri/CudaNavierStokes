#ifndef BOUNDARY_CONDITION_H_
#define BOUNDARY_CONDITION_H_

#include "boundary.h"
#include "sponge.h"

extern __device__ __forceinline__ void BCzderVel(myprec s_f[mz+stencilSize*2][lPencils], myprec *f, myprec *fref, Indices id, int si, int i, int j, int kNum);

extern __device__ __forceinline__ void BCzNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
												  myprec *s_m, myprec *s_dil,
												  myprec *u, myprec *v, myprec *w,
												  myprec *m, myprec *dil,
												  Indices id, int si, int n, int kNum);

extern __device__ __forceinline__ void BCzNumber2(myprec *s_p, myprec *p, Indices id, int si, int n, int kNum);

extern __device__ __forceinline__ void BCzNumber3(myprec *s_l, myprec *s_t,
												  myprec *l,   myprec *t,
												  Indices id, int si, int n, int kNum);

extern __device__ __forceinline__ void BCzNumber4(myprec *s_r, myprec *s_h,
												  myprec *r,   myprec *h,
												  Indices id, int si, int n, int kNum);

__device__ __forceinline__ __attribute__((always_inline)) void BCzderVel(myprec s_f[mz/nDivZ+stencilSize*2][lPencils], myprec *f, myprec *fref, Indices id, int si, int i, int j, int kNum) {
	int sk = id.tiy + stencilVisc;
	if (sk < stencilVisc*2) {
	if(nDivZ == 1) {
		if(multiGPU) {
			int k = sk - stencilVisc;
			s_f[sk-stencilVisc][si]  = f[mx*my*mz + 2*stencilSize*mx*mz + k + i*stencilSize + j*mx*stencilSize];
			s_f[sk+mz][si]           = f[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + k + i*stencilSize + j*mx*stencilSize];
		} else {
			if(boundaryLayer) {
				//extrapolation
				s_f[sk+mz][si]           = 2.0*s_f[mz+stencilVisc-1][si] - s_f[mz+2*stencilVisc-sk-2][si];
				//extrapolation on reference solution
				//				s_f[sk-stencilVisc][si]  = 2.0*fref[i]              - s_f[3*stencilVisc-sk-1][si];
				//extrapolation
				s_f[sk-stencilVisc][si]  = 2.0*s_f[stencilVisc][si] - s_f[3*stencilVisc-sk][si];
			} else {
				//periodic boundary condition
				s_f[sk-stencilVisc][si]  = s_f[sk+mz-stencilVisc][si];
				s_f[sk+mz][si]           = s_f[sk][si];
			}
		}
	} else {
			int k = sk - stencilVisc + kNum*mz/nDivZ;
			if(kNum==0) {
				s_f[sk+mz/nDivZ][si] = f[idx(i,j,k+mz/nDivZ)];
				if(multiGPU) {
					s_f[sk-stencilVisc][si]  = f[mx*my*mz + 2*stencilSize*mx*mz + k + i*stencilSize + j*mx*stencilSize];
				} else {
					if(boundaryLayer) {
						s_f[sk-stencilVisc][si]  = 2.0*s_f[stencilVisc][si] - s_f[3*stencilVisc-sk][si];
					} else {
						s_f[sk-stencilVisc][si]  = f[idx(i,j,k+mz-1)];
					}
				}
			} else if (kNum==nDivZ-1) {
				s_f[sk-stencilVisc][si]  = f[idx(i,j,k-stencilVisc)];
				if(multiGPU) {
					s_f[sk+mz/nDivZ][si]       = f[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + k + i*stencilSize + j*mx*stencilSize];
				} else {
					if(boundaryLayer) {
						s_f[sk+mz/nDivZ][si]   = 2.0*s_f[mz/nDivZ+stencilVisc-1][si] - s_f[mz/nDivZ+2*stencilVisc-sk-2][si];
					} else {
						s_f[sk+mz/nDivZ][si]   = f[idx(i,j,sk-stencilVisc)];
					}
				}
			} else {
				s_f[sk-stencilVisc][si]  = f[idx(i,j,k-stencilVisc)];
				s_f[sk+mz/nDivZ][si]     = f[idx(i,j,k+mz/nDivZ)];
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCzNumber1(myprec *s_u, myprec *s_v, myprec *s_w,
																		  myprec *s_m, myprec *s_dil,
																		  myprec *u, myprec *v, myprec *w,
																		  myprec *m, myprec *dil,
																		  Indices id, int si, int n, int kNum) {

	if (id.tix < stencilSize) {
		if(nDivZ==1) {
			if(multiGPU) {
				haloBCz(s_u,u,si,id); haloBCz(s_v,v,si,id); haloBCz(s_w,w,si,id);
				haloBCz(s_m,m,si,id); haloBCz(s_dil,dil,si,id);
			} else {
				if(boundaryLayer) {
					topBCzExt(s_u,si);
					topBCzExt(s_v,si);
					topBCzExt(s_w,si);
					topBCzExt(s_m,si);
					topBCzExt(s_dil,si);
					//				botBCzInit(s_u,uInit,si,id.i);
					//				botBCzInit(s_v,vInit,si,id.i);
					//				botBCzInit(s_w,wInit,si,id.i);
					//				botBCzInit(s_m,mInit,si,id.i);
					botBCzExt(s_u,si);
					botBCzExt(s_v,si);
					botBCzExt(s_w,si);
					botBCzExt(s_m,si);
					botBCzExt(s_dil,si);
				} else {
					perBCz(s_u,si);	perBCz(s_v,si); perBCz(s_w,si);
					perBCz(s_m,si); perBCz(s_dil,si);
				}
			}
		} else {
			if(kNum==0) {
				topBCzCpy(s_u,u,si,id); topBCzCpy(s_v,v,si,id); topBCzCpy(s_w,w,si,id);
				topBCzCpy(s_m,m,si,id); topBCzCpy(s_dil,dil,si,id);
				if(multiGPU) {
					haloBCzBot(s_u,u,si,id); haloBCzBot(s_v,v,si,id); haloBCzBot(s_w,w,si,id);
					haloBCzBot(s_m,m,si,id); haloBCzBot(s_dil,dil,si,id);
				} else {
					if(boundaryLayer) {
						botBCzExt(s_u,si);
						botBCzExt(s_v,si);
						botBCzExt(s_w,si);
						botBCzExt(s_m,si);
						botBCzExt(s_dil,si);
					} else {
						perBCzBot(s_u,u,si,id);
						perBCzBot(s_v,v,si,id);
						perBCzBot(s_w,w,si,id);
						perBCzBot(s_m,m,si,id);
						perBCzBot(s_dil,dil,si,id);
					}
				}
			} else if (kNum==nDivZ-1) {
				botBCzCpy(s_u,u,si,id); botBCzCpy(s_v,v,si,id); botBCzCpy(s_w,w,si,id);
				botBCzCpy(s_m,m,si,id); botBCzCpy(s_dil,dil,si,id);
				if(multiGPU) {
					haloBCzTop(s_u,u,si,id); haloBCzTop(s_v,v,si,id); haloBCzTop(s_w,w,si,id);
					haloBCzTop(s_m,m,si,id); haloBCzTop(s_dil,dil,si,id);
				} else {
					if(boundaryLayer) {
						topBCzExt(s_u,si);
						topBCzExt(s_v,si);
						topBCzExt(s_w,si);
						topBCzExt(s_m,si);
						topBCzExt(s_dil,si);
					} else {
						perBCzTop(s_u,u,si,id);
						perBCzTop(s_v,v,si,id);
						perBCzTop(s_w,w,si,id);
						perBCzTop(s_m,m,si,id);
						perBCzTop(s_dil,dil,si,id);
					}
				}
			} else {
				topBCzCpy(s_u,u,si,id); topBCzCpy(s_v,v,si,id); topBCzCpy(s_w,w,si,id);
				topBCzCpy(s_m,m,si,id); topBCzCpy(s_dil,dil,si,id);
				botBCzCpy(s_u,u,si,id); botBCzCpy(s_v,v,si,id); botBCzCpy(s_w,w,si,id);
				botBCzCpy(s_m,m,si,id); botBCzCpy(s_dil,dil,si,id);
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCzNumber2(myprec *s_p, myprec *p, Indices id, int si, int n, int kNum) {
	if (id.tix < stencilSize) {
		if(nDivZ==1) {
			if(multiGPU) {
				haloBCz(s_p,p,si,id);
			} else {
				if(boundaryLayer) {
					topBCzExt(s_p,si);
					botBCzExt(s_p,si);
					//				botBCzInit(s_p,pInit,si,id.i);
				} else {
					perBCz(s_p,si);
				}
			}
		} else {
			if(kNum==0) {
				topBCzCpy(s_p,p,si,id);
				if(multiGPU) {
					haloBCzBot(s_p,p,si,id);
				} else {
					if (boundaryLayer) {
						botBCzExt(s_p,si);
					} else {
						perBCzBot(s_p,p,si,id);
					}
				}
			} else if (kNum==nDivZ-1) {
				botBCzCpy(s_p,p,si,id);
				if(multiGPU) {
					haloBCzTop(s_p,p,si,id);
				} else {
					if (boundaryLayer) {
						topBCzExt(s_p,si);
					} else {
						perBCzTop(s_p,p,si,id);
					}
				}
			} else {
				topBCzCpy(s_p,p,si,id);
				botBCzCpy(s_p,p,si,id);
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCzNumber3(myprec *s_l, myprec *s_t,
																		  myprec *l,   myprec *t,
																		  Indices id, int si, int n, int kNum) {
	if (id.tix < stencilSize) {
		if(nDivZ==1) {
			if(multiGPU) {
				haloBCz(s_l,l,si,id); haloBCz(s_t,t,si,id);
			} else {
				if(boundaryLayer) {
					topBCzExt(s_l,si);
					topBCzExt(s_t,si);
					botBCzExt(s_l,si);
					botBCzExt(s_t,si);
					//				botBCzInit(s_l,lInit,si,id.i);
					//				botBCzInit(s_t,tInit,si,id.i);
				} else {
					perBCz(s_l,si); perBCz(s_t,si);
				}
			}
		} else {
			if(kNum==0) {
				topBCzCpy(s_l,l,si,id);
				topBCzCpy(s_t,t,si,id);
				if(multiGPU) {
					haloBCzBot(s_l,l,si,id);
					haloBCzBot(s_t,t,si,id);
				} else {
					if (boundaryLayer) {
						botBCzExt(s_l,si);
						botBCzExt(s_t,si);
					} else {
						perBCzBot(s_l,l,si,id);
						perBCzBot(s_t,t,si,id);
					}
				}
			} else if (kNum==nDivZ-1) {
				botBCzCpy(s_l,l,si,id);
				botBCzCpy(s_t,t,si,id);
				if(multiGPU) {
					haloBCzTop(s_l,l,si,id);
					haloBCzTop(s_t,t,si,id);
				} else {
					if (boundaryLayer) {
						topBCzExt(s_l,si);
						topBCzExt(s_t,si);
					} else {
						perBCzTop(s_l,l,si,id);
						perBCzTop(s_t,t,si,id);
					}
				}
			} else {
				topBCzCpy(s_l,l,si,id);
				topBCzCpy(s_t,t,si,id);
				botBCzCpy(s_l,l,si,id);
				botBCzCpy(s_t,t,si,id);
			}
		}
	}
	__syncthreads();
}

__device__ __forceinline__ __attribute__((always_inline)) void BCzNumber4(myprec *s_r, myprec *s_h,
																		  myprec *r,   myprec *h,
																		  Indices id, int si, int n, int kNum) {
	if (id.tix < stencilSize) {
		if(nDivZ==1) {
			if(multiGPU) {
				haloBCz(s_h,h,si,id); haloBCz(s_r,r,si,id);
			} else {
				if(boundaryLayer) {
					topBCzExt(s_r,si);
					topBCzExt(s_h,si);
					//				botBCzInit(s_r,rInit,si,id.i);
					//				botBCzInit(s_h,hInit,si,id.i);
					botBCzExt(s_r,si);
					botBCzExt(s_h,si);
				} else {
					perBCz(s_r,si); perBCz(s_h,si);
				}
			}
		} else {
			if(kNum==0) {
				topBCzCpy(s_r,r,si,id);
				topBCzCpy(s_h,h,si,id);
				if(multiGPU) {
					haloBCzBot(s_r,r,si,id);
					haloBCzBot(s_h,h,si,id);
				} else {
					if (boundaryLayer) {
						botBCzExt(s_r,si);
						botBCzExt(s_h,si);
					} else {
						perBCzBot(s_r,r,si,id);
						perBCzBot(s_h,h,si,id);
					}
				}
			} else if (kNum==nDivZ-1) {
				botBCzCpy(s_r,r,si,id);
				botBCzCpy(s_h,h,si,id);
				if(multiGPU) {
					haloBCzTop(s_r,r,si,id);
					haloBCzTop(s_h,h,si,id);
				} else {
					if (boundaryLayer) {
						topBCzExt(s_r,si);
						topBCzExt(s_h,si);
					} else {
						perBCzTop(s_r,r,si,id);
						perBCzTop(s_h,h,si,id);
					}
				}
			} else {
				topBCzCpy(s_r,r,si,id);
				topBCzCpy(s_h,h,si,id);
				botBCzCpy(s_r,r,si,id);
				botBCzCpy(s_h,h,si,id);
			}
		}
	}
	__syncthreads();
}


#endif /* BOUNDARY_CONDITION_H_ */
