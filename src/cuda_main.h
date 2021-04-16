/*
 * rhs.h
 *
 *  Created on: Apr 6, 2021
 *      Author: simone
 */

#ifndef RHS_H_
#define RHS_H_

__device__ myprec d_r[mx*my*mz];
__device__ myprec d_u[mx*my*mz];
__device__ myprec d_v[mx*my*mz];
__device__ myprec d_w[mx*my*mz];
__device__ myprec d_e[mx*my*mz];

__device__ myprec *d_rO;
__device__ myprec *d_eO;
__device__ myprec *d_uO;
__device__ myprec *d_vO;
__device__ myprec *d_wO;

__device__ myprec *d_h;
__device__ myprec *d_t;
__device__ myprec *d_p;
__device__ myprec *d_m;
__device__ myprec *d_l;

__device__ myprec *d_dil;

__device__ myprec *d_rhsr1[5];
__device__ myprec *d_rhsu1[5];
__device__ myprec *d_rhsv1[5];
__device__ myprec *d_rhsw1[5];
__device__ myprec *d_rhse1[5];

__device__ myprec *d_rhsr2[5];
__device__ myprec *d_rhsu2[5];
__device__ myprec *d_rhsv2[5];
__device__ myprec *d_rhsw2[5];
__device__ myprec *d_rhse2[5];

__device__ myprec *d_rhsr3[5];
__device__ myprec *d_rhsu3[5];
__device__ myprec *d_rhsv3[5];
__device__ myprec *d_rhsw3[5];
__device__ myprec *d_rhse3[5];

#if rk == 4
__device__ myprec *d_rhsr4[5];
__device__ myprec *d_rhsu4[5];
__device__ myprec *d_rhsv4[5];
__device__ myprec *d_rhsw4[5];
__device__ myprec *d_rhse4[5];
#endif

__device__ myprec *sij[9];

__device__ myprec dt2,dtC;

__global__ void eulerSum(myprec *a, myprec *b, myprec *c[5], myprec *dt);
__global__ void eulerSumR(myprec *a, myprec *b, myprec *c[5], myprec *r, myprec *dt);

__global__ void eulerSum3(myprec *a, myprec *b, myprec *c1[5], myprec *c2[5], myprec *dt);
__global__ void eulerSum3R(myprec *a, myprec *b, myprec *c1[5], myprec *c2[5], myprec *r, myprec *dt);

__global__ void rk4final(myprec *a1, myprec *a2, myprec *b[5], myprec *c[5], myprec *d[5], myprec *e[5], myprec *dt);
__global__ void rk4finalR(myprec *a1, myprec *a2, myprec *b[5], myprec *c[5], myprec *d[5], myprec *e[5], myprec *r, myprec *dt);

__global__ void rk3final(myprec *a1, myprec *a2, myprec *b[5], myprec *c[5], myprec *d[5], myprec *dt);
__global__ void rk3finalR(myprec *a1, myprec *a2, myprec *b[5], myprec *c[5], myprec *d[5], myprec *r, myprec *dt);

__global__ void calcState(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam);

__device__ void initSolver();
__device__ void clearSolver();


#endif /* RHS_H_ */
