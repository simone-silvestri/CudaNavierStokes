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

__device__ myprec *d_rhsr1[3];
__device__ myprec *d_rhsu1[3];
__device__ myprec *d_rhsv1[3];
__device__ myprec *d_rhsw1[3];
__device__ myprec *d_rhse1[3];

__device__ myprec *d_rhsr2[3];
__device__ myprec *d_rhsu2[3];
__device__ myprec *d_rhsv2[3];
__device__ myprec *d_rhsw2[3];
__device__ myprec *d_rhse2[3];

__device__ myprec *d_rhsr3[3];
__device__ myprec *d_rhsu3[3];
__device__ myprec *d_rhsv3[3];
__device__ myprec *d_rhsw3[3];
__device__ myprec *d_rhse3[3];

#if rk == 4
__device__ myprec *d_rhsr4[3];
__device__ myprec *d_rhsu4[3];
__device__ myprec *d_rhsv4[3];
__device__ myprec *d_rhsw4[3];
__device__ myprec *d_rhse4[3];
#endif

__device__ myprec *sij[9];

__device__ myprec dt2,dtC,dpdz;

__global__ void eulerSum(myprec *a, myprec *b, myprec *c[3], myprec *dt);
__global__ void eulerSumR(myprec *a, myprec *b, myprec *c[3], myprec *r, myprec *dt);

__global__ void eulerSum3(myprec *a, myprec *b, myprec *c1[3], myprec *c2[3], myprec *dt);
__global__ void eulerSum3R(myprec *a, myprec *b, myprec *c1[3], myprec *c2[3], myprec *r, myprec *dt);

__global__ void rk3final(myprec *a1, myprec *a2, myprec *b[3], myprec *c[3], myprec *d[3], myprec *dt);
__global__ void rk3finalR(myprec *a1, myprec *a2, myprec *b[3], myprec *c[3], myprec *d[3], myprec *r, myprec *dt);

__global__ void calcState(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam);

__device__ void initSolver();
__device__ void clearSolver();

#endif /* RHS_H_ */
