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

__device__ myprec d_h[mx*my*mz];
__device__ myprec d_t[mx*my*mz];
__device__ myprec d_p[mx*my*mz];

__device__ myprec d_m[mx*my*mz];
__device__ myprec d_l[mx*my*mz];
__device__ myprec d_dil[mx*my*mz];

__device__ myprec d_tr[mx*my*mz];
__device__ myprec d_tu[mx*my*mz];
__device__ myprec d_tv[mx*my*mz];
__device__ myprec d_tw[mx*my*mz];
__device__ myprec d_te[mx*my*mz];

__device__ myprec *d_rhsr1[5];
__device__ myprec *d_rhsr2[5];
__device__ myprec *d_rhsr3[5];
__device__ myprec *d_rhsr4[5];

__device__ myprec *d_rhsu1[5];
__device__ myprec *d_rhsu2[5];
__device__ myprec *d_rhsu3[5];
__device__ myprec *d_rhsu4[5];

__device__ myprec *d_rhsv1[5];
__device__ myprec *d_rhsv2[5];
__device__ myprec *d_rhsv3[5];
__device__ myprec *d_rhsv4[5];

__device__ myprec *d_rhsw1[5];
__device__ myprec *d_rhsw2[5];
__device__ myprec *d_rhsw3[5];
__device__ myprec *d_rhsw4[5];

__device__ myprec *d_rhse1[5];
__device__ myprec *d_rhse2[5];
__device__ myprec *d_rhse3[5];
__device__ myprec *d_rhse4[5];

__device__ myprec *sij[9];

__device__ myprec dt2,dtC;

__device__ myprec *dttemp;


__global__ void eulerSum(myprec *a, myprec *b,  myprec *cx, myprec *cy, myprec *cz, myprec *dt);
__global__ void rk4final(myprec *a, myprec *bx, myprec *cx, myprec *dx, myprec *ex,
									myprec *by, myprec *cy, myprec *dy, myprec *ey,
									myprec *bz, myprec *cz, myprec *dz, myprec *ez, myprec *dt);
__global__ void calcState(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam);

__global__ void calcTimeStep(myprec *temporary, myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *mu);
__device__ void initSolver();
__device__ void clearSolver();


#endif /* RHS_H_ */
