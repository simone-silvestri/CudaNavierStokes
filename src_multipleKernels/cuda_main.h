/*
 * rhs.h
 *
 *  Created on: Apr 6, 2021
 *      Author: simone
 */

#ifndef RHS_H_
#define RHS_H_

myprec *d_r;
myprec *d_u;
myprec *d_v;
myprec *d_w;
myprec *d_e;

myprec *d_rO;
myprec *d_eO;
myprec *d_uO;
myprec *d_vO;
myprec *d_wO;

myprec *d_h;
myprec *d_t;
myprec *d_p;
myprec *d_m;
myprec *d_l;

myprec *d_dil;

myprec *d_rhsr1[3];
myprec *d_rhsu1[3];
myprec *d_rhsv1[3];
myprec *d_rhsw1[3];
myprec *d_rhse1[3];

myprec *d_rhsr2[3];
myprec *d_rhsu2[3];
myprec *d_rhsv2[3];
myprec *d_rhsw2[3];
myprec *d_rhse2[3];

myprec *d_rhsr3[3];
myprec *d_rhsu3[3];
myprec *d_rhsv3[3];
myprec *d_rhsw3[3];
myprec *d_rhse3[3];

myprec *dtC,*dpdz;

__device__ myprec gij[9][mx*my*mz];

__global__ void eulerSum(myprec *a, myprec *b, myprec *c, myprec *dt, int i);
__global__ void eulerSumR(myprec *a, myprec *b, myprec *c, myprec *r, myprec *dt, int i);

__global__ void eulerSum3(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *dt, int i);
__global__ void eulerSum3R(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *r, myprec *dt, int i);

__global__ void rk3final(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *dt, int i);
__global__ void rk3finalR(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *r, myprec *dt, int i);

__global__ void calcState(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam);


#endif /* RHS_H_ */
