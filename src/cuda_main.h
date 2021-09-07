/*
 * rhs.h
 *
 *  Created on: Apr 6, 2021
 *      Author: simone
 */

#ifndef RHS_H_
#define RHS_H_


/////////////////////////////////////// variables that fill up the memory (each pointer is mx*my*mz large)

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

myprec *d_rhsr1;
myprec *d_rhsu1;
myprec *d_rhsv1;
myprec *d_rhsw1;
myprec *d_rhse1;

myprec *d_rhsr2;
myprec *d_rhsu2;
myprec *d_rhsv2;
myprec *d_rhsw2;
myprec *d_rhse2;

myprec *d_rhsr3;
myprec *d_rhsu3;
myprec *d_rhsv3;
myprec *d_rhsw3;
myprec *d_rhse3;

myprec *gij[9];

///////////////////////////////////////////////////////////////////////

myprec *dtC,*dpdz;

myprec *djm, *djp, *dkm, *dkp;
myprec *djm5,*djp5,*dkm5,*dkp5;

void calcTimeStepPressGrad(int istep, myprec *dtC, myprec *dpdz, myprec *h_dt, myprec *h_dpdz, Communicator rk);

__device__ myprec time_on_GPU;
__device__ Communicator rkGPU;

__global__ void eulerSum(myprec *a, myprec *b, myprec *c, myprec *dt);
__global__ void eulerSumR(myprec *a, myprec *b, myprec *c, myprec *r, myprec *dt);

__global__ void eulerSum3(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *dt);
__global__ void eulerSum3R(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *r, myprec *dt);

__global__ void rk3final(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *dt);
__global__ void rk3finalR(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *r, myprec *dt);

__global__ void sumLowStorageRK3(myprec *var, myprec *rhs1, myprec *rhs2, myprec *dt, int step);

#endif /* RHS_H_ */










