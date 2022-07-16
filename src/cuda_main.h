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


myprec *rm, *um, *wm, *tm;
myprec *rm_in, *um_in, *wm_in, *tm_in;
myprec *a_inpl, *b_inpl;
int    *idxm, *idxp;
myprec *delta_rec, *delta_in ;
myprec *recy_r, *recy_u, *recy_v, *recy_w, *recy_e, *recy_h, *recy_p, *recy_t, *recy_m, *recy_l;
myprec *gij[9];

///////////////////////////////////////////////////////////////////////

myprec *dtC,*dpdz;

myprec *djm, *djp, *dkm, *dkp;
myprec *djm5,*djp5,*dkm5,*dkp5;

void calcTimeStepPressGrad(int istep, myprec *dtC, myprec *dpdz, myprec *h_dt, myprec *h_dpdz, Communicator rk);

__device__ myprec time_on_GPU;
__device__ Communicator rkGPU;

__global__ void eulerSumAll(myprec *r, myprec *r0,myprec *u, myprec *u0, myprec *v, myprec *v0,myprec *w, 
                            myprec *w0, myprec *e, myprec *e0, myprec *rhsr1, myprec *rhsu1, myprec *rhsv1, myprec *rhsw1, myprec *rhse1,
                            myprec *dt);

__global__ void eulerSumAll2(myprec *r, myprec *r0,myprec *u, myprec *u0, myprec *v, myprec *v0,myprec *w, 
                            myprec *w0, myprec *e, myprec *e0, myprec *rhsr1, myprec *rhsu1, myprec *rhsv1, myprec *rhsw1, myprec *rhse1,
                            myprec *rhsr2, myprec *rhsu2, myprec *rhsv2, myprec *rhsw2, myprec *rhse2, myprec *dt);

__global__ void eulerSumAll3(myprec *r, myprec *r0,myprec *u, myprec *u0, myprec *v, myprec *v0,myprec *w, 
                            myprec *w0, myprec *e, myprec *e0, myprec *rhsr1, myprec *rhsu1, myprec *rhsv1, myprec *rhsw1, myprec *rhse1,
                            myprec *rhsr2, myprec *rhsu2, myprec *rhsv2, myprec *rhsw2, myprec *rhse2,
                            myprec *rhsr3, myprec *rhsu3, myprec *rhsv3, myprec *rhsw3, myprec *rhse3, myprec *dt);

__global__ void eulerSum(myprec *a, myprec *b, myprec *c, myprec *dt);
__global__ void eulerSumR(myprec *a, myprec *b, myprec *c, myprec *r, myprec *dt);

__global__ void eulerSum3(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *dt);
__global__ void eulerSum3R(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *r, myprec *dt);

__global__ void rk3final(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *dt);
__global__ void rk3finalR(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *r, myprec *dt);

__global__ void sumLowStorageRK3(myprec *var, myprec *rhs1, myprec *rhs2, myprec *dt, int step);


void calcMeanRec(myprec *r, myprec *u,myprec *w, myprec *t, myprec *rm, myprec *um, myprec *wm, myprec *tm, myprec *a, myprec *b, int *idxm, int *idxp, myprec *delta_rec, myprec *delta_in, Communicator rk);
void Recycle_Rescale(myprec *r, myprec *u, myprec *v, myprec *w, myprec *t, myprec *rm, myprec *um, myprec *wm, myprec *tm,
                     myprec *rm_in, myprec *um_in, myprec *wm_in, myprec *tm_in, myprec *recy_r, myprec *recy_u,
                     myprec *recy_v, myprec *recy_w, myprec *recy_t,myprec *a, myprec *b, int *idxm, int *idxp, myprec *delta_rec, myprec *delta_in, Communicator rk) ;

void InletMeanUpdate(myprec *rm_in, myprec *um_in, myprec *wm_in, myprec *tm_in, myprec *delta_in, Communicator rk);

void Spanwiseshift(myprec *recy_r,myprec *recy_u,myprec *recy_v,myprec *recy_w,myprec *recy_t, Communicator rk);

__global__ void Recy_Resc(myprec *r, myprec *u, myprec *v, myprec *w, myprec *t, myprec *rm, myprec *um, myprec *wm, myprec *tm,
                          myprec *rm_in, myprec *um_in, myprec *wm_in, myprec *tm_in, myprec *recy_r, myprec *recy_u,
                          myprec *recy_v, myprec *recy_w, myprec *recy_t,myprec *a, myprec *b, int *idxm, int *idxp, myprec *delta_rec, myprec *delta_in, Communicator rk) ;
                          
__global__ void YAvg(myprec *f, myprec *fm, int krec, int kRR);

__global__ void deltaRR(myprec *wm, myprec *delta) ;

__global__ void interpRR(myprec *a, myprec *b, int *idxm, int *idxp, myprec *delta_rec, myprec *delta_in);

__global__ void spanshifting(myprec *var);
                        
#endif /* RHS_H_ */










