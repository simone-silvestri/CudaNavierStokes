
#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

#include "cuda_globals.h"

class Indices {
  public:
    int i,j,k,g,tix,tiy,bix,biy,bdx,bdy;

    __device__ __host__ Indices(int _tix, int _tiy, int _bix, int _biy, int _bdx, int _bdy) {   
       tix = _tix;
       tiy = _tiy;
       bix = _bix;
       biy = _biy;
       bdx = _bdx;
       bdy = _bdy;
       mkidX();
    }

    __device__ __host__ void mkidX() {
       i  = tix;
       j  = bix*bdy + tiy;
       k  = biy;
       g = i + j*mx + k*mx*my;
    }

    __device__ __host__ void mkidY() {
       i  = bix*bdx + tix;
       j  = tiy;
       k  = biy;
       g = i + j*mx + k*mx*my;
    }

    __device__ __host__ void mkidZ() {
       i  = bix*bdx + tix;
       j  = biy;
       k  = tiy;
       g = i + j*mx + k*mx*my;
    }

    __device__ __host__ void mkidXFlx(int iNum) {
       i  = tix + iNum*mx/nX;
       j  = bix*bdy + tiy;
       k  = biy;
       g = i + j*mx + k*mx*my;
    }

    __device__ __host__ void mkidYFlx(int jNum) {
        //CHANGED
        i  = bix*bdx + tix;
        j  = tiy + jNum*my/nDivY;
        k  = biy;
        g = i + j*mx + k*mx*my;
     }

    __device__ __host__ void mkidZFlx(int kNum) {
  //changed
        i  = bix*bdx + tix;
        j  = biy;
        k  = tiy + kNum*mz/nDivZ;
        g = i + j*mx + k*mx*my;

     }

    __device__ __host__ void mkidYBC(int dir) {
        i  = bix*bdy + tiy;
        if(dir==1) {
        	j  = my - stencilSize + tix;
        } else {
        	j = tix; }
        k  = biy;
        g = i + j*mx + k*mx*my;
     }

    __device__ __host__ void mkidZBC(int dir) {
        i  = bix*bdy + tiy;
        j  = biy;
        if(dir==1) {
        	k  = mz - stencilSize + tix;
        } else {
        	k = tix; }
        g = i + j*mx + k*mx*my;
    }
    __device__ __host__ void mkidBCwallTop(){
    	i  = mx_tot-1;
    	j  = tix;
    	k  = bix;
    	g  = i + j*mx + k*mx*my;
    }
    __device__ __host__ void mkidBCwallBot(){
    	i  = 0;
    	j  = tix;
    	k  = bix;
    	g  = i + j*mx + k*mx*my;
    }
    __device__ __host__ void mkidBCRecycAvg(int krec){
    	i  = bix;
    	j  = tix;
    	k  = krec;
    	g  = i + j*mx + k*mx*my;
    }
    __device__ __host__ void mkidBCRR(int krec){
    	i  = tix;
    	j  = bix;
    	k  = krec;
    	g  = i + j*mx + k*mx*my;
    }
    __device__ __host__ void mkidSpanShift(int krec, int shift){
    	i  = bix;
    	j  = (tix + shift)%my;
    	k  = krec;
    	g  = i + j*mx + k*mx*my;
    }


};

class recycle{
  public:
    myprec *RRr,*RRu,*RRv,*RRw,*RRe,*RRh,*RRp,*RRt,*RRm,*RRl;

    __device__ __host__ recycle(myprec *r,myprec *u,myprec *v,myprec *w,myprec *e,myprec *h,myprec *t,myprec *p,myprec *m,myprec *l ) {   
       RRr = r;
       RRu = u;
       RRv = v;
       RRw = w;
       RRe = e;
       RRh = h;
       RRp = p;
       RRt = t;
       RRm = m;
       RRl = l;
    }
};
void calcTimeStep(myprec *dt, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu, Communicator rk);
void calcPressureGrad(myprec *dpdx, myprec *r, myprec *w, Communicator rk);
void calcBulk(myprec *par1, myprec *par2, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *dtC , myprec *dpdz ,int file, int istep , Communicator rk);

#include "boundary_condition_z.h"
#include "cuda_derivs.h"

//global functions
__global__ void deviceRHSX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dudx, myprec *dvdx, myprec *dwdx, myprec *dudy, myprec *dudz,
		myprec *dvdy, myprec *dwdz, myprec *dil, myprec *dpdz);
__global__ void deviceRHSY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dvdx, myprec *dudy, myprec *dvdy, myprec *dwdy, myprec *dvdz,
		myprec *dil, myprec *dpdz);
__global__ void deviceRHSZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dwdx, myprec *dwdy, myprec *dudz, myprec *dvdz, myprec *dwdz,
		myprec *dil, myprec *dpdz, Communicator rk, recycle rec);

__global__ void derVelX(myprec *u, myprec *v, myprec *w, myprec *dudx, myprec *dvdx, myprec *dwdx);
__global__ void derVelY(myprec *u, myprec *v, myprec *w, myprec *dudy, myprec *dvdy, myprec *dwdy);
__global__ void derVelZ(myprec *u, myprec *v, myprec *w, myprec *dudz, myprec *dvdz, myprec *dwdz, recycle rec, Communicator rk);
__global__ void derVelYBC(myprec *u, myprec *v, myprec *w, myprec *dudy, myprec *dvdy, myprec *dwdy, int direction);
__global__ void derVelZBC(myprec *u, myprec *v, myprec *w, myprec *dudz, myprec *dvdz, myprec *dwdz, int direction);
__global__ void calcDil(myprec *dil, myprec *dudx, myprec *dvdy, myprec *dwdz);
__global__ void deviceCalcDt(myprec *wrkArray, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu);
__global__ void calcState(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam, int bc);
__global__ void deviceAdvanceTime(myprec *dt);

__global__ void BCwallCenteredTop(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret);
__global__ void BCwallCenteredBot(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret);

__global__ void calcStateRR(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam);


//derivatives
__device__ void derDev1x(myprec *df , myprec *f, Indices id);
__device__ void derDev1y(myprec *df , myprec *f, Indices id);
__device__ void derDev1z(myprec *df , myprec *f, Indices id);
__device__ void derDev2x(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2y(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2z(myprec *d2f , myprec *f, Indices id);
__device__ void derDev1xL(myprec *df , myprec *f, Indices id);
__device__ void derDev2xL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2yL(myprec *d2f, myprec *f, Indices id);
__device__ void derDev2zL(myprec *d2f, myprec *f, Indices id);

#endif
