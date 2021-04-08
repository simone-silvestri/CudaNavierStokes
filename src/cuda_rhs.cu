
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include "globals.h"
#include "cuda_functions.h"


/*
 *  The L-versions of the RHS have to be ran with
 *  - the L-version of the derivatives
 *  i.e.: derDev1xL instead of derDev1x
 *  - the L-version of the grid
 *  i.e.: h_gridL[0] instead of h_grid[0]
 */

__device__ myprec d_workX[mx*my*mz];
__device__ myprec d_workX1[mx*my*mz];
__device__ myprec d_workX2[mx*my*mz];
__device__ myprec d_workY[mx*my*mz];
__device__ myprec d_workY1[mx*my*mz];
__device__ myprec d_workY2[mx*my*mz];
__device__ myprec d_workZ[mx*my*mz];
__device__ myprec d_workZ1[mx*my*mz];
__device__ myprec d_workZ2[mx*my*mz];

/* I can load all variables into the shared memory!! and work only on those until I finish everything! */
/* Additionally, the flux comp cubic can work by just loading in the needed arrays and working on those like derDev */


__global__ void RHSDeviceFlxX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
							  myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
							  myprec *t,  myprec *p,  myprec *mu, myprec *lam) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	// viscous fluxes derivative
	derDev2x(d_workX1,u,id);
	uX[id.g] = d_workX1[id.g]*mu[id.g];
	derDev2x(d_workX1,v,id);
	vX[id.g] = d_workX1[id.g]*mu[id.g];
	derDev2x(d_workX1,w,id);
	wX[id.g] = d_workX1[id.g]*mu[id.g];
	derDev2x(d_workX1,t,id);
	eX[id.g] = d_workX1[id.g]*lam[id.g];
	__syncthreads();

	//Adding here the terms d (phi) dx * d (mu) dx); (lambda in case of h in rhse);

	derDev1x(d_workX2,mu,id); //d_work2 = d (mu) dx
	derDev1x(d_workX,u,id); // d_work = d (u) dx
	uX[id.g] = uX[id.g] + d_workX[id.g]*d_workX2[id.g];
	derDev1x(d_workX,v,id); // d_work = d (v) dx
	vX[id.g] = vX[id.g] + d_workX[id.g]*d_workX2[id.g];
	derDev1x(d_workX,w,id); // d_work = d (w) dx
	wX[id.g] = wX[id.g] + d_workX[id.g]*d_workX2[id.g];
	derDev1x(d_workX2,lam,id); //d_work2 = d (lam) dx
	derDev1x(d_workX,h,id); // d_work = d (h) dx
	eX[id.g] = eX[id.g] + d_workX[id.g]*d_workX2[id.g];

	//Adding here the terms - d (ru*phi) dx in split flux term;
	fluxQuadx(d_workX,r,u,id);  // d_work = d (ru) dx
	rX[id.g] = rX[id.g] + d_workX[id.g];

	fluxCubex(d_workX,r,u,u,id);  // d_work = d (ruu) dx
	uX[id.g] = uX[id.g] + d_workX[id.g];
	fluxCubex(d_workX,r,u,v,id);  // d_work = d (ruu) dx
	vX[id.g] = vX[id.g] + d_workX[id.g];
	fluxCubex(d_workX,r,u,w,id);  // d_work = d (ruu) dx
	wX[id.g] = wX[id.g] + d_workX[id.g];
	fluxCubex(d_workX,r,u,h,id);  // d_work = d (ruu) dx
	eX[id.g] = eX[id.g] + d_workX[id.g];

	// pressure derivatives
	derDev1x(d_workX,p,id);
	uX[id.g] = uX[id.g] - d_workX[id.g];

}


__global__ void RHSDeviceX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX, 
						   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
						   myprec *t,  myprec *p,  myprec *mu, myprec *lam) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	// viscous fluxes derivative
	derDev2x(d_workX1,u,id);
	uX[id.g] = d_workX1[id.g]*mu[id.g];
	derDev2x(d_workX1,v,id);
	vX[id.g] = d_workX1[id.g]*mu[id.g];
	derDev2x(d_workX1,w,id);
	wX[id.g] = d_workX1[id.g]*mu[id.g];
	derDev2x(d_workX1,t,id);
	eX[id.g] = d_workX1[id.g]*lam[id.g];
	__syncthreads();

	// split advection terms 
	d_workX1[id.g] = r[id.g]*u[id.g];
	__syncthreads();

	//Adding here the terms -0.5 * phi * d (ru) dx;

	derDev1x(d_workX,d_workX1,id); // d_work = d (ru) dx
	rX[id.g] =          - 0.5*d_workX[id.g];
	uX[id.g] = uX[id.g] - 0.5*(u[id.g]*d_workX[id.g]);
	vX[id.g] = vX[id.g] - 0.5*(v[id.g]*d_workX[id.g]);
	wX[id.g] = wX[id.g] - 0.5*(w[id.g]*d_workX[id.g]);
	eX[id.g] = eX[id.g] - 0.5*(h[id.g]*d_workX[id.g]);

	//Adding here the terms d (phi) dx * ( d (mu) dx -0.5 * ru); (lambda in case of h in rhse);

	derDev1x(d_workX2,mu,id); //d_work2 = d (mu) dx
	derDev1x(d_workX,u,id); // d_work = d (u) dx
	uX[id.g] = uX[id.g] + d_workX[id.g]*(d_workX2[id.g] - 0.5*d_workX1[id.g]);
	rX[id.g] = rX[id.g] - 0.5 * d_workX[id.g]*r[id.g];
	derDev1x(d_workX,r,id); // d_work = d (r) dx
	rX[id.g] = rX[id.g] - 0.5 * d_workX[id.g]*u[id.g];
	derDev1x(d_workX,v,id); // d_work = d (v) dx
	vX[id.g] = vX[id.g] + d_workX[id.g]*(d_workX2[id.g] - 0.5*d_workX1[id.g]);
	derDev1x(d_workX,w,id); // d_work = d (w) dx
	wX[id.g] = wX[id.g] + d_workX[id.g]*(d_workX2[id.g] - 0.5*d_workX1[id.g]);
	derDev1x(d_workX2,lam,id); //d_work2 = d (lam) dx
	derDev1x(d_workX,h,id); // d_work = d (h) dx
	eX[id.g] = eX[id.g] + d_workX[id.g]*(d_workX2[id.g] - 0.5*d_workX1[id.g]);

	//Adding here the terms -0.5 * d (ru*phi) dx;

	d_workX2[id.g] = d_workX1[id.g]*u[id.g];
	__syncthreads();
	derDev1x(d_workX,d_workX2,id);  // d_work = d (ruu) dx
	uX[id.g] = uX[id.g] - 0.5*d_workX[id.g];

	d_workX2[id.g] = d_workX1[id.g]*v[id.g];
	__syncthreads();
	derDev1x(d_workX,d_workX2,id);  // d_work = d (ruv) dx
	vX[id.g] = vX[id.g] - 0.5*d_workX[id.g];

	d_workX2[id.g] = d_workX1[id.g]*w[id.g];
	__syncthreads();
	derDev1x(d_workX,d_workX2,id);  // d_work = d (ruw) dx
	wX[id.g] = wX[id.g] - 0.5*d_workX[id.g];

	d_workX2[id.g] = d_workX1[id.g]*h[id.g];
	__syncthreads();
	derDev1x(d_workX,d_workX2,id);  // d_work = d (ruh) dx
	eX[id.g] = eX[id.g] - 0.5*d_workX[id.g];

	// pressure derivatives
	derDev1x(d_workX,p,id);
	uX[id.g] = uX[id.g] - d_workX[id.g];

}


__global__ void RHSDeviceY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY, 
						   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
						   myprec *t,  myprec *p,  myprec *mu, myprec *lam) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidY();

	//Adding here the terms mu * d^2 (phi) dy^2;

	derDev2y(d_workY1,u,id);
	uY[id.g] = d_workY1[id.g]*mu[id.g];
	derDev2y(d_workY1,v,id);
	vY[id.g] = d_workY1[id.g]*mu[id.g];
	derDev2y(d_workY1,w,id);
	wY[id.g] = d_workY1[id.g]*mu[id.g];
	derDev2y(d_workY1,t,id);
	eY[id.g] = d_workY1[id.g]*lam[id.g];

	// split advection terms 

	d_workY1[id.g] = r[id.g]*v[id.g];

	//Adding here the terms -0.5 * phi * d (rv) dy;

	__syncthreads();
	derDev1y(d_workY,d_workY1,id); // d_work = d (rv) dy
	rY[id.g] =          - 0.5*d_workY[id.g];
	uY[id.g] = uY[id.g] - 0.5*(u[id.g]*d_workY[id.g]);
	vY[id.g] = vY[id.g] - 0.5*(v[id.g]*d_workY[id.g]);
	wY[id.g] = wY[id.g] - 0.5*(w[id.g]*d_workY[id.g]);
	eY[id.g] = eY[id.g] - 0.5*(h[id.g]*d_workY[id.g]);

	//Adding here the terms d (phi) dy * ( d (mu) dy -0.5 * rv); (lambda in case of h in rhse);

	derDev1y(d_workY2,mu,id); //d_work2 = d (mu) dy
	derDev1y(d_workY ,u ,id); // d_work = d (u) dy
	uY[id.g] = uY[id.g] + d_workY[id.g]*(d_workY2[id.g] - 0.5*d_workY1[id.g]);
	derDev1y(d_workY,v,id); // d_work = d (v) dy
	vY[id.g] = vY[id.g] + d_workY[id.g]*(d_workY2[id.g] - 0.5*d_workY1[id.g]);
	rY[id.g] = rY[id.g] - 0.5 * d_workY[id.g]*r[id.g];
	derDev1y(d_workY,r,id); // d_work = d (r) dy
	rY[id.g] = rY[id.g] - 0.5 * d_workY[id.g]*v[id.g];
	derDev1y(d_workY,w,id); // d_work = d (w) dy
	wY[id.g] = wY[id.g] + d_workY[id.g]*(d_workY2[id.g] - 0.5*d_workY1[id.g]);
	derDev1y(d_workY2,lam,id); //d_work2 = d (lam) dy
	derDev1y(d_workY ,h  ,id); // d_work = d (h) dy
	eY[id.g] = eY[id.g] + d_workY[id.g]*(d_workY2[id.g] - 0.5*d_workY1[id.g]);

	//Adding here the terms -0.5 * d (rv*phi) dy;

	d_workY2[id.g] = d_workY1[id.g]*u[id.g];
	__syncthreads();
	derDev1y(d_workY,d_workY2,id);  // d_work = d (ruv) dy
	uY[id.g] = uY[id.g] - 0.5*d_workY[id.g];

	d_workY2[id.g] = d_workY1[id.g]*v[id.g];
	__syncthreads();
	derDev1y(d_workY,d_workY2,id);  // d_work = d (rvv) dy
	vY[id.g] = vY[id.g] - 0.5*d_workY[id.g];

	d_workY2[id.g] = d_workY1[id.g]*w[id.g];
	__syncthreads();
	derDev1y(d_workY,d_workY2,id);  // d_work = d (rvw) dy
	wY[id.g] = wY[id.g] - 0.5*d_workY[id.g];

	d_workY2[id.g] = d_workY1[id.g]*h[id.g];
	__syncthreads();
	derDev1y(d_workY,d_workY2,id);  // d_work = d (rvh) dy
	eY[id.g] = eY[id.g] - 0.5*d_workY[id.g];

	// pressure derivatives
	derDev1y(d_workY,p,id);
	vY[id.g] = vY[id.g] - d_workY[id.g];
}


__global__ void RHSDeviceZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ, 
						   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
						   myprec *t,  myprec *p,  myprec *mu, myprec *lam) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZ();

	// viscous fluxes derivative
	derDev2z(d_workZ1,u,id);
	uZ[id.g] = 0.0; //d_workZ1[id.g]*mu[id.g];
	derDev2z(d_workZ1,v,id);
	vZ[id.g] = d_workZ1[id.g]*mu[id.g];
	derDev2z(d_workZ1,w,id);
	wZ[id.g] = d_workZ1[id.g]*mu[id.g];
	derDev2z(d_workZ1,t,id);
	eZ[id.g] = 0.0; //d_workZ1[id.g]*lam[id.g];

	// split advection terms 
	d_workZ1[id.g] = r[id.g]*w[id.g]; // d_work = r*w
	__syncthreads();

	//Adding here the terms -0.5 * phi * d (rw) dz;

	derDev1z(d_workZ,d_workZ1,id); // d_work = d (rw) dz
	rZ[id.g] =          - 0.5*d_workZ[id.g];
	uZ[id.g] = uZ[id.g] - 0.5*(u[id.g]*d_workZ[id.g]);
	vZ[id.g] = vZ[id.g] - 0.5*(v[id.g]*d_workZ[id.g]);
	wZ[id.g] = wZ[id.g] - 0.5*(w[id.g]*d_workZ[id.g]);
	eZ[id.g] = eZ[id.g] - 0.5*(h[id.g]*d_workZ[id.g]);

	//Adding here the terms d (phi) dz * ( d (mu) dz -0.5 * rw); (lambda in case of h in rhse);

	derDev1z(d_workZ2,mu,id); //d_work2 = d (mu) dz
	derDev1z(d_workZ,u,id); // d_work = d (u) dz
	uZ[id.g] = uZ[id.g] + d_workZ[id.g]*(d_workZ2[id.g] - 0.5*d_workZ1[id.g]);
	derDev1z(d_workZ,v,id); // d_work = d (v) dz
	vZ[id.g] = vZ[id.g] + d_workZ[id.g]*(d_workZ2[id.g] - 0.5*d_workZ1[id.g]);
	derDev1z(d_workZ,w,id); // d_work = d (w) dz
	wZ[id.g] = wZ[id.g] + d_workZ[id.g]*(d_workZ2[id.g] - 0.5*d_workZ1[id.g]);
	rZ[id.g] = rZ[id.g] - 0.5 * d_workZ[id.g]*r[id.g];
	derDev1z(d_workZ,r,id); // d_work = d (r) dz
	rZ[id.g] = rZ[id.g] - 0.5 * d_workZ[id.g]*w[id.g];
	derDev1z(d_workZ2,lam,id); //d_work2 = d (lam) dz
	derDev1z(d_workZ,h,id); // d_work = d (h) dz
	eZ[id.g] = eZ[id.g] + d_workZ[id.g]*(d_workZ2[id.g] - 0.5*d_workZ1[id.g]);

	//Adding here the terms -0.5 * d (rw*phi) dz;

	d_workZ2[id.g] = d_workZ1[id.g]*u[id.g];
	__syncthreads();
	derDev1z(d_workZ,d_workZ2,id);  // d_work = d (ruw) dz
	uZ[id.g] = uZ[id.g] - 0.5*d_workZ[id.g];

	d_workZ2[id.g] = d_workZ1[id.g]*v[id.g];
	__syncthreads();
	derDev1z(d_workZ,d_workZ2,id);  // d_work = d (rvw) dz
	vZ[id.g] = vZ[id.g] - 0.5*d_workZ[id.g];

	d_workZ2[id.g] = d_workZ1[id.g]*w[id.g];
	__syncthreads();
	derDev1z(d_workZ,d_workZ2,id);  // d_work = d (rww) dz
	wZ[id.g] = wZ[id.g] - 0.5*d_workZ[id.g];

	d_workZ2[id.g] = d_workZ1[id.g]*h[id.g];
	__syncthreads();
	derDev1z(d_workZ,d_workZ2,id);  // d_work = d (rwh) dz
	eZ[id.g] = eZ[id.g] - 0.5*d_workZ[id.g];


	// pressure derivatives
	derDev1z(d_workZ,p,id);
	wZ[id.g] = wZ[id.g] - d_workZ[id.g];

}


__global__ void RHSDeviceXL(myprec *rhsX, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1xL(rhsX,var,id);
	int i     = id.tix;
	int jBase = id.bix*lPencils;
	int k     = id.biy;
	for (int sj = id.tiy; sj < lPencils; sj += id.bdy) {
		int globalIdx = k * mx * my + (jBase + sj) * mx + i;
		rhsX[globalIdx] = -rhsX[globalIdx];
	}
}


__global__ void RHSDeviceYL(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		   	   	   	   	   	myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		   	   	   	   	   	myprec *t,  myprec *p,  myprec *mu, myprec *lam) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	int sum  = id.biy * mx * my + id.bix*id.bdx + id.tix;

	derDev1yL(uY,u,id);
	derDev1yL(vY,v,id);
	derDev1yL(wY,w,id);
	derDev1yL(eY,t,id);
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		uY[glb] = uY[glb]*mu[glb];
		vY[glb] = vY[glb]*mu[glb];
		wY[glb] = wY[glb]*mu[glb];
		eY[glb] = eY[glb]*lam[glb];
		d_workY1[glb] = r[glb]*v[glb]; // d_work = r*v
	}

	__syncthreads();

	//Adding here the terms -0.5 * phi * d (rw) dz;

	derDev1yL(d_workY,d_workY1,id); // d_work = d (rv) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		rY[glb] = - 0.5*d_workY[glb];
		uY[glb] = uY[glb] - 0.5*(u[glb]*d_workY[glb]);
		vY[glb] = vY[glb] - 0.5*(v[glb]*d_workY[glb]);
		wY[glb] = wY[glb] - 0.5*(w[glb]*d_workY[glb]);
		eY[glb] = eY[glb] - 0.5*(h[glb]*d_workY[glb]);
	}

	//Adding here the terms d (phi) dy * ( d (mu) dy -0.5 * rv); (lambda in case of h in rhse);

	derDev1yL(d_workY2,mu,id); //d_work2 = d (mu) dy
	derDev1yL(d_workY ,u,id);  // d_work = d (u) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		uY[glb] = uY[glb] + d_workY[glb]*(d_workY2[glb] - 0.5*d_workY1[glb]);
	}
	derDev1yL(d_workY,v,id); // d_work = d (v) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		vY[glb] = vY[glb] + d_workY[glb]*(d_workY2[glb] - 0.5*d_workY1[glb]);
		rY[glb] = rY[glb] - 0.5 * d_workY[glb]*r[glb];
	}
	derDev1yL(d_workY,w,id); // d_work = d (w) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		wY[glb] = wY[glb] + d_workY[glb]*(d_workY2[glb] - 0.5*d_workY1[glb]);
	}
	derDev1yL(d_workY,r,id); // d_work = d (r) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		rY[glb] = rY[glb] - 0.5 * d_workY[glb]*v[glb];
	}
	derDev1yL(d_workY2,lam,id); //d_work2 = d (lam) dy
	derDev1yL(d_workY ,h,id);   // d_work = d (h) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		eY[glb] = eY[glb] + d_workY[glb]*(d_workY2[glb] - 0.5*d_workY1[glb]);
	}


	//Adding here the terms -0.5 * d (rv*phi) dy;

	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		d_workY2[glb] = d_workY1[glb]*u[glb];
	}
	__syncthreads();
	derDev1yL(d_workY,d_workY2,id);  // d_work = d (ruv) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		uY[glb] = uY[glb] - 0.5*d_workY[glb];
	}

	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		d_workY2[glb] = d_workY1[glb]*v[glb];
	}
	__syncthreads();
	derDev1yL(d_workY,d_workY2,id);  // d_work = d (rvv) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		vY[glb] = vY[glb] - 0.5*d_workY[glb];
	}

	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		d_workY2[glb] = d_workY1[glb]*w[glb];
	}
	__syncthreads();
	derDev1yL(d_workY,d_workY2,id);  // d_work = d (rvw) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		wY[glb] = wY[glb] - 0.5*d_workY[glb];
	}


	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		d_workY2[glb] = d_workY1[glb]*h[glb];
	}
	__syncthreads();
	derDev1yL(d_workY,d_workY2,id);  // d_work = d (rvh) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		eY[glb] = eY[glb] - 0.5*d_workY[glb];
	}

	// pressure derivatives
	derDev1yL(d_workY,p,id);
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		vY[glb] = vY[glb] - d_workY[glb];
	}


}


__global__ void RHSDeviceZL(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		   myprec *t,  myprec *p,  myprec *mu, myprec *lam) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	int sum = id.biy * mx + id.bix*id.bdx + id.tix;

	derDev1zL(uZ,u,id);
	derDev1zL(vZ,v,id);
	derDev1zL(wZ,w,id);
	derDev1zL(eZ,t,id);
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		uZ[glb] = uZ[glb]*mu[glb];
		vZ[glb] = vZ[glb]*mu[glb];
		wZ[glb] = wZ[glb]*mu[glb];
		eZ[glb] = eZ[glb]*lam[glb];
		d_workZ1[glb] = r[glb]*w[glb]; // d_work = r*w
	}

	// split advection terms

	__syncthreads();

	//Adding here the terms -0.5 * phi * d (rw) dz;

	derDev1zL(d_workZ,d_workZ1,id); // d_work = d (rw) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		rZ[glb] = - 0.5*d_workZ[glb];
		uZ[glb] = uZ[glb] - 0.5*(u[glb]*d_workZ[glb]);
		vZ[glb] = vZ[glb] - 0.5*(v[glb]*d_workZ[glb]);
		wZ[glb] = wZ[glb] - 0.5*(w[glb]*d_workZ[glb]);
		eZ[glb] = eZ[glb] - 0.5*(h[glb]*d_workZ[glb]);
	}

	//Adding here the terms d (phi) dz * ( d (mu) dz -0.5 * rw); (lambda in case of h in rhse);

	derDev1zL(d_workZ2,mu,id); //d_work2 = d (mu) dz
	derDev1zL(d_workZ,u,id); // d_work = d (u) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		uZ[glb] = uZ[glb] + d_workZ[glb]*(d_workZ2[glb] - 0.5*d_workZ1[glb]);
	}
	derDev1zL(d_workZ,v,id); // d_work = d (v) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		vZ[glb] = vZ[glb] + d_workZ[glb]*(d_workZ2[glb] - 0.5*d_workZ1[glb]);
	}
	derDev1zL(d_workZ,w,id); // d_work = d (w) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		wZ[glb] = wZ[glb] + d_workZ[glb]*(d_workZ2[glb] - 0.5*d_workZ1[glb]);
		rZ[glb] = rZ[glb] - 0.5 * d_workZ[glb]*r[glb];
	}
	derDev1zL(d_workZ,r,id); // d_work = d (r) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		rZ[glb] = rZ[glb] - 0.5 * d_workZ[glb]*w[glb];
	}
	derDev1zL(d_workZ2,lam,id); //d_work2 = d (lam) dz
	derDev1zL(d_workZ,h,id); // d_work = d (h) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		eZ[glb] = eZ[glb] + d_workZ[glb]*(d_workZ2[glb] - 0.5*d_workZ1[glb]);
	}

	//Adding here the terms -0.5 * d (rw*phi) dz;

	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		d_workZ2[glb] = d_workZ1[glb]*u[glb];
	}
	__syncthreads();
	derDev1zL(d_workZ,d_workZ2,id);  // d_work = d (ruw) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		uZ[glb] = uZ[glb] - 0.5*d_workZ[glb];
	}

	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		d_workZ2[glb] = d_workZ1[glb]*v[glb];
	}
	__syncthreads();
	derDev1zL(d_workZ,d_workZ2,id);  // d_work = d (rvw) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		vZ[glb] = vZ[glb] - 0.5*d_workZ[glb];
	}

	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		d_workZ2[glb] = d_workZ1[glb]*w[glb];
	}
	__syncthreads();
	derDev1zL(d_workZ,d_workZ2,id);  // d_work = d (rww) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		wZ[glb] = wZ[glb] - 0.5*d_workZ[glb];
	}


	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		d_workZ2[glb] = d_workZ1[glb]*h[glb];
	}
	__syncthreads();
	derDev1zL(d_workZ,d_workZ2,id);  // d_work = d (rwh) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		eZ[glb] = eZ[glb] - 0.5*d_workZ[glb];
	}

	// pressure derivatives
	derDev1zL(d_workZ,p,id);
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		wZ[glb] = wZ[glb] - d_workZ[glb];
	}

}


__global__ void RHSDeviceSharedX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
								 myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
								 myprec *t,  myprec *p,  myprec *mu, myprec *lam) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	int si = id.i + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	myprec rXtmp=0;
	myprec uXtmp=0;
	myprec vXtmp=0;
	myprec wXtmp=0;
	myprec eXtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;

	__shared__ myprec s_r[sPencils][mx+stencilSize*2];
	__shared__ myprec s_u[sPencils][mx+stencilSize*2];
	__shared__ myprec s_v[sPencils][mx+stencilSize*2];
	__shared__ myprec s_w[sPencils][mx+stencilSize*2];
	__shared__ myprec s_h[sPencils][mx+stencilSize*2];
	__shared__ myprec s_t[sPencils][mx+stencilSize*2];
	__shared__ myprec s_p[sPencils][mx+stencilSize*2];
	__shared__ myprec s_m[sPencils][mx+stencilSize*2];
	__shared__ myprec s_l[sPencils][mx+stencilSize*2];
	__shared__ myprec s_wrk1[sPencils][mx+stencilSize*2];


	s_r[sj][si] = r[id.g];
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_h[sj][si] = h[id.g];
	s_t[sj][si] = t[id.g];
	s_p[sj][si] = p[id.g];
	s_m[sj][si] = mu[id.g];
	s_l[sj][si] = lam[id.g];
	__syncthreads();

	s_wrk1[sj][si] = s_r[sj][si]*s_u[sj][si];

	__syncthreads();

	// fill in periodic images in shared memory array
	if (id.i < stencilSize) {
		s_r[sj][si-stencilSize]  = s_r[sj][si+mx-stencilSize];
		s_r[sj][si+mx]           = s_r[sj][si];
		s_u[sj][si-stencilSize]  = s_u[sj][si+mx-stencilSize];
		s_u[sj][si+mx]           = s_u[sj][si];
		s_v[sj][si-stencilSize]  = s_v[sj][si+mx-stencilSize];
		s_v[sj][si+mx]           = s_v[sj][si];
		s_w[sj][si-stencilSize]  = s_w[sj][si+mx-stencilSize];
		s_w[sj][si+mx]           = s_w[sj][si];
		s_h[sj][si-stencilSize]  = s_h[sj][si+mx-stencilSize];
		s_h[sj][si+mx]           = s_h[sj][si];
		s_t[sj][si-stencilSize]  = s_t[sj][si+mx-stencilSize];
		s_t[sj][si+mx]           = s_t[sj][si];
		s_p[sj][si-stencilSize]  = s_p[sj][si+mx-stencilSize];
		s_p[sj][si+mx]           = s_p[sj][si];
		s_m[sj][si-stencilSize]  = s_m[sj][si+mx-stencilSize];
		s_m[sj][si+mx]           = s_m[sj][si];
		s_l[sj][si-stencilSize]  = s_l[sj][si+mx-stencilSize];
		s_l[sj][si+mx]           = s_l[sj][si];
		s_wrk1[sj][si-stencilSize]  = s_wrk1[sj][si+mx-stencilSize];
		s_wrk1[sj][si+mx]           = s_wrk1[sj][si];
	}

	__syncthreads();

	// viscous fluxes derivative
	derDevShared2x(&wrk1,s_u[sj],si);
	uXtmp = wrk1*s_m[sj][si];
	derDevShared2x(&wrk1,s_v[sj],si);
	vXtmp = wrk1*s_m[sj][si];
	derDevShared2x(&wrk1,s_w[sj],si);
	wXtmp = wrk1*s_m[sj][si];
	derDevShared2x(&wrk1,s_h[sj],si);
	eXtmp = wrk1*s_l[sj][si];
	__syncthreads();

	// split advection terms

	//Adding here the terms -0.5 * phi * d (ru) dx;

	derDevShared1x(&wrk1,s_wrk1[sj],si); // wrk1 = d (ru) dx
	rXtmp =       - 0.5*wrk1;
	uXtmp = uXtmp - 0.5*(s_u[sj][si]*wrk1);
	vXtmp = vXtmp - 0.5*(s_v[sj][si]*wrk1);
	wXtmp = wXtmp - 0.5*(s_w[sj][si]*wrk1);
	eXtmp = eXtmp - 0.5*(s_h[sj][si]*wrk1);

	//Adding here the terms d (phi) dx * ( d (mu) dx -0.5 * ru); (lambda in case of h in rhse);

	derDevShared1x(&wrk2,s_m[sj],si); //wrk2 = d (mu) dx
	derDevShared1x(&wrk1,s_u[sj],si); //wrk1 = d (u) dx
	uXtmp = uXtmp + wrk1*(wrk2 - 0.5*s_wrk1[sj][si]);
	rXtmp = rXtmp - 0.5 * wrk1*s_r[sj][si];
	derDevShared1x(&wrk1,s_r[sj],si); // wrk1 = d (r) dx
	rXtmp = rXtmp - 0.5 * wrk1*s_u[sj][si];
	derDevShared1x(&wrk1,s_v[sj],si); // wrk1 = d (v) dx
	vXtmp = vXtmp + wrk1*(wrk2 - 0.5*s_wrk1[sj][si]);
	derDevShared1x(&wrk1,s_w[sj],si); // wrk1 = d (w) dx
	wXtmp = wXtmp + wrk1*(wrk2 - 0.5*s_wrk1[sj][si]);
	derDevShared1x(&wrk2,s_l[sj],si); //wrk2 = d (lam) dx
	derDevShared1x(&wrk1,s_h[sj],si); //wrk1 = d (h) dx
	eXtmp = eXtmp + wrk1*(wrk2 - 0.5*s_wrk1[sj][si]);

	//Adding here the terms -0.5 * d (ru*phi) dx;

	s_wrk1[sj][si] = s_r[sj][si]*s_u[sj][si]*s_u[sj][si];
	__syncthreads();
	if (id.i < stencilSize) {
		s_wrk1[sj][si-stencilSize]  = s_wrk1[sj][si+mx-stencilSize];
		s_wrk1[sj][si+mx]           = s_wrk1[sj][si];
	}
	__syncthreads();
	derDevShared1x(&wrk2,s_wrk1[sj],si);  // wrk2 = d (ruu) dx
	uXtmp = uXtmp - 0.5*wrk2;

	s_wrk1[sj][si] = s_r[sj][si]*s_u[sj][si]*s_v[sj][si];
	__syncthreads();
	if (id.i < stencilSize) {
		s_wrk1[sj][si-stencilSize]  = s_wrk1[sj][si+mx-stencilSize];
		s_wrk1[sj][si+mx]           = s_wrk1[sj][si];
	}
	__syncthreads();
	derDevShared1x(&wrk2,s_wrk1[sj],si);  // wrk2 = d (ruv) dx
	vXtmp = vXtmp - 0.5*wrk2;

	s_wrk1[sj][si] = s_r[sj][si]*s_u[sj][si]*s_w[sj][si];
	__syncthreads();
	if (id.i < stencilSize) {
		s_wrk1[sj][si-stencilSize]  = s_wrk1[sj][si+mx-stencilSize];
		s_wrk1[sj][si+mx]           = s_wrk1[sj][si];
	}
	__syncthreads();
	derDevShared1x(&wrk2,s_wrk1[sj],si);  // wrk2 = d (ruw) dx
	wXtmp = wXtmp - 0.5*wrk2;

	s_wrk1[sj][si] = s_r[sj][si]*s_u[sj][si]*s_h[sj][si];
	__syncthreads();
	if (id.i < stencilSize) {
		s_wrk1[sj][si-stencilSize]  = s_wrk1[sj][si+mx-stencilSize];
		s_wrk1[sj][si+mx]           = s_wrk1[sj][si];
	}
	derDevShared1x(&wrk2,s_wrk1[sj],si);  // wrk2 = d (ruh) dx
	eXtmp = eXtmp - 0.5*wrk2;

	// pressure derivatives
	derDevShared1x(&wrk1,s_p[sj],si);


	rX[id.g] = rXtmp;
	uX[id.g] = uXtmp - wrk1;
	vX[id.g] = vXtmp;
	wX[id.g] = wXtmp;
	eX[id.g] = eXtmp;

}
