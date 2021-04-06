
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

 __device__ myprec d_work[mx*my*mz];

__device__ myprec d_work1[mx*my*mz];
__device__ myprec d_work2[mx*my*mz];

 
__global__ void RHSDeviceX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX, 
						   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *e ,
						   myprec *t,  myprec *p) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	// viscous fluxes derivative
	derDev2x(d_work1,u,id);
	uX[id.g] = d_work1[id.g]/Re;
	derDev2x(d_work1,v,id);
	vX[id.g] = d_work1[id.g]/Re;
	derDev2x(d_work1,w,id);
	wX[id.g] = d_work1[id.g]/Re;
	derDev2x(d_work1,t,id);
	eX[id.g] = d_work1[id.g]/Re/Pr/Ec;

	// split advection terms 
	d_work1[id.g] = r[id.g]*u[id.g];
	__syncthreads();
	derDev1x(d_work,d_work1,id); // d_work = d (ru) dx
	rX[id.g] =          - d_work[id.g];
	uX[id.g] = uX[id.g] - 0.5*(u[id.g]*d_work[id.g]); 
	vX[id.g] = vX[id.g] - 0.5*(v[id.g]*d_work[id.g]); 
	wX[id.g] = wX[id.g] - 0.5*(w[id.g]*d_work[id.g]); 
	eX[id.g] = eX[id.g] - 0.5*(e[id.g]*d_work[id.g]); 

	derDev1x(d_work,u,id); // d_work = d (u) dx
	uX[id.g] = uX[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	derDev1x(d_work,v,id); // d_work = d (v) dx
	vX[id.g] = vX[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	derDev1x(d_work,w,id); // d_work = d (w) dx
	wX[id.g] = wX[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	derDev1x(d_work,e,id); // d_work = d (e) dx
	eX[id.g] = eX[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	
	d_work2[id.g] = d_work1[id.g]*u[id.g];
	__syncthreads();
	derDev1x(d_work,d_work2,id);  // d_work = d (ruu) dx
	uX[id.g] = uX[id.g] - 0.5*d_work[id.g];

	d_work2[id.g] = d_work1[id.g]*v[id.g];
	__syncthreads();
	derDev1x(d_work,d_work2,id);  // d_work = d (ruv) dx
	vX[id.g] = vX[id.g] - 0.5*d_work[id.g];

	d_work2[id.g] = d_work1[id.g]*w[id.g];
	__syncthreads();
	derDev1x(d_work,d_work2,id);  // d_work = d (ruw) dx
	wX[id.g] = wX[id.g] - 0.5*d_work[id.g];

	d_work2[id.g] = d_work1[id.g]*e[id.g];
	__syncthreads();
	derDev1x(d_work,d_work2,id);  // d_work = d (rue) dx
	eX[id.g] = eX[id.g] - 0.5*d_work[id.g];	

	// pressure derivatives
	derDev1x(d_work,p,id);
	uX[id.g] = uX[id.g] - d_work[id.g];
}


__global__ void RHSDeviceY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY, 
						   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *e ,
						   myprec *t,  myprec *p) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidY();

	// viscous fluxes derivative
	derDev2y(d_work1,u,id);
	uY[id.g] = d_work1[id.g]/Re;
	derDev2y(d_work1,v,id);
	vY[id.g] = d_work1[id.g]/Re;
	derDev2y(d_work1,w,id);
	wY[id.g] = d_work1[id.g]/Re;
	derDev2y(d_work1,t,id);
	eY[id.g] = d_work1[id.g]/Re/Pr/Ec;

	// split advection terms 
	d_work1[id.g] = r[id.g]*v[id.g];
	__syncthreads();
	derDev1y(d_work,d_work1,id); // d_work = d (rv) dy
	rY[id.g] =          - d_work[id.g];
	uY[id.g] = uY[id.g] - 0.5*(u[id.g]*d_work[id.g]); 
	vY[id.g] = vY[id.g] - 0.5*(v[id.g]*d_work[id.g]); 
	wY[id.g] = wY[id.g] - 0.5*(w[id.g]*d_work[id.g]); 
	eY[id.g] = eY[id.g] - 0.5*(e[id.g]*d_work[id.g]); 

	derDev1y(d_work,u,id); // d_work = d (u) dy
	uY[id.g] = uY[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	derDev1y(d_work,v,id); // d_work = d (v) dy
	vY[id.g] = vY[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	derDev1y(d_work,w,id); // d_work = d (w) dy
	wY[id.g] = wY[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	derDev1y(d_work,e,id); // d_work = d (e) dy
	eY[id.g] = eY[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	
	d_work2[id.g] = d_work1[id.g]*u[id.g];
	__syncthreads();
	derDev1y(d_work,d_work2,id);  // d_work = d (ruv) dy
	uY[id.g] = uY[id.g] - 0.5*d_work[id.g];

	d_work2[id.g] = d_work1[id.g]*v[id.g];
	__syncthreads();
	derDev1y(d_work,d_work2,id);  // d_work = d (rvv) dy
	vY[id.g] = vY[id.g] - 0.5*d_work[id.g];

	d_work2[id.g] = d_work1[id.g]*w[id.g];
	__syncthreads();
	derDev1y(d_work,d_work2,id);  // d_work = d (rvw) dy
	wY[id.g] = wY[id.g] - 0.5*d_work[id.g];

	d_work2[id.g] = d_work1[id.g]*e[id.g];
	__syncthreads();
	derDev1y(d_work,d_work2,id);  // d_work = d (rve) dy
	eY[id.g] = eY[id.g] - 0.5*d_work[id.g];	

	// pressure derivatives
	derDev1y(d_work,p,id);
	vY[id.g] = vY[id.g] - d_work[id.g];
}


__global__ void RHSDeviceZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ, 
						   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *e ,
						   myprec *t,  myprec *p) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZ();

	// viscous fluxes derivative
	derDev2z(d_work1,u,id);
	uZ[id.g] = d_work1[id.g]/Re;
	derDev2z(d_work1,v,id);
	vZ[id.g] = d_work1[id.g]/Re;
	derDev2z(d_work1,w,id);
	wZ[id.g] = d_work1[id.g]/Re;
	derDev2z(d_work1,t,id);
	eZ[id.g] = d_work1[id.g]/Re/Pr/Ec;

	// split advection terms 
	d_work1[id.g] = r[id.g]*w[id.g];
	__syncthreads();
	derDev1z(d_work,d_work1,id); // d_work = d (rw) dz
	rZ[id.g] =          - d_work[id.g];
	uZ[id.g] = uZ[id.g] - 0.5*(u[id.g]*d_work[id.g]); 
	vZ[id.g] = vZ[id.g] - 0.5*(v[id.g]*d_work[id.g]); 
	wZ[id.g] = wZ[id.g] - 0.5*(w[id.g]*d_work[id.g]); 
	eZ[id.g] = eZ[id.g] - 0.5*(e[id.g]*d_work[id.g]); 

	derDev1z(d_work,u,id); // d_work = d (u) dz
	uZ[id.g] = uZ[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	derDev1z(d_work,v,id); // d_work = d (v) dz
	vZ[id.g] = vZ[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	derDev1z(d_work,w,id); // d_work = d (w) dz
	wZ[id.g] = wZ[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	derDev1z(d_work,e,id); // d_work = d (e) dz
	eZ[id.g] = eZ[id.g] - 0.5*(d_work1[id.g]*d_work[id.g]); 
	
	d_work2[id.g] = d_work1[id.g]*u[id.g];
	__syncthreads();
	derDev1z(d_work,d_work2,id);  // d_work = d (ruw) dz
	uZ[id.g] = uZ[id.g] - 0.5*d_work[id.g];

	d_work2[id.g] = d_work1[id.g]*v[id.g];
	__syncthreads();
	derDev1z(d_work,d_work2,id);  // d_work = d (rvw) dz
	vZ[id.g] = vZ[id.g] - 0.5*d_work[id.g];

	d_work2[id.g] = d_work1[id.g]*w[id.g];
	__syncthreads();
	derDev1z(d_work,d_work2,id);  // d_work = d (rww) dz
	wZ[id.g] = wZ[id.g] - 0.5*d_work[id.g];

	d_work2[id.g] = d_work1[id.g]*e[id.g];
	__syncthreads();
	derDev1z(d_work,d_work2,id);  // d_work = d (rwe) dz
	eZ[id.g] = eZ[id.g] - 0.5*d_work[id.g];	

	// pressure derivatives
	derDev1z(d_work,p,id);
	vZ[id.g] = vZ[id.g] - d_work[id.g];
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


__global__ void RHSDeviceYL(myprec *rhsY, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1yL(rhsY,var,id);

	int i  = id.bix*id.bdx + id.tix;
	int k  = id.biy;
	for (int j = id.tiy; j < my; j += id.bdy) {
		int globalIdx = k * mx * my + j * mx + i;
		rhsY[globalIdx] = -rhsY[globalIdx];
	}
}


__global__ void RHSDeviceZL(myprec *rhsZ, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	int i  = id.bix*id.bdx + id.tix;
	int j  = id.biy;
	derDev1zL(rhsZ,var,id);
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int globalIdx = k * mx * my + j * mx + i;
		rhsZ[globalIdx] = -rhsZ[globalIdx];
	}
}
