
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

/* The whole RHS in the X direction is calculated in RHSDeviceSharedFlxX thanks to the beneficial memory layout that allows to use small pencils */
/* For the Y and Z direction, fluxes require a small pencil discretization while the rest of the RHS can be calculated on large pencils which speed
 * up significantly the computation. Therefore 5 streams are used
 * stream 0 -> complete X RHS (in RHSDeviceSharedFlxX) (small pencil grid)
 * stream 1 -> viscous terms and pressure terms in Y (in RHSDeviceFullYL) (large pencil grid)
 * stream 2 -> viscous terms and pressure terms in Z (in RHSDeviceFullZL) (large pencil grid)
 * stream 3 -> advective fluxes in Y direction (in FLXDeviceY) (small pencil transposed grid)
 * stream 4 -> advective fluxes in Z direction (in FLXDeviceZ) (small pencil transposed grid)*/

__global__ void RHSDeviceSharedFlxX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil) {

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
	__shared__ myprec s_wrk[sPencils][mx+stencilSize*2];

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
	}

	__syncthreads();

	// viscous fluxes derivative
	derDevShared2x(&wrk1,s_u[sj],si);
	uXtmp = 2.0*wrk1*s_m[sj][si];
	derDevShared2x(&wrk1,s_v[sj],si);
	vXtmp = wrk1*s_m[sj][si];
	derDevShared2x(&wrk1,s_w[sj],si);
	wXtmp = wrk1*s_m[sj][si];
	derDevShared2x(&wrk1,s_t[sj],si);
	eXtmp = wrk1*s_l[sj][si];
	__syncthreads();

	// split advection terms

	//Adding here the terms - d (ru phi) dx;

	fluxQuadSharedx(&wrk1,s_r[sj],s_u[sj],si);
	rXtmp = wrk1;
	__syncthreads();
	fluxCubeSharedx(&wrk1,s_r[sj],s_u[sj],s_u[sj],si);
	uXtmp = uXtmp + wrk1;
	__syncthreads();
	fluxCubeSharedx(&wrk1,s_r[sj],s_u[sj],s_v[sj],si);
	vXtmp = vXtmp + wrk1;
	__syncthreads();
	fluxCubeSharedx(&wrk1,s_r[sj],s_u[sj],s_w[sj],si);
	wXtmp = wXtmp + wrk1;
	__syncthreads();
	fluxCubeSharedx(&wrk1,s_r[sj],s_u[sj],s_h[sj],si);
	eXtmp = eXtmp + wrk1;
	__syncthreads();

	//Adding here the terms d (phi) dx * d (mu) dx; (lambda in case of h in rhse);

	derDevShared1x(&wrk2,s_m[sj],si); //wrk2 = d (mu) dx
	derDevShared1x(&wrk1,s_u[sj],si); //wrk1 = d (u) dx
	uXtmp = uXtmp + 2.0*wrk1*wrk2;
	derDevShared1x(&wrk1,s_v[sj],si); // wrk1 = d (v) dx
	vXtmp = vXtmp + wrk1*wrk2;
	derDevShared1x(&wrk1,s_w[sj],si); // wrk1 = d (w) dx
	wXtmp = wXtmp + wrk1*wrk2;
	derDevShared1x(&wrk2,s_l[sj],si); //wrk2 = d (lam) dx
	derDevShared1x(&wrk1,s_t[sj],si); //wrk1 = d (t) dx
	eXtmp = eXtmp + wrk1*wrk2;

	__syncthreads();

	// pressure and dilation derivatives
	s_wrk[sj][si] = dil[id.g]*mu[id.g];
	__syncthreads();
	if (id.i < stencilSize) {
		s_wrk[sj][si-stencilSize]  = s_wrk[sj][si+mx-stencilSize];
		s_wrk[sj][si+mx]           = s_wrk[sj][si];
	}
	__syncthreads();
	derDevShared1x(&wrk2,s_wrk[sj],si);
	derDevShared1x(&wrk1,s_p[sj],si);
	uXtmp = uXtmp - wrk1 + wrk2*2.0/3.0;


	//adding cross derivatives
	s_wrk[sj][si] = sij[3][id.g]*mu[id.g];    // s_work = d (mu dudy) dx
	__syncthreads();
	if (id.i < stencilSize) {
		s_wrk[sj][si-stencilSize]  = s_wrk[sj][si+mx-stencilSize];
		s_wrk[sj][si+mx]           = s_wrk[sj][si];
	}
	__syncthreads();
	derDevShared1x(&wrk2,s_wrk[sj],si);
	vXtmp = vXtmp + wrk2;

	s_wrk[sj][si] = sij[6][id.g]*mu[id.g];    // s_work = d (mu dudz) dx
	__syncthreads();
	if (id.i < stencilSize) {
		s_wrk[sj][si-stencilSize]  = s_wrk[sj][si+mx-stencilSize];
		s_wrk[sj][si+mx]           = s_wrk[sj][si];
	}
	__syncthreads();
	derDevShared1x(&wrk2,s_wrk[sj],si);
	wXtmp = wXtmp + wrk2;

	//viscous dissipation
	s_wrk[sj][si] = s_m[sj][si]*(
					s_u[sj][si]*(2*sij[0][id.g]) +
					s_v[sj][si]*(  sij[3][id.g]  + sij[1][id.g]) +
					s_w[sj][si]*(  sij[6][id.g]  + sij[2][id.g])
					);     // s_work = d (mu dudz) dx
	__syncthreads();
	if (id.i < stencilSize) {
		s_wrk[sj][si-stencilSize]  = s_wrk[sj][si+mx-stencilSize];
		s_wrk[sj][si+mx]           = s_wrk[sj][si];
	}
	__syncthreads();
	derDevShared1x(&wrk2,s_wrk[sj],si);

	rX[id.g] = rXtmp;
	uX[id.g] = uXtmp;
	vX[id.g] = vXtmp;
	wX[id.g] = wXtmp;
	eX[id.g] = eXtmp + wrk2;
}


__global__ void RHSDeviceFullYL(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	int sum  = id.biy * mx * my + id.bix*id.bdx + id.tix;

	derDev1yL(uY,sij[3],id);
	derDev1yL(vY,sij[4],id);
	derDev1yL(wY,sij[5],id);
	derDev1yL(d_workY1,t,id);
	derDev1yL(eY,d_workY1,id);
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		uY[glb] = uY[glb]*mu[glb];
		vY[glb] = 2.0*vY[glb]*mu[glb];
		wY[glb] = wY[glb]*mu[glb];
		eY[glb] = eY[glb]*lam[glb];
	}

	__syncthreads();

	//Adding here the terms d (phi) dy * ( d (mu) dy -0.5 * rv); (lambda in case of h in rhse);

	derDev1yL(d_workY2,mu,id); //d_work2 = d (mu) dy
	derDev1yL(d_workY ,u,id);  // d_work = d (u) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		uY[glb] = uY[glb] + d_workY[glb]*d_workY2[glb];
	}
	derDev1yL(d_workY,v,id); // d_work = d (v) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		vY[glb] = vY[glb] + d_workY[glb]*2.0*d_workY2[glb];
	}
	derDev1yL(d_workY,w,id); // d_work = d (w) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		wY[glb] = wY[glb] + d_workY[glb]*d_workY2[glb];
	}
	derDev1yL(d_workY2,lam,id); //d_work2 = d (lam) dy
	derDev1yL(d_workY ,t,id);   // d_work = d (t) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		eY[glb] = eY[glb] + d_workY[glb]*d_workY2[glb];
	}

	// pressure derivative and dilation derivative
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		d_workY2[glb] = dil[glb]*mu[glb];
	}
	__syncthreads();
	derDev1yL(d_workY ,p,id);
	derDev1yL(d_workY1,d_workY2,id);
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		vY[glb] = vY[glb] - d_workY[glb] + d_workY1[glb]*2.0/3.0;
	}

	//adding cross derivatives
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		d_workY2[glb] = sij[1][glb]*mu[glb];
	}
	__syncthreads();
	derDev1yL(d_workY,d_workY2,id);  // d_work = d (mu dvdx) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		uY[glb] = uY[glb] + d_workY[glb];
	}

	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		d_workY2[glb] = sij[7][glb]*mu[glb];
	}
	__syncthreads();
	derDev1yL(d_workY,d_workY2,id);  // d_work = d (mu dvdz) dy
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		wY[glb] = wY[glb] + d_workY[glb];
	}

	//viscous dissipation
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		d_workY2[glb] =  mu[glb]*(
					u[glb]*(  sij[1][glb]  + sij[3][glb]) +
					v[glb]*(2*sij[4][glb]) +
					w[glb]*(  sij[5][glb]  + sij[7][glb])
					);
	}
	__syncthreads();
	derDev1yL(d_workY,d_workY2,id);
	for (int j = id.tiy; j < my; j += id.bdy) {
		int glb = sum + j * mx ;
		eY[glb] = eY[glb] + d_workY[glb];
	}

}


__global__ void RHSDeviceFullZL(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		   myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		   myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		   myprec *sij[9], myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	int sum = id.biy * mx + id.bix*id.bdx + id.tix;

	derDev1zL(uZ,sij[6],id);
	derDev1zL(vZ,sij[7],id);
	derDev1zL(wZ,sij[8],id);
	derDev1zL(d_workZ1,t,id);
	derDev1zL(eZ,d_workZ1,id);
	__syncthreads();
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		uZ[glb] = uZ[glb]*mu[glb];
		vZ[glb] = vZ[glb]*mu[glb];
		wZ[glb] = 2.0*wZ[glb]*mu[glb];
		eZ[glb] = eZ[glb]*lam[glb];
	}

	// split advection terms

	__syncthreads();

	//Adding here the terms d (phi) dz * ( d (mu) dz -0.5 * rw); (lambda in case of h in rhse);

	derDev1zL(d_workZ2,mu,id); //d_work2 = d (mu) dz
	derDev1zL(d_workZ,u,id); // d_work = d (u) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		uZ[glb] = uZ[glb] + d_workZ[glb]*d_workZ2[glb];
	}
	derDev1zL(d_workZ,v,id); // d_work = d (v) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		vZ[glb] = vZ[glb] + d_workZ[glb]*d_workZ2[glb];
	}
	derDev1zL(d_workZ,w,id); // d_work = d (w) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		wZ[glb] = wZ[glb] + d_workZ[glb]*2.0*d_workZ2[glb];
	}
	derDev1zL(d_workZ2,lam,id); //d_work2 = d (lam) dz
	derDev1zL(d_workZ,t,id); // d_work = d (h) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		eZ[glb] = eZ[glb] + d_workZ[glb]*d_workZ2[glb];
	}


	// pressure derivative and dilation derivative
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		d_workZ2[glb] = dil[glb]*mu[glb];
	}
	__syncthreads();
	derDev1zL(d_workZ,p,id);
	derDev1zL(d_workZ1,d_workZ2,id);
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		wZ[glb] = wZ[glb] - d_workZ[glb] + d_workZ1[glb]*2.0/3.0;
	}

	//adding cross derivatives
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		d_workZ2[glb] = sij[2][glb]*mu[glb];
	}
	__syncthreads();
	derDev1zL(d_workZ,d_workZ2,id);  // d_work = d (mu dwdx) dz
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		uZ[glb] = uZ[glb] + d_workZ[glb];
	}

	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		d_workZ2[glb] = sij[5][glb]*mu[glb];
	}
	derDev1zL(d_workZ,d_workZ2,id);  // d_work = d (mu dwdy) dz
	__syncthreads();
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		vZ[glb] = vZ[glb] + d_workZ[glb];
	}

	//viscous dissipation
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		d_workZ2[glb] =  mu[glb]*(
					u[glb]*(  sij[2][glb] + sij[6][glb]) +
					v[glb]*(  sij[5][glb] + sij[7][glb]) +
					w[glb]*(2*sij[8][glb])
					);
	}
	__syncthreads();
	derDev1zL(d_workZ,d_workZ2,id);  // d_work = d (mu dvdz) dy
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int glb = k * mx * my + sum;
		eZ[glb] = eZ[glb] + d_workZ[glb];
	}

}


__global__ void FLXDeviceY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidYFlx();

	int si = id.j + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	myprec wrk1=0;

	__shared__ myprec s_r[sPencils][my+stencilSize*2];
	__shared__ myprec s_u[sPencils][my+stencilSize*2];
	__shared__ myprec s_v[sPencils][my+stencilSize*2];
	__shared__ myprec s_w[sPencils][my+stencilSize*2];
	__shared__ myprec s_h[sPencils][my+stencilSize*2];
	s_r[sj][si] = r[id.g];
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_h[sj][si] = h[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	if (id.j < stencilSize) {
		s_r[sj][si-stencilSize]  = s_r[sj][si+my-stencilSize];
		s_r[sj][si+my]           = s_r[sj][si];
		s_u[sj][si-stencilSize]  = s_u[sj][si+my-stencilSize];
		s_u[sj][si+my]           = s_u[sj][si];
		s_v[sj][si-stencilSize]  = s_v[sj][si+my-stencilSize];
		s_v[sj][si+my]           = s_v[sj][si];
		s_w[sj][si-stencilSize]  = s_w[sj][si+my-stencilSize];
		s_w[sj][si+my]           = s_w[sj][si];
		s_h[sj][si-stencilSize]  = s_h[sj][si+my-stencilSize];
		s_h[sj][si+my]           = s_h[sj][si];
	}

	__syncthreads();

	//Adding here the terms - d (ru phi) dx;

	fluxQuadSharedG(&wrk1,s_r[sj],s_v[sj],si,d_dy);
	rY[id.g] = wrk1;
	__syncthreads();
	fluxCubeSharedG(&wrk1,s_r[sj],s_v[sj],s_u[sj],si,d_dy);
	uY[id.g] = wrk1;
	__syncthreads();
	fluxCubeSharedG(&wrk1,s_r[sj],s_v[sj],s_v[sj],si,d_dy);
	vY[id.g] = wrk1;
	__syncthreads();
	fluxCubeSharedG(&wrk1,s_r[sj],s_v[sj],s_w[sj],si,d_dy);
	wY[id.g] = wrk1;
	__syncthreads();
	fluxCubeSharedG(&wrk1,s_r[sj],s_v[sj],s_h[sj],si,d_dy);
	eY[id.g] = wrk1;
	__syncthreads();

}


__global__ void FLXDeviceZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZFlx();

	int si = id.k + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	myprec wrk1=0;

	__shared__ myprec s_r[sPencils][mz+stencilSize*2];
	__shared__ myprec s_u[sPencils][mz+stencilSize*2];
	__shared__ myprec s_v[sPencils][mz+stencilSize*2];
	__shared__ myprec s_w[sPencils][mz+stencilSize*2];
	__shared__ myprec s_h[sPencils][mz+stencilSize*2];
	s_r[sj][si] = r[id.g];
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_h[sj][si] = h[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	if (id.k < stencilSize) {
		s_r[sj][si-stencilSize]  = s_r[sj][si+mz-stencilSize];
		s_r[sj][si+mz]           = s_r[sj][si];
		s_u[sj][si-stencilSize]  = s_u[sj][si+mz-stencilSize];
		s_u[sj][si+mz]           = s_u[sj][si];
		s_v[sj][si-stencilSize]  = s_v[sj][si+mz-stencilSize];
		s_v[sj][si+mz]           = s_v[sj][si];
		s_w[sj][si-stencilSize]  = s_w[sj][si+mz-stencilSize];
		s_w[sj][si+mz]           = s_w[sj][si];
		s_h[sj][si-stencilSize]  = s_h[sj][si+mz-stencilSize];
		s_h[sj][si+mz]           = s_h[sj][si];
	}

	__syncthreads();

	//Adding here the terms - d (ru phi) dx;

	fluxQuadSharedG(&wrk1,s_r[sj],s_w[sj],si,d_dz);
	rZ[id.g] = wrk1;
	__syncthreads();
	fluxCubeSharedG(&wrk1,s_r[sj],s_w[sj],s_u[sj],si,d_dz);
	uZ[id.g] = wrk1;
	__syncthreads();
	fluxCubeSharedG(&wrk1,s_r[sj],s_w[sj],s_v[sj],si,d_dz);
	vZ[id.g] = wrk1;
	__syncthreads();
	fluxCubeSharedG(&wrk1,s_r[sj],s_w[sj],s_w[sj],si,d_dz);
	wZ[id.g] = wrk1;
	__syncthreads();
	fluxCubeSharedG(&wrk1,s_r[sj],s_w[sj],s_h[sj],si,d_dz);
	eZ[id.g] = wrk1;
	__syncthreads();
}
