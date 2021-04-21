
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
#include "cuda_math.h"
#include "boundary.h"


/*
 *  The L-versions of the RHS have to be ran with
 *  - the L-version of the derivatives
 *  i.e.: derDev1xL instead of derDev1x
 *  - the L-version of the grid
 *  i.e.: h_gridL[0] instead of h_grid[0]
 */

/* The whole RHS in the X direction is calculated in RHSDeviceSharedFlxX_old thanks to the beneficial memory layout that allows to use small pencils */
/* For the Y and Z direction, fluxes require a small pencil discretization while the rest of the RHS can be calculated on large pencils which speed
 * up significantly the computation. Therefore 5 streams are used
 * stream 0 -> complete X RHS (in RHSDeviceSharedFlxX_old) (small pencil grid)
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
	__shared__ myprec s_s0[sPencils][mx+stencilSize*2];
#if !periodicX
	__shared__ myprec s_s4[sPencils][mx+stencilSize*2];
	__shared__ myprec s_s8[sPencils][mx+stencilSize*2];
#endif
	__shared__ myprec s_dil[sPencils][mx+stencilSize*2];

	s_r[sj][si] = r[id.g];
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_h[sj][si] = h[id.g];
	s_t[sj][si] = t[id.g];
	s_p[sj][si] = p[id.g];
	s_m[sj][si] = mu[id.g];
	s_l[sj][si] = lam[id.g];
	s_s0[sj][si]= sij[0][id.g];
#if !periodicX
	s_s4[sj][si]= sij[4][id.g];
	s_s8[sj][si]= sij[8][id.g];
#endif
	s_dil[sj][si] = dil[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	if (id.i < stencilSize) {
#if periodicX
		perBCx(s_r[sj],si); perBCx(s_u[sj],si);
		perBCx(s_v[sj],si); perBCx(s_w[sj],si);
		perBCx(s_h[sj],si); perBCx(s_t[sj],si);
		perBCx(s_p[sj],si); perBCx(s_m[sj],si);
		perBCx(s_l[sj],si);
#else
		wallBCxMir(s_r[sj],si);
		wallBCxVel(s_u[sj],si); wallBCxVel(s_v[sj],si); wallBCxVel(s_w[sj],si);
		wallBCxExt(s_t[sj],si,1.0,1.0);
		stateBoundTr(s_r[sj], s_t[sj], s_u[sj], s_v[sj], s_w[sj], s_h[sj], s_p[sj], s_m[sj], s_l[sj], si);
		wallBCxMir(s_s0[sj],si); wallBCxVel(s_s4[sj],si);  wallBCxVel(s_s8[sj],si);
#endif
	}

	__syncthreads();

	//initialize momentum RHS with stresses so that they can be added for both viscous terms and viscous heating without having to load additional terms
	uXtmp = ( 2 * sij[0][id.g] - 2./3.*s_dil[sj][si] );
	vXtmp = (     sij[1][id.g] + sij[3][id.g]  );
	wXtmp = (     sij[2][id.g] + sij[6][id.g]  );

	//adding the viscous dissipation part duidx*mu*six
	eXtmp = s_m[sj][si]*(uXtmp*s_s0[sj][si] + vXtmp*sij[1][id.g] + wXtmp*sij[2][id.g]);

	//Adding here the terms d (mu) dx * sxj; (lambda in case of h in rhse);
	derDevSharedV1x(&wrk2,s_m[sj],si); //wrk2 = d (mu) dx
    uXtmp *= wrk2;
	vXtmp *= wrk2;
	wXtmp *= wrk2;

	// viscous fluxes derivative mu*d^2ui dx^2
	derDevSharedV2x(&wrk1,s_u[sj],si);
	uXtmp = uXtmp + wrk1*s_m[sj][si];
	derDevSharedV2x(&wrk1,s_v[sj],si);
	vXtmp = uXtmp + wrk1*s_m[sj][si];
	derDevSharedV2x(&wrk1,s_w[sj],si);
	wXtmp = uXtmp + wrk1*s_m[sj][si];

	//adding the viscous dissipation part ui*(mu * d2duidx2 + dmudx * six)
	derDevSharedV2x(&wrk1,s_t[sj],si);
	eXtmp = eXtmp + s_u[sj][si]*uXtmp + s_v[sj][si]*vXtmp + s_w[sj][si]*wXtmp + wrk1*s_l[sj][si];

	derDevSharedV1x(&wrk2,s_l[sj],si); //wrk2 = d (lam) dx
	derDevSharedV1x(&wrk1,s_t[sj],si); //wrk1 = d (t) dx
	eXtmp = eXtmp + wrk1*wrk2;

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

	// pressure and dilation derivatives
	if (id.i < stencilSize) {
#if periodicX
		perBCx(s_dil[sj],si);
#else
		wallBCxDil(s_dil[sj],s_s0[sj],s_s4[sj],s_s8[sj],si);
#endif
	}
	__syncthreads();

	derDevSharedV1x(&wrk2,s_dil[sj],si);
	derDevShared1x(&wrk1 ,s_p[sj],si);
	uXtmp = uXtmp + s_m[sj][si]*wrk2/3.0     - wrk1 ;
	eXtmp = eXtmp + s_m[sj][si]*wrk2/3.0*s_u[sj][si];

	rX[id.g] = rXtmp;
	uX[id.g] = uXtmp;
	vX[id.g] = vXtmp;
	wX[id.g] = wXtmp;
	eX[id.g] = eXtmp ;
}

__global__ void RHSDeviceSharedFlxY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidYFlx();

	int si = id.j + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	myprec rYtmp=0;
	myprec uYtmp=0;
	myprec vYtmp=0;
	myprec wYtmp=0;
	myprec eYtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;

	__shared__ myprec s_r[sPencils][my+stencilSize*2];
	__shared__ myprec s_u[sPencils][my+stencilSize*2];
	__shared__ myprec s_v[sPencils][my+stencilSize*2];
	__shared__ myprec s_w[sPencils][my+stencilSize*2];
	__shared__ myprec s_h[sPencils][my+stencilSize*2];
	__shared__ myprec s_t[sPencils][my+stencilSize*2];
	__shared__ myprec s_p[sPencils][my+stencilSize*2];
	__shared__ myprec s_m[sPencils][my+stencilSize*2];
	__shared__ myprec s_l[sPencils][my+stencilSize*2];
	__shared__ myprec s_s3[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s4[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s5[sPencils][mz+stencilSize*2];
	__shared__ myprec s_dil[sPencils][my+stencilSize*2];

	s_r[sj][si] = r[id.g];
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_h[sj][si] = h[id.g];
	s_t[sj][si] = t[id.g];
	s_p[sj][si] = p[id.g];
	s_m[sj][si] = mu[id.g];
	s_l[sj][si] = lam[id.g];
	s_dil[sj][si] = dil[id.g];
	s_s3[sj][si] = sij[3][id.g];
	s_s4[sj][si] = sij[4][id.g];
	s_s5[sj][si] = sij[5][id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	if (id.j < stencilSize) {
		perBCy(s_r[sj],si); perBCy(s_u[sj],si);
		perBCy(s_v[sj],si); perBCy(s_w[sj],si);
		perBCy(s_h[sj],si); perBCy(s_t[sj],si);
		perBCy(s_p[sj],si); perBCy(s_m[sj],si);
		perBCy(s_l[sj],si);
	}
	__syncthreads();

	//initialize momentum RHS with stresses so that they can be added for both viscous terms and viscous heating without having to load additional terms
	uYtmp =     s_s3[sj][si] + sij[1][id.g];
	vYtmp = 2 * s_s4[sj][si] - 2./3.*s_dil[sj][si];
	wYtmp =     s_s5[sj][si] + sij[7][id.g];

	//adding the viscous dissipation part duidy*mu*siy
	eYtmp = s_m[sj][si]*(uYtmp*s_s3[sj][si] + vYtmp*s_s4[sj][si] + wYtmp*s_s5[sj][si]);

	//Adding here the terms d (mu) dy * siy;
	derDevSharedV1y(&wrk2,s_m[sj],si); //wrk2 = d (mu) dx
	uYtmp *= wrk2;
	vYtmp *= wrk2;
	wYtmp *= wrk2;

	// viscous fluxes derivative mu*d^2dui dy^2
	derDevSharedV2y(&wrk1,s_u[sj],si);
	uYtmp = uYtmp + wrk1*s_m[sj][si];
	derDevSharedV2y(&wrk1,s_v[sj],si);
	vYtmp = vYtmp + wrk1*s_m[sj][si];
	derDevSharedV2y(&wrk1,s_w[sj],si);
	wYtmp = wYtmp + wrk1*s_m[sj][si];

	//adding the viscous dissipation part ui*(mu * d2duidy2 + dmudy * siy)
	derDevSharedV2y(&wrk1,s_t[sj],si);
	eYtmp = eYtmp + s_u[sj][si]*uYtmp + s_v[sj][si]*vYtmp + s_w[sj][si]*wYtmp + wrk1*s_l[sj][si];

	derDevSharedV1y(&wrk2,s_l[sj],si); //wrk2 = d (lam) dx
	derDevSharedV1y(&wrk1,s_t[sj],si); //wrk1 = d (t) dx
	eYtmp = eYtmp + wrk1*wrk2;


	// split advection terms

	//Adding here the terms - d (ru phi) dy;

	fluxQuadSharedy(&wrk1,s_r[sj],s_v[sj],si);
	rYtmp = wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_r[sj],s_v[sj],s_u[sj],si);
	uYtmp = uYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_r[sj],s_v[sj],s_v[sj],si);
	vYtmp = vYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_r[sj],s_v[sj],s_w[sj],si);
	wYtmp = wYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_r[sj],s_v[sj],s_h[sj],si);
	eYtmp = eYtmp + wrk1;
	__syncthreads();

	// pressure and dilation derivatives
	if (id.j < stencilSize) {
		perBCy(s_dil[sj],si);
	}
	__syncthreads();
	derDevSharedV1y(&wrk2,s_dil[sj],si);
	derDevShared1y(&wrk1,s_p[sj],si);
	vYtmp = vYtmp + s_m[sj][si]*wrk2/3.0     - wrk1 ;
	eYtmp = eYtmp + s_m[sj][si]*wrk2/3.0*s_v[sj][si];

	rY[id.g] = rYtmp;
	uY[id.g] = uYtmp;
	vY[id.g] = vYtmp;
	wY[id.g] = wYtmp;
	eY[id.g] = eYtmp;
}

__global__ void RHSDeviceSharedFlxZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZFlx();

	int si = id.k + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	myprec rZtmp=0;
	myprec uZtmp=0;
	myprec vZtmp=0;
	myprec wZtmp=0;
	myprec eZtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;

	__shared__ myprec s_r[sPencils][mz+stencilSize*2];
	__shared__ myprec s_u[sPencils][mz+stencilSize*2];
	__shared__ myprec s_v[sPencils][mz+stencilSize*2];
	__shared__ myprec s_w[sPencils][mz+stencilSize*2];
	__shared__ myprec s_h[sPencils][mz+stencilSize*2];
	__shared__ myprec s_t[sPencils][mz+stencilSize*2];
	__shared__ myprec s_p[sPencils][mz+stencilSize*2];
	__shared__ myprec s_m[sPencils][mz+stencilSize*2];
	__shared__ myprec s_l[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s6[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s7[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s8[sPencils][mz+stencilSize*2];
	__shared__ myprec s_dil[sPencils][mz+stencilSize*2];

	s_r[sj][si] = r[id.g];
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_h[sj][si] = h[id.g];
	s_t[sj][si] = t[id.g];
	s_p[sj][si] = p[id.g];
	s_m[sj][si] = mu[id.g];
	s_l[sj][si] = lam[id.g];
	s_s6[sj][si] = sij[6][id.g];
	s_s7[sj][si] = sij[7][id.g];
	s_s8[sj][si] = sij[8][id.g];
	s_dil[sj][si] = dil[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	if (id.k < stencilSize) {
		perBCz(s_r[sj],si); perBCz(s_u[sj],si);
		perBCz(s_v[sj],si); perBCz(s_w[sj],si); perBCz(s_t[sj],si);
		perBCz(s_h[sj],si);
		perBCz(s_p[sj],si); perBCz(s_m[sj],si);
		perBCz(s_l[sj],si);
	}

	__syncthreads();

	//initialize momentum RHS with stresses so that they can be added for both viscous terms and viscous heating without having to load additional terms
	uZtmp = (    s_s6[sj][si] + sij[2][id.g]        );
	vZtmp = (    s_s7[sj][si] + sij[5][id.g]        );
	wZtmp = (2 * s_s8[sj][si] - 2./3.*s_dil[sj][si] );

	//adding the viscous dissipation part duidz*mu*siz
	eZtmp = s_m[sj][si]*(uZtmp*s_s6[sj][si] + vZtmp*s_s7[sj][si] + wZtmp*s_s8[sj][si]);

	//Adding here the terms d (mu) dz * szj;
	derDevSharedV1z(&wrk2,s_m[sj],si); //wrk2 = d (mu) dz
    uZtmp *= wrk2;
	vZtmp *= wrk2;
	wZtmp *= wrk2;

	// viscous fluxes derivative
	derDevSharedV2z(&wrk1,s_u[sj],si);
	uZtmp = wrk1*s_m[sj][si];
	derDevSharedV2z(&wrk1,s_v[sj],si);
	vZtmp = wrk1*s_m[sj][si];
	derDevSharedV2z(&wrk1,s_w[sj],si);
	wZtmp = wrk1*s_m[sj][si];

	//adding the viscous dissipation part ui*(mu * d2duidz2 + dmudz * siz)
	derDevSharedV2z(&wrk1,s_t[sj],si);
	eZtmp = eZtmp + s_u[sj][si]*uZtmp + s_v[sj][si]*vZtmp + s_w[sj][si]*wZtmp + wrk1*s_l[sj][si];

	derDevSharedV1z(&wrk2,s_l[sj],si); //wrk2 = d (lam) dz
	derDevSharedV1z(&wrk1,s_t[sj],si); //wrk1 = d (t) dx
	eZtmp = eZtmp + wrk1*wrk2;

	//Adding here the terms - d (ru phi) dz;

	fluxQuadSharedz(&wrk1,s_r[sj],s_w[sj],si);
	rZtmp = wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_r[sj],s_w[sj],s_u[sj],si);
	uZtmp = uZtmp + wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_r[sj],s_w[sj],s_v[sj],si);
	vZtmp = vZtmp + wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_r[sj],s_w[sj],s_w[sj],si);
	wZtmp = wZtmp + wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_r[sj],s_w[sj],s_h[sj],si);
	eZtmp = eZtmp + wrk1;
	__syncthreads();

	// pressure and dilation derivatives
	__syncthreads();
	if (id.k < stencilSize) {
		perBCz(s_dil[sj],si);
	}
	__syncthreads();
	derDevSharedV1z(&wrk2,s_dil[sj],si);
	derDevShared1z(&wrk1,s_p[sj],si);
	wZtmp = wZtmp + s_m[sj][si]*wrk2/3.0     - wrk1 ;
	eZtmp = eZtmp + s_m[sj][si]*wrk2/3.0*s_w[sj][si];

	rZ[id.g] = rZtmp;
	uZ[id.g] = uZtmp;
	vZ[id.g] = vZtmp;
	wZ[id.g] = wZtmp;
	eZ[id.g] = eZtmp; // + 1.0*s_w[sj][si] ;
	__syncthreads();
}



__global__ void RHSDeviceSharedFlxX_old(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
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
#if periodicX
		perBCx(s_r[sj],si); perBCx(s_u[sj],si);
		perBCx(s_v[sj],si); perBCx(s_w[sj],si);
		perBCx(s_h[sj],si); perBCx(s_t[sj],si);
		perBCx(s_p[sj],si); perBCx(s_m[sj],si);
		perBCx(s_l[sj],si);
#else
		wallBCxMir(s_r[sj],si);
		wallBCxVel(s_u[sj],si); wallBCxVel(s_v[sj],si); wallBCxVel(s_w[sj],si);
		wallBCxExt(s_t[sj],si,1.0,1.0);
		stateBoundTr(s_r[sj], s_t[sj], s_u[sj], s_v[sj], s_w[sj], s_h[sj], s_p[sj], s_m[sj], s_l[sj], si);
#endif
	}

	__syncthreads();

	// viscous fluxes derivative
	derDevSharedV2x(&wrk1,s_u[sj],si);
	uXtmp = wrk1*s_m[sj][si];
	derDevSharedV2x(&wrk1,s_v[sj],si);
	vXtmp = wrk1*s_m[sj][si];
	derDevSharedV2x(&wrk1,s_w[sj],si);
	wXtmp = wrk1*s_m[sj][si];
	derDevSharedV2x(&wrk1,s_t[sj],si);
	eXtmp = wrk1*s_l[sj][si];
	__syncthreads();
	derDevSharedV1x(&wrk2,s_l[sj],si); //wrk2 = d (lam) dx
	derDevSharedV1x(&wrk1,s_t[sj],si); //wrk1 = d (t) dx
	eXtmp = eXtmp + wrk1*wrk2;

	//Adding here the terms d (mu) dx * sxj; (lambda in case of h in rhse);

	derDevSharedV1x(&wrk2,s_m[sj],si); //wrk2 = d (mu) dx
	uXtmp = uXtmp + wrk2*sij[0][id.g];
	vXtmp = vXtmp + wrk2*sij[1][id.g];
	wXtmp = wXtmp + wrk2*sij[2][id.g];

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

	// pressure and dilation derivatives
	s_wrk[sj][si] = dil[id.g];
	__syncthreads();

	if (id.i < stencilSize) {
#if periodicX
		perBCx(s_wrk[sj],si);
#else
		wallBCxMir(s_wrk[sj],si);
#endif
	}
	__syncthreads();

	derDevSharedV1x(&wrk2,s_wrk[sj],si);
	derDevShared1x(&wrk1,s_p[sj],si);
	uXtmp = uXtmp - wrk1 + s_m[sj][si]*wrk2*1.0/3.0;

	//viscous dissipation
	s_wrk[sj][si] = s_m[sj][si]*(
					s_u[sj][si]*(  sij[0][id.g]  ) +
					s_v[sj][si]*(  sij[1][id.g]  ) +
					s_w[sj][si]*(  sij[2][id.g]  )
					);
	__syncthreads();

	if (id.i < stencilSize) {
#if periodicX
		perBCx(s_wrk[sj],si);
#else
		wallBCxMir(s_wrk[sj],si);
#endif
	}
	__syncthreads();

	derDevSharedV1x(&wrk2,s_wrk[sj],si);

	rX[id.g] = rXtmp;
	uX[id.g] = uXtmp;
	vX[id.g] = vXtmp;
	wX[id.g] = wXtmp;
	eX[id.g] = eXtmp + wrk2;
}

__global__ void RHSDeviceSharedFlxY_old(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidYFlx();

	int si = id.j + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	myprec rYtmp=0;
	myprec uYtmp=0;
	myprec vYtmp=0;
	myprec wYtmp=0;
	myprec eYtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;

	__shared__ myprec s_r[sPencils][my+stencilSize*2];
	__shared__ myprec s_u[sPencils][my+stencilSize*2];
	__shared__ myprec s_v[sPencils][my+stencilSize*2];
	__shared__ myprec s_w[sPencils][my+stencilSize*2];
	__shared__ myprec s_h[sPencils][my+stencilSize*2];
	__shared__ myprec s_t[sPencils][my+stencilSize*2];
	__shared__ myprec s_p[sPencils][my+stencilSize*2];
	__shared__ myprec s_m[sPencils][my+stencilSize*2];
	__shared__ myprec s_l[sPencils][my+stencilSize*2];
	__shared__ myprec s_s1[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s2[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s3[sPencils][mz+stencilSize*2];
	__shared__ myprec s_wrk[sPencils][my+stencilSize*2];

	s_r[sj][si] = r[id.g];
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_h[sj][si] = h[id.g];
	s_t[sj][si] = t[id.g];
	s_p[sj][si] = p[id.g];
	s_m[sj][si] = mu[id.g];
	s_s1[sj][si] = sij[3][id.g];
	s_s2[sj][si] = sij[4][id.g];
	s_s3[sj][si] = sij[5][id.g];
	s_l[sj][si] = lam[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	if (id.j < stencilSize) {
		perBCy(s_r[sj],si); perBCy(s_u[sj],si);
		perBCy(s_v[sj],si); perBCy(s_w[sj],si);
		perBCy(s_h[sj],si); perBCy(s_t[sj],si);
		perBCy(s_p[sj],si); perBCy(s_m[sj],si);
		perBCy(s_l[sj],si);
	}

	__syncthreads();

	// viscous fluxes derivative
	derDevSharedV2y(&wrk1,s_u[sj],si);
	uYtmp = wrk1*s_m[sj][si];
	derDevSharedV2y(&wrk1,s_v[sj],si);
	vYtmp = wrk1*s_m[sj][si];
	derDevSharedV2y(&wrk1,s_w[sj],si);
	wYtmp = wrk1*s_m[sj][si];
	derDevSharedV2y(&wrk1,s_t[sj],si);
	eYtmp = wrk1*s_l[sj][si];
	__syncthreads();
	derDevSharedV1y(&wrk2,s_l[sj],si); //wrk2 = d (lam) dx
	derDevSharedV1y(&wrk1,s_t[sj],si); //wrk1 = d (t) dx
	eYtmp = eYtmp + wrk1*wrk2;

	//Adding here the terms d (mu) dy * syj; (lambda in case of h in rhse);

	derDevSharedV1y(&wrk2,s_m[sj],si); //wrk2 = d (mu) dx
	uYtmp = uYtmp + wrk2*s_s1[sj][si];
	vYtmp = vYtmp + wrk2*s_s2[sj][si];
	wYtmp = wYtmp + wrk2*s_s3[sj][si];

	// split advection terms

	//Adding here the terms - d (ru phi) dy;

	fluxQuadSharedy(&wrk1,s_r[sj],s_v[sj],si);
	rYtmp = wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_r[sj],s_v[sj],s_u[sj],si);
	uYtmp = uYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_r[sj],s_v[sj],s_v[sj],si);
	vYtmp = vYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_r[sj],s_v[sj],s_w[sj],si);
	wYtmp = wYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_r[sj],s_v[sj],s_h[sj],si);
	eYtmp = eYtmp + wrk1;
	__syncthreads();

	// pressure and dilation derivatives
	s_wrk[sj][si] = dil[id.g];
	__syncthreads();
	if (id.j < stencilSize) {
		perBCy(s_wrk[sj],si);
	}
	__syncthreads();
	derDevSharedV1y(&wrk2,s_wrk[sj],si);
	derDevShared1y(&wrk1,s_p[sj],si);
	vYtmp = vYtmp - wrk1 + s_m[sj][si]*wrk2*1.0/3.0;

	//viscous dissipation
	s_wrk[sj][si] = s_m[sj][si]*(
					s_u[sj][si]*(  s_s1[sj][si]  ) +
					s_v[sj][si]*(  s_s2[sj][si]  ) +
					s_w[sj][si]*(  s_s3[sj][si]  )
					);
	__syncthreads();
	if (id.j < stencilSize) {
		perBCy(s_wrk[sj],si);
	}
	__syncthreads();
	derDevSharedV1y(&wrk2,s_wrk[sj],si);

	rY[id.g] = rYtmp;
	uY[id.g] = uYtmp;
	vY[id.g] = vYtmp;
	wY[id.g] = wYtmp;
	eY[id.g] = eYtmp + wrk2;
}

__global__ void RHSDeviceSharedFlxZ_old(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *sij[9], myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZFlx();

	int si = id.k + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	myprec rZtmp=0;
	myprec uZtmp=0;
	myprec vZtmp=0;
	myprec wZtmp=0;
	myprec eZtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;

	__shared__ myprec s_r[sPencils][mz+stencilSize*2];
	__shared__ myprec s_u[sPencils][mz+stencilSize*2];
	__shared__ myprec s_v[sPencils][mz+stencilSize*2];
	__shared__ myprec s_w[sPencils][mz+stencilSize*2];
	__shared__ myprec s_h[sPencils][mz+stencilSize*2];
	__shared__ myprec s_t[sPencils][mz+stencilSize*2];
	__shared__ myprec s_p[sPencils][mz+stencilSize*2];
	__shared__ myprec s_m[sPencils][mz+stencilSize*2];
	__shared__ myprec s_l[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s1[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s2[sPencils][mz+stencilSize*2];
	__shared__ myprec s_s3[sPencils][mz+stencilSize*2];
	__shared__ myprec s_wrk[sPencils][mz+stencilSize*2];

	s_r[sj][si] = r[id.g];
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_h[sj][si] = h[id.g];
	s_t[sj][si] = t[id.g];
	s_p[sj][si] = p[id.g];
	s_m[sj][si] = mu[id.g];
	s_l[sj][si] = lam[id.g];
	s_s1[sj][si] = sij[6][id.g];
	s_s2[sj][si] = sij[7][id.g];
	s_s3[sj][si] = sij[8][id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	if (id.k < stencilSize) {
		perBCz(s_r[sj],si); perBCz(s_u[sj],si);
		perBCz(s_v[sj],si); perBCz(s_w[sj],si);
		perBCz(s_t[sj],si); perBCz(s_h[sj],si);
		perBCz(s_p[sj],si); perBCz(s_m[sj],si);
		perBCz(s_l[sj],si);
	}

	__syncthreads();

	// viscous fluxes derivative
	derDevSharedV2z(&wrk1,s_u[sj],si);
	uZtmp = wrk1*s_m[sj][si];
	derDevSharedV2z(&wrk1,s_v[sj],si);
	vZtmp = wrk1*s_m[sj][si];
	derDevSharedV2z(&wrk1,s_w[sj],si);
	wZtmp = wrk1*s_m[sj][si];
	derDevSharedV2z(&wrk1,s_t[sj],si);
	eZtmp = wrk1*s_l[sj][si];
	__syncthreads();
	derDevSharedV1z(&wrk2,s_l[sj],si); //wrk2 = d (lam) dz
	derDevSharedV1z(&wrk1,s_t[sj],si); //wrk1 = d (t) dx
	eZtmp = eZtmp + wrk1*wrk2;

	//Adding here the terms d (mu) dz * szj; (lambda in case of h in rhse);

	derDevSharedV1z(&wrk2,s_m[sj],si); //wrk2 = d (mu) dz
	uZtmp = uZtmp + wrk2*s_s1[sj][si];
	vZtmp = vZtmp + wrk2*s_s2[sj][si];
	wZtmp = wZtmp + wrk2*s_s3[sj][si];

	// split advection terms

	//Adding here the terms - d (ru phi) dz;

	fluxQuadSharedz(&wrk1,s_r[sj],s_w[sj],si);
	rZtmp = wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_r[sj],s_w[sj],s_u[sj],si);
	uZtmp = uZtmp + wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_r[sj],s_w[sj],s_v[sj],si);
	vZtmp = vZtmp + wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_r[sj],s_w[sj],s_w[sj],si);
	wZtmp = wZtmp + wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_r[sj],s_w[sj],s_h[sj],si);
	eZtmp = eZtmp + wrk1;
	__syncthreads();

	// pressure and dilation derivatives
	s_wrk[sj][si] = dil[id.g];
	__syncthreads();
	if (id.k < stencilSize) {
		perBCz(s_wrk[sj],si);
	}
	__syncthreads();
	derDevSharedV1z(&wrk2,s_wrk[sj],si);
	derDevShared1z(&wrk1,s_p[sj],si);
	wZtmp = wZtmp - wrk1 + s_m[sj][si]*wrk2*1.0/3.0;

	//viscous dissipation
	s_wrk[sj][si] = s_m[sj][si]*(
					s_u[sj][si]*(  s_s1[sj][si]  ) +
					s_v[sj][si]*(  s_s2[sj][si]  ) +
					s_w[sj][si]*(  s_s3[sj][si]  )
					);
	__syncthreads();
	if (id.k < stencilSize) {
		perBCz(s_wrk[sj],si);
	}
	__syncthreads();
	derDevSharedV1z(&wrk2,s_wrk[sj],si);

	rZ[id.g] = rZtmp;
	uZ[id.g] = uZtmp;
	vZ[id.g] = vZtmp;
	wZ[id.g] = wZtmp;
	eZ[id.g] = eZtmp + wrk2;
}

