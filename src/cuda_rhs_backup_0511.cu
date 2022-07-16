#include "globals.h"
#include "cuda_functions.h"
#include "cuda_math.h"
#include "sponge.h"
#include "boundary_condition_x.h"
#include "boundary_condition_y.h"
#include "boundary_condition_z.h"

__global__ void deviceRHSX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dudx, myprec *dvdx, myprec *dwdx, myprec *dudy, myprec *dudz,
		myprec *dil, myprec *dpdz, int iNum) {

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

	__shared__ myprec s_u[sPencils][mx+stencilSize*2];
	__shared__ myprec s_v[sPencils][mx+stencilSize*2];
	__shared__ myprec s_w[sPencils][mx+stencilSize*2];
	__shared__ myprec s_t[sPencils][mx+stencilSize*2];
	__shared__ myprec s_p[sPencils][mx+stencilSize*2];
	__shared__ myprec s_prop1[sPencils][mx+stencilSize*2];
	__shared__ myprec s_prop2[sPencils][mx+stencilSize*2];

	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_t[sj][si] = t[id.g];
	s_p[sj][si] = p[id.g];
	s_prop1[sj][si] = mu[id.g];
	s_prop2[sj][si] = lam[id.g];
	__syncthreads();

	// Boundary conditions in the x-direction are in boundary_condition_x.h
	// these are the BCs for u,v,w,p,t,mu,lambda
	BCxNumber1(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],id,si,mx);
	__syncthreads();

	//initialize momentum RHS with stresses so that they can be added for both viscous terms and viscous heating without having to load additional terms
	uXtmp = ( 2 * dudx[id.g] - 2./3.*dil[id.g] );
	vXtmp = (     dvdx[id.g] + dudy[id.g]  );
	wXtmp = (     dwdx[id.g] + dudz[id.g]  );

	//adding the viscous dissipation part duidx*mu*six
	eXtmp = s_prop1[sj][si]*(uXtmp*dudx[id.g] + vXtmp*dvdx[id.g] + wXtmp*dwdx[id.g]);

	//Adding here the terms d (mu) dx * sxj; (lambda in case of h in rhse);
	derDevSharedV1x(&wrk2,s_prop1[sj],si); //wrk2 = d (mu) dx
    uXtmp *= wrk2;
	vXtmp *= wrk2;
	wXtmp *= wrk2;

	// viscous fluxes derivative mu*d^2ui dx^2
	derDevSharedV2x(&wrk1,s_u[sj],si);
	uXtmp = uXtmp + wrk1*s_prop1[sj][si];
	derDevSharedV2x(&wrk1,s_v[sj],si);
	vXtmp = vXtmp + wrk1*s_prop1[sj][si];
	derDevSharedV2x(&wrk1,s_w[sj],si);
	wXtmp = wXtmp + wrk1*s_prop1[sj][si];

	//adding the viscous dissipation part ui*(mu * d2duidx2 + dmudx * six)
	eXtmp = eXtmp + s_u[sj][si]*uXtmp + s_v[sj][si]*vXtmp + s_w[sj][si]*wXtmp;

	//adding the molecular conduction part (d2 temp dx2*lambda + dlambda dx * d temp dx)
	derDevSharedV2x(&wrk1,s_t[sj],si);
	eXtmp = eXtmp + wrk1*s_prop2[sj][si];
	derDevSharedV1x(&wrk2,s_prop2[sj],si); //wrk2 = d (lam) dx
	derDevSharedV1x(&wrk1,s_t[sj],si); //wrk1 = d (t) dx
	eXtmp = eXtmp + wrk1*wrk2;

	// pressure and dilation derivatives
	s_prop2[sj][si] = dil[id.g];
	__syncthreads();

	// these are the BCs for the dilatation
	BCxNumber2(s_prop2[sj],id,si,mx);
	__syncthreads();

	derDevSharedV1x(&wrk2,s_prop2[sj],si);
	derDevShared1x(&wrk1 ,s_p[sj],si);
	uXtmp = uXtmp + s_prop1[sj][si]*wrk2/3.0     - wrk1 ;
	eXtmp = eXtmp + s_prop1[sj][si]*wrk2/3.0*s_u[sj][si];

	//Adding here the terms - d (ru phi) dx;
	s_prop1[sj][si] = r[id.g];
	s_prop2[sj][si] = h[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	// these are the BCs for rho (prop1) and enthalpy (prop2)
	BCxNumber3(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],id,si,mx);
	__syncthreads();

	fluxQuadSharedx(&wrk1,s_prop1[sj],s_u[sj],si);
	rXtmp = wrk1;
	__syncthreads();
	fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_u[sj],si);
	uXtmp = uXtmp + wrk1;
	__syncthreads();
	fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_v[sj],si);
	vXtmp = vXtmp + wrk1;
	__syncthreads();
	fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_w[sj],si);
	wXtmp = wXtmp + wrk1;
	__syncthreads();
	fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_prop2[sj],si);
	eXtmp = eXtmp + wrk1;
	__syncthreads();

	rX[id.g] = rXtmp;
	uX[id.g] = uXtmp;
	vX[id.g] = vXtmp;
	wX[id.g] = wXtmp;
	eX[id.g] = eXtmp;
}

__global__ void deviceRHSY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dvdx, myprec *dudy, myprec *dvdy, myprec *dwdy, myprec *dvdz,
		myprec *dil, myprec *dpdz, int jNum) {

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

	__shared__ myprec s_u[sPencils][my+stencilSize*2];
	__shared__ myprec s_v[sPencils][my+stencilSize*2];
	__shared__ myprec s_w[sPencils][my+stencilSize*2];
	__shared__ myprec s_prop[sPencils][my+stencilSize*2];
	__shared__ myprec s_dil[sPencils][my+stencilSize*2];

	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_prop[sj][si] = mu[id.g];
	s_dil[sj][si] = dil[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	// these are boundary conditions for u,v,w,mu and dilatation
	BCyNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,my);
	__syncthreads();

	//initialize momentum RHS with stresses so that they can be added for both viscous terms and viscous heating without having to load additional terms
	uYtmp = (    dudy[id.g] + dvdx[id.g]        );
	vYtmp = (2 * dvdy[id.g] - 2./3.*s_dil[sj][si] );
	wYtmp = (    dwdy[id.g] + dvdz[id.g]        );

	//adding the viscous dissipation part duidy*mu*siy
	eYtmp = s_prop[sj][si]*(uYtmp*dudy[id.g] + vYtmp*dvdy[id.g] + wYtmp*dvdz[id.g]);

	//Adding here the terms d (mu) dy * syj;
	derDevSharedV1y(&wrk2,s_prop[sj],si); //wrk2 = d (mu) dy
    uYtmp *= wrk2;
	vYtmp *= wrk2;
	wYtmp *= wrk2;

	// viscous fluxes derivative
	derDevSharedV2y(&wrk1,s_u[sj],si);
	uYtmp = uYtmp + wrk1*s_prop[sj][si];
	derDevSharedV2y(&wrk1,s_v[sj],si);
	vYtmp = vYtmp + wrk1*s_prop[sj][si];
	derDevSharedV2y(&wrk1,s_w[sj],si);
	wYtmp = wYtmp + wrk1*s_prop[sj][si];

	//adding the viscous dissipation part ui*(mu * d2duidy2 + dmudy * siy)
	eYtmp = eYtmp + s_u[sj][si]*uYtmp + s_v[sj][si]*vYtmp + s_w[sj][si]*wYtmp;

	//dilation derivatives
	derDevSharedV1y(&wrk2,s_dil[sj],si);
	vYtmp = vYtmp + s_prop[sj][si]*wrk2/3.0;
	eYtmp = eYtmp + s_prop[sj][si]*wrk2/3.0*s_v[sj][si];

	// pressure derivatives
	s_dil[sj][si] = p[id.g];
	__syncthreads();

	// Boundary condition for pressure
	BCyNumber2(s_dil[sj],p,id,si,my);
	__syncthreads();

	derDevShared1y(&wrk1,s_dil[sj],si);
	vYtmp = vYtmp - wrk1;

	// fourier terms
	s_prop[sj][si] = lam[id.g];
	s_dil[sj][si]  = t[id.g];
	__syncthreads();

	// Boundary condition for temperature and thermal conductivity
	BCyNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,my);
	__syncthreads();

	derDevSharedV2y(&wrk1,s_dil[sj],si);
	eYtmp = eYtmp + wrk1*s_prop[sj][si];
	derDevSharedV1y(&wrk2,s_prop[sj],si); //wrk2 = d (lam) dy
	derDevSharedV1y(&wrk1,s_dil[sj] ,si); //wrk1 = d (t) dy
	eYtmp = eYtmp + wrk1*wrk2;

	//Adding here the terms - d (ru phi) dy;
	s_prop[sj][si] = r[id.g];
	s_dil[sj][si]  = h[id.g];
	__syncthreads();
	// Boundary condition for density and enthalpy
	BCyNumber3(s_prop[sj],s_dil[sj],r,h,id,si,my);
	__syncthreads();
	fluxQuadSharedy(&wrk1,s_prop[sj],s_v[sj],si);
	rYtmp = wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_prop[sj],s_v[sj],s_u[sj],si);
	uYtmp = uYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_prop[sj],s_v[sj],s_v[sj],si);
	vYtmp = vYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_prop[sj],s_v[sj],s_w[sj],si);
	wYtmp = wYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_prop[sj],s_v[sj],s_dil[sj],si);
	eYtmp = eYtmp + wrk1;
	__syncthreads();

#if useStreams
	rY[id.g] = rYtmp;
	uY[id.g] = uYtmp;
	vY[id.g] = vYtmp;
	wY[id.g] = wYtmp;
	eY[id.g] = eYtmp;
#else
	rY[id.g] += rYtmp;
	uY[id.g] += uYtmp;
	vY[id.g] += vYtmp;
	wY[id.g] += wYtmp;
	eY[id.g] += eYtmp;
#endif
	__syncthreads();
}

__global__ void deviceRHSZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dwdx, myprec *dwdy, myprec *dudz, myprec *dvdz, myprec *dwdz,
		myprec *dil, myprec *dpdz, int kNum) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	id.mkidZFlx(kNum);

	int si = id.tix + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	myprec rZtmp=0;
	myprec uZtmp=0;
	myprec vZtmp=0;
	myprec wZtmp=0;
	myprec eZtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;

	myprec tmpreg1=0;
	myprec tmpreg2=0;
	myprec tmpreg3=0;
	myprec tmpreg4=0;
	myprec tmpreg5=0;

	__shared__ myprec s_u[sPencils][mz/nDivZ+stencilSize*2];
	__shared__ myprec s_v[sPencils][mz/nDivZ+stencilSize*2];
	__shared__ myprec s_w[sPencils][mz/nDivZ+stencilSize*2];
	__shared__ myprec s_prop[sPencils][mz/nDivZ+stencilSize*2];
	__shared__ myprec s_dil[sPencils][mz/nDivZ+stencilSize*2];

	s_u[sj][si] = u[id.g];
	s_prop[sj][si] = mu[id.g];
	s_dil[sj][si] = dil[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
        tmpreg1 = dudz[id.g];
        tmpreg2 = dwdx[id.g];
        tmpreg3 = dvdz[id.g];
        tmpreg4 = dwdy[id.g];
        tmpreg5 = dwdz[id.g];

	uZtmp = (    tmpreg1 + tmpreg2        );
	vZtmp = (    tmpreg3 + tmpreg4        );
        //__syncthreads(); // for dil - not needed as s_dil is the one updated by the same thread and hence RAW dependent
	wZtmp = (2 * tmpreg5 - 2./3.*s_dil[sj][si] );
	eZtmp = s_prop[sj][si]*(uZtmp*tmpreg1 + vZtmp*tmpreg3 + wZtmp*tmpreg5);
	
        __syncthreads();
	BCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,kNum);
	__syncthreads();

	derDevSharedV1z(&wrk2,s_prop[sj],si); //wrk2 = d (mu) dz
        derDevSharedV2z(&tmpreg1,s_u[sj],si); //& will come or not?
	derDevSharedV2z(&tmpreg2,s_v[sj],si);
	derDevSharedV2z(&tmpreg3,s_w[sj],si);
        derDevSharedV1z(&tmpreg5,s_dil[sj],si);

        uZtmp *= wrk2;
	vZtmp *= wrk2;
	wZtmp *= wrk2;
       
        tmpreg4 = s_prop[sj][si];

	uZtmp = uZtmp + tmpreg1*tmpreg4;
	vZtmp = vZtmp + tmpreg2*tmpreg4;
	wZtmp = wZtmp + tmpreg3*tmpreg4;
	eZtmp = eZtmp + s_u[sj][si]*uZtmp + s_v[sj][si]*vZtmp + s_w[sj][si]*wZtmp;

	wZtmp = wZtmp + tmpreg4*tmpreg5/3.0;
	eZtmp = eZtmp + tmpreg4*tmpreg5/3.0*s_w[sj][si];





	s_dil[sj][si] = p[id.g];
	__syncthreads();
	BCzNumber2(s_dil[sj],p,id,si,mz,kNum);

	__syncthreads();
	derDevShared1z(&wrk1,s_dil[sj],si);
	wZtmp = wZtmp - wrk1;

	// fourier terms
	s_prop[sj][si] = lam[id.g];
	s_dil[sj][si]  = t[id.g];
	__syncthreads();

	// Boundary conditions for thermal conductivity and temperature
	BCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,kNum);
	__syncthreads();

	derDevSharedV2z(&wrk1,s_dil[sj],si);
	eZtmp = eZtmp + wrk1*s_prop[sj][si];
	derDevSharedV1z(&wrk2,s_prop[sj],si); //wrk2 = d (lam) dz
	derDevSharedV1z(&wrk1,s_dil[sj],si); //wrk1 = d (t) dz
	eZtmp = eZtmp + wrk1*wrk2;

	//Adding here the terms - d (ru phi) dz;
	s_prop[sj][si] = r[id.g];
	s_dil[sj][si]  = h[id.g];
	__syncthreads();

	// Boundary conditions for denisty and enthalpy
	BCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,kNum);
	__syncthreads();

	fluxQuadSharedz(&wrk1,s_prop[sj],s_w[sj],si);
	rZtmp = wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_prop[sj],s_w[sj],s_u[sj],si);
	uZtmp = uZtmp + wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_prop[sj],s_w[sj],s_v[sj],si);
	vZtmp = vZtmp + wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_prop[sj],s_w[sj],s_w[sj],si);
	wZtmp = wZtmp + wrk1;
	__syncthreads();
	fluxCubeSharedz(&wrk1,s_prop[sj],s_w[sj],s_dil[sj],si);
	eZtmp = eZtmp + wrk1;
	__syncthreads();

#if useStreams
	rZ[id.g] = rZtmp;
	uZ[id.g] = uZtmp;
	vZ[id.g] = vZtmp;
	wZ[id.g] = wZtmp + *dpdz;
	eZ[id.g] = eZtmp + *dpdz*s_w[sj][si] ;
#else
	rZ[id.g] += rZtmp;
	uZ[id.g] += uZtmp;
	vZ[id.g] += vZtmp;
	wZ[id.g] += wZtmp + *dpdz;
	eZ[id.g] += eZtmp + *dpdz*s_w[sj][si] ;
#endif
	__syncthreads();
}

//__global__ void deviceRHSX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
//		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
//		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
//		myprec *dil, myprec *dpdz) {
//
//	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
//	id.mkidX();
//
//	int si = id.i + stencilSize;       // local i for shared memory access + halo offset
//	int sj = id.tiy;                   // local j for shared memory access
//
//	myprec rXtmp=0;
//	myprec uXtmp=0;
//	myprec vXtmp=0;
//	myprec wXtmp=0;
//	myprec eXtmp=0;
//
//	myprec wrk1=0;
//	myprec wrk2=0;
//
//	__shared__ myprec s_u[sPencils][mx+stencilSize*2];
//	__shared__ myprec s_v[sPencils][mx+stencilSize*2];
//	__shared__ myprec s_w[sPencils][mx+stencilSize*2];
//	__shared__ myprec s_t[sPencils][mx+stencilSize*2];
//	__shared__ myprec s_p[sPencils][mx+stencilSize*2];
//	__shared__ myprec s_prop1[sPencils][mx+stencilSize*2];
//	__shared__ myprec s_prop2[sPencils][mx+stencilSize*2];
//#if !periodicX
//	__shared__ myprec s_s0[sPencils][mx+stencilSize*2];
//	__shared__ myprec s_s4[sPencils][mx+stencilSize*2];
//	__shared__ myprec s_s8[sPencils][mx+stencilSize*2];
//#endif
//	__shared__ myprec s_dil[sPencils][mx+stencilSize*2];
//
//	s_u[sj][si] = u[id.g];
//	s_v[sj][si] = v[id.g];
//	s_w[sj][si] = w[id.g];
//	s_t[sj][si] = t[id.g];
//	s_p[sj][si] = p[id.g];
//	s_prop1[sj][si] = mu[id.g];
//	s_prop2[sj][si] = lam[id.g];
//#if !periodicX
//	s_s0[sj][si]= gij[0][id.g];
//	s_s4[sj][si]= gij[4][id.g];
//	s_s8[sj][si]= gij[8][id.g];
//#endif
//	s_dil[sj][si] = dil[id.g];
//	__syncthreads();
//
//	// fill in periodic images in shared memory array
//	if (id.i < stencilSize) {
//#if periodicX
//		perBCx(s_u[sj],si); perBCx(s_v[sj],si); perBCx(s_w[sj],si);
//		perBCx(s_t[sj],si); perBCx(s_p[sj],si); perBCx(s_prop1[sj],si);
//		perBCx(s_prop2[sj],si);
//#else
//		wallBCxMir(s_p[sj],si);
//		wallBCxVel(s_u[sj],si); wallBCxVel(s_v[sj],si); wallBCxVel(s_w[sj],si);
//		wallBCxExt(s_t[sj],si,TwallTop,TwallBot);
//		mlBoundPT(s_prop1[sj], s_prop2[sj],  s_p[sj], s_t[sj], s_u[sj], s_v[sj], s_w[sj], si);
//		wallBCxMir(s_s0[sj],si); wallBCxVel(s_s4[sj],si);  wallBCxVel(s_s8[sj],si);
//#endif
//	}
//
//	__syncthreads();
//
//	//initialize momentum RHS with stresses so that they can be added for both viscous terms and viscous heating without having to load additional terms
//	uXtmp = ( 2 * gij[0][id.g] - 2./3.*s_dil[sj][si] );
//	vXtmp = (     gij[1][id.g] + gij[3][id.g]  );
//	wXtmp = (     gij[2][id.g] + gij[6][id.g]  );
//
//	//adding the viscous dissipation part duidx*mu*six
//	eXtmp = s_prop1[sj][si]*(uXtmp*gij[0][id.g] + vXtmp*gij[1][id.g] + wXtmp*gij[2][id.g]);
//
//	//Adding here the terms d (mu) dx * sxj; (lambda in case of h in rhse);
//	derDevSharedV1x(&wrk2,s_prop1[sj],si); //wrk2 = d (mu) dx
//    uXtmp *= wrk2;
//	vXtmp *= wrk2;
//	wXtmp *= wrk2;
//
//	// viscous fluxes derivative mu*d^2ui dx^2
//	derDevSharedV2x(&wrk1,s_u[sj],si);
//	uXtmp = uXtmp + wrk1*s_prop1[sj][si];
//	derDevSharedV2x(&wrk1,s_v[sj],si);
//	vXtmp = vXtmp + wrk1*s_prop1[sj][si];
//	derDevSharedV2x(&wrk1,s_w[sj],si);
//	wXtmp = wXtmp + wrk1*s_prop1[sj][si];
//
//	//adding the viscous dissipation part ui*(mu * d2duidx2 + dmudx * six)
//	eXtmp = eXtmp + s_u[sj][si]*uXtmp + s_v[sj][si]*vXtmp + s_w[sj][si]*wXtmp;
//
//	//adding the molecular conduction part (d2 temp dx2*lambda + dlambda dx * d temp dx)
//	derDevSharedV2x(&wrk1,s_t[sj],si);
//	eXtmp = eXtmp + wrk1*s_prop2[sj][si];
//	derDevSharedV1x(&wrk2,s_prop2[sj],si); //wrk2 = d (lam) dx
//	derDevSharedV1x(&wrk1,s_t[sj],si); //wrk1 = d (t) dx
//	eXtmp = eXtmp + wrk1*wrk2;
//
//	// pressure and dilation derivatives
//	if (id.i < stencilSize) {
//#if periodicX
//		perBCx(s_dil[sj],si);
//#else
//		wallBCxDil(s_dil[sj],s_s0[sj],s_s4[sj],s_s8[sj],si);
//#endif
//	}
//	__syncthreads();
//
//	derDevSharedV1x(&wrk2,s_dil[sj],si);
//	derDevShared1x(&wrk1 ,s_p[sj],si);
//	uXtmp = uXtmp + s_prop1[sj][si]*wrk2/3.0     - wrk1 ;
//	eXtmp = eXtmp + s_prop1[sj][si]*wrk2/3.0*s_u[sj][si];
//
//	//Adding here the terms - d (ru phi) dx;
//	s_prop1[sj][si] = r[id.g];
//	s_prop2[sj][si] = h[id.g];
//	__syncthreads();
//	// fill in periodic images in shared memory array
//	if (id.i < stencilSize) {
//#if periodicX
//		perBCx(s_prop1[sj],si); perBCx(s_prop2[sj],si);
//#else
//		rhBoundPT(s_prop1[sj], s_prop2[sj],  s_p[sj], s_t[sj], s_u[sj], s_v[sj], s_w[sj], si);
//#endif
//	}
//
//	fluxQuadSharedx(&wrk1,s_prop1[sj],s_u[sj],si);
//	rXtmp = wrk1;
//	__syncthreads();
//	fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_u[sj],si);
//	uXtmp = uXtmp + wrk1;
//	__syncthreads();
//	fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_v[sj],si);
//	vXtmp = vXtmp + wrk1;
//	__syncthreads();
//	fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_w[sj],si);
//	wXtmp = wXtmp + wrk1;
//	__syncthreads();
//	fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_prop2[sj],si);
//	eXtmp = eXtmp + wrk1;
//	__syncthreads();
//
//	rX[id.g] = rXtmp;
//	uX[id.g] = uXtmp;
//	vX[id.g] = vXtmp;
//	wX[id.g] = wXtmp;
//	eX[id.g] = eXtmp;
//}
