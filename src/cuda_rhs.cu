#include "globals.h"
#include "cuda_functions.h"
#include "cuda_math.h"
#include "sponge.h"
#include "boundary_condition_x.h"
#include "boundary_condition_y.h"
#include "boundary_condition_z.h"
#include "boundary.h"
#include "nrbcX.h"
#include "nrbcZ.h"


__global__ void deviceRHSX(myprec *rX, myprec *uX, myprec *vX, myprec *wX, myprec *eX,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dudx, myprec *dvdx, myprec *dwdx, myprec *dudy, myprec *dudz,
		myprec *dvdy, myprec *dwdz, myprec *dil, myprec *dpdz) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	int iNum = blockIdx.z ;
	id.mkidXFlx(iNum);

	int si = id.tix + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	bool boolIsothnrbc_top = ((topbc==3) && (id.i == mx_tot-1));
	bool boolIsothnrbc_bot = ((bottombc==3) && (id.i == 0));
	bool boolFreestreamnrbc_top = ((topbc==4) && (id.i == mx_tot-1));
	bool boolAdiabnrbc_top = ((topbc==6) && (id.i == mx_tot-1));
	bool boolAdiabnrbc_bot = ((bottombc==6) && (id.i == 0));

	myprec rXtmp=0;
	myprec uXtmp=0;
	myprec vXtmp=0;
	myprec wXtmp=0;
	myprec eXtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;
	myprec wrk3=0;

	__shared__ myprec s_u[sPencils/2][mx/nX+stencilSize*2];
	__shared__ myprec s_v[sPencils/2][mx/nX+stencilSize*2];
	__shared__ myprec s_w[sPencils/2][mx/nX+stencilSize*2];
	__shared__ myprec s_t[sPencils/2][mx/nX+stencilSize*2];
	__shared__ myprec s_p[sPencils/2][mx/nX+stencilSize*2];
	__shared__ myprec s_prop1[sPencils/2][mx/nX+stencilSize*2];
	__shared__ myprec s_prop2[sPencils/2][mx/nX+stencilSize*2];

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
	if (id.tix < stencilSize){
		if (nX ==1){
			TopBCxNumber1(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],u,v,w,p,t,mu,lam,id,si,mx, topbc);
			BotBCxNumber1(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],u,v,w,p,t,mu,lam,id,si,mx, bottombc);
		} else {
			if (iNum ==0){
				TopBCxCpy(s_u[sj],u,si,id);TopBCxCpy(s_v[sj],v,si,id);TopBCxCpy(s_w[sj],w,si,id);TopBCxCpy(s_p[sj],p,si,id);
				TopBCxCpy(s_t[sj],t,si,id);TopBCxCpy(s_prop1[sj],mu,si,id);TopBCxCpy(s_prop2[sj],lam,si,id);

				BotBCxNumber1(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],u,v,w,p,t,mu,lam,id,si,mx, bottombc);
			} else if (iNum == nX-1){
				BotBCxCpy(s_u[sj],u,si,id);BotBCxCpy(s_v[sj],v,si,id);BotBCxCpy(s_w[sj],w,si,id);BotBCxCpy(s_p[sj],p,si,id);
				BotBCxCpy(s_t[sj],t,si,id);BotBCxCpy(s_prop1[sj],mu,si,id);BotBCxCpy(s_prop2[sj],lam,si,id);

				TopBCxNumber1(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],u,v,w,p,t,mu,lam,id,si,mx, topbc);
			} else {
				TopBCxCpy(s_u[sj],u,si,id);TopBCxCpy(s_v[sj],v,si,id);TopBCxCpy(s_w[sj],w,si,id);TopBCxCpy(s_p[sj],p,si,id);
				TopBCxCpy(s_t[sj],t,si,id);TopBCxCpy(s_prop1[sj],mu,si,id);TopBCxCpy(s_prop2[sj],lam,si,id);

				BotBCxCpy(s_u[sj],u,si,id);BotBCxCpy(s_v[sj],v,si,id);BotBCxCpy(s_w[sj],w,si,id);BotBCxCpy(s_p[sj],p,si,id);
				BotBCxCpy(s_t[sj],t,si,id);BotBCxCpy(s_prop1[sj],mu,si,id);BotBCxCpy(s_prop2[sj],lam,si,id);
			}
		}
	}

	__syncthreads();

	derDevSharedV1x(&wrk1,s_u[sj],si);
	derDevSharedV1x(&wrk2,s_v[sj],si);
	derDevSharedV1x(&wrk3,s_w[sj],si);

	dudx[id.g] = wrk1;
	dvdx[id.g] = wrk2;
	dwdx[id.g] = wrk3;

	rXtmp  = wrk1 + dvdy[id.g] + dwdz[id.g]; // rXtmp is for rho but here it just acts acts as a temporary variable.
	dil[id.g] = rXtmp;
	//initialize momentum RHS with stresses so that they can be added for both viscous terms and viscous heating without having to load additional terms
	uXtmp = ( 2 * wrk1 - 2./3.*rXtmp ); //rXtmp stores dilatation
	vXtmp = (     wrk2 + dudy[id.g]  );
	wXtmp = (     wrk3 + dudz[id.g]  );

	//adding the viscous dissipation part duidx*mu*six
	eXtmp = s_prop1[sj][si]*(uXtmp*wrk1 + vXtmp*wrk2 + wXtmp*wrk3);

	//Adding here the terms d (mu) dx * sxj; (lambda in case of h in rhse);
	derDevSharedV1x(&wrk2,s_prop1[sj],si); //wrk2 = d (mu) dx
	uXtmp *= wrk2;
	vXtmp *= wrk2;
	wXtmp *= wrk2;

	// viscous fluxes derivative mu*d^2ui dx^2
	derDevSharedV2x(&wrk1,s_u[sj],si);
	uXtmp = uXtmp + wrk1*s_prop1[sj][si];
	derDevSharedV2x(&wrk2,s_v[sj],si);
	vXtmp = vXtmp + wrk2*s_prop1[sj][si];
	derDevSharedV2x(&wrk3,s_w[sj],si);
	wXtmp = wXtmp + wrk3*s_prop1[sj][si];

	if ( boolFreestreamnrbc_top ) {// This is what the NSCBC says (to put derivative of tangential stress ==0. BUT NOTE THAT WE ARE not setting THE CROSS DERIVATIVE TERMS COMING OUT OF TAU12 AND TAU 13 I.E. mu d2u/dxdy and  mu d2u/dxdz equal to zero. This is because to do it we need to edit the Y and Z kernel which will make the code complicated.;
		vXtmp = 0;
		wXtmp = 0;
	}
	//adding the viscous dissipation part ui*(mu * d2duidx2 + dmudx * six)
	eXtmp = eXtmp + s_u[sj][si]*uXtmp + s_v[sj][si]*vXtmp + s_w[sj][si]*wXtmp;

	//adding the molecular conduction part (d2 temp dx2*lambda + dlambda dx * d temp dx)

	if ( boolFreestreamnrbc_top ) { // dqdx = 0;
		eXtmp = eXtmp + 0;
	} else {
		derDevSharedV2x(&wrk1,s_t[sj],si);
		eXtmp = eXtmp + wrk1*s_prop2[sj][si];
		derDevSharedV1x(&wrk2,s_prop2[sj],si); //wrk2 = d (lam) dx
		derDevSharedV1x(&wrk3,s_t[sj],si); //wrk1 = d (t) dx
		eXtmp = eXtmp + wrk2*wrk3;
	}
	// pressure and dilation derivatives
	__syncthreads();
	s_prop2[sj][si] = rXtmp; // rXtmp stores dilatation temporarily
	__syncthreads();

	// these are the BCs for the dilatation

	if (id.tix < stencilSize){
		if (nX ==1){
			TopBCxNumber2(s_prop2[sj],dil,id,si,mx, topbc);
			BotBCxNumber2(s_prop2[sj],dil,id,si,mx, bottombc);
		} else {
			if (iNum ==0){
				TopBCxCpy(s_prop2[sj],dil,si,id);
				BotBCxNumber2(s_prop2[sj],dil,id,si,mx, bottombc);
			} else if (iNum == nX-1){
				BotBCxCpy(s_prop2[sj],dil,si,id);
				TopBCxNumber2(s_prop2[sj],dil,id,si,mx, topbc);
			} else {
				TopBCxCpy(s_prop2[sj],dil,si,id);
				BotBCxCpy(s_prop2[sj],dil,si,id);
			}
		}
	}
	__syncthreads();

	derDevSharedV1x(&wrk2,s_prop2[sj],si);
	derDevShared1x(&wrk1 ,s_p[sj],si);

	if (boolFreestreamnrbc_top){ // pressure is taken care in the nrbc
		wrk1 = 0;
	}
	uXtmp = uXtmp + s_prop1[sj][si]*wrk2/3.0     - wrk1 ;
	eXtmp = eXtmp + s_prop1[sj][si]*wrk2/3.0*s_u[sj][si];


	//Adding here the terms - d (ru phi) dx;
	s_prop1[sj][si] = r[id.g];
	s_prop2[sj][si] = h[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	// these are the BCs for rho (prop1) and enthalpy (prop2)


	if (id.tix < stencilSize){
		if (nX ==1){
			TopBCxNumber3(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],r,h,id,si,mx, topbc);
			BotBCxNumber3(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],r,h,id,si,mx, bottombc);
		} else {
			if (iNum ==0){
				TopBCxCpy(s_prop1[sj],r,si,id);TopBCxCpy(s_prop2[sj],h,si,id);
				BotBCxNumber3(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],r,h,id,si,mx, bottombc);
			} else if (iNum == nX-1){
				BotBCxCpy(s_prop1[sj],r,si,id);BotBCxCpy(s_prop2[sj],h,si,id);
				TopBCxNumber3(s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_t[sj],s_prop1[sj],s_prop2[sj],r,h,id,si,mx, topbc);
			} else {
				TopBCxCpy(s_prop1[sj],r,si,id); TopBCxCpy(s_prop2[sj],h,si,id);
				BotBCxCpy(s_prop1[sj],r,si,id); BotBCxCpy(s_prop2[sj],h,si,id);
			}
		}
	}
	__syncthreads();


	if ( boolIsothnrbc_top){
		IsothnrbcX_top(&rXtmp,&uXtmp,&vXtmp,&wXtmp,&eXtmp, s_prop1[sj],s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_prop2[sj], si, id);
	} else if (boolIsothnrbc_bot){
		IsothnrbcX_bot(&rXtmp,&uXtmp,&vXtmp,&wXtmp,&eXtmp, s_prop1[sj],s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_prop2[sj], si, id);
	} else if (boolFreestreamnrbc_top){
		FreestreamnrbcX_top(&rXtmp,&uXtmp,&vXtmp,&wXtmp,&eXtmp, s_prop1[sj],s_u[sj],s_v[sj],s_w[sj],s_p[sj],s_prop2[sj], si, id);
	} else if (boolAdiabnrbc_top){

	} else if (boolAdiabnrbc_bot){

	} else {

		fluxQuadSharedx(&wrk3,s_prop1[sj],s_u[sj],si); // DONT FORGET TO REMOVE SYNCTHREADS FROM THESE FUNCTIONS AS IT IS PLACED IN A DIVERGENT PATH.
		rXtmp = wrk3;
		fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_u[sj],si);
		uXtmp = uXtmp + wrk1;
		fluxCubeSharedx(&wrk2,s_prop1[sj],s_u[sj],s_v[sj],si);
		vXtmp = vXtmp + wrk2;
		fluxCubeSharedx(&wrk3,s_prop1[sj],s_u[sj],s_w[sj],si);
		wXtmp = wXtmp + wrk3;
		fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_prop2[sj],si);
		eXtmp = eXtmp + wrk1;
	}	



	rX[id.g] = rXtmp;
	uX[id.g] = uXtmp;
	vX[id.g] = vXtmp;
	wX[id.g] = wXtmp;
	eX[id.g] = eXtmp;

	__syncthreads();

}

__global__ void deviceRHSY(myprec *rY, myprec *uY, myprec *vY, myprec *wY, myprec *eY,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dvdx, myprec *dudy, myprec *dvdy, myprec *dwdy, myprec *dvdz,
		myprec *dil, myprec *dpdz) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	int jNum = blockIdx.z;
	id.mkidYFlx(jNum);

	int tid = id.tix + id.bdx*id.tiy;

	int si  = id.tiy + stencilSize;       // local i for shared memory access + halo offset
	int sj  = id.tix;                   // local j for shared memory access
	int si1 = tid%id.bdy +  stencilSize;       // local i for shared memory access + halo offset
	int sj1 = tid/id.bdy;



	// Indices id1(sj1,si1-stencilSize, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
	// id1.mkidYFlx(jNum);

	myprec rYtmp=0;
	myprec uYtmp=0;
	myprec vYtmp=0;
	myprec wYtmp=0;
	myprec eYtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;
	myprec wrk3=0;

	__shared__ myprec s_u[mx/nDivX][my/nDivY+stencilSize*2];
	__shared__ myprec s_v[mx/nDivX][my/nDivY+stencilSize*2];
	__shared__ myprec s_w[mx/nDivX][my/nDivY+stencilSize*2];
	__shared__ myprec s_prop[mx/nDivX][my/nDivY+stencilSize*2];
	__shared__ myprec s_dil[mx/nDivX][my/nDivY+stencilSize*2];
	__shared__ myprec s_tmp[mx/nDivX][my/nDivY+stencilSize*2];

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_prop[sj][si] = mu[id.g];
	s_dil[sj][si] = dil[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	// these are boundary conditions for u,v,w,mu and dilatation

	if (id.tiy < stencilSize){
		if (nDivY ==1){
			if(pRow>1) {
				haloBCyTop(s_u[sj],u,si,id); haloBCyTop(s_v[sj],v,si,id); haloBCyTop(s_w[sj],w,si,id);
				haloBCyTop(s_prop[sj],mu,si,id); haloBCyTop(s_dil[sj],dil,si,id);

				haloBCyBot(s_u[sj],u,si,id); haloBCyBot(s_v[sj],v,si,id); haloBCyBot(s_w[sj],w,si,id);
				haloBCyBot(s_prop[sj],mu,si,id); haloBCyBot(s_dil[sj],dil,si,id);
			} else{
				TopBCyNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,my);
				BotBCyNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,my);
			}
		} else {
			if (jNum ==0){
				TopBCyCpy(s_u[sj],u,si,id); TopBCyCpy(s_v[sj],v,si,id); TopBCyCpy(s_w[sj],w,si,id);
				TopBCyCpy(s_prop[sj],mu,si,id); TopBCyCpy(s_dil[sj],dil,si,id);

				if(pRow>1) {

					haloBCyBot(s_u[sj],u,si,id); haloBCyBot(s_v[sj],v,si,id); haloBCyBot(s_w[sj],w,si,id);
					haloBCyBot(s_prop[sj],mu,si,id); haloBCyBot(s_dil[sj],dil,si,id);
				} else{
					BotBCyNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,my);
				}
			} else if (jNum == nDivY-1){
				BotBCyCpy(s_u[sj],u,si,id); BotBCyCpy(s_v[sj],v,si,id); BotBCyCpy(s_w[sj],w,si,id);
				BotBCyCpy(s_prop[sj],mu,si,id); BotBCyCpy(s_dil[sj],dil,si,id);
				if(pRow>1) {
					haloBCyTop(s_u[sj],u,si,id); haloBCyTop(s_v[sj],v,si,id); haloBCyTop(s_w[sj],w,si,id);
					haloBCyTop(s_prop[sj],mu,si,id); haloBCyTop(s_dil[sj],dil,si,id);

				} else {
					TopBCyNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,my);
				}
			} else {
				TopBCyCpy(s_u[sj],u,si,id); TopBCyCpy(s_v[sj],v,si,id); TopBCyCpy(s_w[sj],w,si,id);
				TopBCyCpy(s_prop[sj],mu,si,id); TopBCyCpy(s_dil[sj],dil,si,id);

				BotBCyCpy(s_u[sj],u,si,id); BotBCyCpy(s_v[sj],v,si,id); BotBCyCpy(s_w[sj],w,si,id);
				BotBCyCpy(s_prop[sj],mu,si,id);BotBCyCpy(s_dil[sj],dil,si,id);
			}
		}
	}

	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	derDevSharedV1y(&wrk1,s_u[sj1],si1); 
	derDevSharedV1y(&wrk2,s_v[sj1],si1); 
	derDevSharedV1y(&wrk3,s_w[sj1],si1); 

	//initialize momentum RHS with stresses so that they can be added for both viscous terms and viscous heating without having to load additional terms

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	s_tmp[sj][si] = dvdx[id.g];
	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	uYtmp = (    wrk1 + s_tmp[sj1][si1]        );
	__syncthreads();
        s_tmp[sj1][si1] = wrk1;
        __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
        dudy[id.g] = s_tmp[sj][si];
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
        __syncthreads();
	vYtmp = (2 * wrk2 - 2./3.*s_dil[sj1][si1] );
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	__syncthreads();
	s_tmp[sj][si] = dvdz[id.g];
	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	wYtmp = (    wrk3 + s_tmp[sj1][si1]        );

	__syncthreads();
        s_tmp[sj1][si1] = wrk2;
        __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
        dvdy[id.g] = s_tmp[sj][si];
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	__syncthreads();
	s_tmp[sj1][si1] = wrk3;
	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	dwdy[id.g] = s_tmp[sj][si];
	__syncthreads();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP



	//adding the viscous dissipation part duidy*mu*siy
	eYtmp = s_prop[sj1][si1]*(uYtmp*wrk1 + vYtmp*wrk2 + wYtmp*wrk3);

	//Adding here the terms d (mu) dy * syj;
	derDevSharedV1y(&wrk2,s_prop[sj1],si1); //wrk2 = d (mu) dy
	uYtmp *= wrk2;
	vYtmp *= wrk2;
	wYtmp *= wrk2;

	// viscous fluxes derivative
	derDevSharedV2y(&wrk1,s_u[sj1],si1);
	uYtmp = uYtmp + wrk1*s_prop[sj1][si1];
	derDevSharedV2y(&wrk1,s_v[sj1],si1);
	vYtmp = vYtmp + wrk1*s_prop[sj1][si1];
	derDevSharedV2y(&wrk1,s_w[sj1],si1);
	wYtmp = wYtmp + wrk1*s_prop[sj1][si1];

	//adding the viscous dissipation part ui*(mu * d2duidy2 + dmudy * siy)
	eYtmp = eYtmp + s_u[sj1][si1]*uYtmp + s_v[sj1][si1]*vYtmp + s_w[sj1][si1]*wYtmp;

	//dilation derivatives
	derDevSharedV1y(&wrk2,s_dil[sj1],si1);
	vYtmp = vYtmp + s_prop[sj1][si1]*wrk2/3.0;
	eYtmp = eYtmp + s_prop[sj1][si1]*wrk2/3.0*s_v[sj1][si1];

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM

	// pressure derivatives
	__syncthreads();
	s_dil[sj][si] = p[id.g];
	__syncthreads();

	// Boundary condition for pressure
	if (id.tiy < stencilSize){
		if (nDivY ==1){
			if(pRow>1) {
				haloBCyTop(s_dil[sj],p,si,id);
				haloBCyBot(s_dil[sj],p,si,id);
			} else {
				TopBCyNumber2(s_dil[sj],p,id,si,my);
				BotBCyNumber2(s_dil[sj],p,id,si,my);
			}
		} else {
			if (jNum ==0){
				TopBCyCpy(s_dil[sj],p,si,id);
				if(pRow>1) {
					haloBCyBot(s_dil[sj],p,si,id);
				} else {
					BotBCyNumber2(s_dil[sj],p,id,si,my);
				}
			} else if (jNum == nDivY-1){
				BotBCyCpy(s_dil[sj],p,si,id);
				if(pRow>1) {
					haloBCyTop(s_dil[sj],p,si,id);
				} else {
					TopBCyNumber2(s_dil[sj],p,id,si,my);
				}
			} else {
				TopBCyCpy(s_dil[sj],p,si,id);
				BotBCyCpy(s_dil[sj],p,si,id);
			}
		}
	}
	__syncthreads();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	derDevShared1y(&wrk1,s_dil[sj1],si1);
	vYtmp = vYtmp - wrk1;


	// fourier terms
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	__syncthreads();
	s_prop[sj][si] = lam[id.g];
	s_dil[sj][si]  = t[id.g];
	__syncthreads();

	// Boundary condition for temperature and thermal conductivity

	if (id.tiy < stencilSize){
		if (nDivY ==1){
			if(pRow>1) {
				haloBCyTop(s_prop[sj],lam,si,id); haloBCyTop(s_dil[sj],t,si,id);

				haloBCyBot(s_prop[sj],lam,si,id); haloBCyBot(s_dil[sj],t,si,id);
			} else{
				TopBCyNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,my) ;
				BotBCyNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,my) ;
			}
		} else {
			if (jNum ==0){
				TopBCyCpy(s_prop[sj],lam,si,id); TopBCyCpy(s_dil[sj],t,si,id);
				if(pRow>1) {
					haloBCyBot(s_prop[sj],lam,si,id); haloBCyBot(s_dil[sj],t,si,id);
				} else {
					BotBCyNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,my) ;
				}
			} else if (jNum == nDivY-1){
				BotBCyCpy(s_prop[sj],lam,si,id); BotBCyCpy(s_dil[sj],t,si,id);
				if(pRow>1) {
					haloBCyTop(s_prop[sj],lam,si,id); haloBCyTop(s_dil[sj],t,si,id);
				} else {
					TopBCyNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,my) ;
				}
			} else {
				TopBCyCpy(s_prop[sj],lam,si,id); TopBCyCpy(s_dil[sj],t,si,id);
				BotBCyCpy(s_prop[sj],lam,si,id); BotBCyCpy(s_dil[sj],t,si,id);
			}
		}
	}
	__syncthreads();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP

	derDevSharedV2y(&wrk1,s_dil[sj1],si1);
	eYtmp = eYtmp + wrk1*s_prop[sj1][si1];
	derDevSharedV1y(&wrk2,s_prop[sj1],si1); //wrk2 = d (lam) dy
	derDevSharedV1y(&wrk1,s_dil[sj1] ,si1); //wrk1 = d (t) dy
	eYtmp = eYtmp + wrk1*wrk2;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	//Adding here the terms - d (ru phi) dy;
	__syncthreads();
	s_prop[sj][si] = r[id.g];
	s_dil[sj][si]  = h[id.g];
	__syncthreads();
	// Boundary condition for density and enthalpy
	if (id.tiy < stencilSize){
		if (nDivY ==1){
			if(pRow>1) {
				haloBCyTop(s_prop[sj],r,si,id); haloBCyTop(s_dil[sj],h,si,id);
				haloBCyBot(s_prop[sj],r,si,id); haloBCyBot(s_dil[sj],h,si,id);
			} else{
				TopBCyNumber3(s_prop[sj],s_dil[sj],r,h,id,si,my) ;
				BotBCyNumber3(s_prop[sj],s_dil[sj],r,h,id,si,my) ;
			}
		} else {
			if (jNum ==0){
				TopBCyCpy(s_prop[sj],r,si,id); TopBCyCpy(s_dil[sj],h,si,id);
				if(pRow>1) {
					haloBCyBot(s_prop[sj],r,si,id); haloBCyBot(s_dil[sj],h,si,id);
				} else{
					BotBCyNumber3(s_prop[sj],s_dil[sj],r,h,id,si,my) ;
				}
			} else if (jNum == nDivY-1){
				BotBCyCpy(s_prop[sj],r,si,id); BotBCyCpy(s_dil[sj],h,si,id);
				if(pRow>1) {
					haloBCyTop(s_prop[sj],r,si,id); haloBCyTop(s_dil[sj],h,si,id);
				} else{
					TopBCyNumber3(s_prop[sj],s_dil[sj],r,h,id,si,my) ;
				}
			} else {
				TopBCyCpy(s_prop[sj],r,si,id); TopBCyCpy(s_dil[sj],h,si,id);
				BotBCyCpy(s_prop[sj],r,si,id); BotBCyCpy(s_dil[sj],h,si,id);
			}
		}
	}
	__syncthreads();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	fluxQuadSharedy(&wrk1,s_prop[sj1],s_v[sj1],si1);
	rYtmp = wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_prop[sj1],s_v[sj1],s_u[sj1],si1);
	uYtmp = uYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_prop[sj1],s_v[sj1],s_v[sj1],si1);
	vYtmp = vYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_prop[sj1],s_v[sj1],s_w[sj1],si1);
	wYtmp = wYtmp + wrk1;
	__syncthreads();
	fluxCubeSharedy(&wrk1,s_prop[sj1],s_v[sj1],s_dil[sj1],si1);
	eYtmp = eYtmp + wrk1;
	__syncthreads();
	//USE SHARED ARRAYS TO STORE OUTPUT AND THEN WRITE USING MEMORY TILE.
	s_prop[sj1][si1] = rYtmp ; s_u[sj1][si1] = uYtmp ; s_v[sj1][si1] = vYtmp ; s_w[sj1][si1] = wYtmp ; s_dil[sj1][si1] = eYtmp ;
	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM

#if useStreams
	rY[id.g] = rYtmp;
	uY[id.g] = uYtmp;
	vY[id.g] = vYtmp;
	wY[id.g] = wYtmp;
	eY[id.g] = eYtmp;
#else
	rY[id.g] += s_prop[sj][si];
	uY[id.g] += s_u[sj][si];
	vY[id.g] += s_v[sj][si];
	wY[id.g] += s_w[sj][si];
	eY[id.g] += s_dil[sj][si];
#endif
	__syncthreads();
}

__global__ void deviceRHSZ(myprec *rZ, myprec *uZ, myprec *vZ, myprec *wZ, myprec *eZ,
		myprec *r,  myprec *u,  myprec *v,  myprec *w,  myprec *h ,
		myprec *t,  myprec *p,  myprec *mu, myprec *lam,
		myprec *dwdx, myprec *dwdy, myprec *dudz, myprec *dvdz, myprec *dwdz,
		myprec *dil, myprec *dpdz , Communicator rk, recycle rec) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	int kNum = blockIdx.z;
	id.mkidZFlx(kNum);

	int tid = id.tix + id.bdx*id.tiy;

	int si = id.tiy + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tix;                   // local j for shared memory access
	int si1 = tid%id.bdy +  stencilSize;       // local i for shared memory access + halo offset
	int sj1 = tid/id.bdy;


	Indices id1(sj1,si1-stencilSize, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
	id1.mkidZFlx(kNum);

	bool CellAtOutlet = ( (rk.kp == mz_tot - 1) && (id1.k == mz-1) );
	bool CellAtInlet  = ( (rk.km == 0) && (id1.k == 0) );
	bool boolOutflownrbc_top = ( (outletbc==4) && CellAtOutlet );
	bool boolInflownrbc_bot  = ( (inletbc ==5) && CellAtInlet  );

	myprec rZtmp=0;
	myprec uZtmp=0;
	myprec vZtmp=0;
	myprec wZtmp=0;
	myprec eZtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;
	myprec wrk3=0;

	//myprec tmpreg1[stencilSize*2+1];
	//myprec tmpreg2[stencilSize*2+1];
	//myprec tmpreg3[stencilSize*2+1];

	__shared__ myprec s_u[mx/nDivX][mz/nDivZ+stencilSize*2];
	__shared__ myprec s_v[mx/nDivX][mz/nDivZ+stencilSize*2];
	__shared__ myprec s_w[mx/nDivX][mz/nDivZ+stencilSize*2];
	__shared__ myprec s_prop[mx/nDivX][mz/nDivZ+stencilSize*2];
	__shared__ myprec s_dil[mx/nDivX][mz/nDivZ+stencilSize*2];
	__shared__ myprec s_tmp[mx/nDivX][mz/nDivZ+stencilSize*2];

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];
	s_prop[sj][si] = mu[id.g];
	s_dil[sj][si] = dil[id.g];
	__syncthreads();

	// fill in periodic images in shared memory array
	// these are boundary conditions for u,v,w,mu and dilatation
	if (id.tiy < stencilSize){
		if (nDivZ ==1){
			if(pCol > 1) { // because even if its multi gpu but pcol == 1 then we can use directly TopBCzNumber 1 and Bot..
				if (inletbc == 1 || outletbc ==1 ){ // if periodic
					haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);
					haloBCzTop(s_prop[sj],mu,si,id); haloBCzTop(s_dil[sj],dil,si,id);

					haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);
					haloBCzBot(s_prop[sj],mu,si,id); haloBCzBot(s_dil[sj],dil,si,id);
				} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
					if (rk.km == pCol-2){ // last block
						haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);
						haloBCzBot(s_prop[sj],mu,si,id); haloBCzBot(s_dil[sj],dil,si,id);

						TopBCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,outletbc);
					} else if (rk.kp == 1) {// first block
						haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);
						haloBCzTop(s_prop[sj],mu,si,id); haloBCzTop(s_dil[sj],dil,si,id);

						BotBCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,inletbc,rec);
					} else { // all internal blocks
						haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);
						haloBCzTop(s_prop[sj],mu,si,id); haloBCzTop(s_dil[sj],dil,si,id);

						haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);
						haloBCzBot(s_prop[sj],mu,si,id); haloBCzBot(s_dil[sj],dil,si,id);
					}
				}

			} else{
				TopBCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,outletbc);
				BotBCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,inletbc,rec);
			}
		} else {
			if (kNum ==0){
				TopBCzCpy(s_u[sj],u,si,id); TopBCzCpy(s_v[sj],v,si,id); TopBCzCpy(s_w[sj],w,si,id);
				TopBCzCpy(s_prop[sj],mu,si,id); TopBCzCpy(s_dil[sj],dil,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);
						haloBCzBot(s_prop[sj],mu,si,id); haloBCzBot(s_dil[sj],dil,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.kp == 1){ // first block
							BotBCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,inletbc,rec);
						} else { // all other blocks
							haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);
							haloBCzBot(s_prop[sj],mu,si,id); haloBCzBot(s_dil[sj],dil,si,id);
						}
					}

				} else{
					BotBCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,inletbc,rec);
				}
			} else if (kNum == nDivZ-1){
				BotBCzCpy(s_u[sj],u,si,id); BotBCzCpy(s_v[sj],v,si,id); BotBCzCpy(s_w[sj],w,si,id);
				BotBCzCpy(s_prop[sj],mu,si,id); BotBCzCpy(s_dil[sj],dil,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);
						haloBCzTop(s_prop[sj],mu,si,id); haloBCzTop(s_dil[sj],dil,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.km == pCol -2 ){ // last block
							TopBCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,outletbc);
						} else { // all other blocks
							haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);
							haloBCzTop(s_prop[sj],mu,si,id); haloBCzTop(s_dil[sj],dil,si,id);
						}
					}
				} else{
					TopBCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,outletbc);
				}
			} else {
				TopBCzCpy(s_u[sj],u,si,id); TopBCzCpy(s_v[sj],v,si,id); TopBCzCpy(s_w[sj],w,si,id);
				TopBCzCpy(s_prop[sj],mu,si,id); TopBCzCpy(s_dil[sj],dil,si,id);

				BotBCzCpy(s_u[sj],u,si,id); BotBCzCpy(s_v[sj],v,si,id); BotBCzCpy(s_w[sj],w,si,id);
				BotBCzCpy(s_prop[sj],mu,si,id); BotBCzCpy(s_dil[sj],dil,si,id);
			}
		}
	}
	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	////BY COMPUTATION THREAD ARRANGEMENT
	derDevSharedV1z(&wrk1,s_u[sj1],si1);
	derDevSharedV1z(&wrk2,s_v[sj1],si1);
	derDevSharedV1z(&wrk3,s_w[sj1],si1);
	//initialize momentum RHS with stresses so that they can be added for both viscous terms and viscous heating without having to load additional terms

	// WRK1 CORRESPONDS TO THE NEW THREAD ARRANGEMENT.

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	s_tmp[sj][si] = dwdx[id.g];
	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	uZtmp = (    wrk1 + s_tmp[sj1][si1]        );
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	__syncthreads();
	s_tmp[sj][si] = dwdy[id.g];
	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	vZtmp = (    wrk2 + s_tmp[sj1][si1]        );
	wZtmp = (2 * wrk3 - 2./3.*s_dil[sj1][si1] );

	__syncthreads();
        s_tmp[sj1][si1] = wrk1;
        __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM 
        dudz[id.g] = s_tmp[sj][si];
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	__syncthreads();
        s_tmp[sj1][si1] = wrk2;
        __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM 
        dvdz[id.g] = s_tmp[sj][si];
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	__syncthreads();
        s_tmp[sj1][si1] = wrk3;
        __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM 
        dwdz[id.g] = s_tmp[sj][si];
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP

        __syncthreads();


	//adding the viscous dissipation part duidz*mu*siz
	eZtmp = s_prop[sj1][si1]*(uZtmp*wrk1 + vZtmp*wrk2 + wZtmp*wrk3);

	//Adding here the terms d (mu) dz * szj;
	derDevSharedV1z(&wrk2,s_prop[sj1],si1); //wrk2 = d (mu) dz
	uZtmp *= wrk2;
	vZtmp *= wrk2;
	wZtmp *= wrk2;

	//viscous fluxes derivative
	derDevSharedV2z(&wrk1,s_u[sj1],si1);
	uZtmp = uZtmp + wrk1*s_prop[sj1][si1];
	derDevSharedV2z(&wrk1,s_v[sj1],si1);
	vZtmp = vZtmp + wrk1*s_prop[sj1][si1];
	derDevSharedV2z(&wrk1,s_w[sj1],si1);
	wZtmp = wZtmp + wrk1*s_prop[sj1][si1];

	if ( boolOutflownrbc_top ) {
		uZtmp = 0; // derivative of tangential stress = 0 as per NSCBC
		vZtmp = 0;
	}
	//adding the viscous dissipation part ui*(mu * d2duidz2 + dmudz * siz)
	eZtmp = eZtmp + s_u[sj1][si1]*uZtmp + s_v[sj1][si1]*vZtmp + s_w[sj1][si1]*wZtmp;

	//dilation derivatives
	derDevSharedV1z(&wrk2,s_dil[sj1],si1);
	wZtmp = wZtmp + s_prop[sj1][si1]*wrk2/3.0;
	eZtmp = eZtmp + s_prop[sj1][si1]*wrk2/3.0*s_w[sj1][si1];

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	//SHARED MEM LOADING OPERATION. - ID

	// pressure derivatives
	__syncthreads();
	s_tmp[sj][si] = p[id.g];
	__syncthreads();

	if (id.tiy < stencilSize){
		if (nDivZ ==1){
			if(pCol > 1) { // because even if its multi gpu but pcol == 1 then we can use directly TopBCzNumber 1 and Bot..
				if (inletbc == 1 || outletbc ==1 ){ // if periodic
					haloBCzTop(s_tmp[sj],p,si,id);
					haloBCzBot(s_tmp[sj],p,si,id);
				} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
					if (rk.km == pCol-2){ // last block
						haloBCzBot(s_tmp[sj],p,si,id);
						TopBCzNumber2(s_tmp[sj],p,id,si,mz,outletbc);
					} else if (rk.kp == 1) {// first block
						haloBCzTop(s_tmp[sj],p,si,id);
						BotBCzNumber2(s_tmp[sj],p,id,si,mz,inletbc,rec);
					} else { // all internal blocks
						haloBCzTop(s_tmp[sj],p,si,id);
						haloBCzBot(s_tmp[sj],p,si,id);
					}
				}
			} else{
				TopBCzNumber2(s_tmp[sj],p,id,si,mz,outletbc);
				BotBCzNumber2(s_tmp[sj],p,id,si,mz,inletbc,rec);
			}
		} else {
			if (kNum ==0){
				TopBCzCpy(s_tmp[sj],p,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzBot(s_tmp[sj],p,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.kp == 1){ // first block
							BotBCzNumber2(s_tmp[sj],p,id,si,mz,inletbc,rec);
						} else { // all other blocks
							haloBCzBot(s_tmp[sj],p,si,id);
						}
					}
				} else{
					BotBCzNumber2(s_tmp[sj],p,id,si,mz,inletbc,rec);
				}
			} else if (kNum == nDivZ-1){
				BotBCzCpy(s_tmp[sj],p,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzTop(s_tmp[sj],p,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.km == pCol -2 ){ // last block
							TopBCzNumber2(s_tmp[sj],p,id,si,mz,outletbc);
						} else { // all other blocks
							haloBCzTop(s_tmp[sj],p,si,id);
						}
					}
				} else{
					TopBCzNumber2(s_tmp[sj],p,id,si,mz,outletbc);
				}
			} else {
				TopBCzCpy(s_tmp[sj],p,si,id);
				BotBCzCpy(s_tmp[sj],p,si,id);

			}
		}
	}
	__syncthreads();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	derDevShared1z(&wrk1,s_tmp[sj1],si1);
	if ( boolOutflownrbc_top ||  boolInflownrbc_bot) {
		wrk1 = 0; // pressure derivative taken care in the nrbc computation.
	}
	wZtmp = wZtmp - wrk1;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	// fourier terms
	__syncthreads();
	s_prop[sj][si] = lam[id.g];
	s_dil[sj][si]  = t[id.g];
	__syncthreads();
	// Boundary conditions for thermal conductivity and temperature

	if (id.tiy < stencilSize){
		if (nDivZ ==1){
			if(pCol > 1) { // because even if its multi gpu but pcol == 1 then we can use directly TopBCzNumber 1 and Bot..
				if (inletbc == 1 || outletbc ==1 ){ // if periodic
					haloBCzTop(s_prop[sj],lam,si,id); haloBCzTop(s_dil[sj],t,si,id);
					haloBCzBot(s_prop[sj],lam,si,id); haloBCzBot(s_dil[sj],t,si,id);
				} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
					if (rk.km == pCol-2){ // last block
						haloBCzBot(s_prop[sj],lam,si,id); haloBCzBot(s_dil[sj],t,si,id);
						TopBCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,outletbc);
					} else if (rk.kp == 1) {// first block
						haloBCzTop(s_prop[sj],lam,si,id); haloBCzTop(s_dil[sj],t,si,id);
						BotBCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,inletbc,rec);
					} else { // all internal blocks
						haloBCzTop(s_prop[sj],lam,si,id); haloBCzTop(s_dil[sj],t,si,id);
						haloBCzBot(s_prop[sj],lam,si,id); haloBCzBot(s_dil[sj],t,si,id);
					}
				}
			} else{
				TopBCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,outletbc);
				BotBCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,inletbc,rec);
			}
		} else {
			if (kNum ==0){
				TopBCzCpy(s_prop[sj],lam,si,id); TopBCzCpy(s_dil[sj],t,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzBot(s_prop[sj],lam,si,id); haloBCzBot(s_dil[sj],t,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.kp == 1){ // first block
							BotBCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,inletbc,rec);
						} else { // all other blocks
							haloBCzBot(s_prop[sj],lam,si,id); haloBCzBot(s_dil[sj],t,si,id);
						}
					}
				} else{
					BotBCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,inletbc,rec);
				}
			} else if (kNum == nDivZ-1){
				BotBCzCpy(s_prop[sj],lam,si,id); BotBCzCpy(s_dil[sj],t,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzTop(s_prop[sj],lam,si,id); haloBCzTop(s_dil[sj],t,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.km == pCol -2 ){ // last block
							TopBCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,outletbc);
						} else { // all other blocks
							haloBCzTop(s_prop[sj],lam,si,id); haloBCzTop(s_dil[sj],t,si,id);
						}
					}
				} else{
					TopBCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,outletbc);
				}
			} else {
				TopBCzCpy(s_prop[sj],lam,si,id); TopBCzCpy(s_dil[sj],t,si,id);

				BotBCzCpy(s_prop[sj],lam,si,id); BotBCzCpy(s_dil[sj],t,si,id);
			}
		}
	}

	__syncthreads();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP


	if ( boolOutflownrbc_top ) {
		eZtmp = eZtmp + 0.0; // dqdz = 0 at the outflow as per NSCBC
	} else {
		derDevSharedV2z(&wrk1,s_dil[sj1],si1);
		eZtmp = eZtmp + wrk1*s_prop[sj1][si1];
		derDevSharedV1z(&wrk2,s_prop[sj1],si1); //wrk2 = d (lam) dz
		derDevSharedV1z(&wrk1,s_dil[sj1],si1); //wrk1 = d (t) dz
		eZtmp = eZtmp + wrk1*wrk2;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
	//Adding here the terms - d (ru phi) dz;
	__syncthreads();
	s_prop[sj][si] = r[id.g];
	s_dil[sj][si]  = h[id.g];
	__syncthreads();
	// Boundary conditions for denisty and enthalpy

	if (id.tiy < stencilSize){
		if (nDivZ ==1){
			if(pCol > 1) { // because even if its multi gpu but pcol == 1 then we can use directly TopBCzNumber 1 and Bot..
				if (inletbc == 1 || outletbc ==1 ){ // if periodic
					haloBCzTop(s_prop[sj],r,si,id); haloBCzTop(s_dil[sj],h,si,id);
					haloBCzBot(s_prop[sj],r,si,id); haloBCzBot(s_dil[sj],h,si,id);
				} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
					if (rk.km == pCol-2){ // last block
						haloBCzBot(s_prop[sj],r,si,id); haloBCzBot(s_dil[sj],h,si,id);
						TopBCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,outletbc);
					} else if (rk.kp == 1) {// first block
						haloBCzTop(s_prop[sj],r,si,id); haloBCzTop(s_dil[sj],h,si,id);
						BotBCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,inletbc,rec);
					} else { // all internal blocks
						haloBCzTop(s_prop[sj],r,si,id); haloBCzTop(s_dil[sj],h,si,id);
						haloBCzBot(s_prop[sj],r,si,id); haloBCzBot(s_dil[sj],h,si,id);
					}
				}
			} else{
				TopBCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,outletbc);
				BotBCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,inletbc,rec);
			}
		} else {
			if (kNum ==0){
				TopBCzCpy(s_prop[sj],r,si,id); TopBCzCpy(s_dil[sj],h,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzBot(s_prop[sj],r,si,id); haloBCzBot(s_dil[sj],h,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.kp == 1){ // first block
							BotBCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,inletbc,rec);
						} else { // all other blocks
							haloBCzBot(s_prop[sj],r,si,id); haloBCzBot(s_dil[sj],h,si,id);
						}
					}
				} else{
					BotBCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,inletbc,rec);
				}
			} else if (kNum == nDivZ-1){
				BotBCzCpy(s_prop[sj],r,si,id); BotBCzCpy(s_dil[sj],h,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzTop(s_prop[sj],r,si,id); haloBCzTop(s_dil[sj],h,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.km == pCol -2 ){ // last block
							TopBCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,outletbc);
						} else { // all other blocks
							haloBCzTop(s_prop[sj],r,si,id); haloBCzTop(s_dil[sj],h,si,id);
						}
					}
				} else{
					TopBCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,outletbc);
				}
			} else {
				TopBCzCpy(s_prop[sj],r,si,id); TopBCzCpy(s_dil[sj],h,si,id);
				BotBCzCpy(s_prop[sj],r,si,id); BotBCzCpy(s_dil[sj],h,si,id);
			}
		}
	}

	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
	if ( boolOutflownrbc_top){
		OutflownrbcZ_top(&rZtmp,&uZtmp,&vZtmp,&wZtmp,&eZtmp, s_prop[sj1],s_u[sj1],s_v[sj1],s_w[sj1],s_tmp[sj1],s_dil[sj1], si1, id1);
	} else if (boolInflownrbc_bot){
		InflownrbcZ_bot(&rZtmp,&uZtmp,&vZtmp,&wZtmp,&eZtmp, s_prop[sj1],s_u[sj1],s_v[sj1],s_w[sj1],s_tmp[sj1],s_dil[sj1], si1, id1);
	} else {
		fluxQuadSharedz(&wrk1,s_prop[sj1],s_w[sj1],si1);
		rZtmp = wrk1;
		fluxCubeSharedz(&wrk1,s_prop[sj1],s_w[sj1],s_u[sj1],si1);
		uZtmp = uZtmp + wrk1;
		fluxCubeSharedz(&wrk1,s_prop[sj1],s_w[sj1],s_v[sj1],si1);
		vZtmp = vZtmp + wrk1;
		fluxCubeSharedz(&wrk1,s_prop[sj1],s_w[sj1],s_w[sj1],si1);
		wZtmp = wZtmp + wrk1;
		fluxCubeSharedz(&wrk1,s_prop[sj1],s_w[sj1],s_dil[sj1],si1);
		eZtmp = eZtmp + wrk1;
	}

	//USE SHARED ARRAYS TO STORE OUTPUT AND THEN WRITE USING MEMORY TILE.
	__syncthreads();
	s_prop[sj1][si1] = rZtmp ; s_u[sj1][si1] = uZtmp ; s_v[sj1][si1] = vZtmp ; s_tmp[sj1][si1] = wZtmp ; s_dil[sj1][si1] = eZtmp ;
	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM

	if (forcing) {
		rZ[id.g] += s_prop[sj][si];
		uZ[id.g] += s_u[sj][si];
		vZ[id.g] += s_v[sj][si];
		wZ[id.g] += s_tmp[sj][si] + *dpdz;
		eZ[id.g] += s_dil[sj][si] + *dpdz*s_w[sj][si];
	} else {
		rZ[id.g] += s_prop[sj][si];
		uZ[id.g] += s_u[sj][si];
		vZ[id.g] += s_v[sj][si];
		wZ[id.g] += s_tmp[sj][si] ;
		eZ[id.g] += s_dil[sj][si] ;
	}
	//#if useStreams
	//	rZ[id.g] = rZtmp;
	//	uZ[id.g] = uZtmp;
	//	vZ[id.g] = vZtmp;
	//	wZ[id.g] = wZtmp + *dpdz;
	//	eZ[id.g] = eZtmp + *dpdz*s_w[sj][si] ;
	//#else





	//#endif
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
