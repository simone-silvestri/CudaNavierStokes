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
		myprec *dvdy, myprec *dwdz, myprec *dil, myprec *dpdz, int iNum) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	int si = id.tix + stencilSize;       // local i for shared memory access + halo offset
	int sj = id.tiy;                   // local j for shared memory access

	myprec rXtmp=0;
	myprec uXtmp=0;
	myprec vXtmp=0;
	myprec wXtmp=0;
	myprec eXtmp=0;

	myprec wrk1=0;
	myprec wrk2=0;
	myprec wrk3=0;

        
	__shared__ myprec s_u[sPencils/2][mx+stencilSize*2];
	__shared__ myprec s_v[sPencils/2][mx+stencilSize*2];
	__shared__ myprec s_w[sPencils/2][mx+stencilSize*2];
	__shared__ myprec s_t[sPencils/2][mx+stencilSize*2];
	__shared__ myprec s_p[sPencils/2][mx+stencilSize*2];
	__shared__ myprec s_prop1[sPencils/2][mx+stencilSize*2];
	__shared__ myprec s_prop2[sPencils/2][mx+stencilSize*2];

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

	//adding the viscous dissipation part ui*(mu * d2duidx2 + dmudx * six)
	eXtmp = eXtmp + s_u[sj][si]*uXtmp + s_v[sj][si]*vXtmp + s_w[sj][si]*wXtmp;

	//adding the molecular conduction part (d2 temp dx2*lambda + dlambda dx * d temp dx)
	derDevSharedV2x(&wrk1,s_t[sj],si);
	eXtmp = eXtmp + wrk1*s_prop2[sj][si];
	derDevSharedV1x(&wrk2,s_prop2[sj],si); //wrk2 = d (lam) dx
	derDevSharedV1x(&wrk3,s_t[sj],si); //wrk1 = d (t) dx
	eXtmp = eXtmp + wrk2*wrk3;

	// pressure and dilation derivatives
	s_prop2[sj][si] = rXtmp; // rXtmp stores dilatation temporarily
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

	fluxQuadSharedx(&wrk3,s_prop1[sj],s_u[sj],si);
	rXtmp = wrk3;
	__syncthreads();
	fluxCubeSharedx(&wrk1,s_prop1[sj],s_u[sj],s_u[sj],si);
	uXtmp = uXtmp + wrk1;
	__syncthreads();
	fluxCubeSharedx(&wrk2,s_prop1[sj],s_u[sj],s_v[sj],si);
	vXtmp = vXtmp + wrk2;
	__syncthreads();
	fluxCubeSharedx(&wrk3,s_prop1[sj],s_u[sj],s_w[sj],si);
	wXtmp = wXtmp + wrk3;
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

	BCyNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,my,jNum);
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
	vYtmp = (2 * wrk2 - 2./3.*s_dil[sj1][si1] );
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
        __syncthreads();
        s_tmp[sj][si] = dvdz[id.g];
        __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP 
	wYtmp = (    wrk3 + s_tmp[sj1][si1]        );

        /*__syncthreads();
        s_tmp[sj1][si1] = wrk2;
        __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
        dvdy[id.g] = s_tmp[sj][si];*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP 
        __syncthreads();
        s_tmp[sj1][si1] = wrk3;
        __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM 
        dwdy[id.g] = s_tmp[sj][si];

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
	BCyNumber2(s_dil[sj],p,id,si,my, jNum);
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
	BCyNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,my,jNum);
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
	BCyNumber3(s_prop[sj],s_dil[sj],r,h,id,si,my,jNum);
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
        myprec *dil, myprec *dpdz) {

    Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
    int kNum = blockIdx.z;
    id.mkidZFlx(kNum);
       
    int tid = id.tix + id.bdx*id.tiy;

    int si = id.tiy + stencilSize;       // local i for shared memory access + halo offset
    int sj = id.tix;                   // local j for shared memory access
    int si1 = tid%id.bdy +  stencilSize;       // local i for shared memory access + halo offset
    int sj1 = tid/id.bdy;

    //Indices id1(sj1,si1-stencilSize, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
    //id1.mkidZFlx(kNum);

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

   
        BCzNumber1(s_u[sj],s_v[sj],s_w[sj],s_prop[sj],s_dil[sj],u,v,w,mu,dil,id,si,mz,kNum);
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

        /*__syncthreads();
        s_tmp[sj1][si1] = wrk3;
        __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM 
        dwdz[id.g] = s_tmp[sj][si];*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP



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
    s_dil[sj][si] = p[id.g];
    __syncthreads();
    BCzNumber2(s_dil[sj],p,id,si,mz,kNum);
    __syncthreads();

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
    derDevShared1z(&wrk1,s_dil[sj1],si1);
    wZtmp = wZtmp - wrk1;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
    // fourier terms
    __syncthreads();   
        s_prop[sj][si] = lam[id.g];
    s_dil[sj][si]  = t[id.g];
    __syncthreads();
    // Boundary conditions for thermal conductivity and temperature
    BCzNumber3(s_prop[sj],s_dil[sj],lam,t,id,si,mz,kNum);
    __syncthreads();

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP

    derDevSharedV2z(&wrk1,s_dil[sj1],si1);
    eZtmp = eZtmp + wrk1*s_prop[sj1][si1];
    derDevSharedV1z(&wrk2,s_prop[sj1],si1); //wrk2 = d (lam) dz
    derDevSharedV1z(&wrk1,s_dil[sj1],si1); //wrk1 = d (t) dz
    eZtmp = eZtmp + wrk1*wrk2;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
    //Adding here the terms - d (ru phi) dz;
    __syncthreads();
    s_prop[sj][si] = r[id.g];
    s_dil[sj][si]  = h[id.g];
    __syncthreads();
    // Boundary conditions for denisty and enthalpy
    BCzNumber4(s_prop[sj],s_dil[sj],r,h,id,si,mz,kNum);
    __syncthreads();

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////COMP
    fluxQuadSharedz(&wrk1,s_prop[sj1],s_w[sj1],si1);
    rZtmp = wrk1;
    __syncthreads();
    fluxCubeSharedz(&wrk1,s_prop[sj1],s_w[sj1],s_u[sj1],si1);
    uZtmp = uZtmp + wrk1;
    __syncthreads();
    fluxCubeSharedz(&wrk1,s_prop[sj1],s_w[sj1],s_v[sj1],si1);
    vZtmp = vZtmp + wrk1;
    __syncthreads();
    fluxCubeSharedz(&wrk1,s_prop[sj1],s_w[sj1],s_w[sj1],si1);
    wZtmp = wZtmp + wrk1;
    __syncthreads();
    fluxCubeSharedz(&wrk1,s_prop[sj1],s_w[sj1],s_dil[sj1],si1);
    eZtmp = eZtmp + wrk1;
    __syncthreads();
//USE SHARED ARRAYS TO STORE OUTPUT AND THEN WRITE USING MEMORY TILE.      
    s_prop[sj1][si1] = rZtmp ; s_u[sj1][si1] = uZtmp ; s_v[sj1][si1] = vZtmp ; s_tmp[sj1][si1] = wZtmp ; s_dil[sj1][si1] = eZtmp ;
    __syncthreads();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////MEM
        
    rZ[id.g] += s_prop[sj][si];
    uZ[id.g] += s_u[sj][si];
    vZ[id.g] += s_v[sj][si];
    wZ[id.g] += s_tmp[sj][si] + *dpdz;
    eZ[id.g] += s_dil[sj][si] + *dpdz*s_w[sj][si];
    //dil[id.g] = dudx[id.g] + dvdy[id.g] + dwdz[id.g];

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
