
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_math.h"
#include "boundary.h"

__device__ myprec *d_workSX;
__device__ myprec *d_workSY;
__device__ myprec *d_workSZ;


__device__ void initStress() {
        checkCudaDev( cudaMalloc((void**)&d_workSX,mx*my*mz*sizeof(myprec)) );
        checkCudaDev( cudaMalloc((void**)&d_workSY,mx*my*mz*sizeof(myprec)) );
        checkCudaDev( cudaMalloc((void**)&d_workSZ,mx*my*mz*sizeof(myprec)) );
}

__device__ void clearStress() {
        checkCudaDev( cudaFree(d_workSX) );
        checkCudaDev( cudaFree(d_workSY) );
        checkCudaDev( cudaFree(d_workSZ) );
}

__global__ void calcStressX(myprec *u, myprec *v, myprec *w, myprec *stress[9]) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	__shared__ myprec s_u[sPencils][mx+stencilSize*2];
	__shared__ myprec s_v[sPencils][mx+stencilSize*2];
	__shared__ myprec s_w[sPencils][mx+stencilSize*2];

	int si = id.i + stencilSize;
	int sj = id.tiy;

	s_u[sj][si] = u[id.g];
	s_v[sj][si] = v[id.g];
	s_w[sj][si] = w[id.g];

	__syncthreads();

	if(id.i<stencilSize) {
#if periodicX
		perBCx(s_u[sj],si);perBCx(s_v[sj],si);perBCx(s_w[sj],si);
#else
		wallBCxVel(s_u[sj],si);wallBCxVel(s_v[sj],si);wallBCxVel(s_w[sj],si);
#endif
	}

	__syncthreads();

	myprec wrk1;
	derDevShared1x(&wrk1,s_u[sj],si); stress[0][id.g] = wrk1;
	derDevShared1x(&wrk1,s_v[sj],si); stress[1][id.g] = wrk1;
	derDevShared1x(&wrk1,s_w[sj],si); stress[2][id.g] = wrk1;

}

__global__ void calcStressY(myprec *u, myprec *v, myprec *w, myprec *stress[9]) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1yL(stress[3],u,id);
	derDev1yL(stress[4],v,id);
	derDev1yL(stress[5],w,id);
}

__global__ void calcStressZ(myprec *u, myprec *v, myprec *w, myprec *stress[9]) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1zL(stress[6],u,id);
	derDev1zL(stress[7],v,id);
	derDev1zL(stress[8],w,id);
}

__global__ void calcDil(myprec *stress[9], myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	//Stress goes with RHS old
//
//	myprec dudx = stress[0][id.g];
//	myprec dvdx = stress[1][id.g];
//	myprec dwdx = stress[2][id.g];
//	myprec dudy = stress[3][id.g];
//	myprec dvdy = stress[4][id.g];
//	myprec dwdy = stress[5][id.g];
//	myprec dudz = stress[6][id.g];
//	myprec dvdz = stress[7][id.g];
//	myprec dwdz = stress[8][id.g];

	dil[id.g] = stress[0][id.g] + stress[4][id.g] + stress[8][id.g]; //dudx + dvdy + dwdz; //

//	stress[0][id.g] = 2.0*dudx - 2.0/3.0*dil[id.g];
//	stress[1][id.g] = dudy + dvdx;
//	stress[2][id.g] = dudz + dwdx;
//	stress[3][id.g] = dudy + dvdx;
//	stress[4][id.g] = 2.0*dvdy - 2.0/3.0*dil[id.g];
//	stress[5][id.g] = dvdz + dwdy;
//	stress[6][id.g] = dudz + dwdx;
//	stress[7][id.g] = dvdz + dwdy;
//	stress[8][id.g] = 2.0*dwdz - 2.0/3.0*dil[id.g];


}

__device__ void calcPressureGrad(myprec *dpdz, myprec *w) {
	myprec dpdz_prev = *dpdz;
	reduceToOne(dpdz,w);
	*dpdz = *dpdz/mx/my/mz;
	*dpdz = 0.99*dpdz_prev - 0.5*(*dpdz - 1.0);
}

__device__ void calcTimeStep(myprec *dt, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu) {

#if (capability < capabilityMin)
	dim3 gr0,bl0;
	gr0 = dim3(grid0[0],grid0[1],1); bl0 = dim3(block0[0],block0[1],1);
	deviceCalcDt<<<gr0,bl0>>>(d_workSX,r,u,v,w,e,mu);
#else
	deviceCalcDt<<<grid0,block0>>>(d_workSX,r,u,v,w,e,mu);
#endif
	cudaDeviceSynchronize();
	reduceToMin(dt,d_workSX);
	cudaDeviceSynchronize();
}

__global__ void deviceCalcDt(myprec *wrkArray, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	myprec dtConvInv = 0.0;
    myprec dtViscInv = 0.0;

    myprec ien = e[id.g]/r[id.g] - 0.5*(u[id.g]*u[id.g] + v[id.g]*v[id.g] + w[id.g]*w[id.g]);
    myprec sos = pow(gamma*(gamma-1)*ien,0.5);

    myprec dx,d2x;

    if(id.i==0) {
    	dx = (d_x[id.i+1] + d_x[id.i])/2.0;
    } else if (id.i==mx-1) {
    	dx = Lx - (d_x[id.i] + d_x[id.i-1])/2.0;
    } else {
    	dx = (d_x[id.i+1] - d_x[id.i-1])/2.0;
    }

    d2x = dx*dx;

    dtConvInv =  MAX( (abs(u[id.g]) + sos)/dx, MAX( (abs(v[id.g]) + sos)*d_dy, (abs(w[id.g]) + sos)*d_dz) );
    dtViscInv =  MAX( mu[id.g]/d2x, MAX( mu[id.g]*d_d2y, mu[id.g]*d_d2z) );

    wrkArray[id.g] = CFL/MAX(dtConvInv, dtViscInv);
    __syncthreads();

}

__device__ void calcIntegrals(myprec *r, myprec *u, myprec *v, myprec *w, myprec *stress[9], myprec *kin, myprec *enst) {

	*kin  = 0;
	*enst = 0;

	myprec dV = 1.0/d_dx/d_dy/d_dz;
#if (capability < capabilityMin)
	dim3 gr0,bl0;
	gr0 = dim3(grid0[0],grid0[1],1); bl0 = dim3(block0[0],block0[1],1);
	deviceSca<<<gr0,bl0>>>(d_workSX,u,v,w,u,v,w);
	deviceMul<<<gr0,bl0>>>(d_workSX,r,d_workSX);
#else
	deviceSca<<<grid0,block0>>>(d_workSX,u,v,w,u,v,w);
	deviceMul<<<grid0,block0>>>(d_workSX,r,d_workSX);
#endif
	cudaDeviceSynchronize();
	reduceToOne(kin,d_workSX);
	*kin *= dV/2.0/Lx/Ly/Lz;
#if (capability < capabilityMin)
	deviceSub<<<gr0,bl0>>>(d_workSX,stress[5],stress[7]);
	deviceSub<<<gr0,bl0>>>(d_workSY,stress[6],stress[2]);
	deviceSub<<<gr0,bl0>>>(d_workSZ,stress[1],stress[3]);

	deviceSca<<<gr0,bl0>>>(d_workSX,d_workSX,d_workSY,d_workSZ,d_workSX,d_workSY,d_workSZ);
	deviceMul<<<gr0,bl0>>>(d_workSX,r,d_workSX);
#else
	deviceSub<<<grid0,block0>>>(d_workSX,stress[5],stress[7]);
	deviceSub<<<grid0,block0>>>(d_workSY,stress[6],stress[2]);
	deviceSub<<<grid0,block0>>>(d_workSZ,stress[1],stress[3]);

	deviceSca<<<grid0,block0>>>(d_workSX,d_workSX,d_workSY,d_workSZ,d_workSX,d_workSY,d_workSZ);
	deviceMul<<<grid0,block0>>>(d_workSX,r,d_workSX);
#endif
	cudaDeviceSynchronize();
	reduceToOne(enst,d_workSX);
	*enst *= dV/Lx/Ly/Lz/Re;
}
