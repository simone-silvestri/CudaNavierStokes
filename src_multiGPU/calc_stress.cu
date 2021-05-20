
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_math.h"
#include "boundary.h"
#include "comm.h"


__global__ void deviceCalcPress(myprec *a, myprec *b, myprec *c) {
	*a = 0.99*(*b) - 0.5*(*a/(*c)-1);
}

__global__ void calcStressX(myprec *u, myprec *v, myprec *w) {

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
	derDevShared1x(&wrk1,s_u[sj],si); gij[0][id.g] = wrk1;
	derDevShared1x(&wrk1,s_v[sj],si); gij[1][id.g] = wrk1;
	derDevShared1x(&wrk1,s_w[sj],si); gij[2][id.g] = wrk1;

}

__global__ void calcStressY(myprec *u, myprec *v, myprec *w) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1yL(gij[3],u,id);
	derDev1yL(gij[4],v,id);
	derDev1yL(gij[5],w,id);
}

__global__ void calcStressZ(myprec *u, myprec *v, myprec *w) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1zL(gij[6],u,id);
	derDev1zL(gij[7],v,id);
	derDev1zL(gij[8],w,id);
}

__global__ void calcDil(myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	//Stress goes with RHS old

	dil[id.g] = gij[0][id.g] + gij[4][id.g] + gij[8][id.g];

}

void calcPressureGrad(myprec *dpdz, myprec *r, myprec *w) {

	myprec *workA, *dpdz_prev, *rbulk;
	dim3 gr0  = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);

	checkCuda( cudaMalloc((void**)&workA,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&dpdz_prev,     sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&rbulk    ,     sizeof(myprec)) );

	deviceCpyOne<<<1,1>>>(dpdz_prev,dpdz);
	hostVolumeIntegral(rbulk,r);
	deviceMul<<<gr0,bl0>>>(workA,r,w);
	hostVolumeIntegral(dpdz,workA);

	deviceCalcPress<<<1,1>>>(dpdz,dpdz_prev,rbulk);

	checkCuda( cudaFree(workA) );
	checkCuda( cudaFree(dpdz_prev) );
	checkCuda( cudaFree(rbulk) );
}

void calcTimeStep(myprec *dt, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu) {

	myprec *workA;

	dim3 gr0 = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);
    checkCuda( cudaMalloc((void**)&workA,mx*my*mz*sizeof(myprec)) );
    deviceCalcDt<<<gr0,bl0>>>(workA,r,u,v,w,e,mu);

	cudaDeviceSynchronize();
	hostReduceToMin(dt,workA);

	checkCuda( cudaFree(workA) );
	cudaDeviceSynchronize();
}

__global__ void deviceCalcDt(myprec *wrkArray, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	myprec dtConvInv = 0.0;
    myprec dtViscInv = 0.0;

    myprec ien = e[id.g]/r[id.g] - 0.5*(u[id.g]*u[id.g] + v[id.g]*v[id.g] + w[id.g]*w[id.g]);
    myprec sos = pow(gamma*(gamma-1)*ien,0.5);

    myprec dx,d2x;
    dx = d_dxv[id.i];
    d2x = dx*dx;

    dtConvInv =  MAX( (abs(u[id.g]) + sos)/dx, MAX( (abs(v[id.g]) + sos)*d_dy, (abs(w[id.g]) + sos)*d_dz) );
    dtViscInv =  MAX( mu[id.g]/d2x, MAX( mu[id.g]*d_d2y, mu[id.g]*d_d2z) );

    wrkArray[id.g] = CFL/MAX(dtConvInv, dtViscInv);
    __syncthreads();

}

void calcBulk(myprec *par1, myprec *par2, myprec *r, myprec *w, myprec *e, Communicator rk) {

	myprec *workA, *rbulk;
	myprec *hostWork = (myprec*)malloc(sizeof(myprec));
	dim3 gr0  = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);

	checkCuda( cudaMalloc((void**)&workA,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&rbulk    ,     sizeof(myprec)) );

	hostVolumeIntegral(rbulk,r);
	deviceMul<<<gr0,bl0>>>(workA,r,w);
	hostVolumeIntegral(par1,workA);
	checkCuda( cudaMemcpy(hostWork, par1, sizeof(myprec), cudaMemcpyDeviceToHost) );
	allReduceSum(hostWork,1);
	checkCuda( cudaMemcpy(par1, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );
	deviceDivOne<<<1,1>>>(par1,par1,rbulk);
	hostVolumeIntegral(par2,e);
	checkCuda( cudaMemcpy(hostWork, par2, sizeof(myprec), cudaMemcpyDeviceToHost) );
	allReduceSum(hostWork,1);
	checkCuda( cudaMemcpy(par2, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );

	free(hostWork);
	checkCuda( cudaFree(workA) );
	checkCuda( cudaFree(rbulk) );
}
