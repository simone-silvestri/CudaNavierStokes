
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_math.h"
#include "boundary_condition_x.h"
#include "comm.h"
#include "sponge.h"

__global__ void deviceAdvanceTime(myprec *dt) {
	time_on_GPU += *dt;
}

__global__ void deviceCalcPress(myprec *a, myprec *b, myprec *c) {
	*a = 0.99*(*b) - 0.5*(*a/(*c)-1);
}

__global__ void derVelX(myprec *u, myprec *v, myprec *w, myprec *dudx, myprec *dvdx, myprec *dwdx) {

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

	BCxderVel(s_u[sj],s_v[sj],s_w[sj],id,si,mx);
	__syncthreads();

	myprec wrk1;
	derDevSharedV1x(&wrk1,s_u[sj],si); dudx[id.g] = wrk1;
	derDevSharedV1x(&wrk1,s_v[sj],si); dvdx[id.g] = wrk1;
	derDevSharedV1x(&wrk1,s_w[sj],si); dwdx[id.g] = wrk1;

}

__global__ void derVelY(myprec *u, myprec *v, myprec *w, myprec *dudy, myprec *dvdy, myprec *dwdy) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDevV1yL(dudy,u,id);
	derDevV1yL(dvdy,v,id);
	derDevV1yL(dwdy,w,id);
}

__global__ void derVelZ(myprec *u, myprec *v, myprec *w, myprec *dudz, myprec *dvdz, myprec *dwdz, int kNum) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDevV1zL(dudz,u,uInit,id,kNum);
	derDevV1zL(dvdz,v,vInit,id,kNum);
	derDevV1zL(dwdz,w,wInit,id,kNum);
}

__global__ void derVelYBC(myprec *u, myprec *v, myprec *w, myprec *dudy, myprec *dvdy, myprec *dwdy, int direction) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	id.mkidYBC(direction);

	derDev1yBC(dudy,u,id,direction);
	derDev1yBC(dvdy,v,id,direction);
	derDev1yBC(dwdy,w,id,direction);
}

__global__ void derVelZBC(myprec *u, myprec *v, myprec *w, myprec *dudz, myprec *dvdz, myprec *dwdz, int direction) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	id.mkidZBC(direction);

	derDev1zBC(dudz,u,id,direction);
	derDev1zBC(dvdz,v,id,direction);
	derDev1zBC(dwdz,w,id,direction);
}

__global__ void calcDil(myprec *dil, myprec *dudx, myprec *dvdy, myprec *dwdz) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	//Stress goes with RHS old

	dil[id.g] = dudx[id.g] + dvdy[id.g] + dwdz[id.g];

}

void calcPressureGrad(myprec *dpdz, myprec *r, myprec *w, Communicator rk) {

    cudaSetDevice(rk.nodeRank);

	myprec *workA, *dpdz_prev, *rbulk;
	dim3 gr0  = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);

	checkCuda( cudaMalloc((void**)&workA,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&dpdz_prev,     sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&rbulk    ,     sizeof(myprec)) );

	deviceCpyOne<<<1,1>>>(dpdz_prev,dpdz);
	hostVolumeIntegral(rbulk,r,rk);
	deviceMul<<<gr0,bl0>>>(workA,r,w);
	hostVolumeIntegral(dpdz,workA,rk);

	deviceCalcPress<<<1,1>>>(dpdz,dpdz_prev,rbulk);

	checkCuda( cudaFree(workA) );
	checkCuda( cudaFree(dpdz_prev) );
	checkCuda( cudaFree(rbulk) );
}

void calcTimeStep(myprec *dt, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu, Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	myprec *workA;

	dim3 gr0 = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);
    checkCuda( cudaMalloc((void**)&workA,mx*my*mz*sizeof(myprec)) );
    deviceCalcDt<<<gr0,bl0>>>(workA,r,u,v,w,e,mu);

	cudaDeviceSynchronize();
	hostReduceToMin(dt,workA,rk);

	checkCuda( cudaFree(workA) );
	cudaDeviceSynchronize();
}

__global__ void deviceCalcDt(myprec *wrkArray, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	myprec dtConvInv = 0.0;
    myprec dtViscInv = 0.0;

    myprec ien = e[id.g]/r[id.g] - 0.5*(u[id.g]*u[id.g] + v[id.g]*v[id.g] + w[id.g]*w[id.g]);
    myprec sos = pow(gam*(gam-1)*ien,0.5);

    myprec dx,d2x;
    dx = d_dxv[id.i];
    d2x = dx*dx;

    dtConvInv =  MAX( (abs(u[id.g]) + sos)/dx, MAX( (abs(v[id.g]) + sos)*d_dy, (abs(w[id.g]) + sos)*d_dz) );
    dtViscInv =  MAX( mu[id.g]/d2x, MAX( mu[id.g]*d_d2y, mu[id.g]*d_d2z) );

    wrkArray[id.g] = CFL/MAX(dtConvInv, dtViscInv);
    __syncthreads();

}

void calcBulk(myprec *par1, myprec *par2, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	myprec *workA, *rbulk;
	myprec *hostWork = (myprec*)malloc(sizeof(myprec));
	dim3 gr0  = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);

	checkCuda( cudaMalloc((void**)&workA,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&rbulk    ,     sizeof(myprec)) );

	if(forcing) {
		hostVolumeIntegral(rbulk,r,rk);
		checkCuda( cudaMemcpy(hostWork, rbulk, sizeof(myprec), cudaMemcpyDeviceToHost) );
		allReduceSum(hostWork,1);
		checkCuda( cudaMemcpy(rbulk, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );

		deviceMul<<<gr0,bl0>>>(workA,r,w);
		hostVolumeIntegral(par1,workA,rk);
		checkCuda( cudaMemcpy(hostWork, par1, sizeof(myprec), cudaMemcpyDeviceToHost) );
		allReduceSum(hostWork,1);
		checkCuda( cudaMemcpy(par1, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );
		deviceDivOne<<<1,1>>>(par1,par1,rbulk);

		hostVolumeIntegral(par2,e,rk);
		checkCuda( cudaMemcpy(hostWork, par2, sizeof(myprec), cudaMemcpyDeviceToHost) );
		allReduceSum(hostWork,1);
		checkCuda( cudaMemcpy(par2, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );
	} else {
		deviceSca<<<gr0,bl0>>>(workA,u,v,w,u,v,w);
		hostVolumeAverage(par1,workA,rk);
		checkCuda( cudaMemcpy(hostWork, par1, sizeof(myprec), cudaMemcpyDeviceToHost) );
		allReduceSum(hostWork,1);
		checkCuda( cudaMemcpy(par1, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );
	}
	free(hostWork);
	checkCuda( cudaFree(workA) );
	checkCuda( cudaFree(rbulk) );
}
