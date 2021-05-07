
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


__global__ void initStress() {
        checkCudaDev( cudaMalloc((void**)&d_workSX,mx*my*mz*sizeof(myprec)) );
        checkCudaDev( cudaMalloc((void**)&d_workSY,mx*my*mz*sizeof(myprec)) );
        checkCudaDev( cudaMalloc((void**)&d_workSZ,mx*my*mz*sizeof(myprec)) );
}

__global__ void clearStress() {
        checkCudaDev( cudaFree(d_workSX) );
        checkCudaDev( cudaFree(d_workSY) );
        checkCudaDev( cudaFree(d_workSZ) );
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
	derDevShared1x(&wrk1,s_u[sj],si); sij[0][id.g] = wrk1;
	derDevShared1x(&wrk1,s_v[sj],si); sij[1][id.g] = wrk1;
	derDevShared1x(&wrk1,s_w[sj],si); sij[2][id.g] = wrk1;

}

__global__ void calcStressY(myprec *u, myprec *v, myprec *w) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1yL(sij[3],u,id);
	derDev1yL(sij[4],v,id);
	derDev1yL(sij[5],w,id);
}

__global__ void calcStressZ(myprec *u, myprec *v, myprec *w) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1zL(sij[6],u,id);
	derDev1zL(sij[7],v,id);
	derDev1zL(sij[8],w,id);
}

__global__ void calcDil(myprec *dil) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	//Stress goes with RHS old

	dil[id.g] = sij[0][id.g] + sij[4][id.g] + sij[8][id.g];

}

__global__ void calcPressureGrad(myprec *dpdz, myprec *r, myprec *w) {
	myprec dpdz_prev = *dpdz;
	volumeIntegral(dpdz,r);
	myprec rbulk = *dpdz;

	dim3 gr0  = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);

	deviceMul<<<gr0,bl0>>>(d_workSX,r,w);
	volumeIntegral(dpdz,d_workSX);
	*dpdz = *dpdz/rbulk;
	*dpdz = 0.99*dpdz_prev - 0.5*(*dpdz - 1.0);
}

__global__ void calcTimeStep(myprec *dt, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu) {

	dim3 gr0  = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);
	deviceCalcDt<<<gr0,bl0>>>(d_workSX,r,u,v,w,e,mu);

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
    dx = d_dxv[id.i];
    d2x = dx*dx;

    dtConvInv =  MAX( (abs(u[id.g]) + sos)/dx, MAX( (abs(v[id.g]) + sos)*d_dy, (abs(w[id.g]) + sos)*d_dz) );
    dtViscInv =  MAX( mu[id.g]/d2x, MAX( mu[id.g]*d_d2y, mu[id.g]*d_d2z) );

    wrkArray[id.g] = CFL/MAX(dtConvInv, dtViscInv);
    __syncthreads();

}

__global__ void calcIntegrals(myprec *r, myprec *u, myprec *v, myprec *w, myprec *kin, myprec *enst) {

	*kin  = 0;
	*enst = 0;

	dim3 gr0  = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);

	deviceSca<<<gr0,bl0>>>(d_workSX,u,v,w,u,v,w);
	deviceMul<<<gr0,bl0>>>(d_workSX,r,d_workSX);

	cudaDeviceSynchronize();
	volumeIntegral(kin,d_workSX);
	*kin *= 1.0/Lx/Ly/Lz/2.0;

	deviceSub<<<gr0,bl0>>>(d_workSX,sij[5],sij[7]);
	deviceSub<<<gr0,bl0>>>(d_workSY,sij[6],sij[2]);
	deviceSub<<<gr0,bl0>>>(d_workSZ,sij[1],sij[3]);

	deviceSca<<<gr0,bl0>>>(d_workSX,d_workSX,d_workSY,d_workSZ,d_workSX,d_workSY,d_workSZ);
	deviceMul<<<gr0,bl0>>>(d_workSX,r,d_workSX);

	cudaDeviceSynchronize();
	volumeIntegral(enst,d_workSX);
	*enst = *enst/Lx/Ly/Lz/Re;
}
