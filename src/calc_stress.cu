
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_math.h"

__device__ myprec d_workSX[mx*my*mz];
__device__ myprec d_workSY[mx*my*mz];
__device__ myprec d_workSZ[mx*my*mz];

__global__ void calcStressX(myprec *u, myprec *v, myprec *w, myprec *stress[9]) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	derDev1x(stress[0],u,id);
	derDev1x(stress[1],v,id);
	derDev1x(stress[2],w,id);
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

	myprec dudx = stress[0][id.g];
	myprec dvdx = stress[1][id.g];
	myprec dwdx = stress[2][id.g];
	myprec dudy = stress[3][id.g];
	myprec dvdy = stress[4][id.g];
	myprec dwdy = stress[5][id.g];
	myprec dudz = stress[6][id.g];
	myprec dvdz = stress[7][id.g];
	myprec dwdz = stress[8][id.g];

	dil[id.g] = dudx + dvdy + dwdz; // stress[0][id.g] + stress[4][id.g] + stress[8][id.g]; //

	stress[0][id.g] = 2.0*dudx - 2.0/3.0*dil[id.g];
	stress[1][id.g] = dudy + dvdx;
	stress[2][id.g] = dudz + dwdx;
	stress[3][id.g] = dudy + dvdx;
	stress[4][id.g] = 2.0*dvdy - 2.0/3.0*dil[id.g];
	stress[5][id.g] = dvdz + dwdy;
	stress[6][id.g] = dudz + dwdx;
	stress[7][id.g] = dvdz + dwdy;
	stress[8][id.g] = 2.0*dwdz - 2.0/3.0*dil[id.g];

}

__device__ void calcTimeStep(myprec *dt, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu) {
	deviceCalcDt<<<grid0,block0>>>(d_workSX,r,u,v,w,e,mu);
	cudaDeviceSynchronize();
	reduceToMin(dt,d_workSX);
	cudaDeviceSynchronize();
}

__global__ void deviceCalcDt(myprec *wrkArray, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu) {

	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int gt = blockNumInGrid * threadsPerBlock + threadNumInBlock;

    myprec dtConvInv = 0.0;
    myprec dtViscInv = 0.0;

    myprec ien = e[gt]/r[gt] - 0.5*(u[gt]*u[gt] + v[gt]*v[gt] + w[gt]*w[gt]);
    myprec sos = pow(gamma*(gamma-1)*ien,0.5);

    dtConvInv =  MAX( (abs(u[gt]) + sos)*d_dx, MAX( (abs(v[gt]) + sos)*d_dy, (abs(w[gt]) + sos)*d_dz) );
    dtViscInv =  MAX( mu[gt]*d_d2x, MAX( mu[gt]*d_d2y, mu[gt]*d_d2z) );

    wrkArray[gt] = CFL/MAX(dtConvInv, dtViscInv);
    __syncthreads();

}

__device__ void calcIntegrals(myprec *r, myprec *u, myprec *v, myprec *w, myprec *stress[9], myprec *kin, myprec *enst) {

	*kin  = 0;
	*enst = 0;

	myprec dV = 1.0/d_dx/d_dy/d_dz;

	deviceSca<<<grid0,block0>>>(d_workSX,u,v,w,u,v,w);
	deviceMul<<<grid0,block0>>>(d_workSX,r,d_workSX);
	cudaDeviceSynchronize();
	reduceToOne(kin,d_workSX);
	*kin *= dV/2.0/Lx/Ly/Lz;

	deviceSub<<<grid0,block0>>>(d_workSX,stress[5],stress[7]);
	deviceSub<<<grid0,block0>>>(d_workSY,stress[6],stress[2]);
	deviceSub<<<grid0,block0>>>(d_workSZ,stress[1],stress[3]);

	deviceSca<<<grid0,block0>>>(d_workSX,d_workSX,d_workSY,d_workSZ,d_workSX,d_workSY,d_workSZ);
	deviceMul<<<grid0,block0>>>(d_workSX,r,d_workSX);
	cudaDeviceSynchronize();
	reduceToOne(enst,d_workSX);
	*enst *= dV/Lx/Ly/Lz/Re;
}

__global__ void calcIntegrals2(myprec *r, myprec *u, myprec *v, myprec *w, myprec *stress[9], myprec *kin, myprec *enst) {

	*kin  = 0;
	*enst = 0;

	myprec dV = 1.0/d_dx/d_dy/d_dz;

	deviceSca<<<grid0,block0>>>(d_workSX,u,v,w,u,v,w);
	deviceMul<<<grid0,block0>>>(d_workSX,r,d_workSX);
	for (int it=0; it<mx*my*mz; it++) 	{
		*kin += d_workSX[it];
	}
	*kin *= dV/2.0/Lx/Ly/Lz;
	deviceSub<<<grid0,block0>>>(d_workSX,stress[5],stress[7]);
	deviceSub<<<grid0,block0>>>(d_workSY,stress[6],stress[2]);
	deviceSub<<<grid0,block0>>>(d_workSZ,stress[1],stress[3]);

	deviceSca<<<grid0,block0>>>(d_workSX,d_workSX,d_workSY,d_workSZ,d_workSX,d_workSY,d_workSZ);
	deviceMul<<<grid0,block0>>>(d_workSX,r,d_workSX);
	for (int it=0; it<mx*my*mz; it++) 	{
		*enst += d_workSX[it];
	}
	*enst *= dV/Lx/Ly/Lz/Re;
}
