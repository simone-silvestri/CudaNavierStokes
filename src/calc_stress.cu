
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_functions.h"
#include "cuda_globals.h"

__device__ myprec d_workSX[mx*my*mz];
__device__ myprec d_workSY[mx*my*mz];
__device__ myprec d_workSZ[mx*my*mz];

__device__ float integral;

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

	dil[id.g] = stress[0][id.g] + stress[4][id.g] + stress[8][id.g];

}

__global__ void calcIntegrals(myprec *r, myprec *u, myprec *v, myprec *w, myprec *stress[9], myprec *kin, myprec *enst) {

	*kin  = 0;
	*enst = 0;

	myprec dV = 1.0/d_dx/d_dy/d_dz;

	deviceSca<<<grid0,block0>>>(d_workSX,u,v,w,u,v,w);
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
