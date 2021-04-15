
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_math.h"

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
