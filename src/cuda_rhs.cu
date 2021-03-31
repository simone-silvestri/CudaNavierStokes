
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include "globals.h"
#include "cuda_functions.h"

__device__ myprec d_work[mx*my*mz];


__global__ void RHSDeviceX(myprec *rhsX, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	derDev2x(rhsX,var,id);
	derDev1x(d_work,var,id);
	rhsX[id.g] = rhsX[id.g]*visc - d_work[id.g]*U;
}

__global__ void RHSDeviceY(myprec *rhsY, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidY();

	derDev1y(rhsY,var,id);
	rhsY[id.g] = - rhsY[id.g]*U;
}

__global__ void RHSDeviceZ(myprec *rhsZ, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZ();

	derDev1z(rhsZ,var,id);
	rhsZ[id.g] = -rhsZ[id.g]*U;
}

__global__ void RHSDeviceYSum(myprec *rhsY, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidY();

	derDev1y(d_work,var,id); // do stuff here
	rhsY[id.g] = rhsY[id.g] - d_work[id.g]*U;
}

__global__ void RHSDeviceZSum(myprec *rhsZ, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZ();

	derDev1z(d_work,var,id); // do stuff here
	rhsZ[id.g] = rhsZ[id.g] - d_work[id.g]*U;
}

