
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


/*
 *  The L-versions of the RHS have to be ran with
 *  - the L-version of the derivatives
 *  i.e.: derDev1xL instead of derDev1x
 *  - the L-version of the grid
 *  i.e.: h_gridL[0] instead of h_grid[0]
 */

__device__ myprec d_work[mx*my*mz];


__global__ void RHSDeviceX(myprec *rhsX, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	derDev2x(rhsX,var,id);
	rhsX[id.g] = rhsX[id.g]*visc;
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


__global__ void RHSDeviceXL(myprec *rhsX, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1xL(rhsX,var,id);
	int i     = id.tix;
	int jBase = id.bix*lPencils;
	int k     = id.biy;
	for (int sj = id.tiy; sj < lPencils; sj += id.bdy) {
		int globalIdx = k * mx * my + (jBase + sj) * mx + i;
		rhsX[globalIdx] = -rhsX[globalIdx]*U;
	}
}


__global__ void RHSDeviceYL(myprec *rhsY, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	derDev1yL(rhsY,var,id);

	int i  = id.bix*id.bdx + id.tix;
	int k  = id.biy;
	for (int j = id.tiy; j < my; j += id.bdy) {
		int globalIdx = k * mx * my + j * mx + i;
		rhsY[globalIdx] = -rhsY[globalIdx]*U;
	}
}


__global__ void RHSDeviceZL(myprec *rhsZ, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	int i  = id.bix*id.bdx + id.tix;
	int j  = id.biy;
	derDev1zL(rhsZ,var,id);
	for (int k = id.tiy; k < mz; k += id.bdy) {
		int globalIdx = k * mx * my + j * mx + i;
		rhsZ[globalIdx] = -rhsZ[globalIdx]*U;
	}
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

