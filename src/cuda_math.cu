#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_math.h"

__device__ myprec *wrkM;
__device__ int block, grid, total;

__global__ void deviceSum(myprec *a, myprec *b, myprec *c) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = b[id.g] + c[id.g];
}

__global__ void deviceSub(myprec *a, myprec *b, myprec *c) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = b[id.g] - c[id.g];
}

__global__ void deviceSca(myprec *a, myprec *bx, myprec *by, myprec *bz, myprec *cx, myprec *cy, myprec *cz) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = bx[id.g]*cx[id.g] + by[id.g]*cy[id.g] + bz[id.g]*cz[id.g];
}

__global__ void deviceMul(myprec *a, myprec *b, myprec *c) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = b[id.g]*c[id.g];
}

__global__ void deviceDiv(myprec *a, myprec *b, myprec *c) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = b[id.g]/c[id.g];
}

__global__ void deviceCpy(myprec *a, myprec *b) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = b[id.g];
}

__global__ void reduceToMax(myprec *gOut, myprec *var) {

	checkCudaDev( cudaMalloc((void**)&wrkM ,grid0.x*grid0.y*sizeof(myprec)) );

	total = mx*my*mz;
	grid  = grid0.x*grid0.y;
	block = block0.x*block0.y;

	maxOfThreads<<<grid, block, block*sizeof(myprec)>>>(wrkM , var,  total);
	cudaDeviceSynchronize();
	maxOfThreads<<<   1, block, block*sizeof(myprec)>>>(wrkM, wrkM, grid);
	cudaDeviceSynchronize();

	*gOut = wrkM[0];

	checkCudaDev( cudaFree(wrkM  ) );
}

__global__ void reduceToMin(myprec *gOut, myprec *var) {

	checkCudaDev( cudaMalloc((void**)&wrkM ,grid0.x*grid0.y*sizeof(myprec)) );

	total = mx*my*mz;
	grid  = grid0.x*grid0.y;
	block = block0.x*block0.y;

	minOfThreads<<<grid, block, block*sizeof(myprec)>>>(wrkM, var,  total);
	cudaDeviceSynchronize();
	minOfThreads<<<   1, block, block*sizeof(myprec)>>>(wrkM, wrkM, grid);
	cudaDeviceSynchronize();

	*gOut = wrkM[0];

	checkCudaDev( cudaFree(wrkM  ) );

}

__global__ void reduceToOne(myprec *gOut, myprec *var) {

	checkCudaDev( cudaMalloc((void**)&wrkM ,grid0.x*grid0.y*sizeof(myprec)) );

	total = mx*my*mz;
	grid  = grid0.x*grid0.y;
	block = block0.x*block0.y;

	reduceThreads<<<grid, block, block*sizeof(myprec)>>>(wrkM, var , total);
	cudaDeviceSynchronize();
	reduceThreads<<<   1, block, block*sizeof(myprec)>>>(wrkM, wrkM, grid);
	cudaDeviceSynchronize();

	*gOut = wrkM[0];

	checkCudaDev( cudaFree(wrkM  ) );

}

__global__ void reduceThreads(myprec *gOut, myprec *gArr, int arraySize) {

	int bdim  = blockDim.x;
	int tix   = threadIdx.x;
	int bix   = blockIdx.x;
	int gdim  = gridDim.x*blockDim.x;

	int glb   = tix + bix * bdim;

	myprec sum = 0;
    for (int i = glb; i < arraySize; i += gdim)
        sum += gArr[i];

	extern __shared__ myprec shArr[];
	shArr[tix] = sum;
	__syncthreads();
	for (int size = bdim/2; size>0; size/=2) {
		if (tix<size)
			shArr[tix] += shArr[tix+size];
		__syncthreads();
	}
	if (tix == 0) {
			gOut[bix] = shArr[0];
	}

	__syncthreads();

}

__global__ void minOfThreads(myprec *gOut, myprec *gArr, int arraySize) {

	int bdim  = blockDim.x;
	int tix   = threadIdx.x;
	int bix   = blockIdx.x;
	int gdim  = gridDim.x*blockDim.x;

	int glb   = tix + bix * bdim;

	myprec sum = 100000;
    for (int i = glb; i < arraySize; i += gdim)
        sum = MIN(sum,gArr[i]);

	extern __shared__ myprec shArr[];
	shArr[tix] = sum;
	__syncthreads();
	for (int size = bdim/2; size>0; size/=2) {
		if (tix<size)
			shArr[tix] = MIN(shArr[tix],shArr[tix+size]);
		__syncthreads();
	}
	if (tix == 0) {
			gOut[bix] = shArr[0];
	}

	__syncthreads();

}

__global__ void maxOfThreads(myprec *gOut, myprec *gArr, int arraySize) {

	int bdim  = blockDim.x;
	int tix   = threadIdx.x;
	int bix   = blockIdx.x;
	int gdim  = gridDim.x*blockDim.x;

	int glb   = tix + bix * bdim;

	myprec sum = -100000;
    for (int i = glb; i < arraySize; i += gdim)
        sum = MAX(sum,gArr[i]);

	extern __shared__ myprec shArr[];
	shArr[tix] = sum;
	__syncthreads();
	for (int size = bdim/2; size>0; size/=2) {
		if (tix<size)
			shArr[tix] = MAX(shArr[tix],shArr[tix+size]);
		__syncthreads();
	}
	if (tix == 0) {
			gOut[bix] = shArr[0];
	}

	__syncthreads();

}





