#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_math.h"

__device__ myprec *wrkM,*wrkM1;
__device__ int block, grid;

__global__ void reduceThreads(myprec *gOut, myprec *gArr);

__global__ void deviceSum(myprec *a, myprec *b, myprec *c) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g] + c[id.g];
}

__global__ void deviceSub(myprec *a, myprec *b, myprec *c) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g] - c[id.g];
}

__global__ void deviceSca(myprec *a, myprec *bx, myprec *by, myprec *bz, myprec *cx, myprec *cy, myprec *cz) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = bx[id.g]*cx[id.g] + by[id.g]*cy[id.g] + bz[id.g]*cz[id.g];
}

__global__ void deviceMul(myprec *a, myprec *b, myprec *c) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g]*c[id.g];
}

__global__ void deviceCpy(myprec *a, myprec *b) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g];
}

__global__ void reduceToOne(myprec *gOut, myprec *var) {

	checkCudaDev( cudaMalloc((void**)&wrkM ,grid0.x*grid0.y*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&wrkM1,                sizeof(myprec)) );

	grid  = grid0.x*grid0.y;
	block = block0.x*block0.y;

	reduceThreads<<<grid, block, block*sizeof(myprec)>>>(wrkM , var);
	reduceThreads<<<   1, grid , grid *sizeof(myprec)>>>(wrkM1, wrkM);

	*gOut = *wrkM1;

	checkCudaDev( cudaFree(wrkM ) );
	checkCudaDev( cudaFree(wrkM1) );

}

__global__ void reduceThreads(myprec *gOut, myprec *gArr) {

	int bdim  = blockDim.x;
	int tix   = threadIdx.x;
	int bix   = blockIdx.x;


	int glb = tix + bix * bdim;

	extern __shared__ myprec shArr[];
	shArr[tix] = gArr[glb];
	__syncthreads();
	for (int size = bdim/2; size>0; size/=2) { //uniform
		if (tix<size)
			shArr[tix] += shArr[tix+size];
		__syncthreads();
	}
	if (tix == 0)
		gOut[bix] = shArr[0];

}





