#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_math.h"

__device__ myprec *wrkM;
__device__ unsigned int block, grid, total;

__global__ void deviceBlocker() {
}

__global__ void deviceSum(myprec *a, myprec *b, myprec *c) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = b[id.g] + c[id.g];
}

__global__ void deviceSumOne(myprec *a, myprec *b, myprec *c) {
	*a = *b + *c;
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

__global__ void deviceDivOne(myprec *a, myprec *b, myprec *c) {
	*a = *b/(*c);
}

__global__ void deviceCpy(myprec *a, myprec *b) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = b[id.g];
}

__global__ void deviceCpyOne(myprec *a, myprec *b) {
	*a = *b;
}

__device__ void reduceToMax(myprec *gOut, myprec *var) {

	total = mx*my*mz;

	grid  = my / sPencils *  mz;
	block = mx * sPencils;

	checkCudaDev( cudaMalloc((void**)&wrkM ,grid*sizeof(myprec)) );
	cudaDeviceSynchronize();

	block = findPreviousPowerOf2(block);

	maxOfThreads<<<grid, block, block*sizeof(myprec)>>>(wrkM , var,  total);
	cudaDeviceSynchronize();
	maxOfThreads<<<   1, block, block*sizeof(myprec)>>>(wrkM, wrkM, grid);
	cudaDeviceSynchronize();

	*gOut = wrkM[0];

	checkCudaDev( cudaFree(wrkM  ) );
}

__device__ void reduceToMin(myprec *gOut, myprec *var) {


	total = mx*my*mz;

	grid  = my / sPencils *  mz;
	block = mx * sPencils;

	checkCudaDev( cudaMalloc((void**)&wrkM ,grid*sizeof(myprec)) );
	cudaDeviceSynchronize();

	block = findPreviousPowerOf2(block);

	minOfThreads<<<grid, block, block*sizeof(myprec)>>>(wrkM, var,  total);
	cudaDeviceSynchronize();
	minOfThreads<<<   1, block, block*sizeof(myprec)>>>(wrkM, wrkM, grid);
	cudaDeviceSynchronize();

	*gOut = wrkM[0];

	checkCudaDev( cudaFree(wrkM  ) );

}

__device__ void volumeIntegral(myprec *gOut, myprec *var) {

	grid  = my*mz;
	block = mx;
	total = mx*my*mz;

	checkCudaDev( cudaMalloc((void**)&wrkM ,grid*sizeof(myprec)) );
	cudaDeviceSynchronize();

	block = findPreviousPowerOf2(block);

	integrateThreads<<<grid, block, block*sizeof(myprec)>>>(wrkM, var , total);
	cudaDeviceSynchronize();
	reduceThreads<<<   1, block, block*sizeof(myprec)>>>(wrkM, wrkM, grid);
	cudaDeviceSynchronize();

	*gOut = wrkM[0];
	checkCudaDev( cudaFree( wrkM ) );
}

__device__ void reduceToOne(myprec *gOut, myprec *var) {

	total = mx*my*mz;

	grid  = my / sPencils *  mz;
	block = mx * sPencils;


	checkCudaDev( cudaMalloc((void**)&wrkM ,grid*sizeof(myprec)) );
	cudaDeviceSynchronize();

	block = findPreviousPowerOf2(block);

	reduceThreads<<<grid, block, block*sizeof(myprec)>>>(wrkM, var , total);
	cudaDeviceSynchronize();
	reduceThreads<<<   1, block, block*sizeof(myprec)>>>(wrkM, wrkM, grid);
	cudaDeviceSynchronize();

	*gOut = wrkM[0];
	checkCudaDev( cudaFree(wrkM  ) );

}

__global__ void integrateThreads(myprec *gOut, myprec *gArr, int arraySize) {

	int bdim  = blockDim.x;
	int tix   = threadIdx.x;
	int bix   = blockIdx.x;
	int gdim  = gridDim.x*blockDim.x;

	int glb   = tix + bix * bdim;

	myprec sum = 0;
	for (int it = glb; it < arraySize; it += gdim) {
		int i = it % mx;
		sum += gArr[it]*d_dxv[i]/d_dy/d_dz /(Lx*Ly*Lz);
	}
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

__global__ void AverageThreads(myprec *gOut, myprec *gArr, int arraySize) {

	int bdim  = blockDim.x;
	int tix   = threadIdx.x;
	int bix   = blockIdx.x;
	int gdim  = gridDim.x*blockDim.x;

	int glb   = tix + bix * bdim;

	myprec sum = 0;
	for (int it = glb; it < arraySize; it += gdim) {
		int i = it % mx;
		sum += gArr[it]*d_dxv[i]/d_dy/d_dz/Lx/Ly/Lz;
	}
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

__device__ unsigned int findPreviousPowerOf2(unsigned int n)
{
    while (n & n - 1) {
        n = n & n - 1;        // unset rightmost bit
    }

    return n;
}

__device__ unsigned int findNextPowerOf2(unsigned int n)
{
    unsigned count = 0;
    while(n != 0)
    {
        n >>= 1;
        count += 1;
    }
    
    n = 1 << count;
    return n;
}

unsigned int hostFindPreviousPowerOf2(unsigned int n)
{
    while (n & n - 1) {
        n = n & n - 1;        // unset rightmost bit
    }

    return n;
}

unsigned int hostfindNextPowerOf2(unsigned int n)
{
    unsigned count = 0;
    while(n != 0)
    {
        n >>= 1;
        count += 1;
    }

    n = 1 << count;
    return n;
}


void hostReduceToMin(myprec *gOut, myprec *var, Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	myprec *dwrkM;

	int tot = mx*my*mz;

	int gr  = my / sPencils *  mz;
	int bl = mx * sPencils;

	checkCuda( cudaMalloc((void**)&dwrkM ,gr*sizeof(myprec)) );
	cudaDeviceSynchronize();

	bl = hostFindPreviousPowerOf2(bl);

	minOfThreads<<< gr, bl, bl*sizeof(myprec)>>>(dwrkM, var,  tot);
	cudaDeviceSynchronize();
	minOfThreads<<< 1 , bl, bl*sizeof(myprec)>>>(dwrkM, dwrkM, gr);
	cudaDeviceSynchronize();

	deviceCpyOne<<<1,1>>>(gOut,&dwrkM[0]);

	checkCuda( cudaFree(dwrkM  ) );

}

void hostVolumeIntegral(myprec *gOut, myprec *var, Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	myprec *dwrkM;

	int tot = mx*my*mz;

	int gr  = my *  mz;
	int bl  = mx ;

	bl = hostFindPreviousPowerOf2(bl);

	checkCuda( cudaMalloc((void**)&dwrkM ,gr*sizeof(myprec)) );
	cudaDeviceSynchronize();
	integrateThreads<<<gr, bl, bl*sizeof(myprec)>>>(dwrkM, var , tot);
	cudaDeviceSynchronize();
	reduceThreads<<<   1 , bl, bl*sizeof(myprec)>>>(dwrkM, dwrkM, gr);
	cudaDeviceSynchronize();

	deviceCpyOne<<<1,1>>>(gOut,&dwrkM[0]);

	checkCuda( cudaFree(dwrkM  ) );
}

void hostVolumeAverage(myprec *gOut, myprec *var, Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	myprec *dwrkM;

	int tot = mx*my*mz;

	int gr  = my *  mz;
	int bl = mx ;

	checkCuda( cudaMalloc((void**)&dwrkM ,gr*sizeof(myprec)) );
	cudaDeviceSynchronize();

	bl = hostFindPreviousPowerOf2(bl);

	AverageThreads<<<gr, bl, bl*sizeof(myprec)>>>(dwrkM, var , tot);
	cudaDeviceSynchronize();
	reduceThreads<<<   1 , bl, bl*sizeof(myprec)>>>(dwrkM, dwrkM, gr);
	cudaDeviceSynchronize();

	deviceCpyOne<<<1,1>>>(gOut,&dwrkM[0]);

	checkCuda( cudaFree(dwrkM  ) );
}


