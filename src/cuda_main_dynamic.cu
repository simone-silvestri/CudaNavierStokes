#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"


__device__ myprec d_phi[mx*my*mz];
__device__ myprec d_temp[mx*my*mz];


__device__ myprec *d_rhs1[3];
__device__ myprec *d_rhs2[3];
__device__ myprec *d_rhs3[3];
__device__ myprec *d_rhs4[3];

__device__ myprec dt2;


__global__ void eulerSum(myprec *a, myprec *b,  myprec *cx, myprec *cy, myprec *cz, myprec *dt);
__global__ void rk4final(myprec *a, myprec *bx, myprec *cx, myprec *dx, myprec *ex,
									myprec *by, myprec *cy, myprec *dy, myprec *ey,
									myprec *bz, myprec *cz, myprec *dz, myprec *ez, myprec *dt);

__global__ void runDevice() {

	dt2 = d_dt/2;
	__syncthreads();

	/* We use streams for an additional parallelization
	 * (X, Y and Z RHs will be calculated on different streams
	 */

	/* allocating temporary arrays and streams */
	void (*RHSDeviceDir[3])(myprec*, myprec*);
	RHSDeviceDir[0] = RHSDeviceXL;
	RHSDeviceDir[1] = RHSDeviceY;
	RHSDeviceDir[2] = RHSDeviceZL;

	cudaStream_t s[3];
    for (int i=0; i<3; i++) {
    	checkCudaDev( cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking) );
    	checkCudaDev( cudaMalloc((void**)&d_rhs1[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhs2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhs3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhs4[i],mx*my*mz*sizeof(myprec)) );
    }

    for (int istep = 0; istep < nsteps; istep++) {

    	/* rk step 1 */

    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_gridL[d],d_blockL[d],0,s[d]>>>(d_rhs1[d],d_phi);
    	cudaDeviceSynchronize();
    	eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs1[0],d_rhs1[1],d_rhs1[2],&dt2);

    	/* rk step 2 */

    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_gridL[d],d_blockL[d],0,s[d]>>>(d_rhs2[d],d_temp);
    	cudaDeviceSynchronize();
    	eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs2[0],d_rhs2[1],d_rhs2[2],&dt2);

    	/* rk step 3 */

    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_gridL[d],d_blockL[d],0,s[d]>>>(d_rhs3[d],d_temp);
    	cudaDeviceSynchronize();
    	eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs3[0],d_rhs3[1],d_rhs3[2],&d_dt);

    	/* rk step 4 */

    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_gridL[d],d_blockL[d],0,s[d]>>>(d_rhs4[d],d_temp);
		cudaDeviceSynchronize();
		rk4final<<<d_grid[0],d_block[0]>>>(d_phi,d_rhs1[0],d_rhs2[0],d_rhs3[0],d_rhs4[0],
												 d_rhs1[1],d_rhs2[1],d_rhs3[1],d_rhs4[1],
												 d_rhs1[2],d_rhs2[2],d_rhs3[2],d_rhs4[2],&d_dt);
	}

	for (int i=0; i<3; i++) {
		checkCudaDev( cudaStreamDestroy(s[i]) );
		checkCudaDev( cudaFree(d_rhs1[i]) );
		checkCudaDev( cudaFree(d_rhs2[i]) );
		checkCudaDev( cudaFree(d_rhs3[i]) );
		checkCudaDev( cudaFree(d_rhs4[i]) );
	}
}

__global__ void eulerSum(myprec *a, myprec *b, myprec *cx, myprec *cy, myprec *cz, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g] + ( cx[id.g] + cy[id.g] + cz[id.g] )*(*dt);
}

__global__ void rk4final(myprec *a, myprec *bx, myprec *cx, myprec *dx, myprec *ex,
									myprec *by, myprec *cy, myprec *dy, myprec *ey,
									myprec *bz, myprec *cz, myprec *dz, myprec *ez, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = a[id.g] + (*dt)*( bx[id.g] + 2*cx[id.g] + 2*dx[id.g] + ex[id.g] +
								by[id.g] + 2*cy[id.g] + 2*dy[id.g] + ey[id.g] +
								bz[id.g] + 2*cz[id.g] + 2*dz[id.g] + ez[id.g])/6.;
}



