#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"


__device__ myprec d_phi[mx*my*mz];

__device__ myprec d_rhs1x[mx*my*mz];
__device__ myprec d_rhs2x[mx*my*mz];
__device__ myprec d_rhs3x[mx*my*mz];
__device__ myprec d_rhs4x[mx*my*mz];
__device__ myprec d_rhs1y[mx*my*mz];
__device__ myprec d_rhs2y[mx*my*mz];
__device__ myprec d_rhs3y[mx*my*mz];
__device__ myprec d_rhs4y[mx*my*mz];
__device__ myprec d_rhs1z[mx*my*mz];
__device__ myprec d_rhs2z[mx*my*mz];
__device__ myprec d_rhs3z[mx*my*mz];
__device__ myprec d_rhs4z[mx*my*mz];
__device__ myprec d_temp[mx*my*mz];

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

    cudaStream_t s[3];
    for (int i=0; i<3; i++)
    	checkCudaDev( cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking) );

	for (int istep=0; istep < nsteps; istep++) {

		RHSDeviceX<<<d_grid[0],d_block[0],0,s[0]>>>(d_rhs1x,d_phi);
		RHSDeviceY<<<d_grid[1],d_block[1],0,s[1]>>>(d_rhs1y,d_phi);
		RHSDeviceZ<<<d_grid[2],d_block[2],0,s[2]>>>(d_rhs1z,d_phi);
		cudaDeviceSynchronize();
		eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs1x,d_rhs1y,d_rhs1z,&dt2);

		RHSDeviceX<<<d_grid[0],d_block[0],0,s[0]>>>(d_rhs2x,d_temp);
		RHSDeviceY<<<d_grid[1],d_block[1],0,s[1]>>>(d_rhs2y,d_temp);
		RHSDeviceZ<<<d_grid[2],d_block[2],0,s[2]>>>(d_rhs2z,d_temp);
		cudaDeviceSynchronize();
		eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs2x,d_rhs2y,d_rhs2z,&dt2);

		RHSDeviceX<<<d_grid[0],d_block[0],0,s[0]>>>(d_rhs3x,d_temp);
		RHSDeviceY<<<d_grid[1],d_block[1],0,s[1]>>>(d_rhs3y,d_temp);
		RHSDeviceZ<<<d_grid[2],d_block[2],0,s[2]>>>(d_rhs3z,d_temp);
		cudaDeviceSynchronize();
		eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs3x,d_rhs3y,d_rhs3z,&d_dt);

		RHSDeviceX<<<d_grid[0],d_block[0],0,s[0]>>>(d_rhs4x,d_temp);
		RHSDeviceY<<<d_grid[1],d_block[1],0,s[1]>>>(d_rhs4y,d_temp);
		RHSDeviceZ<<<d_grid[2],d_block[2],0,s[2]>>>(d_rhs4z,d_temp);
		cudaDeviceSynchronize();
		rk4final<<<d_grid[0],d_block[0]>>>(d_phi,d_rhs1x,d_rhs2x,d_rhs3x,d_rhs4x,
												 d_rhs1y,d_rhs2y,d_rhs3y,d_rhs4y,
												 d_rhs1z,d_rhs2z,d_rhs3z,d_rhs4z,&d_dt);

	}

    for (int i=0; i<3; i++)
    	checkCudaDev( cudaStreamDestroy(s[i]) );
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

