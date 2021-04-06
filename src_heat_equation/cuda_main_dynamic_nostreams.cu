#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"


__device__ myprec d_phi[mx*my*mz];

__device__ myprec d_rhs1[mx*my*mz];
__device__ myprec d_rhs2[mx*my*mz];
__device__ myprec d_rhs3[mx*my*mz];
__device__ myprec d_rhs4[mx*my*mz];
__device__ myprec d_temp[mx*my*mz];

__device__ myprec dt2;

__global__ void eulerSum(myprec *a, myprec *b, myprec *c, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g] + c[id.g]*(*dt);
}

__global__ void rkfinal(myprec *a, myprec *b, myprec *c, myprec *d, myprec *e, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = a[id.g] + (*dt)*(b[id.g] + 2*c[id.g] + 2*d[id.g] + e[id.g])/6.;
}

__global__ void runDevice() {

	dt2 = d_dt/2;

	__syncthreads();

	for (int istep=0; istep < nsteps; istep++) {

		// WE can stream the RHS for an additional parallelism level!
		RHSDeviceX<<<d_grid[0],d_block[0]>>>(d_rhs1,d_phi);
		RHSDeviceYSum<<<d_grid[1],d_block[1]>>>(d_rhs1,d_phi);
		RHSDeviceZSum<<<d_grid[2],d_block[2]>>>(d_rhs1,d_phi);
		eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs1,&dt2);

		RHSDeviceX<<<d_grid[0],d_block[0]>>>(d_rhs2,d_temp);
		RHSDeviceYSum<<<d_grid[1],d_block[1]>>>(d_rhs2,d_temp);
		RHSDeviceZSum<<<d_grid[2],d_block[2]>>>(d_rhs2,d_temp);
		eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs2,&dt2);

		RHSDeviceX<<<d_grid[0],d_block[0]>>>(d_rhs3,d_temp);
		RHSDeviceYSum<<<d_grid[1],d_block[1]>>>(d_rhs3,d_temp);
		RHSDeviceZSum<<<d_grid[2],d_block[2]>>>(d_rhs3,d_temp);
		eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs3,&d_dt);

		RHSDeviceX<<<d_grid[0],d_block[0]>>>(d_rhs4,d_temp);
		RHSDeviceYSum<<<d_grid[1],d_block[1]>>>(d_rhs4,d_temp);
		RHSDeviceZSum<<<d_grid[2],d_block[2]>>>(d_rhs4,d_temp);
		rkfinal<<<d_grid[0],d_block[0]>>>(d_phi,d_rhs1,d_rhs2,d_rhs3,d_rhs4,&d_dt);
	}

}
