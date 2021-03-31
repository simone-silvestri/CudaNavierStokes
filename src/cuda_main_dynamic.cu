#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"


__device__ myprec d_phi[mx*my*mz];

__device__ myprec d_rhs1[mx*my*mz];
__device__ myprec d_rhs2[mx*my*mz];
__device__ myprec d_rhs3[mx*my*mz];
__device__ myprec d_rhs4[mx*my*mz];
__device__ myprec d_temp[mx*my*mz];
__device__ myprec d_work[mx*my*mz];

__device__ myprec dt2;

__global__ void threadBlockDeviceSynchronize() {
	__syncthreads();
}

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

/*	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

//#if parentGrid==0
//	id.mkidX();
//	if(id.g==0) {
//		printf("\n");
//		printf("Using X-Grid\n");
//		printf("Grid: {%d %d %d}. Blocks: {%d %d %d}.\n",gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
//		printf("\n");
//	}
//
//#elif parentGrid==1
//	id.mkidY();
//	if(id.g==0) {
//		printf("\n");
//		printf("Using Y-Grid\n");
//		printf("Grid: {%d %d %d}. Blocks: {%d %d %d}.\n",gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
//		printf("\n");
//	}
//#else
//	id.mkidZ();
//	if(id.g==0) {
//		printf("\n");
//		printf("Using Z-Grid\n");
//		printf("Grid: {%d %d %d}. Blocks: {%d %d %d}.\n",gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
//		printf("\n");
//	}
#endif */

	dt2 = d_dt/2;

	__syncthreads();
	for (int istep=0; istep < nsteps; istep++) {

		/* Just running this in the loop works!! (rk separate for separate directions, no way to do cross-derivatives)
		//		if(id.g==64) {
		//			 RHSDeviceZ2<<<d_grid[2],d_block[2]>>>(d_rhs1,d_rhs2,d_rhs3,d_rhs4,d_temp,d_phi,&d_dt);
		//		} */

		// WE can stream the RHS for an additional parallelism level!

		RHSDeviceX<<<d_grid[0],d_block[0]>>>(d_rhs1,d_phi);
//		RHSDeviceY<<<d_grid[1],d_block[1]>>>(d_rhs1,d_temp);
		RHSDeviceZ<<<d_grid[2],d_block[2]>>>(d_rhs1,d_temp);
		eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs1,&dt2);

		RHSDeviceX<<<d_grid[0],d_block[0]>>>(d_rhs2,d_temp);
//		RHSDeviceY<<<d_grid[1],d_block[1]>>>(d_rhs2,d_temp);
		RHSDeviceZ<<<d_grid[2],d_block[2]>>>(d_rhs2,d_temp);
		eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs2,&dt2);

		RHSDeviceX<<<d_grid[0],d_block[0]>>>(d_rhs3,d_temp);
//		RHSDeviceY<<<d_grid[1],d_block[2]>>>(d_rhs3,d_temp);
		RHSDeviceZ<<<d_grid[2],d_block[2]>>>(d_rhs3,d_temp);
		eulerSum<<<d_grid[0],d_block[0]>>>(d_temp,d_phi,d_rhs3,&d_dt);

		RHSDeviceX<<<d_grid[0],d_block[0]>>>(d_rhs4,d_temp);
//		RHSDeviceY<<<d_grid[1],d_block[1]>>>(d_rhs4,d_temp);
		RHSDeviceZ<<<d_grid[2],d_block[2]>>>(d_rhs4,d_temp);
		rkfinal<<<d_grid[0],d_block[0]>>>(d_phi,d_rhs1,d_rhs2,d_rhs3,d_rhs4,&d_dt);
	}
}


__global__ void RHSDeviceX(myprec *rhsX, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	derDev1x(rhsX,var,id);
	rhsX[id.g] = -rhsX[id.g]*U;
}

__global__ void RHSDeviceY(myprec *rhsY, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidY();

	derDev1y(d_work,var,id);
	rhsY[id.g] = rhsY[id.g] - d_work[id.g]*U;
}

__global__ void RHSDeviceZ(myprec *rhsZ, myprec *var) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZ();

	derDev1z(d_work,var,id);
	rhsZ[id.g] = rhsZ[id.g] - d_work[id.g]*U;
}

__global__ void RHSDeviceZ2(myprec *rhs1, myprec *rhs2, myprec *rhs3, myprec *rhs4, myprec *temp, myprec *phi, myprec *dt) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZ();

	derDev1z(rhs1,phi,id);
	rhs1[id.g] = -rhs1[id.g]*U;

	temp[id.g] = (phi[id.g] + rhs1[id.g]*(*dt)/2);

	derDev1z(rhs2,temp,id);
	rhs2[id.g] = -rhs2[id.g]*U;

	temp[id.g] = (phi[id.g] + rhs2[id.g]*(*dt)/2);

	derDev1z(rhs3,temp,id);
	rhs3[id.g] = -rhs3[id.g]*U;

	temp[id.g] = (phi[id.g] + rhs3[id.g]*(*dt));

	derDev1z(rhs4,temp,id);
	rhs4[id.g] = -rhs4[id.g]*U;

	phi[id.g] = phi[id.g] + (*dt)*
			( rhs1[id.g] +
					2*rhs2[id.g] +
					2*rhs3[id.g] +
					rhs4[id.g])/6.;
}
