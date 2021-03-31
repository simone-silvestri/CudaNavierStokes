#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"


__device__ myprec d_phi[mx*my*mz];

__device__ myprec d_rhs1[mx*my*mz];
__device__ myprec d_rhs2[mx*my*mz];
__device__ myprec d_rhs3[mx*my*mz];
__device__ myprec d_rhs4[mx*my*mz];
__device__ myprec d_temp[mx*my*mz];

__global__ void sumArrays(myprec *a, myprec *b, myprec *c, myprec lambda) {
	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int glb = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	a[glb] = b[glb] + c[glb]*(d_dt)*lambda;
}

__global__ void rkfinal() {
	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int glb = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	d_phi[glb] = d_phi[glb] + d_dt*
			( d_rhs1[glb] +
					2*d_rhs2[glb] +
					2*d_rhs3[glb] +
					d_rhs4[glb])/6.;
}

void runHost() {

	printf("Grid 0: {%d, %d, %d} blocks. Blocks 0: {%d, %d, %d} threads.\n",hgrid[0].x, hgrid[0].y, hgrid[0].z, hblock[0].x, hblock[0].y, hblock[0].z);
	printf("Grid 1: {%d, %d, %d} blocks. Blocks 1: {%d, %d, %d} threads.\n",hgrid[1].x, hgrid[1].y, hgrid[1].z, hblock[1].x, hblock[1].y, hblock[1].z);
	printf("Grid 2: {%d, %d, %d} blocks. Blocks 2: {%d, %d, %d} threads.\n",hgrid[2].x, hgrid[2].y, hgrid[2].z, hblock[2].x, hblock[2].y, hblock[2].z);

	for (int istep=0; istep < nsteps; istep++) {
		RHSDeviceZ2_host<<<hgrid[2],hblock[2]>>>();
		//sumArrays<<<hgrid[0],hblock[0]>>>(d_temp,d_phi,d_rhs1,0.5);

		//RHS2DeviceZ<<<hgrid[2],hblock[2]>>>(d_rhs2,d_phi);
		//sumArrays<<<hgrid[0],hblock[0]>>>(d_temp,d_phi,d_rhs2,0.5);

		//RHS3DeviceZ<<<hgrid[2],hblock[2]>>>(d_rhs3,d_phi);
		//sumArrays<<<hgrid[0],hblock[0]>>>(d_temp,d_phi,d_rhs3,1.0);

		//RHSDeviceZ<<<hgrid[2],hblock[2]>>>(d_rhs4,d_phi);
		//rkfinal<<<hgrid[0],hblock[0]>>>();
	}
}

__global__ void RHSDeviceZ2_host() {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidZ();

	derDev1z(d_rhs1,d_phi,id);
	d_rhs1[id.g] = -d_rhs1[id.g]*U;

	d_temp[id.g] = (d_phi[id.g] + d_rhs1[id.g]*(d_dt)/2);

	derDev1z(d_rhs2,d_temp,id);
	d_rhs2[id.g] = -d_rhs2[id.g]*U;

	d_temp[id.g] = (d_phi[id.g] + d_rhs2[id.g]*(d_dt)/2);

	derDev1z(d_rhs3,d_temp,id);
	d_rhs3[id.g] = -d_rhs3[id.g]*U;

	d_temp[id.g] = (d_phi[id.g] + d_rhs3[id.g]*(d_dt));

	derDev1z(d_rhs4,d_temp,id);
	d_rhs4[id.g] = -d_rhs4[id.g]*U;

	d_phi[id.g] = d_phi[id.g] + (d_dt)*
			( d_rhs1[id.g] +
					2*d_rhs2[id.g] +
					2*d_rhs3[id.g] +
					d_rhs4[id.g])/6.;
}
