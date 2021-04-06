#include "cuda_functions.h"
#include "cuda_globals.h"


__device__ myprec d_phi[mx*my*mz];

__device__ myprec d_rhs1[mx*my*mz];
__device__ myprec d_rhs2[mx*my*mz];
__device__ myprec d_rhs3[mx*my*mz];
__device__ myprec d_rhs4[mx*my*mz];
__device__ myprec d_temp[mx*my*mz];
__device__ myprec d_tmp[mx*my*mz];


__device__ void RHSDevice(myprec *rhs, myprec *var, Indices id) {
  
#if parentGrid==0
  derDev1x(rhs,var,id);
#elif parentGrid==1
  derDev1y(rhs,var,id);
#else
  derDev1z(rhs,var,id);
#endif
  rhs[id.g] = -rhs[id.g]*U;

}


__device__ void rk4Device(Indices id) {

  RHSDevice(d_rhs1,d_phi,id); 

  d_temp[id.g] = (d_phi[id.g] + d_rhs1[id.g]*d_dt/2);
  RHSDevice(d_rhs2,d_temp,id); 

  d_temp[id.g] = (d_phi[id.g] + d_rhs2[id.g]*d_dt/2);
  RHSDevice(d_rhs3,d_temp,id);

  d_temp[id.g] = (d_phi[id.g] + d_rhs3[id.g]*d_dt);
  RHSDevice(d_rhs4,d_temp,id);
 
  d_phi[id.g] = d_phi[id.g] + d_dt*
                              ( d_rhs1[id.g] +
			      2*d_rhs2[id.g] + 	
			      2*d_rhs3[id.g] + 	
			        d_rhs4[id.g])/6.; 	


}


__global__ void runDevice() {

  Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

#if parentGrid==0
  id.mkidX();
  if(id.g==0) {
        printf("\n");
        printf("Using X-Grid\n");
        printf("Grid: {%d %d %d}. Blocks: {%d %d %d}.\n",gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
        printf("\n");
  }
#elif parentGrid==1
  id.mkidY();
  if(id.g==0) {
        printf("\n");
        printf("Using Y-Grid\n");
        printf("Grid: {%d %d %d}. Blocks: {%d %d %d}.\n",gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
        printf("\n");
  }
#else
  id.mkidZ();
  if(id.g==0) {
        printf("\n");
        printf("Using Z-Grid\n");
        printf("Grid: {%d %d %d}. Blocks: {%d %d %d}.\n",gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z);
        printf("\n");
  }
#endif


  for (int istep=0; istep < nsteps; istep++) {
          rk4Device(id);
  }
}


