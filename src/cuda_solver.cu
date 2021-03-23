#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"

// host routine to set constant data
void setDerivativeParameters(dim3 &grid, dim3 &block)
{

  // check to make sure dimensions are integral multiples of sPencils
  if ((mx % sPencils != 0) || (my %sPencils != 0) || (mz % sPencils != 0)) {
    printf("'mx', 'my', and 'mz' must be integral multiples of sPencils\n");
    exit(1);
  }
  
  if ((mx % lPencils != 0) || (my % lPencils != 0)) {
    printf("'mx' and 'my' must be multiples of lPencils\n");
    exit(1);
  }

  myprec h_dt = (myprec) dt;
  myprec h_dx = (myprec) x[1] - x[0];

  myprec *h_coeff = new myprec[stencilSize];
  
  for (int it=0; it<stencilSize; it++) 
   h_coeff[it] = (myprec) coeffS[it]; 

  checkCuda( cudaMemcpyToSymbol(dcoeff, h_coeff, stencilSize*sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_dt  , &h_dt  ,             sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_dx  , &h_dx  ,             sizeof(myprec), 0, cudaMemcpyHostToDevice) );

  grid  = dim3(my / sPencils, mz, 1);
  block = dim3(mx, sPencils, 1);

  delete [] h_coeff;

}



__device__ void derivative_x_lPencils(myprec *dydx, myprec *y, Indices id)
{
  __shared__ myprec s_f[lPencils][mx+stencilSize*2]; // 4-wide halo

  int i     = id.tix;
  int jBase = id.bix*lPencils;
  int k     = id.biy;
  int si    = i + 4; // local i for shared memory access + halo offset

  for (int sj = id.tiy; sj < lPencils; sj += id.bdy) {
    int globalIdx = k * mx * my + (jBase + sj) * mx + i;
    s_f[sj][si] = y[globalIdx];
  }

  __syncthreads();

  // fill in periodic images in shared memory array 
  if (i < stencilSize) {
    for (int sj = id.tiy; sj < lPencils; sj += id.bdy) {
      s_f[sj][si-stencilSize]  = s_f[sj][si+mx-stencilSize-1];
      s_f[sj][si+mx] = s_f[sj][si+1];
    }
  }

  __syncthreads();

  for (int sj = id.tiy; sj < lPencils; sj += id.bdy) {
     int globalIdx = k * mx * my + (jBase + sj) * mx + i;
     myprec dftemp = 0.0;
     for (int it=0; it<stencilSize; it++)  { 
      dftemp += dcoeff[it]*(s_f[sj][si+it-stencilSize]-s_f[sj][si+stencilSize-it])/d_dx;
     }
     dydx[globalIdx] = dftemp; 
  }
}



__device__ void derivative_x(myprec *dydx, myprec *y, Indices id)
{  

  int si = id.i + stencilSize;       // local i for shared memory access + halo offset
  int sj = id.tiy;                   // local j for shared memory access

  __shared__ myprec s_f[sPencils][mx+stencilSize*2]; // 4-wide halo

  s_f[sj][si] = y[id.g];

  __syncthreads();

  // fill in periodic images in shared memory array 
  if (id.i < stencilSize) {
    s_f[sj][si-stencilSize]  = s_f[sj][si+mx-stencilSize]; // CHANGED SIMONE: s_f[sj][si+mx-stencilSize-1];
    s_f[sj][si+mx]           = s_f[sj][si];                // CHANGED SIMONE: s_f[sj][si+1];
  }

  __syncthreads();

  myprec dftemp = 0.0;
  for (int it=0; it<stencilSize; it++)  { 
   dftemp += dcoeff[it]*(s_f[sj][si+it-stencilSize]-s_f[sj][si+stencilSize-it])/d_dx;
  }

  __syncthreads();
 
  dydx[id.g] = dftemp; 
}

void copyInit(int direction, dim3 grid, dim3 block) {

  myprec *f = new myprec[mx*my*mz];
  myprec *d_f;
  int bytes = mx*my*mz * sizeof(myprec);
  checkCuda( cudaMalloc((void**)&d_f, bytes) );

  if(direction == 1) {

     for (int k=0; k<mz; k++)
     for (int j=0; j<my; j++)
     for (int i=0; i<mx; i++)  
      f[idx(i,j,k)] = (myprec) phi[idx(i,j,k)];

     // device arrays
     checkCuda( cudaMemcpy(d_f, f , bytes, cudaMemcpyHostToDevice) );  

     initDevice<<<grid, block>>>(d_f);

     checkCuda( cudaFree(d_f) );

  } else {

     checkCuda( cudaMemset(d_f, 0, bytes) );

     getResults<<<grid, block>>>(d_f);

     checkCuda( cudaMemcpy(f, d_f, bytes, cudaMemcpyDeviceToHost) );

     for (int k=0; k<mz; k++)
     for (int j=0; j<my; j++)
     for (int i=0; i<mx; i++)  
      phi[idx(i,j,k)] = (myprec) f[idx(i,j,k)];

     checkCuda( cudaFree(d_f) );
 
  }
  
  delete []  f;  

}


__global__ void getResults(myprec *d_f) {
  
  int i2  = threadIdx.x;
  int j2  = blockIdx.x*blockDim.y + threadIdx.y;
  int k2  = blockIdx.y;

  int globalIdx2 = k2 * mx * my + j2 * mx + i2;
  d_f[globalIdx2] = d_phi[globalIdx2];

}

__global__ void initDevice(myprec *d_f) {
  
  int i2  = threadIdx.x;
  int j2  = blockIdx.x*blockDim.y + threadIdx.y;
  int k2  = blockIdx.y;

  int globalIdx2 = k2 * mx * my + j2 * mx + i2;
  d_phi[globalIdx2] = d_f[globalIdx2];
}


__device__ void RHSDevice(myprec *var, myprec *rhs, Indices id) {
  
  derivative_x(var,rhs,id);
  var[id.g] = -var[id.g]*U;
}


__device__ void rk4Device(Indices id) {

  RHSDevice(d_rhs1,d_phi,id); 

  d_temp[id.g] = (d_phi[id.g] + d_rhs1[id.g]*d_dt/2);
  RHSDevice(d_rhs2,d_temp, id); 
  

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

  Indices id;

  id.mkidx(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.y);
  
  for (int istep=0; istep < nsteps; istep++) {
	  rk4Device(id);
  }
}
