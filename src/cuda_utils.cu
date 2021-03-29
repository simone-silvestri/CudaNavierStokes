#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"

__constant__ myprec dcoeffF[stencilSize];
__constant__ myprec dcoeffS[stencilSize+1];
__constant__ myprec d_dt, d_dx, d_dy, d_dz;

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
  myprec h_dy = (myprec) y[1] - y[0];
  myprec h_dz = (myprec) z[1] - z[0];

  myprec *h_coeffF = new myprec[stencilSize];
  myprec *h_coeffS = new myprec[stencilSize+1];
  
  for (int it=0; it<stencilSize; it++) 
   h_coeffF[it] = (myprec) coeffF[it]; 
  for (int it=0; it<stencilSize+1; it++) 
   h_coeffS[it] = (myprec) coeffS[it]; 

  checkCuda( cudaMemcpyToSymbol(dcoeffF, h_coeffF,  stencilSize   *sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(dcoeffS, h_coeffS, (stencilSize+1)*sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  
  checkCuda( cudaMemcpyToSymbol(d_dt  , &h_dt  ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_dx  , &h_dx  ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_dy  , &h_dy  ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_dz  , &h_dz  ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );

  
  // X-grid spencils
//  grid  = dim3(my / sPencils, mz, 1);
//  block = dim3(mx, sPencils, 1);     

  // X-grid lpencils
//  grid  = dim3(my / lPencils, mz, 1);
//  block = dim3(mx, sPencils, 1);     

  // Y-grid spencils
//  grid  = dim3(mx / sPencils, mz, 1);
//  block = dim3(sPencils, my, 1);     

  // Y-grid lpencils
//  grid  = dim3(mx / lPencils, mz, 1);
//  block = dim3(lPencils, (my / lPencils) * sPencils, 1);     

  // Z-grid spencils
  grid  = dim3(mx / sPencils, my, 1);
  block = dim3(sPencils, mz, 1);     

  // Z-grid lpencils
//  grid  = dim3(mx / lPencils, my, 1);
//  block = dim3(lPencils, (mz / lPencils) * sPencils, 1);     


  delete [] h_coeffF;
  delete [] h_coeffS;

}

void copyInit(int direction, dim3 grid, dim3 block) {

  myprec *f = new myprec[mx*my*mz];
  myprec *d_f;
  int bytes = mx*my*mz * sizeof(myprec);
  checkCuda( cudaMalloc((void**)&d_f, bytes) );

  if(direction == 1) {

     for (int it=0; it<mx*my*mz; it++)  
      f[it] = (myprec) phi[it];

     // device arrays
     checkCuda( cudaMemcpy(d_f, f , bytes, cudaMemcpyHostToDevice) );  

     initDevice<<<grid, block>>>(d_f);

     checkCuda( cudaFree(d_f) );

  } else {

     checkCuda( cudaMemset(d_f, 0, bytes) );

     getResults<<<grid, block>>>(d_f);

     checkCuda( cudaMemcpy(f, d_f, bytes, cudaMemcpyDeviceToHost) );

     for (int it=0; it<mx*my*mz; it++)  
      phi[it] = (double) f[it];

     checkCuda( cudaFree(d_f) );
 
  }
  
  delete []  f;  

}


__global__ void getResults(myprec *d_f) {
 
  int threadsPerBlock  = blockDim.x * blockDim.y;
  int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
  int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

  int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

  d_f[globalThreadNum] = d_phi[globalThreadNum];


  /* when your grid elements < mx*my*mz
  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int k  = blockIdx.y;
  int si = threadIdx.x;

  for (int j = threadIdx.y; j < my; j += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
     d_f[globalIdx] = d_phi[globalIdx]; 
  } */

}

__global__ void initDevice(myprec *d_f) {
  
  int threadsPerBlock  = blockDim.x * blockDim.y;
  int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
  int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y; 

  int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

  d_phi[globalThreadNum] = d_f[globalThreadNum];


  /* when your grid elements < mx*my*mz
  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int k  = blockIdx.y;
  int si = threadIdx.x;

  for (int j = threadIdx.y; j < my; j += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
     d_phi[globalIdx] = d_f[globalIdx]; 
  } */

}



