/* To run the debugger!!
 * CUDA_VISIBLE_DEVICES="0" cuda-gdb -tui ns
 *  */

#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"

__constant__ myprec dcoeffF[stencilSize];
__constant__ myprec dcoeffS[stencilSize+1];
__constant__ myprec d_dt, d_dx, d_dy, d_dz;
__constant__ dim3 d_block[3];
__constant__ dim3 d_grid[3];

dim3 hgrid[3],hblock[3];

// host routine to set constant data
void setDerivativeParameters(dim3 &grid, dim3 &block)
{

  // check to make sure dimensions are integral multiples of sPencils
  if ((mx % sPencils != 0) || (my %sPencils != 0) || (mz % sPencils != 0)) {
    printf("'mx', 'my', and 'mz' must be integral multiples of sPencils\n");
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

  dim3 *h_grid, *h_block;
  h_grid  = new dim3[3]; 
  h_block = new dim3[3]; 


  // X-grid spencils
  dim3 gridX  = dim3(my / sPencils, mz, 1);
  dim3 blockX = dim3(mx, sPencils, 1);     

  // Y-grid spencils
  dim3 gridY  = dim3(mx / sPencils, mz, 1);
  dim3 blockY = dim3(sPencils, my, 1);     

  // Z-grid spencils
  dim3 gridZ  = dim3(mx / sPencils, my, 1);
  dim3 blockZ = dim3(sPencils, mz, 1);     

  h_grid[0]  =  gridX; h_grid[1]  =  gridY; h_grid[2]  =  gridZ;
  h_block[0] = blockX; h_block[1] = blockY; h_block[2] =  blockZ;

  for (int it=0; it<3; it++) {
	  hgrid[it]  = h_grid[it];
	  hblock[it] = h_block[it];
  }
  
  checkCuda( cudaMemcpyToSymbol(d_grid  , h_grid  , 3*sizeof(dim3), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_block , h_block , 3*sizeof(dim3), 0, cudaMemcpyHostToDevice) );


  printf("Grid 0: {%d, %d, %d} blocks. Blocks 0: {%d, %d, %d} threads.\n",h_grid[0].x, h_grid[0].y, h_grid[0].z, h_block[0].x, h_block[0].y, h_block[0].z);
  printf("Grid 1: {%d, %d, %d} blocks. Blocks 1: {%d, %d, %d} threads.\n",h_grid[1].x, h_grid[1].y, h_grid[1].z, h_block[1].x, h_block[1].y, h_block[1].z);
  printf("Grid 2: {%d, %d, %d} blocks. Blocks 2: {%d, %d, %d} threads.\n",h_grid[2].x, h_grid[2].y, h_grid[2].z, h_block[2].x, h_block[2].y, h_block[2].z);


  grid  = 1; //h_grid[parentGrid];
  block = 1; //h_block[parentGrid];

  delete [] h_coeffF;
  delete [] h_coeffS;
  delete [] h_grid;
  delete [] h_block;

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

     initDevice<<<hgrid[0], hblock[0]>>>(d_f);

     checkCuda( cudaFree(d_f) );

  } else {

     checkCuda( cudaMemset(d_f, 0, bytes) );

     getResults<<<hgrid[0], hblock[0]>>>(d_f);

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
}

__global__ void initDevice(myprec *d_f) {

	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	d_phi[globalThreadNum] = d_f[globalThreadNum];
}


