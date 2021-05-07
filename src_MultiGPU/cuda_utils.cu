/* To run the debugger!!
 * CUDA_VISIBLE_DEVICES="0" cuda-gdb -tui ns
 *
 * To run the profiler on Reynolds!!
 * nvvp -vmargs -Dosgi.locking=none
 *
 *  */

#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"

__constant__ myprec dcoeffF[stencilSize];
__constant__ myprec dcoeffS[stencilSize+1];
__constant__ myprec dcoeffVF[stencilVisc];
__constant__ myprec dcoeffVS[stencilVisc+1];
__constant__ myprec dcoeffSx[mx*(2*stencilSize+1)];
__constant__ myprec dcoeffVSx[mx*(2*stencilVisc+1)];
__constant__ myprec d_dx, d_dy, d_dz, d_d2x, d_d2y, d_d2z, d_x[mx], d_xp[mx], d_dxv[mx];

dim3 d_block[5], grid0;
dim3 d_grid[5], block0;

void copyThreadGridsToDevice();

// host routine to set constant data

void setDerivativeParameters()
{

  // check to make sure dimensions are integral multiples of sPencils
  if ((mx % sPencils != 0) || (my %sPencils != 0) || (mz % sPencils != 0)) {
    printf("'mx', 'my', and 'mz' must be integral multiples of sPencils\n");
    exit(1);
  }

  myprec h_dt = (myprec) dt;
  myprec h_dx = (myprec) 1.0/(dx);
  myprec h_dy = (myprec) 1.0/(y[1] - y[0]);
  myprec h_dz = (myprec) 1.0/(z[1] - z[0]);
  myprec *h_x   = new myprec[mx];
  myprec *h_xp  = new myprec[mx];
  myprec *h_dxv = new myprec[mx];

  myprec h_dpdz = 0.00372;

  checkCuda( cudaMalloc((void**)&dtC ,sizeof(myprec)) );
  checkCuda( cudaMalloc((void**)&dpdz,sizeof(myprec)) );

  checkCuda( cudaMemcpy(dtC  , &h_dt  ,   sizeof(myprec), cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(dpdz , &h_dpdz,   sizeof(myprec), cudaMemcpyHostToDevice) );

  for (int i=0; i<mx; i++) {
	  h_x[i]   = x[i];
	  h_xp[i]  = xp[i];
  }
  h_dxv[0] = (x[1]+x[0])/2.0;
  for (int i=1; i<mx-1; i++) {
	  h_dxv[i]   = (x[i+1]-x[i-1])/2.0;
  }
  h_dxv[mx-1] = Lx - (x[mx-1]+x[mx-2])/2.0;

  myprec h_d2x = h_dx*h_dx;
  myprec h_d2y = h_dy*h_dy;
  myprec h_d2z = h_dz*h_dz;

  myprec *h_coeffF  = new myprec[stencilSize];
  myprec *h_coeffS  = new myprec[stencilSize+1];
  myprec *h_coeffVF = new myprec[stencilSize];
  myprec *h_coeffVS = new myprec[stencilSize+1];

  for (int it=0; it<stencilSize; it++) {
	  h_coeffF[it]  = (myprec) coeffF[it]; }
  for (int it=0; it<stencilVisc; it++) {
	  h_coeffVF[it] = (myprec) coeffVF[it]; }
  for (int it=0; it<stencilSize+1; it++) {
	  h_coeffS[it]  = (myprec) coeffS[it]; }
  for (int it=0; it<stencilVisc+1; it++) {
	  h_coeffVS[it] = (myprec) coeffVS[it]; }

  //constructing the second order derivative coefficients in the x-direction
  myprec *h_coeffVSx = new myprec[mx*(2*stencilVisc+1)];
  for (int it=0; it<stencilVisc; it++)
	  for (int i=0; i<mx; i++) {
		  int idx = i + it*mx;
		  h_coeffVSx[idx] = (coeffVS[it]*(xp[i]*xp[i])*h_d2x - coeffVF[it]*xpp[i]*(xp[i]*xp[i]*xp[i])*h_dx);
	  }
  for (int i=0; i<mx; i++) {
	  int idx = i + stencilVisc*mx;
	  h_coeffVSx[idx] = coeffVS[stencilVisc]*(xp[i]*xp[i])*h_d2x;
  }
  for (int it=stencilVisc+1; it<2*stencilVisc+1; it++)
	  for (int i=0; i<mx; i++) {
		  int idx = i + it*mx;
		  h_coeffVSx[idx] = ( coeffVS[2*stencilVisc - it]*(xp[i]*xp[i])*h_d2x +
				  	  	  	  coeffVF[2*stencilVisc - it]*xpp[i]*(xp[i]*xp[i]*xp[i])*h_dx);
	  }

  myprec *h_coeffSx = new myprec[mx*(2*stencilSize+1)];
  for (int it=0; it<stencilSize; it++)
	  for (int i=0; i<mx; i++) {
		  int idx = i + it*mx;
		  h_coeffSx[idx] = (coeffS[it]*(xp[i]*xp[i])*h_d2x - coeffF[it]*xpp[i]*(xp[i]*xp[i]*xp[i])*h_dx);
	  }
  for (int i=0; i<mx; i++) {
	  int idx = i + stencilSize*mx;
	  h_coeffSx[idx] = coeffS[stencilSize]*(xp[i]*xp[i])*h_d2x;
  }
  for (int it=stencilSize+1; it<2*stencilSize+1; it++)
	  for (int i=0; i<mx; i++) {
		  int idx = i + it*mx;
		  h_coeffSx[idx] = ( coeffVS[2*stencilSize - it]*(xp[i]*xp[i])*h_d2x +
				  	  	  	 coeffVF[2*stencilSize - it]*xpp[i]*(xp[i]*xp[i]*xp[i])*h_dx);
	  }


  checkCuda( cudaMemcpyToSymbol(dcoeffF , h_coeffF ,  stencilSize   *sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(dcoeffS , h_coeffS , (stencilSize+1)*sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(dcoeffVF, h_coeffVF,  stencilVisc   *sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(dcoeffVS, h_coeffVS, (stencilVisc+1)*sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(dcoeffSx , h_coeffSx , mx*(2*stencilSize+1)*sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(dcoeffVSx, h_coeffVSx, mx*(2*stencilVisc+1)*sizeof(myprec), 0, cudaMemcpyHostToDevice) );

  checkCuda( cudaMemcpyToSymbol(d_dx  , &h_dx  ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_dy  , &h_dy  ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_dz  , &h_dz  ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_d2x , &h_d2x ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_d2y , &h_d2y ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_d2z , &h_d2z ,   sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_x   ,  h_x   ,mx*sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_xp  ,  h_xp  ,mx*sizeof(myprec), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(d_dxv ,  h_dxv ,mx*sizeof(myprec), 0, cudaMemcpyHostToDevice) );

  copyThreadGridsToDevice();

  delete [] h_coeffF;
  delete [] h_coeffS;
  delete [] h_coeffVF;
  delete [] h_coeffVS;
  delete [] h_coeffSx;
  delete [] h_coeffVSx;
  delete [] h_x;
  delete [] h_xp;
  delete [] h_dxv;
}

void copyThreadGridsToDevice() {

	  // X-grid
	  d_grid[0]  = dim3(my / sPencils, mz, 1);
	  d_block[0] = dim3(mx, sPencils, 1);

	  // Y-grid (1) for viscous fluxes and (3) for advective fluxes
	  d_grid[3]  = dim3(mx / lPencils, mz, 1);
	  d_block[3] = dim3(lPencils, (my * sPencils) / lPencils, 1);

	  d_grid[1]  = dim3(mx / sPencils, mz, 1);
	  d_block[1] = dim3(my , sPencils, 1); //if not using shared change!!

	  // Z-grid (2) for viscous fluxes and (4) for advective fluxes
	  d_grid[4]  = dim3(mx / lPencils, my, 1);
	  d_block[4] = dim3(lPencils, (mz * sPencils) / lPencils, 1);

	  d_grid[2]  = dim3(mx / sPencils, my, 1);
	  d_block[2] = dim3(mz , sPencils, 1); //if not using shared change!!


	  printf("Grid configuration:\n");
	  printf("Grid 0: {%d, %d, %d} blocks. Blocks 0: {%d, %d, %d} threads.\n",d_grid[0].x, d_grid[0].y, d_grid[0].z, d_block[0].x, d_block[0].y, d_block[0].z);
	  printf("Grid 1: {%d, %d, %d} blocks. Blocks 1: {%d, %d, %d} threads.\n",d_grid[1].x, d_grid[1].y, d_grid[1].z, d_block[1].x, d_block[1].y, d_block[1].z);
	  printf("Grid 2: {%d, %d, %d} blocks. Blocks 2: {%d, %d, %d} threads.\n",d_grid[2].x, d_grid[2].y, d_grid[2].z, d_block[2].x, d_block[2].y, d_block[2].z);
	  printf("Grid 3: {%d, %d, %d} blocks. Blocks 1: {%d, %d, %d} threads.\n",d_grid[3].x, d_grid[3].y, d_grid[3].z, d_block[3].x, d_block[3].y, d_block[3].z);
	  printf("Grid 4: {%d, %d, %d} blocks. Blocks 2: {%d, %d, %d} threads.\n",d_grid[4].x, d_grid[4].y, d_grid[4].z, d_block[4].x, d_block[4].y, d_block[4].z);
	  printf("\n");

	  grid0 = d_grid[0];
	  block0= d_block[0];
}


void copyField(int direction) {

  myprec *fr  = new myprec[mx*my*mz];
  myprec *fu  = new myprec[mx*my*mz];
  myprec *fv  = new myprec[mx*my*mz];
  myprec *fw  = new myprec[mx*my*mz];
  myprec *fe  = new myprec[mx*my*mz];
  int bytes = mx*my*mz * sizeof(myprec);

  if(direction == 0) {

     for (int it=0; it<mx*my*mz; it++)  {
      fr[it] = (myprec) r[it];
      fu[it] = (myprec) u[it];
      fv[it] = (myprec) v[it];
      fw[it] = (myprec) w[it];
      fe[it] = (myprec) e[it];
     }

     // device arrays
     checkCuda( cudaMemcpy(d_r, fr, bytes, cudaMemcpyHostToDevice) );
     checkCuda( cudaMemcpy(d_u, fu, bytes, cudaMemcpyHostToDevice) );
     checkCuda( cudaMemcpy(d_v, fv, bytes, cudaMemcpyHostToDevice) );
     checkCuda( cudaMemcpy(d_w, fw, bytes, cudaMemcpyHostToDevice) );
     checkCuda( cudaMemcpy(d_e, fe, bytes, cudaMemcpyHostToDevice) );

  } else if (direction == 1) {

     checkCuda( cudaMemcpy(fr, d_r, bytes, cudaMemcpyDeviceToHost) );
     checkCuda( cudaMemcpy(fu, d_u, bytes, cudaMemcpyDeviceToHost) );
     checkCuda( cudaMemcpy(fv, d_v, bytes, cudaMemcpyDeviceToHost) );
     checkCuda( cudaMemcpy(fw, d_w, bytes, cudaMemcpyDeviceToHost) );
     checkCuda( cudaMemcpy(fe, d_e, bytes, cudaMemcpyDeviceToHost) );

     for (int it=0; it<mx*my*mz; it++)  {
      r[it]   = (double) fr[it];
      u[it]   = (double) fu[it];
      v[it]   = (double) fv[it];
      w[it]   = (double) fw[it];
      e[it]   = (double) fe[it];
     }
 
  }
  
  delete []  fr;  
  delete []  fu;  
  delete []  fv;  
  delete []  fw;  
  delete []  fe;  

}


void checkGpuMem() {

	float free_m;
	size_t free_t,total_t;

	cudaMemGetInfo(&free_t,&total_t);

	free_m =(uint)free_t/1048576.0 ;

	printf ( "mem free %zu\t (%f MB mem)\n",free_t,free_m);

}

__device__ void threadBlockDeviceSynchronize(void) {
  __syncthreads();
  if(threadIdx.x == 0)
    cudaDeviceSynchronize();
  __syncthreads();
}
