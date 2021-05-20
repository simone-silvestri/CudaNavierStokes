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

dim3 d_block[5], grid0, gridBC,  gridHalo,  gridHaloY,  gridHaloZ;
dim3 d_grid[5], block0, blockBC, blockHalo, blockHaloY, blockHaloZ;

void copyThreadGridsToDevice(Communicator rk);
__global__ void fillBCValuesY(myprec *m, myprec *p, myprec *var, int direction);
__global__ void fillBCValuesZ(myprec *m, myprec *p, myprec *var, int direction);
__global__ void fillBCValuesYFive(myprec *m, myprec *p, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, int direction);
__global__ void fillBCValuesZFive(myprec *m, myprec *p, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, int direction);
__global__ void initDevice(myprec *d_fr, myprec *d_fu, myprec *d_fv, myprec *d_fw, myprec *d_fe, myprec *r, myprec *u,  myprec *v,  myprec *w,  myprec *e);
__global__ void getResults(myprec *d_fr, myprec *d_fu, myprec *d_fv, myprec *d_fw, myprec *d_fe, myprec *r, myprec *u,  myprec *v,  myprec *w,  myprec *e);

void setGPUParameters(Communicator rk)
{

	// Setting the appropriate GPU (using number of cores per node = number of GPUs per node)
	cudaSetDevice(rk.nodeRank);

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

	myprec h_dpdz;
	if(forcing)  h_dpdz = 0.00372;
	if(!forcing) h_dpdz = 0.0;

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

	copyThreadGridsToDevice(rk);

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

void copyThreadGridsToDevice(Communicator rk) {

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

	  grid0 = d_grid[0];
	  block0= d_block[0];

	  gridBC  = dim3((my+2*stencilSize) / sPencils, (mz+2*stencilSize), 1);
	  blockBC = dim3(mx, sPencils, 1);

	  gridHalo  = dim3((my+mz) / sPencils, 2*stencilSize, 1);
	  blockHalo = dim3(mx, sPencils, 1);

	  gridHaloY  = dim3(mx / sPencils, mz, 1);
	  blockHaloY = dim3(stencilSize, sPencils, 1);

	  gridHaloZ  = dim3(mx / sPencils, my, 1);
	  blockHaloZ = dim3(stencilSize, sPencils, 1);

	  if(rk.rank==0) {
		  printf("Grid configuration:\n");
		  printf("Grid 0:  {%d, %d, %d} blocks. Blocks 0:  {%d, %d, %d} threads.\n",d_grid[0].x, d_grid[0].y, d_grid[0].z, d_block[0].x, d_block[0].y, d_block[0].z);
		  printf("Grid 1:  {%d, %d, %d} blocks. Blocks 1:  {%d, %d, %d} threads.\n",d_grid[1].x, d_grid[1].y, d_grid[1].z, d_block[1].x, d_block[1].y, d_block[1].z);
		  printf("Grid 2:  {%d, %d, %d} blocks. Blocks 2:  {%d, %d, %d} threads.\n",d_grid[2].x, d_grid[2].y, d_grid[2].z, d_block[2].x, d_block[2].y, d_block[2].z);
		  printf("Grid 3:  {%d, %d, %d} blocks. Blocks 3:  {%d, %d, %d} threads.\n",d_grid[3].x, d_grid[3].y, d_grid[3].z, d_block[3].x, d_block[3].y, d_block[3].z);
		  printf("Grid 4:  {%d, %d, %d} blocks. Blocks 4:  {%d, %d, %d} threads.\n",d_grid[4].x, d_grid[4].y, d_grid[4].z, d_block[4].x, d_block[4].y, d_block[4].z);
		  printf("Grid BC: {%d, %d, %d} blocks. Blocks BC: {%d, %d, %d} threads.\n",gridBC.x, gridBC.y, gridBC.z, blockBC.x, blockBC.y, blockBC.z);
		  printf("Grid H:  {%d, %d, %d} blocks. Blocks H:  {%d, %d, %d} threads.\n",gridHalo.x, gridHalo.y, gridHalo.z, blockHalo.x, blockHalo.y, blockHalo.z);
		  printf("Grid HY: {%d, %d, %d} blocks. Blocks HY: {%d, %d, %d} threads.\n",gridHaloY.x, gridHaloY.y, gridHaloY.z, blockHaloY.x, blockHaloY.y, blockHaloY.z);
		  printf("Grid HZ: {%d, %d, %d} blocks. Blocks HZ: {%d, %d, %d} threads.\n",gridHaloZ.x, gridHaloZ.y, gridHaloZ.z, blockHaloZ.x, blockHaloZ.y, blockHaloZ.z);
		  printf("\n");
	  }
}

void copyField(int direction) {

  myprec *fr  = new myprec[mx*my*mz];
  myprec *fu  = new myprec[mx*my*mz];
  myprec *fv  = new myprec[mx*my*mz];
  myprec *fw  = new myprec[mx*my*mz];
  myprec *fe  = new myprec[mx*my*mz];
  myprec *d_fr, *d_fu, *d_fv, *d_fw, *d_fe;
  int bytes = mx*my*mz * sizeof(myprec);
  checkCuda( cudaMalloc((void**)&d_fr, bytes) );
  checkCuda( cudaMalloc((void**)&d_fu, bytes) );
  checkCuda( cudaMalloc((void**)&d_fv, bytes) );
  checkCuda( cudaMalloc((void**)&d_fw, bytes) );
  checkCuda( cudaMalloc((void**)&d_fe, bytes) );

  if(direction == 0) {
     for (int it=0; it<mx*my*mz; it++)  {
      fr[it] = (myprec) r[it];
      fu[it] = (myprec) u[it];
      fv[it] = (myprec) v[it];
      fw[it] = (myprec) w[it];
      fe[it] = (myprec) e[it];
     }
     // device arrays
     checkCuda( cudaMemcpy(d_fr, fr, bytes, cudaMemcpyHostToDevice) );
     checkCuda( cudaMemcpy(d_fu, fu, bytes, cudaMemcpyHostToDevice) );
     checkCuda( cudaMemcpy(d_fv, fv, bytes, cudaMemcpyHostToDevice) );
     checkCuda( cudaMemcpy(d_fw, fw, bytes, cudaMemcpyHostToDevice) );
     checkCuda( cudaMemcpy(d_fe, fe, bytes, cudaMemcpyHostToDevice) );

     initDevice<<<grid0, block0>>>(d_fr,d_fu,d_fv,d_fw,d_fe,d_r,d_u,d_v,d_w,d_e);
 	 calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l,0);

  } else if (direction == 1) {
     checkCuda( cudaMemset(d_fr, 0, bytes) );
     checkCuda( cudaMemset(d_fu, 0, bytes) );
     checkCuda( cudaMemset(d_fv, 0, bytes) );
     checkCuda( cudaMemset(d_fw, 0, bytes) );
     checkCuda( cudaMemset(d_fe, 0, bytes) );

     getResults<<<grid0, block0>>>(d_fr,d_fu,d_fv,d_fw,d_fe,d_r,d_u,d_v,d_w,d_e);

     checkCuda( cudaMemcpy(fr, d_fr, bytes, cudaMemcpyDeviceToHost) );
     checkCuda( cudaMemcpy(fu, d_fu, bytes, cudaMemcpyDeviceToHost) );
     checkCuda( cudaMemcpy(fv, d_fv, bytes, cudaMemcpyDeviceToHost) );
     checkCuda( cudaMemcpy(fw, d_fw, bytes, cudaMemcpyDeviceToHost) );
     checkCuda( cudaMemcpy(fe, d_fe, bytes, cudaMemcpyDeviceToHost) );

     for (int it=0; it<mx*my*mz; it++)  {
      r[it]   = (double) fr[it];
      u[it]   = (double) fu[it];
      v[it]   = (double) fv[it];
      w[it]   = (double) fw[it];
      e[it]   = (double) fe[it];
     }

  }

  checkCuda( cudaFree(d_fr) );
  checkCuda( cudaFree(d_fu) );
  checkCuda( cudaFree(d_fv) );
  checkCuda( cudaFree(d_fw) );
  checkCuda( cudaFree(d_fe) );
  delete []  fr;
  delete []  fu;
  delete []  fv;
  delete []  fw;
  delete []  fe;

}

__global__ void initDevice(myprec *d_fr, myprec *d_fu, myprec *d_fv, myprec *d_fw, myprec *d_fe, myprec *r, myprec *u,  myprec *v,  myprec *w,  myprec *e) {

	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	r[globalThreadNum]   = d_fr[globalThreadNum];
	u[globalThreadNum]   = d_fu[globalThreadNum];
	v[globalThreadNum]   = d_fv[globalThreadNum];
	w[globalThreadNum]   = d_fw[globalThreadNum];
	e[globalThreadNum]   = d_fe[globalThreadNum];
}

__global__ void getResults(myprec *d_fr, myprec *d_fu, myprec *d_fv, myprec *d_fw, myprec *d_fe, myprec *r, myprec *u,  myprec *v,  myprec *w,  myprec *e) {

	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	d_fr[globalThreadNum] = r[globalThreadNum];
	d_fu[globalThreadNum] = u[globalThreadNum];
	d_fv[globalThreadNum] = v[globalThreadNum];
	d_fw[globalThreadNum] = w[globalThreadNum];
	d_fe[globalThreadNum] = e[globalThreadNum];
}

void initSolver() {

    for (int i=0; i<fin; i++) {
    	checkCuda( cudaMalloc((void**)&d_rhsr1[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsu1[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsv1[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsw1[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhse1[i],mx*my*mz*sizeof(myprec)) );

    	checkCuda( cudaMalloc((void**)&d_rhsr2[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsu2[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsv2[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsw2[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhse2[i],mx*my*mz*sizeof(myprec)) );

    	checkCuda( cudaMalloc((void**)&d_rhsr3[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsu3[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsv3[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsw3[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhse3[i],mx*my*mz*sizeof(myprec)) );
    }

    //Boundary condition pointer to pass from GPU to CPU
	checkCuda( cudaMalloc((void**)&djm5, 5*mz*mx*stencilSize*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&djp5, 5*mz*mx*stencilSize*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&dkm5, 5*my*mx*stencilSize*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&dkp5, 5*my*mx*stencilSize*sizeof(myprec)) );

	checkCuda( cudaMalloc((void**)&djm, mz*mx*stencilSize*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&djp, mz*mx*stencilSize*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&dkm, my*mx*stencilSize*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&dkp, my*mx*stencilSize*sizeof(myprec)) );

    //fill variables with domain size plus boundary location
    int bytes = (mx*(my+2*stencilSize)*(mz+2*stencilSize))*sizeof(myprec);

    //Remember!!! -> the boundary conditions will be located at the end of the variable in this fashion:
    // from (mx*my*mz)                                           to (mx*my*mz +   stencilSize*mx*mz - 1) -> boundary Ym
    // from (mx*my*mz +   stencilSize*mx*mz)                     to (mx*my*mz + 2*stencilSize*mx*mz - 1) -> boundary Yp
    // from (mx*my*mz + 2*stencilSize*mx*mz)                     to (mx*my*mz + 2*stencilSize*mx*mz +   stencilSize*mx*my - 1) -> boundary Zm
    // from (mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my) to (mx*my*mz + 2*stencilSize*mx*mz + 2*stencilSize*mx*my - 1) -> boundary Zp

	checkCuda( cudaMalloc((void**)&d_r,  bytes) );
	checkCuda( cudaMalloc((void**)&d_u,  bytes) );
	checkCuda( cudaMalloc((void**)&d_v,  bytes) );
	checkCuda( cudaMalloc((void**)&d_w,  bytes) );
	checkCuda( cudaMalloc((void**)&d_e,  bytes) );

	checkCuda( cudaMalloc((void**)&d_h,  bytes) );
	checkCuda( cudaMalloc((void**)&d_t,  bytes) );
	checkCuda( cudaMalloc((void**)&d_p,  bytes) );
	checkCuda( cudaMalloc((void**)&d_m,  bytes) );
	checkCuda( cudaMalloc((void**)&d_l,  bytes) );

	checkCuda( cudaMalloc((void**)&d_rO, bytes) );
	checkCuda( cudaMalloc((void**)&d_eO, bytes) );
	checkCuda( cudaMalloc((void**)&d_uO, bytes) );
	checkCuda( cudaMalloc((void**)&d_vO, bytes) );
	checkCuda( cudaMalloc((void**)&d_wO, bytes) );

	checkCuda( cudaMalloc((void**)&d_dil, bytes) );

	int bytesY = mz*mx*stencilSize*sizeof(myprec);
	int bytesZ = my*mx*stencilSize*sizeof(myprec);
	checkCuda( cudaMallocHost(&senYm, bytesY) );
	checkCuda( cudaMallocHost(&senYp, bytesY) );
	checkCuda( cudaMallocHost(&rcvYm, bytesY) );
	checkCuda( cudaMallocHost(&rcvYp, bytesY) );
	checkCuda( cudaMallocHost(&senZm, bytesZ) );
	checkCuda( cudaMallocHost(&senZp, bytesZ) );
	checkCuda( cudaMallocHost(&rcvZm, bytesZ) );
	checkCuda( cudaMallocHost(&rcvZp, bytesZ) );

	checkCuda( cudaMallocHost(&senYm5, 5*bytesY) );
	checkCuda( cudaMallocHost(&senYp5, 5*bytesY) );
	checkCuda( cudaMallocHost(&rcvYm5, 5*bytesY) );
	checkCuda( cudaMallocHost(&rcvYp5, 5*bytesY) );
	checkCuda( cudaMallocHost(&senZm5, 5*bytesZ) );
	checkCuda( cudaMallocHost(&senZp5, 5*bytesZ) );
	checkCuda( cudaMallocHost(&rcvZm5, 5*bytesZ) );
	checkCuda( cudaMallocHost(&rcvZp5, 5*bytesZ) );

    for (int i=0; i<16; i++) {
    	checkCuda( cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking) );
    }


}

void clearSolver() {

	for(int i=0; i<fin; i++) {
		checkCuda( cudaFree(d_rhsr1[i]) );
		checkCuda( cudaFree(d_rhsu1[i]) );
		checkCuda( cudaFree(d_rhsv1[i]) );
		checkCuda( cudaFree(d_rhsw1[i]) );
		checkCuda( cudaFree(d_rhse1[i]) );

		checkCuda( cudaFree(d_rhsr2[i]) );
		checkCuda( cudaFree(d_rhsu2[i]) );
		checkCuda( cudaFree(d_rhsv2[i]) );
		checkCuda( cudaFree(d_rhsw2[i]) );
		checkCuda( cudaFree(d_rhse2[i]) );

		checkCuda( cudaFree(d_rhsr3[i]) );
		checkCuda( cudaFree(d_rhsu3[i]) );
		checkCuda( cudaFree(d_rhsv3[i]) );
		checkCuda( cudaFree(d_rhsw3[i]) );
		checkCuda( cudaFree(d_rhse3[i]) );
	}

	checkCuda( cudaFree(djm) );
	checkCuda( cudaFree(djp) );
	checkCuda( cudaFree(dkm) );
	checkCuda( cudaFree(dkp) );

	checkCuda( cudaFree(djm5) );
	checkCuda( cudaFree(djp5) );
	checkCuda( cudaFree(dkm5) );
	checkCuda( cudaFree(dkp5) );


	checkCuda( cudaFree(d_h) );
	checkCuda( cudaFree(d_t) );
	checkCuda( cudaFree(d_p) );
	checkCuda( cudaFree(d_m) );
	checkCuda( cudaFree(d_l) );

	checkCuda( cudaFree(d_rO) );
	checkCuda( cudaFree(d_eO) );
	checkCuda( cudaFree(d_uO) );
	checkCuda( cudaFree(d_vO) );
	checkCuda( cudaFree(d_wO) );

	checkCuda( cudaFree(d_dil) );

	checkCuda( cudaFreeHost(senYm)  );
	checkCuda( cudaFreeHost(senYp)  );
	checkCuda( cudaFreeHost(senZm)  );
	checkCuda( cudaFreeHost(senZp)  );
	checkCuda( cudaFreeHost(rcvYm)  );
	checkCuda( cudaFreeHost(rcvYp)  );
	checkCuda( cudaFreeHost(rcvZm)  );
	checkCuda( cudaFreeHost(rcvZp)  );
	checkCuda( cudaFreeHost(senYm5) );
	checkCuda( cudaFreeHost(senYp5) );
	checkCuda( cudaFreeHost(senZm5) );
	checkCuda( cudaFreeHost(senZp5) );
	checkCuda( cudaFreeHost(rcvYm5) );
	checkCuda( cudaFreeHost(rcvYp5) );
	checkCuda( cudaFreeHost(rcvZm5) );
	checkCuda( cudaFreeHost(rcvZp5) );

	for (int i=0; i<16; i++) {
		checkCuda( cudaStreamDestroy(s[i]) );
	}
}

void fillBoundaries(myprec *jm, myprec *jp, myprec *km, myprec *kp, myprec *var, int direction) {

	dim3 gridY = dim3(mz,1,1);
	dim3 gridZ = dim3(my,1,1);
	dim3 block = dim3(mx,stencilSize,1);

	if(direction == 0) {
		fillBCValuesY<<<gridY,block,0,s[10]>>>(djm,djp,var,0);
		fillBCValuesZ<<<gridZ,block,0,s[11]>>>(dkm,dkp,var,0);
		cudaStreamSynchronize(s[10]);
		cudaStreamSynchronize(s[11]);
		checkCuda( cudaMemcpyAsync(jm,djm,mz*mx*stencilSize*sizeof(myprec),cudaMemcpyDeviceToHost,s[12]) );
		checkCuda( cudaMemcpyAsync(jp,djp,mz*mx*stencilSize*sizeof(myprec),cudaMemcpyDeviceToHost,s[13]) );
		checkCuda( cudaMemcpyAsync(km,dkm,my*mx*stencilSize*sizeof(myprec),cudaMemcpyDeviceToHost,s[14]) );
		checkCuda( cudaMemcpyAsync(kp,dkp,my*mx*stencilSize*sizeof(myprec),cudaMemcpyDeviceToHost,s[15]) );
		cudaStreamSynchronize(s[12]);
		cudaStreamSynchronize(s[13]);
		cudaStreamSynchronize(s[14]);
		cudaStreamSynchronize(s[15]);
	} else {
		checkCuda( cudaMemcpyAsync(djm,jm,mz*mx*stencilSize*sizeof(myprec),cudaMemcpyHostToDevice,s[12]) );
		checkCuda( cudaMemcpyAsync(djp,jp,mz*mx*stencilSize*sizeof(myprec),cudaMemcpyHostToDevice,s[13]) );
		checkCuda( cudaMemcpyAsync(dkm,km,my*mx*stencilSize*sizeof(myprec),cudaMemcpyHostToDevice,s[14]) );
		checkCuda( cudaMemcpyAsync(dkp,kp,my*mx*stencilSize*sizeof(myprec),cudaMemcpyHostToDevice,s[15]) );
		cudaStreamSynchronize(s[12]);
		cudaStreamSynchronize(s[13]);
		cudaStreamSynchronize(s[14]);
		cudaStreamSynchronize(s[15]);
		fillBCValuesY<<<gridY,block,0,s[10]>>>(djm,djp,var,1);
		fillBCValuesZ<<<gridZ,block,0,s[11]>>>(dkm,dkp,var,1);
		cudaStreamSynchronize(s[10]);
		cudaStreamSynchronize(s[11]);
	}
}

__global__ void fillBCValuesY(myprec *m, myprec *p, myprec *var, int direction) {
	if(direction == 0) {
		int k  = blockIdx.x;
		int it = threadIdx.y;
		int i  = threadIdx.x;
		p[it + i*stencilSize + k*mx*stencilSize] = var[i + it*mx + k*mx*my];
		m[it + i*stencilSize + k*mx*stencilSize] = var[i + (my - stencilSize + it)*mx + k*mx*my];
		__syncthreads();
	} else {
		int k  = blockIdx.x;
		int it = threadIdx.y;
		int i  = threadIdx.x;
		var[mx*my*mz                     + it + i*stencilSize + k*mx*stencilSize] = m[it + i*stencilSize + k*mx*stencilSize];
		var[mx*my*mz + stencilSize*mx*mz + it + i*stencilSize + k*mx*stencilSize] = p[it + i*stencilSize + k*mx*stencilSize];
		__syncthreads();
	}
}

__global__ void fillBCValuesZ(myprec *m, myprec *p, myprec *var, int direction) {
	if(direction == 0) {
		int j  = blockIdx.x;
		int it = threadIdx.y;
		int i  = threadIdx.x;
		p[it + i*stencilSize + j*mx*stencilSize] = var[i + j*mx + it*mx*my];
		m[it + i*stencilSize + j*mx*stencilSize] = var[i + j*mx + (mz - stencilSize + it)*mx*my];
		__syncthreads();
	} else {
		int j  = blockIdx.x;
		int it = threadIdx.y;
		int i  = threadIdx.x;
		var[mx*my*mz + 2*stencilSize*mx*mz                     + it + i*stencilSize + j*mx*stencilSize] = m[it + i*stencilSize + j*mx*stencilSize];
		var[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + it + i*stencilSize + j*mx*stencilSize] = p[it + i*stencilSize + j*mx*stencilSize];

		__syncthreads();
	}
}

void fillBoundariesFive(myprec *jm, myprec *jp, myprec *km, myprec *kp, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, int direction) {

	dim3 gridY = dim3(mz,1,1);
	dim3 gridZ = dim3(my,1,1);
	dim3 block = dim3(mx,stencilSize,1);

	if(direction == 0) {
		fillBCValuesYFive<<<gridY,block,0,s[10]>>>(djm5,djp5,r,u,v,w,e,0);
		fillBCValuesZFive<<<gridZ,block,0,s[11]>>>(dkm5,dkp5,r,u,v,w,e,0);
		cudaStreamSynchronize(s[10]);
		cudaStreamSynchronize(s[11]);
		checkCuda( cudaMemcpyAsync(jm,djm5,5*mz*mx*stencilSize*sizeof(myprec),cudaMemcpyDeviceToHost,s[12]) );
		checkCuda( cudaMemcpyAsync(jp,djp5,5*mz*mx*stencilSize*sizeof(myprec),cudaMemcpyDeviceToHost,s[13]) );
		checkCuda( cudaMemcpyAsync(km,dkm5,5*my*mx*stencilSize*sizeof(myprec),cudaMemcpyDeviceToHost,s[14]) );
		checkCuda( cudaMemcpyAsync(kp,dkp5,5*my*mx*stencilSize*sizeof(myprec),cudaMemcpyDeviceToHost,s[15]) );
		cudaStreamSynchronize(s[12]);
		cudaStreamSynchronize(s[13]);
		cudaStreamSynchronize(s[14]);
		cudaStreamSynchronize(s[15]);
	} else {
		checkCuda( cudaMemcpyAsync(djm5,jm,5*mz*mx*stencilSize*sizeof(myprec),cudaMemcpyHostToDevice,s[12]) );
		checkCuda( cudaMemcpyAsync(djp5,jp,5*mz*mx*stencilSize*sizeof(myprec),cudaMemcpyHostToDevice,s[13]) );
		checkCuda( cudaMemcpyAsync(dkm5,km,5*my*mx*stencilSize*sizeof(myprec),cudaMemcpyHostToDevice,s[14]) );
		checkCuda( cudaMemcpyAsync(dkp5,kp,5*my*mx*stencilSize*sizeof(myprec),cudaMemcpyHostToDevice,s[15]) );
		cudaStreamSynchronize(s[12]);
		cudaStreamSynchronize(s[13]);
		cudaStreamSynchronize(s[14]);
		cudaStreamSynchronize(s[15]);
		fillBCValuesYFive<<<gridY,block,0,s[10]>>>(djm5,djp5,r,u,v,w,e,1);
		fillBCValuesZFive<<<gridZ,block,0,s[11]>>>(dkm5,dkp5,r,u,v,w,e,1);
		cudaStreamSynchronize(s[10]);
		cudaStreamSynchronize(s[11]);
	}
}

__global__ void fillBCValuesXChannel(myprec *r, myprec *u, myprec *v, myprec *w, myprec *h, myprec *p, myprec *t, myprec *m, myprec *l) {
		int k   = blockIdx.x;
		int j   = threadIdx.y;
		int it  = threadIdx.x;
		int add = mx*my*mz + 2*stencilSize*mx*(mz+my);
		int gl  = it + j*stencilSize + k*my*stencilSize;
		r[add                     + gl] = r[2*stencilSize-it-1];
		r[add + stencilSize*my*mz + gl] = r[2*stencilSize-it-1];
		u[add                     + gl] = u[2*stencilSize-it-1];
		u[add + stencilSize*my*mz + gl] = u[2*stencilSize-it-1];
		v[add                     + gl] = v[2*stencilSize-it-1];
		v[add + stencilSize*my*mz + gl] = v[2*stencilSize-it-1];
		w[add                     + gl] = w[2*stencilSize-it-1];
		w[add + stencilSize*my*mz + gl] = w[2*stencilSize-it-1];
		p[add                     + gl] = p[2*stencilSize-it-1];
		p[add + stencilSize*my*mz + gl] = p[2*stencilSize-it-1];
		t[add                     + gl] = t[2*stencilSize-it-1];
		t[add + stencilSize*my*mz + gl] = t[2*stencilSize-it-1];
		h[add                     + gl] = h[2*stencilSize-it-1];
		h[add + stencilSize*my*mz + gl] = h[2*stencilSize-it-1];
		m[add                     + gl] = m[2*stencilSize-it-1];
		m[add + stencilSize*my*mz + gl] = m[2*stencilSize-it-1];
		l[add                     + gl] = l[2*stencilSize-it-1];
		l[add + stencilSize*my*mz + gl] = l[2*stencilSize-it-1];
		__syncthreads();
}

__global__ void fillBCValuesYFive(myprec *m, myprec *p, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, int direction) {
	if(direction == 0) {
		int k  = blockIdx.x;
		int it = threadIdx.y;
		int i  = threadIdx.x;
		int gl = it + i*stencilSize + k*mx*stencilSize;
		m[gl]                       = r[i + (my - stencilSize + it)*mx + k*mx*my];
		p[gl]                       = r[i + it*mx + k*mx*my];
		m[gl +   stencilSize*mx*mz] = u[i + (my - stencilSize + it)*mx + k*mx*my];
		p[gl +   stencilSize*mx*mz] = u[i + it*mx + k*mx*my];
		m[gl + 2*stencilSize*mx*mz] = v[i + (my - stencilSize + it)*mx + k*mx*my];
		p[gl + 2*stencilSize*mx*mz] = v[i + it*mx + k*mx*my];
		m[gl + 3*stencilSize*mx*mz] = w[i + (my - stencilSize + it)*mx + k*mx*my];
		p[gl + 3*stencilSize*mx*mz] = w[i + it*mx + k*mx*my];
		m[gl + 4*stencilSize*mx*mz] = e[i + (my - stencilSize + it)*mx + k*mx*my];
		p[gl + 4*stencilSize*mx*mz] = e[i + it*mx + k*mx*my];
		__syncthreads();
	} else {
		int k  = blockIdx.x;
		int it = threadIdx.y;
		int i  = threadIdx.x;
		int gl = it + i*stencilSize + k*mx*stencilSize;
		r[mx*my*mz                     + gl] = m[gl];
		r[mx*my*mz + stencilSize*mx*mz + gl] = p[gl];
		u[mx*my*mz                     + gl] = m[gl +   stencilSize*mx*mz];
		u[mx*my*mz + stencilSize*mx*mz + gl] = p[gl +   stencilSize*mx*mz];
		v[mx*my*mz                     + gl] = m[gl + 2*stencilSize*mx*mz];
		v[mx*my*mz + stencilSize*mx*mz + gl] = p[gl + 2*stencilSize*mx*mz];
		w[mx*my*mz                     + gl] = m[gl + 3*stencilSize*mx*mz];
		w[mx*my*mz + stencilSize*mx*mz + gl] = p[gl + 3*stencilSize*mx*mz];
		e[mx*my*mz                     + gl] = m[gl + 4*stencilSize*mx*mz];
		e[mx*my*mz + stencilSize*mx*mz + gl] = p[gl + 4*stencilSize*mx*mz];
		__syncthreads();
	}
}

__global__ void fillBCValuesZFive(myprec *m, myprec *p, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, int direction) {
	if(direction == 0) {
		int j  = blockIdx.x;
		int it = threadIdx.y;
		int i  = threadIdx.x;
		int gl = it + i*stencilSize + j*mx*stencilSize;
		m[gl]                       = r[i + j*mx + (mz - stencilSize + it)*mx*my];
		p[gl]                       = r[i + j*mx + it*mx*my];
		m[gl +   stencilSize*mx*my] = u[i + j*mx + (mz - stencilSize + it)*mx*my];
		p[gl +   stencilSize*mx*my] = u[i + j*mx + it*mx*my];
		m[gl + 2*stencilSize*mx*my] = v[i + j*mx + (mz - stencilSize + it)*mx*my];
		p[gl + 2*stencilSize*mx*my] = v[i + j*mx + it*mx*my];
		m[gl + 3*stencilSize*mx*my] = w[i + j*mx + (mz - stencilSize + it)*mx*my];
		p[gl + 3*stencilSize*mx*my] = w[i + j*mx + it*mx*my];
		m[gl + 4*stencilSize*mx*my] = e[i + j*mx + (mz - stencilSize + it)*mx*my];
		p[gl + 4*stencilSize*mx*my] = e[i + j*mx + it*mx*my];
		__syncthreads();
	} else {
		int j  = blockIdx.x;
		int it = threadIdx.y;
		int i  = threadIdx.x;
		int gl = it + i*stencilSize + j*mx*stencilSize;
		r[mx*my*mz + 2*stencilSize*mx*mz                     + gl] = m[gl];
		r[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + gl] = p[gl];
		u[mx*my*mz + 2*stencilSize*mx*mz                     + gl] = m[gl +   stencilSize*mx*my];
		u[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + gl] = p[gl +   stencilSize*mx*my];
		v[mx*my*mz + 2*stencilSize*mx*mz                     + gl] = m[gl + 2*stencilSize*mx*my];
		v[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + gl] = p[gl + 2*stencilSize*mx*my];
		w[mx*my*mz + 2*stencilSize*mx*mz                     + gl] = m[gl + 3*stencilSize*mx*my];
		w[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + gl] = p[gl + 3*stencilSize*mx*my];
		e[mx*my*mz + 2*stencilSize*mx*mz                     + gl] = m[gl + 4*stencilSize*mx*my];
		e[mx*my*mz + 2*stencilSize*mx*mz + stencilSize*mx*my + gl] = p[gl + 4*stencilSize*mx*my];
		__syncthreads();
	}
}

void checkGpuMem() {

	float free_m;
	size_t free_t,total_t;

	cudaMemGetInfo(&free_t,&total_t);

	free_m =(uint)free_t/1048576.0 ;

	printf ( "mem free %zu\t (%f MB mem)\n",free_t,free_m);

}

