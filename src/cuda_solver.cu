#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"


__device__ myprec d_phi[mx*my*mz];

__device__ myprec d_rhs1[mx*my*mz];
__device__ myprec d_rhs2[mx*my*mz];
__device__ myprec d_rhs3[mx*my*mz];
__device__ myprec d_rhs4[mx*my*mz];
__device__ myprec d_temp[mx*my*mz];
__device__ myprec d_tmp[mx*my*mz];

__device__ void derDev2x(myprec *d2f, myprec *f, Indices id)
{  

  int si = id.i + stencilSize;       // local i for shared memory access + halo offset
  int sj = id.tiy;                   // local j for shared memory access

  __shared__ myprec s_f[sPencils][mx+stencilSize*2]; // 4-wide halo

  myprec d2x = 1.0/d_dx/d_dx;

  s_f[sj][si] = f[id.g];

  __syncthreads();

  // fill in periodic images in shared memory array 
  if (id.i < stencilSize) {
    s_f[sj][si-stencilSize]  = s_f[sj][si+mx-stencilSize]; // CHANGED SIMONE: s_f[sj][si+mx-stencilSize-1];
    s_f[sj][si+mx]           = s_f[sj][si];                // CHANGED SIMONE: s_f[sj][si+1];
  }

  __syncthreads();

  myprec dftemp = dcoeffS[stencilSize]*s_f[sj][si]*d2x;
  for (int it=0; it<stencilSize; it++)  { 
   dftemp += dcoeffS[it]*(s_f[sj][si+it-stencilSize]-s_f[sj][si+stencilSize-it])*d2x;
  }

  __syncthreads();
 
  d2f[id.g] = dftemp; 
}


__device__ void derDev1x(myprec *df, myprec *f, Indices id)
{  

  int si = id.i + stencilSize;       // local i for shared memory access + halo offset
  int sj = id.tiy;                   // local j for shared memory access

  __shared__ myprec s_f[sPencils][mx+stencilSize*2]; // 4-wide halo

  s_f[sj][si] = f[id.g];

  __syncthreads();

  // fill in periodic images in shared memory array 
  if (id.i < stencilSize) {
    s_f[sj][si-stencilSize]  = s_f[sj][si+mx-stencilSize]; // CHANGED SIMONE: s_f[sj][si+mx-stencilSize-1];
    s_f[sj][si+mx]           = s_f[sj][si];                // CHANGED SIMONE: s_f[sj][si+1];
  }

  __syncthreads();

  myprec dftemp = 0.0;
  for (int it=0; it<stencilSize; it++)  { 
   dftemp += dcoeffF[it]*(s_f[sj][si+it-stencilSize]-s_f[sj][si+stencilSize-it])/d_dx;
  }

  __syncthreads();
 
  df[id.g] = dftemp; 
}

__device__ void derDev1y(myprec *df, myprec *f, Indices id)
{
  __shared__ myprec s_f[my+stencilSize*2][sPencils];

  int si = id.tix;
  int sj = id.j + stencilSize;

  s_f[sj][si] = f[id.g];

  __syncthreads();

  if (id.j < stencilSize) {
    s_f[sj-stencilSize][si]  = s_f[sj+my-stencilSize][si];
    s_f[sj+my][si]           = s_f[sj][si];
  }

  __syncthreads();

  myprec dftemp = 0.0;
  for (int jt=0; jt<stencilSize; jt++)  { 
   dftemp += dcoeffF[jt]*(s_f[sj+jt-stencilSize][si]-s_f[sj+stencilSize-jt][si])/d_dy;
  }

  __syncthreads();
 
  df[id.g] = dftemp;
}

__device__ void derDev1z(myprec *df, myprec *f, Indices id)
{
  __shared__ myprec s_f[mz+stencilSize*2][sPencils];

  int si = id.tix;
  int sk = id.k + stencilSize;

  s_f[sk][si] = f[id.g];

  __syncthreads();

  if (id.k < stencilSize) {
    s_f[sk-stencilSize][si]  = s_f[sk+mz-stencilSize][si];
    s_f[sk+mz][si]           = s_f[sk][si];
  }

  __syncthreads();

  myprec dftemp = 0.0;
  for (int kt=0; kt<stencilSize; kt++)  { 
   dftemp += dcoeffF[kt]*(s_f[sk+kt-stencilSize][si]-s_f[sk+stencilSize-kt][si])/d_dz;
  }

  __syncthreads();
 
  df[id.g] = dftemp;
}



__device__ void RHSDevice(myprec *rhs, myprec *var, Indices id) {
  
  derDev1z(d_tmp,var,id);
  rhs[id.g] = -d_tmp[id.g]*U;

  /* if using a large pencil discretization of the cuda block grid
  for (int j = id.tiy; j < my; j += id.bdy) {
    int gblIdx = id.k*mx*my + j*mx + id.i;
    rhs[gblIdx] = -d_tmp[gblIdx]*U; 
  } */
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


  /* if using a large pencil discretization of the cuda block grid
  for (int j = id.tiy; j < my; j += id.bdy) {
    int gblIdx = id.k*mx*my + j*mx + id.i;
    d_temp[gblIdx] = d_phi[gblIdx] + d_rhs1[gblIdx]*d_dt/2;
  }
  for (int j = id.tiy; j < my; j += id.bdy) {
    int gblIdx = id.k*mx*my + j*mx + id.i;
    d_temp[gblIdx] = d_phi[gblIdx] + d_rhs2[gblIdx]*d_dt/2;
  }
  for (int j = id.tiy; j < my; j += id.bdy) {
    int gblIdx = id.k*mx*my + j*mx + id.i;
    d_temp[gblIdx] = d_phi[gblIdx] + d_rhs3[gblIdx]*d_dt;
  }
  for (int j = id.tiy; j < my; j += id.bdy) {
    int gblIdx = id.k*mx*my + j*mx + id.i;
    d_phi[gblIdx] = d_phi[gblIdx] + d_dt*(d_rhs1[gblIdx] + 
					2*d_rhs2[gblIdx] + 
					2*d_rhs3[gblIdx] + 
					  d_rhs4[gblIdx]);
  } */

}


__global__ void runDevice() {

  Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);


  id.mkidZ();

  for (int istep=0; istep < nsteps; istep++) {
          rk4Device(id);
  }
}



/* Large pencil functions not used at the moment


__device__ void derDev1yL(myprec *df, myprec *f, Indices id)
{
  __shared__ myprec s_f[my+stencilSize*2][lPencils];

  int i  = id.bix*id.bdx + id.tix; 
  int k  = id.biy; 
  int si = id.tix;

  for (int j = id.tiy; j < my; j += id.bdy) {
    int gblIdx = k*mx*my + j*mx + i;
    int sj = j + stencilSize;
    s_f[sj][si] = f[gblIdx]; 
  }

  __syncthreads();
  
  int sj = id.tiy + stencilSize;

  if (sj < stencilSize*2) {
    s_f[sj-stencilSize][si]  = s_f[sj+my-stencilSize][si];
    s_f[sj+my][si]           = s_f[sj][si];
  }

  __syncthreads();

  for (int j = id.tiy; j < my; j += id.bdy) {
    int gblIdx = k*mx*my + j*mx + i;
    int sj = j + stencilSize;
    myprec dftemp = 0.0;
    for (int jt=0; jt<stencilSize; jt++)  { 
       dftemp += dcoeffF[jt]*(s_f[sj+jt-stencilSize][si]-s_f[sj+stencilSize-jt][si])/d_dy;
    }
    df[gblIdx] = dftemp;
  }

  __syncthreads();
}

*/
