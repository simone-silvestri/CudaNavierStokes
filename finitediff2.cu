
#include <stdio.h>
#include <assert.h>

#define nTimeSteps 10000      // maximum number of time steps
#define alpha      0.01f      // diffusivity
#define timeStep   0.0001f    // time step  

float fx = 1.0f, fy = 10.0f, fz = 1.0f;
const int mx = 256, my = 256, mz = 256;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}
  
// shared memory tiles will be m*-by-*Pencils
// sPencils is used when each thread calculates the derivative at one point
// lPencils is used for coalescing in y and z where each thread has to 
//     calculate the derivative at mutiple points
const int sPencils = 1 ; // small # pencils
const int lPencils = 32; // large # pencils
  
dim3 grid[3][2], block[3][2];


// stencil coefficients
__constant__ float c_ax , c_bx , c_cx , c_dx ;
__constant__ float c_ay , c_by , c_cy , c_dy ;
__constant__ float c_az , c_bz , c_cz , c_dz ;
__constant__ float c2_ax, c2_bx, c2_cx, c2_dx, c2_px;
__constant__ float c2_ay, c2_by, c2_cy, c2_dy, c2_py;
__constant__ float c2_az, c2_bz, c2_cz, c2_dz, c2_pz;
__constant__ float d_dt, d_alpha;
__constant__ bool  periodX, periodY,periodZ;
// host routine to set constant data

void setDerivativeParameters()
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

  // device time step
  float dt = timeStep;
  checkCuda( cudaMemcpyToSymbol(d_dt, &dt, sizeof(float), 0, cudaMemcpyHostToDevice) );
  
  // device  
  float diff = alpha;
  checkCuda( cudaMemcpyToSymbol(d_alpha, &diff, sizeof(float), 0, cudaMemcpyHostToDevice) );
  

  // stencil weights (for unit length problem)
  float dsinv = mx-1.f;
  
  float ax =  4.f / 5.f   * dsinv;
  float bx = -1.f / 5.f   * dsinv;
  float cx =  4.f / 105.f * dsinv;
  float dx = -1.f / 280.f * dsinv;
  checkCuda( cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c_bx, &bx, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c_cx, &cx, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c_dx, &dx, sizeof(float), 0, cudaMemcpyHostToDevice) );

  dsinv = my-1.f;
  
  float ay =  4.f / 5.f   * dsinv;
  float by = -1.f / 5.f   * dsinv;
  float cy =  4.f / 105.f * dsinv;
  float dy = -1.f / 280.f * dsinv;
  checkCuda( cudaMemcpyToSymbol(c_ay, &ay, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c_by, &by, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c_cy, &cy, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c_dy, &dy, sizeof(float), 0, cudaMemcpyHostToDevice) );

  dsinv = mz-1.f;
  
  float az =  4.f / 5.f   * dsinv;
  float bz = -1.f / 5.f   * dsinv;
  float cz =  4.f / 105.f * dsinv;
  float dz = -1.f / 280.f * dsinv;
  checkCuda( cudaMemcpyToSymbol(c_az, &az, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c_bz, &bz, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c_cz, &cz, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c_dz, &dz, sizeof(float), 0, cudaMemcpyHostToDevice) );

  dsinv = mx-1.f;
  
  float a2x =  8.f   / 5.f   * dsinv * dsinv;
  float b2x = -1.f   / 5.f   * dsinv * dsinv;
  float c2x =  8.f   / 315.f * dsinv * dsinv;
  float d2x = -1.f   / 560.f * dsinv * dsinv;
  float p2x = -205.f / 72.f  * dsinv * dsinv;
  checkCuda( cudaMemcpyToSymbol(c2_ax, &a2x, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_bx, &b2x, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_cx, &c2x, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_dx, &d2x, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_px, &p2x, sizeof(float), 0, cudaMemcpyHostToDevice) );

  dsinv = my-1.f;
  
  float a2y =  8.f   / 5.f   * dsinv * dsinv;
  float b2y = -1.f   / 5.f   * dsinv * dsinv;
  float c2y =  8.f   / 315.f * dsinv * dsinv;
  float d2y = -1.f   / 560.f * dsinv * dsinv;
  float p2y = -205.f / 72.f  * dsinv * dsinv;
  checkCuda( cudaMemcpyToSymbol(c2_ay, &a2y, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_by, &b2y, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_cy, &c2y, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_dy, &d2y, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_py, &p2y, sizeof(float), 0, cudaMemcpyHostToDevice) );
  
  dsinv = mz-1.f;
  
  float a2z =  8.f   / 5.f   * dsinv * dsinv;
  float b2z = -1.f   / 5.f   * dsinv * dsinv;
  float c2z =  8.f   / 315.f * dsinv * dsinv;
  float d2z = -1.f   / 560.f * dsinv * dsinv;
  float p2z = -205.f / 72.f  * dsinv * dsinv;
  checkCuda( cudaMemcpyToSymbol(c2_az, &a2z, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_bz, &b2z, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_cz, &c2z, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_dz, &d2z, sizeof(float), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(c2_pz, &p2z, sizeof(float), 0, cudaMemcpyHostToDevice) );

  bool perX = false;
  bool perY = false;
  bool perZ = false;
  checkCuda( cudaMemcpyToSymbol(periodX, &perX, sizeof(bool), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(periodY, &perY, sizeof(bool), 0, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpyToSymbol(periodZ, &perZ, sizeof(bool), 0, cudaMemcpyHostToDevice) );


  // Execution configurations for small and large pencil tiles

  grid[0][0]  = dim3(my / sPencils, mz, 1);
  block[0][0] = dim3(mx, sPencils, 1);

  grid[0][1]  = dim3(my / lPencils, mz, 1);
  block[0][1] = dim3(mx, sPencils, 1);

  grid[1][0]  = dim3(mx / sPencils, mz, 1);
  block[1][0] = dim3(sPencils, my, 1);

  grid[1][1]  = dim3(mx / lPencils, mz, 1);
  // we want to use the same number of threads as above,
  // so when we use lPencils instead of sPencils in one
  // dimension, we multiply the other by sPencils/lPencils
  block[1][1] = dim3(lPencils, my * sPencils / lPencils, 1);

  grid[2][0]  = dim3(mx / sPencils, my, 1);
  block[2][0] = dim3(sPencils, mz, 1);

  grid[2][1]  = dim3(mx / lPencils, my, 1);
  block[2][1] = dim3(lPencils, mz * sPencils / lPencils, 1);
}

void initInput(float *f, int dim)
{
  const float twopi = 8.f * (float)atan(1.0);

  for (int k = 0; k < mz; k++) {
    for (int j = 0; j < my; j++) {
      for (int i = 0; i < mx; i++) {
        switch (dim) {
          case 0: 
            f[k*mx*my+j*mx+i] = cos(fx*twopi*(i-1.f)/(mx-1.f));
            break;
          case 1:
            f[k*mx*my+j*mx+i] = cos(fy*twopi*(j-1.f)/(my-1.f));
            break;
          case 2:
            f[k*mx*my+j*mx+i] = cos(fz*twopi*(k-1.f)/(mz-1.f));
            break;
        }
      }
    }
  }     
}

void initSol(float *sol, float *sol2, int dim)
{
  const float twopi = 8.f * (float)atan(1.0);

  for (int k = 0; k < mz; k++) {
    for (int j = 0; j < my; j++) {
      for (int i = 0; i < mx; i++) {
        switch (dim) {
          case 0: 
            sol[k*mx*my+j*mx+i]  = -fx*twopi*sin(fx*twopi*(i-1.f)/(mx-1.f));
            sol2[k*mx*my+j*mx+i] = -fx*twopi*fx*twopi*cos(fx*twopi*(i-1.f)/(mx-1.f));
            break;
          case 1:
            sol[k*mx*my+j*mx+i]  = -fy*twopi*sin(fy*twopi*(j-1.f)/(my-1.f));
            sol2[k*mx*my+j*mx+i] = -fy*twopi*fy*twopi*cos(fy*twopi*(j-1.f)/(my-1.f));
            break;
          case 2:
            sol[k*mx*my+j*mx+i]  = -fz*twopi*sin(fz*twopi*(k-1.f)/(mz-1.f));
            sol2[k*mx*my+j*mx+i] = -fz*twopi*fz*twopi*cos(fz*twopi*(k-1.f)/(mz-1.f));
            break;
        }
      }
    }
  }    
}

void checkResults(double &error, double &maxError, float *sol, float *df)
{
  // error = sqrt(sum((sol-df)**2)/(mx*my*mz))
  // maxError = maxval(abs(sol-df))
  maxError = 0;
  error = 0;
  for (int k = 0; k < mz; k++) {
    for (int j = 0; j < my; j++) {
      for (int i = 0; i < mx; i++) {
        float s = sol[k*mx*my+j*mx+i];
        float f = df[k*mx*my+j*mx+i];
        //printf("%d %d %d: %f %f\n", i, j, k, s, f);
        error += (s-f)*(s-f);
        if (fabs(s-f) > maxError) maxError = fabs(s-f);
      }
    }
  }
  error = sqrt(error / (mx*my*mz));
}
  

///__global__ void runSimul(float *f)
///{
///  
///  for(int t=0; t<timeSteps; t++)
///    rk3(f);
///}


// -------------
// Second Order derivatives
// -------------

// -------------
// x derivatives
// -------------

__global__ void derivative2_x(float *f, float *df)
{  
  __shared__ float s_f[sPencils][mx+8]; // 4-wide halo

  int i  = threadIdx.x;
  int j  = blockIdx.x*blockDim.y + threadIdx.y;
  int k  = blockIdx.y;
  int si = i + 4;       // local i for shared memory access + halo offset
  int sj = threadIdx.y; // local j for shared memory access

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj][si] = f[globalIdx];

  __syncthreads();

  // fill in periodic images in shared memory array 

  //if(periodicX) {
    if (i < 4) {
      s_f[sj][si-4]  = s_f[sj][si+mx-5];
      s_f[sj][si+mx] = s_f[sj][si+1];   
    }
  //} else {

  //}

  __syncthreads();
  
  df[globalIdx] = 
    ( c2_ax * ( s_f[sj][si+1] + s_f[sj][si-1] )
    + c2_bx * ( s_f[sj][si+2] + s_f[sj][si-2] )
    + c2_cx * ( s_f[sj][si+3] + s_f[sj][si-3] )
    + c2_dx * ( s_f[sj][si+4] + s_f[sj][si-4] )
    + c2_px *   s_f[sj][si] );
}


// this version uses a 64x32 shared memory tile, 
// still with 64*sPencils threads

__global__ void derivative2_x_lPencils(float *f, float *df)
{
  __shared__ float s_f[lPencils][mx+8]; // 4-wide halo
  
  int i     = threadIdx.x;
  int jBase = blockIdx.x*lPencils;
  int k     = blockIdx.y;
  int si    = i + 4; // local i for shared memory access + halo offset

  for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
    int globalIdx = k * mx * my + (jBase + sj) * mx + i;      
    s_f[sj][si] = f[globalIdx];
  }

  __syncthreads();

  // fill in periodic images in shared memory array 
  if (i < 4) {
    for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
      s_f[sj][si-4]  = s_f[sj][si+mx-5];
      s_f[sj][si+mx] = s_f[sj][si+1];
    }
  }

  __syncthreads();

  for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
     int globalIdx = k * mx * my + (jBase + sj) * mx + i;      
     df[globalIdx] = 
      ( c2_ax * ( s_f[sj][si+1] + s_f[sj][si-1] )
      + c2_bx * ( s_f[sj][si+2] + s_f[sj][si-2] )
      + c2_cx * ( s_f[sj][si+3] + s_f[sj][si-3] )
      + c2_dx * ( s_f[sj][si+4] + s_f[sj][si-4] ) 
      + c2_px *   s_f[sj][si] );
  }
}

// -------------
// y derivatives
// -------------

__global__ void derivative2_y(float *f, float *df)
{
  __shared__ float s_f[my+8][sPencils];

  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int j  = threadIdx.y;
  int k  = blockIdx.y;
  int si = threadIdx.x;
  int sj = j + 4;

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj][si] = f[globalIdx];
  
  __syncthreads();

  if (j < 4) {
    s_f[sj-4][si]  = s_f[sj+my-5][si];
    s_f[sj+my][si] = s_f[sj+1][si];
  }

  __syncthreads();

  df[globalIdx] = 
    ( c2_ay * ( s_f[sj+1][si] + s_f[sj-1][si] )
    + c2_by * ( s_f[sj+2][si] + s_f[sj-2][si] )
    + c2_cy * ( s_f[sj+3][si] + s_f[sj-3][si] )
    + c2_dy * ( s_f[sj+4][si] + s_f[sj-4][si] ) 
    + c2_py *   s_f[sj][si] );
}

// y derivative using a tile of 32x64,
// launch with thread block of 32x8
__global__ void derivative2_y_lPencils(float *f, float *df)
{
  __shared__ float s_f[my+8][lPencils];

  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int k  = blockIdx.y;
  int si = threadIdx.x;
  
  for (int j = threadIdx.y; j < my; j += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + 4;
    s_f[sj][si] = f[globalIdx];
  }

  __syncthreads();

  int sj = threadIdx.y + 4;
  if (sj < 8) {
     s_f[sj-4][si]  = s_f[sj+my-5][si];
     s_f[sj+my][si] = s_f[sj+1][si];   
  }

  __syncthreads();

  for (int j = threadIdx.y; j < my; j += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + 4;
    df[globalIdx] = 
      ( c2_ay * ( s_f[sj+1][si] + s_f[sj-1][si] )
      + c2_by * ( s_f[sj+2][si] + s_f[sj-2][si] )
      + c2_cy * ( s_f[sj+3][si] + s_f[sj-3][si] )
      + c2_dy * ( s_f[sj+4][si] + s_f[sj-4][si] ) 
      + c2_py *   s_f[sj][si]  );
  }
}


// ------------
// z derivative
// ------------

__global__ void derivative2_z(float *f, float *df)
{
  __shared__ float s_f[mz+8][sPencils];

  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int j  = blockIdx.y;
  int k  = threadIdx.y;
  int si = threadIdx.x;
  int sk = k + 4; // halo offset

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sk][si] = f[globalIdx];

  __syncthreads();

  if (k < 4) {
     s_f[sk-4][si]  = s_f[sk+mz-5][si];
     s_f[sk+mz][si] = s_f[sk+1][si];
  }

  __syncthreads();

  df[globalIdx] = 
    ( c2_az * ( s_f[sk+1][si] + s_f[sk-1][si] )
    + c2_bz * ( s_f[sk+2][si] + s_f[sk-2][si] )
    + c2_cz * ( s_f[sk+3][si] + s_f[sk-3][si] )
    + c2_dz * ( s_f[sk+4][si] + s_f[sk-4][si] ) 
    + c2_pz *   s_f[sk][si] );
}

__global__ void derivative2_z_lPencils(float *f, float *df)
{
  __shared__ float s_f[mz+8][lPencils];

  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int j  = blockIdx.y;
  int si = threadIdx.x;

  for (int k = threadIdx.y; k < mz; k += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + 4;
    s_f[sk][si] = f[globalIdx];
  }

  __syncthreads();

  int k = threadIdx.y + 4;
  if (k < 8) {
     s_f[k-4][si]  = s_f[k+mz-5][si];
     s_f[k+mz][si] = s_f[k+1][si];
  }

  __syncthreads();

  for (int k = threadIdx.y; k < mz; k += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + 4;
    df[globalIdx] = 
        ( c2_az * ( s_f[sk+1][si] + s_f[sk-1][si] )
        + c2_bz * ( s_f[sk+2][si] + s_f[sk-2][si] )
        + c2_cz * ( s_f[sk+3][si] + s_f[sk-3][si] )
        + c2_dz * ( s_f[sk+4][si] + s_f[sk-4][si] ) 
	+ c2_pz *   s_f[sk][si]   );  
  }
}

// -------------
// First Order derivatives
// -------------

// -------------
// x derivatives
// -------------

__global__ void derivative_x(float *f, float *df)
{  
  __shared__ float s_f[sPencils][mx+8]; // 4-wide halo

  int i   = threadIdx.x;
  int j   = blockIdx.x*blockDim.y + threadIdx.y;
  int k  = blockIdx.y;
  int si = i + 4;       // local i for shared memory access + halo offset
  int sj = threadIdx.y; // local j for shared memory access

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj][si] = f[globalIdx];

  __syncthreads();

  // fill in periodic images in shared memory array 
  if (i < 4) {
    s_f[sj][si-4]  = s_f[sj][si+mx-5];
    s_f[sj][si+mx] = s_f[sj][si+1];   
  }

  __syncthreads();
  
  df[globalIdx] = 
    ( c_ax * ( s_f[sj][si+1] - s_f[sj][si-1] )
    + c_bx * ( s_f[sj][si+2] - s_f[sj][si-2] )
    + c_cx * ( s_f[sj][si+3] - s_f[sj][si-3] )
    + c_dx * ( s_f[sj][si+4] - s_f[sj][si-4] ) );
}


// this version uses a 64x32 shared memory tile, 
// still with 64*sPencils threads

__global__ void derivative_x_lPencils(float *f, float *df)
{
  __shared__ float s_f[lPencils][mx+8]; // 4-wide halo
  
  int i     = threadIdx.x;
  int jBase = blockIdx.x*lPencils;
  int k     = blockIdx.y;
  int si    = i + 4; // local i for shared memory access + halo offset

  for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
    int globalIdx = k * mx * my + (jBase + sj) * mx + i;      
    s_f[sj][si] = f[globalIdx];
  }

  __syncthreads();

  // fill in periodic images in shared memory array 
  if (i < 4) {
    for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
      s_f[sj][si-4]  = s_f[sj][si+mx-5];
      s_f[sj][si+mx] = s_f[sj][si+1];
    }
  }

  __syncthreads();

  for (int sj = threadIdx.y; sj < lPencils; sj += blockDim.y) {
     int globalIdx = k * mx * my + (jBase + sj) * mx + i;      
     df[globalIdx] = 
      ( c_ax * ( s_f[sj][si+1] - s_f[sj][si-1] )
      + c_bx * ( s_f[sj][si+2] - s_f[sj][si-2] )
      + c_cx * ( s_f[sj][si+3] - s_f[sj][si-3] )
      + c_dx * ( s_f[sj][si+4] - s_f[sj][si-4] ) );
  }
}

// -------------
// y derivatives
// -------------

__global__ void derivative_y(float *f, float *df)
{
  __shared__ float s_f[my+8][sPencils];

  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int j  = threadIdx.y;
  int k  = blockIdx.y;
  int si = threadIdx.x;
  int sj = j + 4;

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj][si] = f[globalIdx];
  
  __syncthreads();

  if (j < 4) {
    s_f[sj-4][si]  = s_f[sj+my-5][si];
    s_f[sj+my][si] = s_f[sj+1][si];
  }

  __syncthreads();

  df[globalIdx] = 
    ( c_ay * ( s_f[sj+1][si] - s_f[sj-1][si] )
    + c_by * ( s_f[sj+2][si] - s_f[sj-2][si] )
    + c_cy * ( s_f[sj+3][si] - s_f[sj-3][si] )
    + c_dy * ( s_f[sj+4][si] - s_f[sj-4][si] ) );
}

// y derivative using a tile of 32x64,
// launch with thread block of 32x8
__global__ void derivative_y_lPencils(float *f, float *df)
{
  __shared__ float s_f[my+8][lPencils];

  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int k  = blockIdx.y;
  int si = threadIdx.x;
  
  for (int j = threadIdx.y; j < my; j += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + 4;
    s_f[sj][si] = f[globalIdx];
  }

  __syncthreads();

  int sj = threadIdx.y + 4;
  if (sj < 8) {
     s_f[sj-4][si]  = s_f[sj+my-5][si];
     s_f[sj+my][si] = s_f[sj+1][si];   
  }

  __syncthreads();

  for (int j = threadIdx.y; j < my; j += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sj = j + 4;
    df[globalIdx] = 
      ( c_ay * ( s_f[sj+1][si] - s_f[sj-1][si] )
      + c_by * ( s_f[sj+2][si] - s_f[sj-2][si] )
      + c_cy * ( s_f[sj+3][si] - s_f[sj-3][si] )
      + c_dy * ( s_f[sj+4][si] - s_f[sj-4][si] ) );
  }
}


// ------------
// z derivative
// ------------

__global__ void derivative_z(float *f, float *df)
{
  __shared__ float s_f[mz+8][sPencils];

  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int j  = blockIdx.y;
  int k  = threadIdx.y;
  int si = threadIdx.x;
  int sk = k + 4; // halo offset

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sk][si] = f[globalIdx];

  __syncthreads();

  if (k < 4) {
     s_f[sk-4][si]  = s_f[sk+mz-5][si];
     s_f[sk+mz][si] = s_f[sk+1][si];
  }

  __syncthreads();

  df[globalIdx] = 
    ( c_az * ( s_f[sk+1][si] - s_f[sk-1][si] )
    + c_bz * ( s_f[sk+2][si] - s_f[sk-2][si] )
    + c_cz * ( s_f[sk+3][si] - s_f[sk-3][si] )
    + c_dz * ( s_f[sk+4][si] - s_f[sk-4][si] ) );
}

__global__ void derivative_z_lPencils(float *f, float *df)
{
  __shared__ float s_f[mz+8][lPencils];

  int i  = blockIdx.x*blockDim.x + threadIdx.x;
  int j  = blockIdx.y;
  int si = threadIdx.x;

  for (int k = threadIdx.y; k < mz; k += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + 4;
    s_f[sk][si] = f[globalIdx];
  }

  __syncthreads();

  int k = threadIdx.y + 4;
  if (k < 8) {
     s_f[k-4][si]  = s_f[k+mz-5][si];
     s_f[k+mz][si] = s_f[k+1][si];
  }

  __syncthreads();

  for (int k = threadIdx.y; k < mz; k += blockDim.y) {
    int globalIdx = k * mx * my + j * mx + i;
    int sk = k + 4;
    df[globalIdx] = 
        ( c_az * ( s_f[sk+1][si] - s_f[sk-1][si] )
        + c_bz * ( s_f[sk+2][si] - s_f[sk-2][si] )
        + c_cz * ( s_f[sk+3][si] - s_f[sk-3][si] )
        + c_dz * ( s_f[sk+4][si] - s_f[sk-4][si] ) );  
  }
}

// Run the kernels for a given dimension. One for sPencils, one for lPencils
void runTest(int dimension, int deriv)
{
  void (*fpDeriv[2])(float*, float*);
 
  switch(deriv) {
    case 1:
       switch(dimension) {
         case 0:
           fpDeriv[0] = derivative_x;
           fpDeriv[1] = derivative_x_lPencils;
           break;
         case 1:
           fpDeriv[0] = derivative_y;
           fpDeriv[1] = derivative_y_lPencils;
           break;
         case 2:
           fpDeriv[0] = derivative_z;
           fpDeriv[1] = derivative_z_lPencils;
           break;
       }
       break;
    case 2:
       switch(dimension) {
         case 0:
           fpDeriv[0] = derivative2_x;
           fpDeriv[1] = derivative2_x_lPencils;
           break;
         case 1:
           fpDeriv[0] = derivative2_y;
           fpDeriv[1] = derivative2_y_lPencils;
           break;
         case 2:
           fpDeriv[0] = derivative2_z;
           fpDeriv[1] = derivative2_z_lPencils;
           break;
       }
    break;
  }
  int sharedDims[3][2][2] = { mx, sPencils, 
                              mx, lPencils,
                              sPencils, my,
                              lPencils, my,
                              sPencils, mz,
                              lPencils, mz };

  float *f = new float[mx*my*mz];
  float *df = new float[mx*my*mz];
  float *sol = new float[mx*my*mz];                           
  float *sol2 = new float[mx*my*mz];                           
    
  initInput(f, dimension);
  initSol(sol, sol2, dimension);

  // device arrays
  int bytes = mx*my*mz * sizeof(float);
  float *d_f, *d_df;
  checkCuda( cudaMalloc((void**)&d_f, bytes) );
  checkCuda( cudaMalloc((void**)&d_df, bytes) );

  const int nReps = 20;
  float milliseconds;
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  double error, maxError;

  printf("%c, %d-order derivatives\n\n", (char)(0x58 + dimension), deriv);

  checkCuda( cudaMemcpy(d_f, f, bytes, cudaMemcpyHostToDevice) );  
  checkCuda( cudaMemset(d_df, 0, bytes) );

  fpDeriv[0]<<<grid[dimension][0],block[dimension][0]>>>(d_f, d_df); // warm up
  checkCuda( cudaMemcpy(df, d_df, bytes, cudaMemcpyDeviceToHost) );
  
  cudaDeviceSynchronize();

  char filename[20];
  sprintf(filename,"%d-order-in-%c.txt", deriv, (char)(0x58 + dimension));
  FILE *file; 
  file = fopen(filename,"w+");
  int skip,tot; 
  switch(dimension) {
    case 0:
      skip = 1;
      tot  = mx;
      break;
    case 1:
      skip = mx;
      tot  = mx*my;
      break;
    case 2:
      skip = mx*my;
      tot  = mx*my*mz;
      break;
  }
  if(deriv==1) {
    for (int idx=0; idx<tot; idx+=skip) {
     fprintf(file,"%d %f %f %f\n",idx,f[idx],sol[idx],df[idx]); } 
  }
  else {
    for (int idx=0; idx<tot; idx+=skip) {
     fprintf(file,"%d %f %f %f\n",idx,f[idx],sol2[idx],df[idx]); } 
  }
  fclose(file);

  for (int fp = 0; fp < 2; fp++) { 
    checkCuda( cudaMemcpy(d_f, f, bytes, cudaMemcpyHostToDevice) );  
    checkCuda( cudaMemset(d_df, 0, bytes) );

    fpDeriv[fp]<<<grid[dimension][fp],block[dimension][fp]>>>(d_f, d_df); // warm up
    checkCuda( cudaEventRecord(startEvent, 0) );
    for (int i = 0; i < nReps; i++)
       fpDeriv[fp]<<<grid[dimension][fp],block[dimension][fp]>>>(d_f, d_df);
    
    checkCuda( cudaMemcpy(df, d_df, bytes, cudaMemcpyDeviceToHost) );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

    switch(deriv) {
      case 1:
        checkResults(error, maxError, sol, df);
        break;
      case 2:
        checkResults(error, maxError, sol2,df);
	break;
    }

    printf("  Using shared memory tile of %d x %d\n", 
           sharedDims[dimension][fp][0], sharedDims[dimension][fp][1]);
    printf("   RMS error: %e\n", error);
    printf("   MAX error: %e\n", maxError);
    printf("   Average time (ms): %f\n", milliseconds / nReps);
    printf("   Average Bandwidth (GB/s): %f\n\n", 
           2.f * 1e-6 * mx * my * mz * nReps * sizeof(float) / milliseconds);
  }

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );

  checkCuda( cudaFree(d_f) );
  checkCuda( cudaFree(d_df) );

  delete [] f;
  delete [] df;
  delete [] sol;
  delete [] sol2;
}


// This the main host code for the finite difference 
// example.  The kernels are contained in the derivative_m module

int main(void)
{
  // Print device and precision
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, 0) );
  printf("\nDevice Name: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

  setDerivativeParameters(); // initialize 

  runTest(0, 1); // x First derivative
  runTest(1, 1); // y First derivative
  runTest(2, 1); // z First derivative

  runTest(0, 2); // x Second derivative
  runTest(1, 2); // y Second derivative
  runTest(2, 2); // z Second derivative

  return 0;
}

