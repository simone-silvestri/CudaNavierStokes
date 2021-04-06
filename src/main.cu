#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "globals.h"
#if arch==1
#include "cuda_functions.h"
#else
#include "display.h" 
#endif

using namespace std;


double dt;

double x[mx],y[my],z[mz],phi[mx*my*mz];

double r[mx*my*mz];
double u[mx*my*mz];
double v[mx*my*mz];
double w[mx*my*mz];
double e[mx*my*mz];

double rhs1[mx*my*mz];
double rhs2[mx*my*mz];
double rhs3[mx*my*mz];
double rhs4[mx*my*mz];
double temp[mx*my*mz];
double tmp[mx*my*mz];


int gui=0;

int main(int argc, char** argv) {

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    initProfile();
    calcdt();
#if arch==0
    cout <<"\n"; 
    cout <<"recompile with the GPU option \n";
    cout <<"\n"; 
    run();
    if(gui==1) glutMainLoop();
#elif arch==1
    dim3 grid, block;

    cout <<"\n"; 
    cout <<"code is running on -> GPU \n";
    cout <<"\n"; 

    setDerivativeParameters(grid, block);

    copyInit(1,grid,block);

    /* to allocate 4GB of heap size on the GPU */
    size_t rsize = 1024ULL*1024ULL*1024ULL*8ULL;  // allocate 8GB
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, rsize);

    runDevice<<<grid,block>>>();
    cudaDeviceSynchronize();

    copyInit(0,grid,block);
    cudaDeviceSynchronize();
#endif

    FILE *fp = fopen("final.txt","w+");
    for(int k=0; k<mz; k++)
    for(int i=0; i<mx; i++)
      fprintf(fp,"%lf %lf %lf \n",x[i],z[k],r[idx(i,0,k)]);
    fclose(fp);
  
 
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::seconds>     (end - begin).count() << "[s]" << std::endl;

    cout << "Simulation is finished! \n";

    return 0;
}

void calcdt() {

    double dx = x[1] - x[0];
    double dy = y[1] - y[0];
    double dz = z[1] - z[0];
    double dx2 = dx*dx;
    double dy2 = dy*dy;
    double dz2 = dz*dz;
   
    dt = CFL*MIN(1.0/viscdt,1.0/veldt);
    printf("this is the dt %lf\n",dt);

}


void initProfile() {
    double fact;

    for(int i=0;i<mx;i++) {
     x[i]=Lx*(0.5+i*1.0)/(mx);  }

    for(int j=0;j<my;j++) {
     y[j]=Ly*(0.5+j*1.0)/(my);  }

    for(int k=0;k<mz;k++) {
     z[k]=Lz*(0.5+k*1.0)/(mz);  }

    // Initial profile
    for (int k = 0; k < mz; k++) {
    for (int j = 0; j < my; j++) {
    for (int i = 0; i < mx; i++) {
        fact = 0.02;
        phi[idx(i,j,k)]  = (tanh((z[k]-0.3*Lz)/fact) + tanh((0.7*Lz-z[k])/fact));
        phi[idx(i,j,k)] *= (tanh((x[i]-0.3*Lx)/fact) + tanh((0.7*Lx-x[i])/fact));
        r[idx(i,j,k)] = phi[idx(i,j,k)];
        u[idx(i,j,k)] = phi[idx(i,j,k)];
        v[idx(i,j,k)] = phi[idx(i,j,k)];
        w[idx(i,j,k)] = phi[idx(i,j,k)];
        e[idx(i,j,k)] = phi[idx(i,j,k)];
    } } }
    FILE *fp = fopen("initial.txt","w+");
    for(int k=0; k<mz; k++)
    for(int i=0; i<mx; i++)
      fprintf(fp,"%lf %lf %lf \n",x[i],z[k],r[idx(i,0,k)]);
    fclose(fp);
}

