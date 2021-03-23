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


double *coeff;
double dt;

double x[mx],phi[mx*my*mz];

double rhs1[mx*my*mz];
double rhs2[mx*my*mz];
double rhs3[mx*my*mz];
double rhs4[mx*my*mz];
double temp[mx*my*mz];


int gui=0;

void higherOrder_FD(double *dydx, double *y, double *g) {

    double dx = g[1]-g[0];
 
    double ybound[mx+stencilSize*2];
    for (int k=0; k<mz; k++) {
    for (int j=0; j<my; j++) {
      for(int i=stencilSize; i<mx+stencilSize; i++) 
        ybound[i] = y[idx(i,j,k)-stencilSize];
      
      if(periodic) {
        for(int i=0; i<stencilSize; i++) {
          ybound[i]   = y[mx-stencilSize+idx(i,j,k)];
          ybound[mx+stencilSize+i] = y[idx(i,j,k)];
        }
      }

      for (int i = 0; i < mx; i++){
          dydx[idx(i,j,k)] = 0.0;
          for (int it = 0; it < stencilSize; it++){
              dydx[idx(i,j,k)] += coeffS[it]*(ybound[i+it]-ybound[i+stencilSize*2-it])/dx;
          }
      }
    }
    }
}


void RHS(double *rhs, double *var, double *g) {

    higherOrder_FD(rhs, var, g);
    for (int k=0; k<mz; k++) {
    for (int j=0; j<my; j++) {
    for (int i=0; i<mx; i++) {
	rhs[idx(i,j,k)] = -rhs[idx(i,j,k)]*U; } } } 
}


void rk4() {

    RHS(rhs1, phi,x);

    for (int k=0; k<mz; k++) {
    for (int j=0; j<my; j++) {
    for (int i=0; i<mx; i++) {
      temp[idx(i,j,k)] = phi[idx(i,j,k)] + rhs1[idx(i,j,k)]*dt/2; } } }
    RHS(rhs2,temp,x);

    for (int k=0; k<mz; k++) {
    for (int j=0; j<my; j++) {
    for (int i=0; i<mx; i++) {
      temp[idx(i,j,k)] = phi[idx(i,j,k)] + rhs2[idx(i,j,k)]*dt/2; } } }
    RHS(rhs3,temp,x);

    for (int k=0; k<mz; k++) {
    for (int j=0; j<my; j++) {
    for (int i=0; i<mx; i++) {
      temp[idx(i,j,k)] = phi[idx(i,j,k)] + rhs3[idx(i,j,k)]*dt; } } }
    RHS(rhs4,temp,x);

    for (int k=0; k<mz; k++) {
    for (int j=0; j<my; j++) {
    for (int i=0; i<mx; i++) {
      phi[idx(i,j,k)] = phi[idx(i,j,k)] + dt*(rhs1[idx(i,j,k)]+2*rhs2[idx(i,j,k)]+2*rhs3[idx(i,j,k)]+rhs4[idx(i,j,k)])/6.0; } } }


}

void run() {
    for (int istep=0; istep < nsteps; istep++){
        rk4();
#if arch==0 
        if(gui==1) display();
#endif
        if((istep%10)==0)  cout << "Time step  " << istep << "  \n";
    }

}

int main(int argc, char** argv) {

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#if arch==0 
    if(argc>1) gui = 1;
    if(gui==1) initDisplay(argc, argv);
#endif
    for(int i=0;i<mx;i++) {
     x[i]=(0.5+i*1.0)/(mx);  }

    dt = CFL*(x[1]-x[0])/abs(U);

    initProfile();


#if arch==0
    cout <<"\n"; 
    cout <<"code is running on -> CPU \n";
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
    runDevice<<<grid,block>>>();
    cudaDeviceSynchronize();

    copyInit(0,grid,block);
    cudaDeviceSynchronize();
#endif
    FILE *fp = fopen("final.txt","w+");
    for(int i=0; i<mx; i++)
      fprintf(fp,"%lf %lf \n",x[i],phi[i]);
    fclose(fp);
  
 
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::seconds>     (end - begin).count() << "[s]" << std::endl;

    cout << "Simulation is finished! \n";

    return 0;
}


void initProfile() {
    double fact;
    double sigma;
    double t;

    // Initial profile
    for (int k = 0; k < mz; k++) {
    for (int j = 0; j < my; j++) {
    for (int i = 0; i < mx; i++) {
    if (j<=my/2) {
        fact = 0.02;
            phi[idx(i,j,k)] = tanh((x[i]-0.3*L)/fact) + tanh((0.7*L-x[i])/fact);
    } else if (j<=(int)(my/4.0)*3) {
        sigma = 0.1;
        t = x[i]-0.5;
            phi[idx(i,j,k)] = exp(-pow((t/sigma), 2));
    } else if (j<=my-5) {
        sigma = 0.1;
        t = x[i]-0.5;
        phi[idx(i,j,k)] = (1-t/sigma*t/sigma)*exp(-pow((t/sigma), 2));
    } else {
        phi[idx(i,j,k)] = cos(2.0*3.141529*x[i]);
    }
    } } }
    FILE *fp = fopen("initial.txt","w+");
    for(int i=0; i<mx; i++)
      fprintf(fp,"%lf %lf \n",x[i],phi[i]);
    fclose(fp);
}

