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

double rhs1[mx*my*mz];
double rhs2[mx*my*mz];
double rhs3[mx*my*mz];
double rhs4[mx*my*mz];
double temp[mx*my*mz];
double tmp[mx*my*mz];


int gui=0;

void deriv1z(double *df, double *f, double *g) {

    double dz = 1.0/(g[1]-g[0]);
    double fbound[mz+stencilSize*2];
    for (int j=0; j<my; j++) {
    for (int i=0; i<mx; i++) {
      for(int k=stencilSize; k<mz+stencilSize; k++)
        fbound[k] = f[idx(i,j,k-stencilSize)];

      if(periodic) {
        for(int k=0; k<stencilSize; k++) {
          fbound[k]   = f[idx(i,j,mz-stencilSize+k)];
          fbound[mz+stencilSize+k] = f[idx(i,j,k)];
        }
      }

      for (int k = 0; k < mz; k++){
          df[idx(i,j,k)] = 0.0;
          for (int kt = 0; kt < stencilSize; kt++){
              df[idx(i,j,k)] += coeffF[kt]*(fbound[k+kt]-fbound[k+stencilSize*2-kt])*dz;
          }
      }
    }
    }
}

void deriv1y(double *df, double *f, double *g) {

    double dy = 1.0/(g[1]-g[0]);
    double fbound[my+stencilSize*2];
    for (int k=0; k<mz; k++) {
    for (int i=0; i<mx; i++) {
      for(int j=stencilSize; j<my+stencilSize; j++) 
        fbound[j] = f[idx(i,j-stencilSize,k)];
      
      if(periodic) {
        for(int j=0; j<stencilSize; j++) {
          fbound[j]   = f[idx(i,my-stencilSize+j,k)];
          fbound[my+stencilSize+j] = f[idx(i,j,k)];
        }
      }
      
      for (int j = 0; j < my; j++){
          df[idx(i,j,k)] = 0.0;
          for (int jt = 0; jt < stencilSize; jt++){
              df[idx(i,j,k)] += coeffF[jt]*(fbound[j+jt]-fbound[j+stencilSize*2-jt])*dy;
          }
      }
    }
    }
}


void deriv1x(double *df, double *f, double *g) {

    double dx = 1.0/(g[1]-g[0]);
 
    double fbound[mx+stencilSize*2];
    for (int k=0; k<mz; k++) {
    for (int j=0; j<my; j++) {
      for(int i=stencilSize; i<mx+stencilSize; i++) 
        fbound[i] = f[idx(i,j,k)-stencilSize];
      
      if(periodic) {
        for(int i=0; i<stencilSize; i++) {
          fbound[i]   = f[mx-stencilSize+idx(i,j,k)];
          fbound[mx+stencilSize+i] = f[idx(i,j,k)];
        }
      }

      for (int i = 0; i < mx; i++){
          df[idx(i,j,k)] = 0.0;
          for (int it = 0; it < stencilSize; it++){
              df[idx(i,j,k)] += coeffF[it]*(fbound[i+it]-fbound[i+stencilSize*2-it])*dx;
          }
      }
    }
    }
}

void deriv2y(double *d2f, double *f, double *g) {

    double d2y = 1.0/(g[1]-g[0])/(g[1]-g[0]);
 
    double fbound[my+stencilSize*2];
    for (int k=0; k<mz; k++) {
    for (int i=0; i<mx; i++) {
      for(int j=stencilSize; j<my+stencilSize; j++) 
        fbound[j] = f[idx(i,j-stencilSize,k)];
      
      if(periodic) {
        for(int j=0; j<stencilSize; j++) {
          fbound[j]   = f[idx(i,my-stencilSize+j,k)];
          fbound[my+stencilSize+j] = f[idx(i,j,k)];
        }
      }

      for (int j = 0; j < my; j++){
          d2f[idx(i,j,k)] = coeffS[stencilSize]*fbound[j+stencilSize]*d2y;
          for (int jt = 0; jt < stencilSize; jt++){
              d2f[idx(i,j,k)] += coeffS[jt]*(fbound[j+jt]-fbound[j+stencilSize*2-jt])*d2y;
          }
      }
    }
    }
}

void deriv2x(double *d2f, double *f, double *g) {

    double d2x = 1.0/(g[1]-g[0])/(g[1]-g[0]);
 
    double fbound[mx+stencilSize*2];
    for (int k=0; k<mz; k++) {
    for (int j=0; j<my; j++) {
      for(int i=stencilSize; i<mx+stencilSize; i++) 
        fbound[i] = f[idx(i,j,k)-stencilSize];
      
      if(periodic) {
        for(int i=0; i<stencilSize; i++) {
          fbound[i]   = f[mx-stencilSize+idx(i,j,k)];
          fbound[mx+stencilSize+i] = f[idx(i,j,k)];
        }
      }

      for (int i = 0; i < mx; i++){
          d2f[idx(i,j,k)] = coeffS[stencilSize]*fbound[i+stencilSize]*d2x;
          for (int it = 0; it < stencilSize; it++){
              d2f[idx(i,j,k)] += coeffS[it]*(fbound[i+it]-fbound[i+stencilSize*2-it])*d2x;
          }
      }
    }
    }
}


void RHS(double *rhs, double *var, double *gx, double *gy, double *gz) {

	deriv1x(tmp, var, gx);

#if parentGrid==0
	deriv1x(tmp,var,gx);
#elif parentGrid==1
	deriv1y(tmp,var,gy);
#else
	deriv1z(tmp,var,gz);
#endif

	for (int it=0; it<mx*my*mz; it++)
		rhs[it] = - tmp[it]*U;

}


void rk4() {

    RHS(rhs1,phi,x,y,z);

    for (int it=0; it<mx*my*mz; it++) 
      temp[it] = phi[it] + rhs1[it]*dt/2; 
    RHS(rhs2,temp,x,y,z);

    for (int it=0; it<mx*my*mz; it++) 
      temp[it] = phi[it] + rhs2[it]*dt/2; 
    RHS(rhs3,temp,x,y,z);

    for (int it=0; it<mx*my*mz; it++) 
      temp[it] = phi[it] + rhs3[it]*dt;
    RHS(rhs4,temp,x,y,z);

    for (int it=0; it<mx*my*mz; it++) 
      phi[it] = phi[it] + dt*(rhs1[it]+2*rhs2[it]+2*rhs3[it]+rhs4[it])/6.0; 
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

    initProfile();
    calcdt();

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
    for(int k=0; k<mz; k++)
    for(int i=0; i<mx; i++)
      fprintf(fp,"%lf %lf %lf \n",x[i],z[k],phi[idx(i,0,k)]);
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
   
    double viscdt = visc/0.5*(1.0/dx2+1.0/dy2+1.0/dz2);
    double veldt  = abs(Ux)/dx + abs(Uy)/dy + abs(Uz)/dz;
 
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
    } } }
    FILE *fp = fopen("initial.txt","w+");
    for(int k=0; k<mz; k++)
    for(int i=0; i<mx; i++)
      fprintf(fp,"%lf %lf %lf \n",x[i],z[k],phi[idx(i,0,k)]);
    fclose(fp);
}

