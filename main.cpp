#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include "globals.h"
#include "display.h"

using namespace std;

int gui=0;

void higherOrder_FD(double *dydx, double *y, double *g) {

    double dx = g[1]-g[0];
 
    double ybound[mx+stencilSize*2+1];
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
          for (int it = 0; it < stencilSize*2+1; it++){
              dydx[idx(i,j,k)] = dydx[idx(i,j,k)] + coeff[it]*ybound[i+it]/dx;
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
    cout << nsteps << " Time steps total \n ";
    for (int istep=0; istep < nsteps; istep++){
        rk4();
        if(gui==1) display();
        cout << "Time step  " << istep << "  \n";
    }

}

int main(int argc, char** argv) {

    if(argc>1) gui = 1;
	 
    if(gui==1) initDisplay(argc, argv);

    for(int i=0;i<mx;i++) {
     x[i]=(0.5+i*1.0)/(mx);  }

    dt = CFL*(x[1]-x[0])/abs(U);

    coeff=(double*)malloc((stencilSize*2+1)*sizeof(double));
    for(int i=0; i<(stencilSize*2+1); i++) 
     coeff[i]=0.0;

    for(int i=0; i<stencilSize; i++) {
     coeff[i] = coeffS[i];
     coeff[stencilSize*2-i] = -coeffS[i];
    }

    int initialProfile = 2;
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
    } else {
        sigma = 0.1;
        t = x[i]-0.5;
        phi[idx(i,j,k)] = (1-t/sigma*t/sigma)*exp(-pow((t/sigma), 2));
    }
    } } }
    FILE *fp = fopen("initial.txt","w+");
    for(int i=0; i<mx; i++)
      fprintf(fp,"%lf %lf \n",x[i],phi[i]);
    fclose(fp);

    run();

    if(gui==1) glutMainLoop();

    fp = fopen("final.txt","w+");
    for(int i=0; i<mx; i++)
      fprintf(fp,"%lf %lf \n",x[i],phi[i]);
    fclose(fp);

    cout << "Simulation is finished! \n ";

    return 0;
}
