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
 
    double ybound[N+stencilSize*2+1];
    for(int i=stencilSize; i<N+stencilSize; i++) 
     ybound[i] = y[i-stencilSize];
      
    if(periodic) {
      for(int i=0; i<stencilSize; i++) {
        ybound[i]   = y[N-stencilSize+i];
        ybound[N+stencilSize+i] = y[i];
      }
    }

    for (int i = 0; i < N; i++){
        dydx[i] = 0.0;
        for (int idx = 0; idx < stencilSize*2+1; idx++){
            dydx[i] = dydx[i] + coeff[idx]*ybound[i+idx]/dx;
        }
    }
}


void RHS(double *rhs, double *var, double *g) {

    higherOrder_FD(rhs, var, g);
    for(int i=0; i<N; i++)  
	rhs[i] = -rhs[i]*U;
}


void rk4() {

    RHS(rhs1, phi,x);

    for(int i=0; i<N; i++)
      temp[i] = phi[i] + rhs1[i]*dt/2;
    RHS(rhs2,temp,x);

    for(int i=0; i<N; i++)
      temp[i] = phi[i] + rhs2[i]*dt/2;
    RHS(rhs3,temp,x);

    for(int i=0; i<N; i++)
      temp[i] = phi[i] + rhs3[i]*dt;
    RHS(rhs4,temp,x);

    for(int i=0; i<N; i++)
      phi[i] = phi[i] + dt*(rhs1[i]+2*rhs2[i]+2*rhs3[i]+rhs4[i])/6.0;


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

    for(int i=0;i<N;i++) {
     x[i]=(0.5+i*1.0)/(N);  }

    dt = CFL*(x[1]-x[0])/abs(U);

    coeff=(double*)malloc((stencilSize*2+1)*sizeof(double));
    for(int i=0; i<(stencilSize*2+1); i++) 
     coeff[i]=0.0;

    for(int i=0; i<stencilSize; i++) {
     coeff[i] = coeffS[i];
     coeff[stencilSize*2-i] = -coeffS[i];
    }

    int initialProfile = 1;
    double fact;
    double sigma;
    double t;

    // Initial profile
    if (initialProfile == 1) {
        fact = 0.02;
        for (int i = 0; i < N; i++) {
            phi[i] = tanh((x[i]-0.3*L)/fact) + tanh((0.7*L-x[i])/fact);
        }
    } else if (initialProfile == 2) {
        sigma = 0.1;
        t = x[0]-0.5;
        for (int i = 0; i < N; i++) {
            phi[i] = exp(pow(-((x[i]-0.5)/sigma), 2));
        }
    }
    FILE *fp = fopen("initial.txt","w+");
    for(int i=0; i<N; i++)
      fprintf(fp,"%lf %lf \n",x[i],phi[i]);
    fclose(fp);

    run();

    if(gui==1) glutMainLoop();

    fp = fopen("final.txt","w+");
    for(int i=0; i<N; i++)
      fprintf(fp,"%lf %lf \n",x[i],phi[i]);
    fclose(fp);

    cout << "Simulation is finished! \n ";

    return 0;
}
