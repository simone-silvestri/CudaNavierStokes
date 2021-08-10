#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "../src_multiGPU/comm.h"
#include "../src_multiGPU/globals.h"
#include "../src_multiGPU/main.h"

extern double dxv[mx_tot];

void derivGrid(double *d2f, double *df, double *f, double dx);

void initField(int timestep, Communicator rk) {
	readFileMPI('r',timestep,r,rk);
	readFileMPI('u',timestep,u,rk);
	readFileMPI('v',timestep,v,rk);
	readFileMPI('w',timestep,w,rk);
	readFileMPI('e',timestep,e,rk);

    if(rk.rank==0) {
    	printf("loaded field number %d\n",timestep);
    }
}

void writeField(int timestep, Communicator rk) {
	saveFileMPI('r',timestep,r,rk);
	saveFileMPI('u',timestep,u,rk);
	saveFileMPI('v',timestep,v,rk);
	saveFileMPI('w',timestep,w,rk);
	saveFileMPI('e',timestep,e,rk);
}

void initGrid(Communicator rk) {

	//constant grid in y and z and stretched in x

	dx = Lx*(1.0)/(mx_tot);

	double xn[mx_tot+1];
	for (int i=0; i<mx_tot+1; i++)
		xn[i] = tanh(stretch*((i*1.0)/mx_tot-0.5))/tanh(stretch*0.5);

	for (int i=0; i<mx_tot; i++)
			x[i] = Lx * (1.0 + (xn[i] + xn[i+1])/2.0)/2.0;

	derivGrid(xpp,xp,x,dx);

	for (int i=0; i<mx_tot; i++)
		xp[i] = 1.0/xp[i];

#if !nonUniformX
	for(int i=0;i<mx_tot;i++) {
			x[i]=Lx*(0.5+i*1.0)/(mx_tot);  }

	dx = x[1] - x[0];

#endif

	dxv[0] = (x[1]+x[0])/2.0;
	for (int i=1; i<mx-1; i++) {
		dxv[i]   = (x[i+1]-x[i-1])/2.0;
	}
	dxv[mx-1] = Lx - (x[mx-1]+x[mx-2])/2.0;

	for(int j=0;j<my_tot;j++) {
		y[j]=Ly*(0.5+j*1.0)/(my_tot);  }

	for(int k=0;k<mz_tot;k++) {
		z[k]=Lz*(0.5+k*1.0)/(mz_tot);  }

	if(rk.rank==0) printf("grid read\n");

	mpiBarrier();
}

void derivGrid(double *d2f, double *df, double *f, double dx) {

	double fbound[mx_tot+stencilSize*2];
	for(int i=stencilSize; i<mx_tot+stencilSize; i++)
		fbound[i] = f[i-stencilSize];

	for(int i=0; i<stencilSize; i++) {
		fbound[i]   = - fbound[2*stencilSize-i-1];
		fbound[mx_tot+stencilSize+i] = 2*Lx - fbound[mx_tot+stencilSize-i-1];
	}

	for (int i = 0; i < mx_tot; i++){
		df[i]  = 0.0;
		d2f[i] = coeffS[stencilSize]*fbound[i+stencilSize]/dx/dx;
		for (int it = 0; it < stencilSize; it++){
			df[i]  += coeffF[it]*(fbound[i+it]-fbound[i+stencilSize*2-it])/dx;
			d2f[i] += coeffS[it]*(fbound[i+it]+fbound[i+stencilSize*2-it])/dx/dx;
		}
	}
}

void calcdt(Communicator rk) {
	double dx;
	double dy = y[1] - y[0];
	double dz = z[1] - z[0];
	double dx2;
	double dy2 = dy*dy;
	double dz2 = dz*dz;
	double dtConvInv = 0.0;
	double dtViscInv = 0.0;
	for (int gt = 0; gt<mx*my*mz; gt++) {
		double ien = e[gt]/r[gt] - 0.5*(u[gt]*u[gt] + v[gt]*v[gt] + w[gt]*w[gt]);
		double sos = pow(gamma*(gamma-1)*ien,0.5);

		int i = gt%my;

	    if(i==0) {
	    	dx = (x[i+1] + x[i])/2.0;
	    } else if (i==mx-1) {
	    	dx = Lx - (x[i] + x[i-1])/2.0;
	    } else {
	    	dx = (x[i+1] - x[i-1])/2.0;
	    }

	    dx2 = dx*dx;

	    double dtc1, dtv1;

	    dtc1      =  MAX( (abs(u[gt]) + sos)/dx, MAX( (abs(v[gt]) + sos)/dy, (abs(w[gt]) + sos)/dz) );
	    dtv1      =  MAX( 1.0/Re/dx2, MAX( 1.0/Re/dy2, 1.0/Re/dz2) );
		dtConvInv =  MAX( dtConvInv, dtc1 );
		dtViscInv =  MAX( dtViscInv, dtv1 );
	}
	dt = CFL/MAX(dtConvInv, dtViscInv);

	allReduceArrayDouble(&dt,1);
	mpiBarrier();
}
