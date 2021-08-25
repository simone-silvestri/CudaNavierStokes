#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "comm.h"
#include "globals.h"
#include "main.h"

void derivGrid(double *d2f, double *df, double *f, double dx);

void initField(int timestep, Communicator rk) {
	readFileMPI('r',timestep,r,rk);
	readFileMPI('u',timestep,u,rk);
	readFileMPI('v',timestep,v,rk);
	readFileMPI('w',timestep,w,rk);
	readFileMPI('e',timestep,e,rk);
    if(rk.rank==0) {
    	printf("finished initializing field\n");
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
	int denom = mx_tot;
	myprec denom2 = 2.0;
	if(boundaryLayer) {
		denom *= 2;
		denom2/= 2;
	}

	for (int i=0; i<mx_tot+1; i++)
		xn[i] = tanh(stretch*((i*1.0)/denom-0.5))/tanh(stretch*0.5);

	for (int i=0; i<mx_tot; i++)
		x[i] = Lx * (1.0 + (xn[i] + xn[i+1])/2.0)/denom2;

	derivGrid(xpp,xp,x,dx);

	for (int i=0; i<mx_tot; i++)
		xp[i] = 1.0/xp[i];

#if !nonUniformX
	for(int i=0;i<mx_tot;i++) {
			x[i]=Lx*(0.5+i*1.0)/(mx_tot);  }

	dx = x[1] - x[0];

#endif

	for(int j=0;j<my_tot;j++) {
		y[j]=Ly*(0.5+j*1.0)/(my_tot);  }

	for(int k=0;k<mz_tot;k++) {
		z[k]=Lz*(0.5+k*1.0)/(mz_tot);  }

	if(rk.rank==0) {

		FILE *fp = fopen("Grid.txt","w+");
		for (int i=0; i<mx; i++)
			fprintf(fp,"%d %lf %lf %lf\n",i,x[i],xp[i],xpp[i]);
		fclose(fp);

		FILE *fb = fopen("fields/x.bin","wb");
		fwrite(x, mx_tot , sizeof(double) , fb );
		fclose(fb);

		fb = fopen("fields/y.bin","wb");
		fwrite(y , my_tot , sizeof(double) , fb );
		fclose(fb);

		fb = fopen("fields/z.bin","wb");
		fwrite(z , mz_tot , sizeof(double) , fb );
		fclose(fb);
	}

	mpiBarrier();
}

void initChannel(Communicator rk) {

	double U0;
	double T0 = 1.0;
	double P0 = T0*Rgas;
	double R0 = 1.0;

	U0 = pow(gam,0.5)*Ma;

	for (int i=0; i<mx; i++) {
		for (int j=0; j<my; j++) {
			for (int k=0; k<mz; k++) {
				double rr1 = rand()*1.0/(RAND_MAX*1.0) - 0.5;
				double rr2 = rand()*1.0/(RAND_MAX*1.0) - 0.5;
				double rr3 = rand()*1.0/(RAND_MAX*1.0) - 0.5;
//				double ufluc  =  0.02*rr1;
//				double vfluc  =  0.02*rr2;
//				double wfluc  =  0.02*rr3;

				double wmean  = 1.5*U0*R0*x[i]*(1.0-x[i]/Lx);

//				ufluc = ufluc + 0.05*sin(0.5*M_PI*x[i])*cos(2*M_PI*y[j+rk.jstart]);
//				vfluc = vfluc + 0.05*sin(0.5*M_PI*x[i])*sin(2*M_PI*y[j+rk.jstart]);
				u[idx(i,j,k)] = 0.0; //ufluc;
				v[idx(i,j,k)] = 0.0; //vfluc;
				w[idx(i,j,k)] = wmean; //+ wfluc;

				r[idx(i,j,k)] = R0;
				e[idx(i,j,k)] = P0/(gam-1.0) + 0.5 * r[idx(i,j,k)] * (pow(u[idx(i,j,k)],2) + pow(v[idx(i,j,k)],2) + pow(w[idx(i,j,k)],2));
			} } }
}

void initCHIT(Communicator rk) {

	double V0 = 1.0;
	double T0 = 1.0;
	double P0 = T0*Rgas;
	double R0 = 1.0;

	for (int i=0; i<mx; i++) {
		double fx = 2*M_PI*x[i]/Lx;
		for (int j=0; j<my; j++) {
			double fy = 2*M_PI*y[j+rk.jstart]/Ly;
			for (int k=0; k<mz; k++) {
				double fz = 2*M_PI*z[k+rk.kstart]/Lz;
				u[idx(i,j,k)] =  V0*sin(fx/1.0)*cos(fy/1.0)*cos(fz/1.0);
				v[idx(i,j,k)] = -V0*cos(fx/1.0)*sin(fy/1.0)*cos(fz/1.0);
				w[idx(i,j,k)] =  0.0;

				double press = P0 + 1.0/16.0*R0*V0*V0 * (cos(2.0*fx/1.0) + cos(2.0*fy/1.0)) * (cos(2.0*fz/1.0) + 2.0);

				r[idx(i,j,k)] = press/Rgas/T0;
				e[idx(i,j,k)] = press/(gam-1.0) + 0.5 * r[idx(i,j,k)] * (pow(u[idx(i,j,k)],2) + pow(v[idx(i,j,k)],2) + pow(w[idx(i,j,k)],2));
			} } }
}

void calcAvgChan(Communicator rk) {
	double um[mx],vm[mx],wm[mx],rm[mx],em[mx];
	double uf[mx],vf[mx],wf[mx],rf[mx],ef[mx];

	for (int i=0; i<mx; i++) {
		rm[i] = 0.0;
		um[i] = 0.0;
		vm[i] = 0.0;
		wm[i] = 0.0;
		em[i] = 0.0;
		rf[i] = 0.0;
		uf[i] = 0.0;
		vf[i] = 0.0;
		wf[i] = 0.0;
		ef[i] = 0.0;
		for (int k=0; k<mz; k++)
			for (int j=0; j<my; j++) {
				rm[i] += r[idx(i,j,k)]/my/mz;
				um[i] += r[idx(i,j,k)]*u[idx(i,j,k)]/my/mz;
				vm[i] += r[idx(i,j,k)]*v[idx(i,j,k)]/my/mz;
				wm[i] += r[idx(i,j,k)]*w[idx(i,j,k)]/my/mz;
				em[i] += e[idx(i,j,k)]/my/mz;
			}
	}

	allReduceArrayDouble(rm,mx);
	allReduceArrayDouble(um,mx);
	allReduceArrayDouble(vm,mx);
	allReduceArrayDouble(wm,mx);
	allReduceArrayDouble(em,mx);

	for (int i=0; i<mx; i++) {
		um[i] /= rm[i];
		vm[i] /= rm[i];
		wm[i] /= rm[i];
		for (int k=0; k<mz; k++)
			for (int j=0; j<my; j++) {
				rf[i] += (r[idx(i,j,k)]-rm[i])*(r[idx(i,j,k)]-rm[i])/my/mz;
				uf[i] += (u[idx(i,j,k)]-um[i])*(u[idx(i,j,k)]-um[i])/my/mz;
				vf[i] += (v[idx(i,j,k)]-vm[i])*(v[idx(i,j,k)]-vm[i])/my/mz;
				wf[i] += (w[idx(i,j,k)]-wm[i])*(w[idx(i,j,k)]-wm[i])/my/mz;
				ef[i] += (e[idx(i,j,k)]-em[i])*(e[idx(i,j,k)]-em[i])/my/mz;
			}
	}

	allReduceArrayDouble(rf,mx);
	allReduceArrayDouble(uf,mx);
	allReduceArrayDouble(vf,mx);
	allReduceArrayDouble(wf,mx);
	allReduceArrayDouble(ef,mx);

	if(rk.rank==0) {
		FILE *fp = fopen("prof.txt","w+");
		for (int i=0; i<mx; i++)
			fprintf(fp,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",x[i],rm[i],um[i],vm[i],wm[i],em[i],rf[i],uf[i],vf[i],wf[i],ef[i]);
		fclose(fp);
	}
	mpiBarrier();
}

void printRes(Communicator rk) {

	double ut;

	double Ret = 0.0;
	double muw,temw;

	for (int k=0; k<mz; k++)
		for (int j=0; j<my; j++) {
			    temw   = 1.0;
			    double suth = pow(temw,viscexp);
			    muw      = suth/Re;

			    double ub[stencilSize*2+1];
			    for (int i=stencilSize; i<stencilSize*2+1; i++)
			    	ub[i] = w[i-stencilSize + j*mx + k*my*mx];
			    for (int i=0; i<stencilSize; i++)
			    	ub[i] = w[stencilSize-i-1 + j*mx + k*my*mx];

			    double dudx = 0;
			    for (int i=0; i<stencilSize; i++)
			    	dudx += coeffF[i]*(ub[i]-ub[stencilSize*2-i])/dx;

			    dudx *= xp[0];

			    ut  = sqrt(muw*abs(dudx)/r[idx(0,j,k)]);
			    Ret += ut*r[idx(0,j,k)]/muw;
		}

	Ret = Ret/my/mz;

	allReduceArrayDouble(&Ret,1);

	if(rk.rank==0) {
		printf("\n");
		printf("The average friction Reynolds number is: \t %lf\n",Ret);
		printf("Resolutions are: \n");
		printf("wall-normal wall: \t %lf\n",(x[0]+x[1])/2*Ret);
		printf("wall-normal center: \t %lf\n",(x[mx/2+1]-x[mx/2-1])/2*Ret);
		printf("span-wise: \t %lf\n",(y[1]-y[0])*Ret);
		printf("stream-wise: \t %lf\n",(z[1]-z[0])*Ret);
		printf("\n");

		printf("the initial dt and dpdz are : %lf   and    %lf\n",dt,h_dpdz);
		printf("\n");
	}
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
		double sos = pow(gam*(gam-1)*ien,0.5);

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

void restartWrapper(int restartFile, Communicator rk) {
    if(restartFile<0) {
    	if(forcing) {
    		initChannel(rk);
    	} else {
    		if(!boundaryLayer) initCHIT(rk);
    	}
    } else {
    	initField(restartFile,rk); }
}
