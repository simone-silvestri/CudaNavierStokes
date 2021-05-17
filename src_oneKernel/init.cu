#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "globals.h"
#include "main.h"

void derivGrid(double *d2f, double *df, double *f, double dx);

void calcdt() {

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
}

void initFile(int timestep) {

	char str[80];
	size_t result;
	sprintf(str, "fields/r.%07d.bin",timestep);
	int lSize = mx*my*mz;

	FILE *fb = fopen(str,"rb");
	result = fread(r , sizeof(double) , mx*my*mz , fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	fclose(fb);

	sprintf(str, "fields/u.%07d.bin",timestep);
	fb = fopen(str,"rb");
	result = fread(u , sizeof(double) , mx*my*mz ,  fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	fclose(fb);

	sprintf(str, "fields/v.%07d.bin",timestep);
	fb = fopen(str,"rb");
	result = fread(v , sizeof(double) , mx*my*mz , fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	fclose(fb);

	sprintf(str, "fields/w.%07d.bin",timestep);
	fb = fopen(str,"rb");
	result = fread(w , sizeof(double) , mx*my*mz , fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	fclose(fb);

	sprintf(str, "fields/e.%07d.bin",timestep);
	fb = fopen(str,"rb");
	result = fread(e , sizeof(double) , mx*my*mz , fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	fclose(fb);

	fb = fopen("fields/dpdz","r");
	result = fscanf(fb,"%lf",&h_dpdz);
	fclose(fb);

}

void initGrid() {

	//constant grid in y and z and stretched in x

	dx = Lx*(1.0)/(mx);

	double xn[mx+1];
	for (int i=0; i<mx+1; i++)
		xn[i] = tanh(stretch*((i*1.0)/mx-0.5))/tanh(stretch*0.5);

	for (int i=0; i<mx; i++)
			x[i] = Lx * (1.0 + (xn[i] + xn[i+1])/2.0)/2.0;

	derivGrid(xpp,xp,x,dx);

	for (int i=0; i<mx; i++)
		xp[i] = 1.0/xp[i];

#if !nonUniformX
	for(int i=0;i<mx;i++) {
			x[i]=Lx*(0.5+i*1.0)/(mx);  }

	dx = x[1] - x[0];

#endif

	for(int j=0;j<my;j++) {
		y[j]=Ly*(0.5+j*1.0)/(my);  }

	for(int k=0;k<mz;k++) {
		z[k]=Lz*(0.5+k*1.0)/(mz);  }

	FILE *fp = fopen("Grid.txt","w+");
	for (int i=0; i<mx; i++)
		fprintf(fp,"%d %lf %lf %lf\n",i,x[i],xp[i],xpp[i]);
	fclose(fp);

	FILE *fb = fopen("fields/x.bin","wb");
	fwrite(x, mx , sizeof(double) , fb );
	fclose(fb);

	fb = fopen("fields/y.bin","wb");
	fwrite(y , my , sizeof(double) , fb );
	fclose(fb);

	fb = fopen("fields/z.bin","wb");
	fwrite(z , mz , sizeof(double) , fb );
	fclose(fb);
}

void initChannel() {

	double U0;
	double T0 = 1.0;
	double P0 = T0*Rgas;
	double R0 = 1.0;

	U0 = pow(gamma,0.5)*Ma;

	for (int i=0; i<mx; i++) {
		for (int j=0; j<my; j++) {
			for (int k=0; k<mz; k++) {
				double rr1 = rand()*1.0/(RAND_MAX*1.0) - 0.5;
				double rr2 = rand()*1.0/(RAND_MAX*1.0) - 0.5;
				double rr3 = rand()*1.0/(RAND_MAX*1.0) - 0.5;
				double ufluc  =  0.02*rr1;
				double vfluc  =  0.02*rr2;
				double wfluc  =  0.02*rr3;

				double wmean  = 1.5*U0*R0*x[i]*(1.0-x[i]/Lx);

				ufluc = ufluc + 0.05*sin(0.5*M_PI*x[i])*cos(2*M_PI*y[j]);
				vfluc = vfluc + 0.05*sin(0.5*M_PI*x[i])*sin(2*M_PI*y[j]);
				u[idx(i,j,k)] = ufluc;
				v[idx(i,j,k)] = vfluc;
				w[idx(i,j,k)] = wfluc + wmean;

				r[idx(i,j,k)] = R0;
				e[idx(i,j,k)] = P0/(gamma-1.0) + 0.5 * r[idx(i,j,k)] * (pow(u[idx(i,j,k)],2) + pow(v[idx(i,j,k)],2) + pow(w[idx(i,j,k)],2));
			} } }

}

void initCHIT() {

	double V0 = 1.0;
	double T0 = 1.0;
	double P0 = T0*Rgas;
	double R0 = 1.0;

	for (int i=0; i<mx; i++) {
		double fx = 2*M_PI*x[i]/Lx;
		for (int j=0; j<my; j++) {
			double fy = 2*M_PI*y[j]/Ly;
			for (int k=0; k<mz; k++) {
				double fz = 2*M_PI*z[k]/Lz;
				u[idx(i,j,k)] =  V0*sin(fx/1.0)*cos(fy/1.0)*cos(fz/1.0);
				v[idx(i,j,k)] = -V0*cos(fx/1.0)*sin(fy/1.0)*cos(fz/1.0);
				w[idx(i,j,k)] =  0.0;

				double press = P0 + 1.0/16.0*R0*V0*V0 * (cos(2.0*fx/1.0) + cos(2.0*fy/1.0)) * (cos(2.0*fz/1.0) + 2.0);

				r[idx(i,j,k)] = press/Rgas/T0;
				e[idx(i,j,k)] = press/(gamma-1.0) + 0.5 * r[idx(i,j,k)] * (pow(u[idx(i,j,k)],2) + pow(v[idx(i,j,k)],2) + pow(w[idx(i,j,k)],2));
			} } }
}

void writeFields(int timestep) {

	char str[80];
	sprintf(str, "fields/r.%07d.bin",timestep);

	FILE *fb = fopen(str,"wb");
	fwrite(r , mx*my*mz , sizeof(double) , fb );
	fclose(fb);

	sprintf(str, "fields/u.%07d.bin",timestep);
	fb = fopen(str,"wb");
	fwrite(u , mx*my*mz , sizeof(double) , fb );
	fclose(fb);

	sprintf(str, "fields/v.%07d.bin",timestep);
	fb = fopen(str,"wb");
	fwrite(v , mx*my*mz , sizeof(double) , fb );
	fclose(fb);

	sprintf(str, "fields/w.%07d.bin",timestep);
	fb = fopen(str,"wb");
	fwrite(w , mx*my*mz , sizeof(double) , fb );
	fclose(fb);

	sprintf(str, "fields/e.%07d.bin",timestep);
	fb = fopen(str,"wb");
	fwrite(e , mx*my*mz , sizeof(double) , fb );
	fclose(fb);

	fb = fopen("fields/dpdz","w+");
	fprintf(fb,"%lf",h_dpdz);
	fclose(fb);

}

void calcAvgChan() {
	double um[mx],vm[mx],wm[mx],rm[mx],em[mx];
	double uf[mx],vf[mx],wf[mx],rf[mx],ef[mx];

	FILE *fp = fopen("prof.txt","w+");
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
		fprintf(fp,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",x[i],rm[i],um[i],vm[i],wm[i],em[i],rf[i],uf[i],vf[i],wf[i],ef[i]);

	}
	fclose(fp);

}

void printRes() {

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
	printf("\n");
	printf("The average friction Reynolds number is: \t %lf\n",Ret);
	printf("Resolutions are: \n");
	printf("wall-normal: \t %lf\n",(x[0]+x[1])/2*Ret);
	printf("span-wise: \t %lf\n",(y[1]-y[0])*Ret);
	printf("stream-wise: \t %lf\n",(z[1]-z[0])*Ret);
	printf("\n");


	printf("the initial dt and dpdz are : %lf   and    %lf\n",dt,h_dpdz);
	printf("\n");

    FILE *fp = fopen("initial.txt","w+");
	for(int k=0; k<mz; k++)
		for(int i=0; i<mx; i++)
			fprintf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",x[i],z[k],r[idx(i,0,k)],u[idx(i,0,k)],v[idx(i,0,k)],w[idx(i,0,k)],e[idx(i,0,k)]);
	fclose(fp);
}

void derivGrid(double *d2f, double *df, double *f, double dx) {

	double fbound[mx+stencilSize*2];
	for(int i=stencilSize; i<mx+stencilSize; i++)
		fbound[i] = f[i-stencilSize];

	for(int i=0; i<stencilSize; i++) {
		fbound[i]   = - fbound[2*stencilSize-i-1];
		fbound[mx+stencilSize+i] = 2*Lx - fbound[mx+stencilSize-i-1];
	}

	for (int i = 0; i < mx; i++){
		df[i]  = 0.0;
		d2f[i] = coeffS[stencilSize]*fbound[i+stencilSize]/dx/dx;
		for (int it = 0; it < stencilSize; it++){
			df[i]  += coeffF[it]*(fbound[i+it]-fbound[i+stencilSize*2-it])/dx;
			d2f[i] += coeffS[it]*(fbound[i+it]+fbound[i+stencilSize*2-it])/dx/dx;
		}
	}
}
