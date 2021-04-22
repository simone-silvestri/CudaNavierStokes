#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "globals.h"
#include "main.h"

void calcdt() {

	double dx = x[1] - x[0];
	double dy = y[1] - y[0];
	double dz = z[1] - z[0];
	double dx2 = dx*dx;
	double dy2 = dy*dy;
	double dz2 = dz*dz;
	double dtConvInv = 0.0;
	double dtViscInv = 0.0;
	for (int gt = 0; gt<mx*my*mz; gt++) {
		double ien = e[gt]/r[gt] - 0.5*(u[gt]*u[gt] + v[gt]*v[gt] + w[gt]*w[gt]);
		double sos = pow(gamma*(gamma-1)*ien,0.5);

		dtConvInv =  MAX( dtConvInv, MAX( (abs(u[gt]) + sos)/dx, MAX( (abs(v[gt]) + sos)/dy, (abs(w[gt]) + sos)/dz) ) );
		dtViscInv =  MAX( dtViscInv, MAX( 1.0/Re/dx2, MAX( 1.0/Re/dy2, 1.0/Re/dz2) ) );
	}
	dt = CFL/MAX(dtConvInv, dtViscInv);

	printf("this is the dt %lf\n",dt);

}


//void initFile(int timestep) {
//
//	char str[80];
//	sprintf(str, "fields/r.%07d.bin",timestep);
//
//	FILE *fb = fopen(str,"wb");
//	fread(r , mx*my*mz , sizeof(str) , fb );
//	fclose(fb);
//
//	sprintf(str, "fields/u.%07d.bin",timestep);
//	fb = fopen(str,"wb");
//	fread(u , mx*my*mz , sizeof(str) , fb );
//	fclose(fb);
//
//	sprintf(str, "fields/v.%07d.bin",timestep);
//	fb = fopen(str,"wb");
//	fread(v , mx*my*mz , sizeof(str) , fb );
//	fclose(fb);
//
//	sprintf(str, "fields/w.%07d.bin",timestep);
//	fb = fopen(str,"wb");
//	fread(w , mx*my*mz , sizeof(str) , fb );
//	fclose(fb);
//
//	sprintf(str, "fields/e.%07d.bin",timestep);
//	fb = fopen(str,"wb");
//	fread(e , mx*my*mz , sizeof(str) , fb );
//	fclose(fb);
//
//
//}


void initGrid() {

	double stretch = 4.0;

	//constant grid in y and z and stretched in x
	for(int i=0;i<mx;i++) {
		double fact    =  (i*1.0)/(mx) - 0.5;
    	x[i]  =  0.5*(1.0+tanh(stretch*fact)/tanh(stretch*0.5))*Lx;
       xp[i]  =  1./(0.5*stretch/tanh(stretch/2.)/(cosh(stretch*fact)*cosh(stretch*fact)));
      xpp[i]  = - stretch*stretch*tanh(stretch*fact)/tanh(stretch/2.)/(cosh(stretch*fact)*cosh(stretch*fact));
	}
	double bias = (Lx - x[mx-1])/2.0;
	for(int i=0;i<mx;i++) {
		x[i] = x[i] + bias;
	}

	dx = Lx*(1.0)/(mx);

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

    FILE *fp = fopen("initial.txt","w+");
	for(int k=0; k<mz; k++)
		for(int i=0; i<mx; i++)
			fprintf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",x[i],z[k],r[idx(i,0,k)],u[idx(i,0,k)],v[idx(i,0,k)],w[idx(i,0,k)],e[idx(i,0,k)]);
	fclose(fp);
	writeFields(0);
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

    FILE *fp = fopen("initial.txt","w+");
	for(int k=0; k<mz; k++)
		for(int i=0; i<mx; i++)
			fprintf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",x[i],z[k],r[idx(i,0,k)],u[idx(i,0,k)],v[idx(i,0,k)],w[idx(i,0,k)],e[idx(i,0,k)]);
	fclose(fp);
	writeFields(0);
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
}
