#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "../src/globals.h"
#include "../src/main.h"

void derivGrid(double *d2f, double *df, double *f, double dx);

extern double dxv[mx];

void initFile(int timestep) {

	char str[80];
	size_t result;
	sprintf(str, "../fields/r.%07d.bin",timestep);
	int lSize = mx*my*mz;

	FILE *fb = fopen(str,"rb");
	result = fread(r , sizeof(double) , mx*my*mz , fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	fclose(fb);

	sprintf(str, "../fields/u.%07d.bin",timestep);
	fb = fopen(str,"rb");
	result = fread(u , sizeof(double) , mx*my*mz ,  fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	fclose(fb);

	sprintf(str, "../fields/v.%07d.bin",timestep);
	fb = fopen(str,"rb");
	result = fread(v , sizeof(double) , mx*my*mz , fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	fclose(fb);

	sprintf(str, "../fields/w.%07d.bin",timestep);
	fb = fopen(str,"rb");
	result = fread(w , sizeof(double) , mx*my*mz , fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
	fclose(fb);

	sprintf(str, "../fields/e.%07d.bin",timestep);
	fb = fopen(str,"rb");
	result = fread(e , sizeof(double) , mx*my*mz , fb );
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
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

	  dxv[0] = (x[1]+x[0])/2.0;
	  for (int i=1; i<mx-1; i++) {
		  dxv[i]   = (x[i+1]-x[i-1])/2.0;
	  }
	  dxv[mx-1] = Lx - (x[mx-1]+x[mx-2])/2.0;

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
			    	ub[i] = w[i-stencilSize];
			    for (int i=0; i<stencilSize; i++)
			    	ub[i] = w[stencilSize-i-1];

			    double dudx = 0;
			    for (int i=0; i<stencilSize; i++)
			    	dudx += coeffF[i]*(ub[i]-ub[stencilSize*2-i])/dx;

			    dudx *= xp[0];

			    ut  = sqrt(muw*abs(dudx)/r[idx(0,j,k)]);
			    Ret += ut*r[idx(0,j,k)]/muw;
		}

	Ret = Ret/my/mz;
	printf("The average friction Reynolds number is: \t %lf\n",Ret);
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
