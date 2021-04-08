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

double x[mx],y[my],z[mz];

double r[mx*my*mz];
double u[mx*my*mz];
double v[mx*my*mz];
double w[mx*my*mz];
double e[mx*my*mz];

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
	writeFields(2);

	cudaDeviceSynchronize();
#endif

	FILE *fp = fopen("final.txt","w+");
	for(int k=0; k<mz; k++)
		for(int i=0; i<mx; i++)
			fprintf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",x[i],z[k],r[idx(i,0,k)],u[idx(i,0,k)],v[idx(i,0,k)],w[idx(i,0,k)],e[idx(i,0,k)]);
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
	double dtConvInv = 0.0;
	double dtViscInv = 0.0;
	for (int gt = 0; gt<mx*my*mz; gt++) {
		double ien = e[gt]/r[gt] - 0.5*(u[gt]*u[gt] + v[gt]*v[gt] + w[gt]*w[gt]);
		double sos = pow(gamma*(gamma-1)*ien,0.5);

		dtConvInv =  MAX( dtConvInv, MAX( (abs(u[gt]) + sos)/dx, MAX( (abs(v[gt]) + sos)/dy, (abs(w[gt]) + sos)/dz) ) );
		dtViscInv =  0.0; //MAX( dtViscInv, MAX( 1.0/Re/dx2, MAX( 1.0/Re/dy2, 1.0/Re/dz2) ) );
	}
	dt = CFL/MAX(dtConvInv, dtViscInv);

	printf("this is the dt %lf\n",dt);

}


void initProfile() {

	for(int i=0;i<mx;i++) {
		x[i]=Lx*(0.5+i*1.0)/(mx);  }

	for(int j=0;j<my;j++) {
		y[j]=Ly*(0.5+j*1.0)/(my);  }

	for(int k=0;k<mz;k++) {
		z[k]=Lz*(0.5+k*1.0)/(mz);  }

	double V0 = 1.0;
	double T0 = 1.0;
	double P0 = T0*Rgas;
	double R0 = 1.0;


	for (int i=0; i<mx; i++) {
		double fx = x[i];
		for (int j=0; j<my; j++) {
			double fy = y[j];
			for (int k=0; k<mz; k++) {
				double fz = z[k];
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
	fwrite(r , mx*my*mz , sizeof(str) , fb );
	fclose(fb);

	sprintf(str, "fields/u.%07d.bin",timestep);
	fb = fopen(str,"wb");
	fwrite(u , mx*my*mz , sizeof(str) , fb );
	fclose(fb);

	sprintf(str, "fields/v.%07d.bin",timestep);
	fb = fopen(str,"wb");
	fwrite(v , mx*my*mz , sizeof(str) , fb );
	fclose(fb);

	sprintf(str, "fields/w.%07d.bin",timestep);
	fb = fopen(str,"wb");
	fwrite(w , mx*my*mz , sizeof(str) , fb );
	fclose(fb);

	sprintf(str, "fields/e.%07d.bin",timestep);
	fb = fopen(str,"wb");
	fwrite(e , mx*my*mz , sizeof(str) , fb );
	fclose(fb);
}


