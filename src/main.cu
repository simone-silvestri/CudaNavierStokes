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

double dx,x[mx],xp[mx],xpp[mx],y[my],z[mz];

double r[mx*my*mz];
double u[mx*my*mz];
double v[mx*my*mz];
double w[mx*my*mz];
double e[mx*my*mz];


int main(int argc, char** argv) {

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	initGrid();
	initFile(200);
//	initChannel();
	calcdt();
	writeFields(0);

	dim3 grid, block;

	cout <<"\n";
	cout <<"code is running on -> GPU \n";
	cout <<"\n";

	setDerivativeParameters(grid, block);

	myprec *dkin, *denst, *dtime;
	myprec *hkin  = new myprec[nsteps];
	myprec *henst = new myprec[nsteps];
	myprec *htime = new myprec[nsteps];

	checkCuda( cudaMalloc((void**)&dkin , nsteps*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&denst, nsteps*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&dtime, nsteps*sizeof(myprec)) );

	copyField(0);  //Initializing solution on the GPU

	/* Increase GPU default limits to accomodate the computations */

	size_t rsize = 1024ULL*1024ULL*1024ULL*8ULL;  // allocate 10GB of HEAP (dynamic) memory size
	cudaDeviceSetLimit(cudaLimitMallocHeapSize     , rsize);

//	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 4); // allow up to 5 nesteg grids to run cocurrently

	//cudaSetDevice(1);

    FILE *fp = fopen("initial.txt","w+");
	for(int k=0; k<mz; k++)
		for(int i=0; i<mx; i++)
			fprintf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",x[i],z[k],r[idx(i,0,k)],u[idx(i,0,k)],v[idx(i,0,k)],w[idx(i,0,k)],e[idx(i,0,k)]);
	fclose(fp);

	fp = fopen("solution.txt","w+");
	for(int file = 1; file<nfiles+1; file++) {

	    runDevice<<<grid,block>>>(dkin,denst,dtime);  //running the simulation on the GPU
		copyField(1);			  //copying back partial results to CPU

		writeFields(file);

		cudaDeviceSynchronize();

		checkCuda( cudaMemcpy(htime, dtime, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );
	    checkCuda( cudaMemcpy(henst, denst, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );
	    checkCuda( cudaMemcpy(hkin , dkin , nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );

//	    checkGpuMem();
	    printf("file number: %d\t step: %d\t time: %lf\t kin: %lf\t dpdz: %lf\n",file,file*nsteps,htime[nsteps-1],hkin[nsteps-1],henst[nsteps-1]);
		for(int t=0; t<nsteps-1; t++)
				fprintf(fp,"%lf %lf %lf %lf\n",htime[t],hkin[t],henst[t],htime[t+1]-htime[t]);

		FILE *fp2 = fopen("final.txt","w+");
		for(int k=0; k<mz; k++)
			for(int i=0; i<mx; i++)
				fprintf(fp2,"%lf %lf %lf %lf %lf %lf %lf\n",x[i],z[k],r[idx(i,0,k)],u[idx(i,0,k)],v[idx(i,0,k)],w[idx(i,0,k)],e[idx(i,0,k)]);
		fclose(fp2);


	}
	fclose(fp);
	cudaDeviceReset();

	fp = fopen("final.txt","w+");
	for(int k=0; k<mz; k++)
		for(int i=0; i<mx; i++)
			fprintf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",x[i],z[k],r[idx(i,0,k)],u[idx(i,0,k)],v[idx(i,0,k)],w[idx(i,0,k)],e[idx(i,0,k)]);
	fclose(fp);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::seconds>     (end - begin).count() << "[s]" << std::endl;

	double timePerTimeStep = ((double) (std::chrono::duration_cast<std::chrono::seconds>  (end - begin).count()))/(nfiles*nsteps);

	cout << "The simulation time per time step is: " << timePerTimeStep << "\n";

	cout << "Simulation is finished! \n";

	return 0;
}


