#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include "mpi.h"
#include <chrono>
#include "globals.h"

using namespace std;

double dt, h_dpdz;

double dx,x[mx],xp[mx],xpp[mx],y[my],z[mz];

double r[mx*my*mz];
double u[mx*my*mz];
double v[mx*my*mz];
double w[mx*my*mz];
double e[mx*my*mz];


int main(int argc, char** argv) {

	int myRank;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int ierr;

    ierr = MPI_Init(&argc, &argv);
    printf("Hello world\n");

    ierr = MPI_Finalize();


	// ==== Call function 'call_me_maybe' from CUDA file multiply.cu: ==========

	//    initGrid();
//    if(restartFile<0) {
//    	if(forcing) {
//    		initChannel();
//    	} else {
//    		initCHIT();
//    	}
//    } else {
//    	initFile(restartFile); }
//    calcdt();
//    writeFields(0);
//    printRes();
//    calcAvgChan();
//
//    cout <<"\n";
//    cout <<"code is running on -> GPU \n";
//    cout <<"\n";
//
//    setDerivativeParameters();
//    initSolver();
//    copyField(0);
//
//    myprec *dpar1, *dpar2, *dtime;
//    myprec *hpar1 = new myprec[nsteps];
//    myprec *hpar2 = new myprec[nsteps];
//    myprec *htime = new myprec[nsteps];
//
//    checkCuda( cudaMalloc((void**)&dpar1, nsteps*sizeof(myprec)) );
//    checkCuda( cudaMalloc((void**)&dpar2, nsteps*sizeof(myprec)) );
//    checkCuda( cudaMalloc((void**)&dtime, nsteps*sizeof(myprec)) );
//
//    // Increase GPU default limits to accomodate the computations
//    size_t rsize = 1024ULL*1024ULL*1024ULL*8ULL;  // allocate 10GB of HEAP (dynamic) memory size
//    cudaDeviceSetLimit(cudaLimitMallocHeapSize , rsize);
//
//    FILE *fp = fopen("solution.txt","w+");
//    for(int file = 1; file<nfiles+1; file++) {
//
//    	runSimulation(dpar1,dpar2,dtime);  //running the simulation on the GPU
//    	copyField(1);			  //copying back partial results to CPU
//
//    	writeFields(file);
//
//    	cudaDeviceSynchronize();
//
//    	checkCuda( cudaMemcpy(htime, dtime, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );
//    	checkCuda( cudaMemcpy(hpar1, dpar1, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );
//    	checkCuda( cudaMemcpy(hpar2, dpar2, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );
//
//    	calcAvgChan();
//
//    	printf("file number: %d  \t step: %d  \t time: %lf  \t kin: %lf  \t dpdz: %lf\n",file,file*nsteps,htime[nsteps-1],hpar1[nsteps-1],hpar2[nsteps-1]);
//    	for(int t=0; t<nsteps-1; t++)
//    		fprintf(fp,"%lf %lf %lf %lf\n",htime[t],hpar1[t],hpar2[t],htime[t+1]-htime[t]);
//    }
//    fclose(fp);
//
//    clearSolver();
//    cudaDeviceReset();
//
//
//    fp = fopen("final.txt","w+");
//    for(int k=0; k<mz; k++)
//    	for(int i=0; i<mx; i++)
//    		fprintf(fp,"%lf %lf %lf %lf %lf %lf %lf\n",x[i],z[k],r[idx(i,0,k)],u[idx(i,0,k)],v[idx(i,0,k)],w[idx(i,0,k)],e[idx(i,0,k)]);
//    fclose(fp);


    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::seconds>     (end - begin).count() << "[s]" << std::endl;

    double timePerTimeStep = ((double) (std::chrono::duration_cast<std::chrono::seconds>  (end - begin).count()))/(nfiles*nsteps);

    cout << "The simulation time per time step is: " << timePerTimeStep << "\n";

    cout << "Simulation is finished! \n";

    MPI_Finalize();
	return 0;
}


