#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include "mpi.h"
#include "globals.h"
#include "comm.h"

using namespace std;

double dt, h_dpdz;

double dx,x[mx],xp[mx],xpp[mx],y[my_tot],z[mz_tot];

double r[mx*my*mz];
double u[mx*my*mz];
double v[mx*my*mz];
double w[mx*my*mz];
double e[mx*my*mz];

int main(int argc, char** argv) {

	int myRank, nProcs;

	double begin = MPI_Wtime();

    int ierr;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    if(nProcs != pRow*pCol) {
    	if(myRank==0) {
    		printf("Error! -> nProcs different that pRow*pCol");
    	}
    	ierr = MPI_Barrier(MPI_COMM_WORLD);
    	exit(1);
    }

    Communicator rk;
    rk.myRank(myRank);
    splitComm(&rk);
    initGrid(rk);
    begin = MPI_Wtime();
    initField(801,rk);
    ierr = MPI_Barrier(MPI_COMM_WORLD);

//    if(restartFile<0) {
//    	if(forcing) {
//    		initChannel(rk);
//    	} else {
//    		initCHIT(rk);
//    	}
//    } else {
//    	initField(restartFile,rk); }
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

//
//    double end = MPI_Wtime();
//
//    cout << "The total time is: " << end - begin << "\n";
//
//    double timePerTimeStep = (end - begin)/(nfiles*nsteps);
//
//    cout << "The simulation time per time step is: " << timePerTimeStep << "\n";
//
//    cout << "Simulation is finished! \n";

    ierr = MPI_Finalize();
	return 0;
}
