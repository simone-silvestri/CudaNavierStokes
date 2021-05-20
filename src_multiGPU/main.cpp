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

    int ierr;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    double begin = MPI_Wtime();

    if(nProcs != pRow*pCol) {
    	if(myRank==0) {
    		printf("Error! -> nProcs different that pRow*pCol");
    	}
    	ierr = MPI_Barrier(MPI_COMM_WORLD);
    	exit(1);
    }

    //Initialize the 2D processor grid
    Communicator rk;
    splitComm(&rk,myRank);

    //Initialize the computational mesh
    initGrid(rk);

    //Initialize the solution field
    restartWrapper(restartFile,rk);

    //Output the initial field
    calcdt(rk);
    printRes(rk);
    calcAvgChan(rk);
//    writeField(restartFile+1,rk);

    //Set GPU parameters and copying the solution onto the GPU
    setGPUParameters(rk);
    initSolver();
    initHalo();
    copyField(0);

    //Run the solver
    solverWrapper(rk);

    double end = MPI_Wtime();
    cout << "The total time is: " << end - begin << "\n";
    double timePerTimeStep = (end - begin)/(nfiles*nsteps);
    cout << "The simulation time per time step is: " << timePerTimeStep << "\n";
    cout << "Simulation is finished! \n";

    ierr = MPI_Finalize();
	return 0;
}
