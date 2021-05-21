#include "globals.h"
#include "comm.h"
#include "mpi.h"
#include "main.h"
#include <unistd.h>

#define CHECK_ERR(func) { \
    if (ierr != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(ierr, errorString, &errorStringLen); \
        printf("Error at line %d: calling %s (%s)\n",__LINE__, #func, errorString); \
    } \
}

myprec *senYp,*senYm,*senZp,*senZm;
myprec *rcvYp,*rcvYm,*rcvZp,*rcvZm;

myprec *senYp5,*senYm5,*senZp5,*senZm5;
myprec *rcvYp5,*rcvYm5,*rcvZp5,*rcvZm5;

void updateHalo(myprec *var, Communicator rk) {

	int ierr;
	MPI_Status status;

	fillBoundaries(senYm,senYp,senZm,senZp,var,0);

	ierr = MPI_Sendrecv(senYm, mz*mx*stencilSize, MPI_myprec, rk.jp, 0,
					    rcvYm, mz*mx*stencilSize, MPI_myprec, rk.jm, 0,
						 MPI_COMM_WORLD,&status);
	ierr = MPI_Sendrecv(senYp, mz*mx*stencilSize, MPI_myprec, rk.jm, 0,
			            rcvYp, mz*mx*stencilSize, MPI_myprec, rk.jp, 0,
						 MPI_COMM_WORLD,&status);

	ierr = MPI_Sendrecv(senZm, my*mx*stencilSize, MPI_myprec, rk.kp, 0,
					    rcvZm, my*mx*stencilSize, MPI_myprec, rk.km, 0,
						 MPI_COMM_WORLD,&status);
	ierr = MPI_Sendrecv(senZp, my*mx*stencilSize, MPI_myprec, rk.km, 0,
					    rcvZp, my*mx*stencilSize, MPI_myprec, rk.kp, 0,
						 MPI_COMM_WORLD,&status);

	fillBoundaries(rcvYm,rcvYp,rcvZm,rcvZp,var,1);
}

void updateHaloFive(myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, Communicator rk) {
	int ierr;
	MPI_Status status;
	fillBoundariesFive(senYm5,senYp5,senZm5,senZp5,r,u,v,w,e,0);

	ierr = MPI_Sendrecv(senYm5, 5*mz*mx*stencilSize, MPI_myprec, rk.jp, 0,
					    rcvYm5, 5*mz*mx*stencilSize, MPI_myprec, rk.jm, 0,
						 MPI_COMM_WORLD,&status);
	ierr = MPI_Sendrecv(senYp5, 5*mz*mx*stencilSize, MPI_myprec, rk.jm, 0,
			            rcvYp5, 5*mz*mx*stencilSize, MPI_myprec, rk.jp, 0,
						 MPI_COMM_WORLD,&status);

	ierr = MPI_Sendrecv(senZm5, 5*my*mx*stencilSize, MPI_myprec, rk.kp, 0,
					    rcvZm5, 5*my*mx*stencilSize, MPI_myprec, rk.km, 0,
						 MPI_COMM_WORLD,&status);
	ierr = MPI_Sendrecv(senZp5, 5*my*mx*stencilSize, MPI_myprec, rk.km, 0,
					    rcvZp5, 5*my*mx*stencilSize, MPI_myprec, rk.kp, 0,
						 MPI_COMM_WORLD,&status);

	fillBoundariesFive(rcvYm5,rcvYp5,rcvZm5,rcvZp5,r,u,v,w,e,1);
}

long int nameToHash(char *name, int length) {
	long int hash = 0;
	for (int i=0; i<length; i++) {
		hash += (atoi(&name[i])+1)*pow(10,i);
	}
	return hash;
}

void splitComm(Communicator *rk, int myRank) {

    rk->myRank(myRank);

	const int dimens[]  = {pRow,pCol};
	const int periods[] = {1,1};
	int coord[2],cYPos[2],cYNeg[2],cZPos[2],cZNeg[2];
	int ierr;

	MPI_Comm comm_cart;

	ierr = MPI_Cart_create(MPI_COMM_WORLD, 2, dimens, periods, 0, &comm_cart);
	ierr = MPI_Cart_coords(comm_cart, rk->rank, 2, coord);

	cZPos[0] = coord[0];
	cZNeg[0] = coord[0];
	cYPos[1] = coord[1];
	cYNeg[1] = coord[1];

	cZPos[1] = coord[1]+1;
	if(cZPos[1]>pCol-1)
		cZPos[1] = 0;

	cZNeg[1] = coord[1]-1;
	if(cZNeg[1]<0)
		cZNeg[1] = pCol-1;

	cYPos[0] = coord[0]+1;
	if(cYPos[0]>pRow-1)
		cYPos[0] = 0;

	cYNeg[0] = coord[0]-1;
	if(cYNeg[0]<0)
		cYNeg[0] = pRow-1;

	ierr = MPI_Cart_rank(comm_cart, cZPos, &rk->kp);
	ierr = MPI_Cart_rank(comm_cart, cZNeg, &rk->km);
	ierr = MPI_Cart_rank(comm_cart, cYPos, &rk->jp);
	ierr = MPI_Cart_rank(comm_cart, cYNeg, &rk->jm);

	rk->jstart = coord[0]*my;
	rk->jend   = rk->jstart + my;
	rk->kstart = coord[1]*mz;
	rk->kend   = rk->kstart + mz;

	long int hash;
	MPI_Comm comm_node;
    int procnamesize;
	char procname[MPI_MAX_PROCESSOR_NAME];

	ierr = MPI_Get_processor_name(procname, &procnamesize);

    hash = nameToHash(procname, procnamesize);

    ierr = MPI_Comm_split(MPI_COMM_WORLD, hash, rk->rank, &comm_node);
    ierr = MPI_Comm_rank(comm_node , &rk->nodeRank);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    printf("rank number %d, gpu number %d\n",rk->rank,rk->nodeRank);
}

void saveFileMPI(char filename, int timestep,  double *var, Communicator rk) {

	int start_indices[3];
	int ierr;

	MPI_Status status;
	MPI_Offset offset = 0;
	int cmode, omode;
    MPI_File fh;
    MPI_Info info;

	char str[80];
	size_t result;
	sprintf(str, "fields/%c.%07d.bin",filename,timestep);

	const int gsizes[] = {mz_tot, my_tot, mx_tot};
	const int lsizes[] = {mz    , my    , mx    };
	MPI_Datatype view, array;

	start_indices[0] = rk.kstart;
	start_indices[1] = rk.jstart;
	start_indices[2] = 0;

	ierr = MPI_Type_create_subarray(3, gsizes, lsizes, start_indices, MPI_ORDER_C, MPI_DOUBLE, &view);
	ierr = MPI_Type_commit(&view);

	start_indices[0] = 0;
	start_indices[1] = 0;
	start_indices[2] = 0;

	ierr = MPI_Type_create_subarray(3, lsizes, lsizes, start_indices, MPI_ORDER_C, MPI_DOUBLE, &array);
	ierr = MPI_Type_commit(&array);


    /* Users can set customized I/O hints in info object */
    info = MPI_INFO_NULL;  /* no user I/O hint */

    /* set file open mode */
    cmode  = MPI_MODE_CREATE; /* to create a new file */
    cmode |= MPI_MODE_WRONLY; /* with write-only permission */

    /* collectively open a file, shared by all processes in MPI_COMM_WORLD */
    ierr = MPI_File_open(MPI_COMM_WORLD, str, cmode, info, &fh);

    ierr = MPI_File_set_view(fh, offset, MPI_DOUBLE, view, "native", MPI_INFO_NULL);
    ierr = MPI_File_write_all(fh, var, 1, array, &status);

    ierr = MPI_File_close(&fh);
    CHECK_ERR(MPI_File_close);
}

void readFileMPI(char filename, int timestep,  double *var, Communicator rk) {
	double *var_tot = new double[mx_tot*my_tot*mz_tot];
	int ierr;
	int lSize = mx_tot*my_tot*mz_tot;

	if(rk.rank==0) {
		char str[80];
		size_t result;
		sprintf(str, "fields/%c.%07d.bin",filename,timestep);
		FILE *fb = fopen(str,"rb");
		result = fread(var_tot , sizeof(myprec) , lSize , fb );
		fclose(fb);
	}
	ierr = MPI_Barrier(MPI_COMM_WORLD);

	ierr = MPI_Bcast(var_tot, lSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	ierr = MPI_Barrier(MPI_COMM_WORLD);
    for (int k=rk.kstart; k<rk.kend; k++)
    	for (int j=rk.jstart; j<rk.jend; j++)
    		for (int i=0; i<mx; i++)
    			var[i + (j-rk.jstart)*mx + (k-rk.kstart)*mx*my] = var_tot[i + j * mx_tot + k * mx_tot * my_tot];
    delete [] var_tot;
}

void reduceArray(int rcvCore, myprec *sendArr, int sizeArr, Communicator rk) {
	int ierr;
	myprec tmpArr[sizeArr];

	ierr = MPI_Reduce(sendArr, tmpArr, sizeArr, MPI_myprec, MPI_SUM, rcvCore, MPI_COMM_WORLD);

	if(rk.rank == rcvCore) {
		for(int i=0; i<sizeArr; i++)
			sendArr[i] = tmpArr[i]/(pRow*pCol);
	}
	ierr = MPI_Barrier(MPI_COMM_WORLD);
}

void allReduceToMin(myprec *sendArr, int sizeArr) {
	int ierr;
	myprec tmpArr[sizeArr];

	ierr = MPI_Allreduce(sendArr, tmpArr, sizeArr, MPI_myprec, MPI_MIN, MPI_COMM_WORLD);
	for(int i=0; i<sizeArr; i++)
		sendArr[i] = tmpArr[i];
	ierr = MPI_Barrier(MPI_COMM_WORLD);
}

void allReduceArray(myprec *sendArr, int sizeArr) {
	int ierr;
	myprec tmpArr[sizeArr];

	ierr = MPI_Allreduce(sendArr, tmpArr, sizeArr, MPI_myprec, MPI_SUM, MPI_COMM_WORLD);

	for(int i=0; i<sizeArr; i++)
		sendArr[i] = tmpArr[i]/(pRow*pCol);
	ierr = MPI_Barrier(MPI_COMM_WORLD);
}

void allReduceArrayDouble(double *sendArr, int sizeArr) {
	int ierr;
	double tmpArr[sizeArr];

	ierr = MPI_Allreduce(sendArr, tmpArr, sizeArr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	for(int i=0; i<sizeArr; i++)
		sendArr[i] = tmpArr[i]/(pRow*pCol);
	ierr = MPI_Barrier(MPI_COMM_WORLD);
}

void allReduceSum(myprec *sendArr, int sizeArr) {
	int ierr;
	myprec tmpArr[sizeArr];

	ierr = MPI_Allreduce(sendArr, tmpArr, sizeArr, MPI_myprec, MPI_SUM, MPI_COMM_WORLD);

	for(int i=0; i<sizeArr; i++)
		sendArr[i] = tmpArr[i];
	ierr = MPI_Barrier(MPI_COMM_WORLD);
}

void broadCastArray(int bcstCore, double *sendArr, int sizeArr, Communicator rk) {
	int ierr;

	ierr = MPI_Bcast(sendArr, sizeArr, MPI_DOUBLE, bcstCore, MPI_COMM_WORLD);
	ierr = MPI_Barrier(MPI_COMM_WORLD);
}

void mpiBarrier() {
	int ierr = MPI_Barrier(MPI_COMM_WORLD);
}


