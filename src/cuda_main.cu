#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_main.h"
#include "cuda_math.h"
#include "comm.h"
#include "sponge.h"


__device__ __constant__ myprec alpha[] = {0.    , -17./60., -5./12.};
__device__ __constant__ myprec beta[]  = {8./15.,   5./12.,  3./4. };

cudaStream_t s[8+nDivZ];

inline void calcRHS(myprec *rhsr, myprec *rhsu, myprec *rhsv, myprec *rhsw, myprec *rhse, Communicator rk) {
	calcState<<<grid0,block0,0,s[0]>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l,0); //here 0 means interior points
	derVelX<<<d_grid[0],d_block[0],0,s[1]>>>(d_u,d_v,d_w,gij[0],gij[1],gij[2]);
	derVelY<<<d_grid[3],d_block[3],0,s[2]>>>(d_u,d_v,d_w,gij[3],gij[4],gij[5]);
	for (int kNum=0; kNum<nDivZ; kNum++)
		derVelZ<<<d_grid[4],d_block[4],0,s[8+kNum]>>>(d_u,d_v,d_w,gij[6],gij[7],gij[8],kNum);
	if(multiGPU) {
		updateHaloFive(d_r,d_u,d_v,d_w,d_e,rk); cudaDeviceSynchronize();
		calcState<<<gridHalo,blockHalo,0,s[4]>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l,1); //here 1 means halo points
		derVelYBC<<<gridHaloY,blockHaloY,0,s[0]>>>(d_u,d_v,d_w,gij[3],gij[4],gij[5],0);  //here 0 means lower boundary (0-index)
		derVelZBC<<<gridHaloZ,blockHaloZ,0,s[1]>>>(d_u,d_v,d_w,gij[6],gij[7],gij[8],0);	//here 0 means lower boundary (0-index)
		derVelYBC<<<gridHaloY,blockHaloY,0,s[2]>>>(d_u,d_v,d_w,gij[3],gij[4],gij[5],1);	//here 1 means upper boundary (my-index)
		derVelZBC<<<gridHaloZ,blockHaloZ,0,s[3]>>>(d_u,d_v,d_w,gij[6],gij[7],gij[8],1);	//here 1 means upper boundary (mz-index)
	}
	cudaDeviceSynchronize();
	calcDil<<<grid0,block0>>>(d_dil,gij[0],gij[4],gij[8]);
	cudaDeviceSynchronize();
	if(multiGPU) deviceBlocker<<<grid0,block0,0,s[0]>>>();   //in order to hide the halo update with deviceRHSX (on stream s[0])
	deviceRHSX<<<d_grid[0],d_block[0],0,s[0]>>>(rhsr,rhsu,rhsv,rhsw,rhse,d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij[0],gij[1],gij[2],gij[3],gij[6],d_dil,dpdz,0);
	if(multiGPU) updateHalo(d_dil,rk);
	cudaDeviceSynchronize();
	deviceRHSY<<<d_grid[1],d_block[1]>>>(rhsr,rhsu,rhsv,rhsw,rhse,d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij[1],gij[3],gij[4],gij[5],gij[7],d_dil,dpdz,0);
	for (int kNum=0; kNum<nDivZ; kNum++)
		deviceRHSZ<<<d_grid[2],d_block[2],0,s[kNum]>>>(rhsr,rhsu,rhsv,rhsw,rhse,d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij[2],gij[5],gij[6],gij[7],gij[8],d_dil,dpdz,kNum);
	cudaDeviceSynchronize();
	if(boundaryLayer) addSponge<<<d_grid[0],d_block[0]>>>(rhsr,rhsu,rhsv,rhsw,rhse,d_r,d_u,d_v,d_w,d_e);
	cudaDeviceSynchronize();
}

void runSimulation(myprec *par1, myprec *par2, myprec *time, Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	myprec h_dt,h_dpdz;

	for (int istep = 0; istep < nsteps; istep++) {
		if(istep%checkCFLcondition==0) calcTimeStepPressGrad(istep,dtC,dpdz,&h_dt,&h_dpdz,rk);
		if(istep>0)  deviceSumOne<<<1,1>>>(&time[istep],&time[istep-1] ,dtC);
		if(istep==0) deviceSumOne<<<1,1>>>(&time[istep],&time[nsteps-1],dtC);
		deviceAdvanceTime<<<1,1>>>(dtC);
		if(istep%checkBulk==0) calcBulk(&par1[istep],&par2[istep],d_r,d_u,d_v,d_w,d_e,rk);

		deviceMul<<<grid0,block0,0,s[0]>>>(d_uO,d_r,d_u);
		deviceMul<<<grid0,block0,0,s[1]>>>(d_vO,d_r,d_v);
		deviceMul<<<grid0,block0,0,s[2]>>>(d_wO,d_r,d_w);
		deviceCpy<<<grid0,block0,0,s[3]>>>(d_rO,d_r);
		deviceCpy<<<grid0,block0,0,s[4]>>>(d_eO,d_e);

		//Starting the Runge-Kutta Steps

		//runge kutta step 1
		calcRHS(d_rhsr1,d_rhsu1,d_rhsv1,d_rhsw1,d_rhse1,rk);
		eulerSum<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,dtC);
		eulerSum<<<grid0,block0>>>(d_e,d_eO,d_rhse1,dtC);
		cudaDeviceSynchronize();
		eulerSumR<<<grid0,block0,0,s[0]>>>(d_u,d_uO,d_rhsu1,d_r,dtC);
		eulerSumR<<<grid0,block0,0,s[1]>>>(d_v,d_vO,d_rhsv1,d_r,dtC);
		eulerSumR<<<grid0,block0,0,s[2]>>>(d_w,d_wO,d_rhsw1,d_r,dtC);
		cudaDeviceSynchronize();

		if(multiGPU) {  //To initiate slowly the routines so that we have time to initiate the memory transfer
			deviceCpy<<<grid0,block0,0,s[0]>>>(d_r,d_r);
			deviceCpy<<<grid0,block0,0,s[1]>>>(d_u,d_u);
			deviceCpy<<<grid0,block0,0,s[2]>>>(d_v,d_v);
			deviceCpy<<<grid0,block0,0,s[3]>>>(d_w,d_w); }

		//runge kutta step 2
		calcRHS(d_rhsr2,d_rhsu2,d_rhsv2,d_rhsw2,d_rhse2,rk);
		eulerSum3<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,dtC);
		eulerSum3<<<grid0,block0>>>(d_e,d_eO,d_rhse1,d_rhse2,dtC);
		cudaDeviceSynchronize();
		eulerSum3R<<<grid0,block0,0,s[0]>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_r,dtC);
		eulerSum3R<<<grid0,block0,0,s[1]>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_r,dtC);
		eulerSum3R<<<grid0,block0,0,s[2]>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_r,dtC);
		cudaDeviceSynchronize();

		if(multiGPU) {  //To initiate slowly the routines so that we have time to initiate the memory transfer
			deviceCpy<<<grid0,block0,0,s[0]>>>(d_r,d_r);
			deviceCpy<<<grid0,block0,0,s[1]>>>(d_u,d_u);
			deviceCpy<<<grid0,block0,0,s[2]>>>(d_v,d_v);
			deviceCpy<<<grid0,block0,0,s[3]>>>(d_w,d_w); }

		//runge kutta step 3
		calcRHS(d_rhsr3,d_rhsu3,d_rhsv3,d_rhsw3,d_rhse3,rk);
		rk3final<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,d_rhsr3,dtC);
		rk3final<<<grid0,block0>>>(d_e,d_eO,d_rhse1,d_rhse2,d_rhse3,dtC);
		cudaDeviceSynchronize();
		rk3finalR<<<grid0,block0,0,s[0]>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_rhsu3,d_r,dtC);
		rk3finalR<<<grid0,block0,0,s[1]>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_rhsv3,d_r,dtC);
		rk3finalR<<<grid0,block0,0,s[2]>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_rhsw3,d_r,dtC);
		cudaDeviceSynchronize();
	}
}

void runSimulationLowStorage(myprec *par1, myprec *par2, myprec *time, Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	myprec h_dt,h_dpdz;

	for (int istep = 0; istep < nsteps; istep++) {
		if(istep%checkCFLcondition==0) calcTimeStepPressGrad(istep,dtC,dpdz,&h_dt,&h_dpdz,rk);
		if(istep>0)  deviceSumOne<<<1,1>>>(&time[istep],&time[istep-1] ,dtC);
		if(istep==0) deviceSumOne<<<1,1>>>(&time[istep],&time[nsteps-1],dtC);
		deviceAdvanceTime<<<1,1>>>(dtC);
		if(istep%checkBulk==0) calcBulk(&par1[istep],&par2[istep],d_r,d_u,d_v,d_w,d_e,rk);


		//Starting the Runge-Kutta Steps

		//runge kutta step 1
		calcRHS(d_rhsr1,d_rhsu1,d_rhsv1,d_rhsw1,d_rhse1,rk);
		deviceMul<<<grid0,block0,0,s[1]>>>(d_u,d_u,d_r);
		deviceMul<<<grid0,block0,0,s[2]>>>(d_v,d_v,d_r);
		deviceMul<<<grid0,block0,0,s[3]>>>(d_w,d_w,d_r);
		sumLowStorageRK3<<<grid0,block0,0,s[0]>>>(d_r, d_rhsr1, d_rhsr1, dtC, 0);
		sumLowStorageRK3<<<grid0,block0,0,s[1]>>>(d_u, d_rhsu1, d_rhsu1, dtC, 0);
		sumLowStorageRK3<<<grid0,block0,0,s[2]>>>(d_v, d_rhsv1, d_rhsv1, dtC, 0);
		sumLowStorageRK3<<<grid0,block0,0,s[3]>>>(d_w, d_rhsw1, d_rhsw1, dtC, 0);
		sumLowStorageRK3<<<grid0,block0,0,s[4]>>>(d_e, d_rhse1, d_rhse1, dtC, 0);
		cudaStreamSynchronize(s[0]);
		deviceDiv<<<grid0,block0,0,s[1]>>>(d_u,d_u,d_r);
		deviceDiv<<<grid0,block0,0,s[2]>>>(d_v,d_v,d_r);
		deviceDiv<<<grid0,block0,0,s[3]>>>(d_w,d_w,d_r);
		cudaDeviceSynchronize();

		if(multiGPU) {  //To initiate slowly the routines so that we have time to initiate the memory transfer
			deviceCpy<<<grid0,block0,0,s[0]>>>(d_r,d_r);
			deviceCpy<<<grid0,block0,0,s[1]>>>(d_u,d_u);
			deviceCpy<<<grid0,block0,0,s[2]>>>(d_v,d_v);
			deviceCpy<<<grid0,block0,0,s[3]>>>(d_w,d_w); }


		//runge kutta step 2
		calcRHS(d_rhsr2,d_rhsu2,d_rhsv2,d_rhsw2,d_rhse2,rk);
		deviceMul<<<grid0,block0,0,s[1]>>>(d_u,d_u,d_r);
		deviceMul<<<grid0,block0,0,s[2]>>>(d_v,d_v,d_r);
		deviceMul<<<grid0,block0,0,s[3]>>>(d_w,d_w,d_r);
		sumLowStorageRK3<<<grid0,block0,0,s[0]>>>(d_r, d_rhsr1, d_rhsr2, dtC, 1);
		sumLowStorageRK3<<<grid0,block0,0,s[1]>>>(d_u, d_rhsu1, d_rhsu2, dtC, 1);
		sumLowStorageRK3<<<grid0,block0,0,s[2]>>>(d_v, d_rhsv1, d_rhsv2, dtC, 1);
		sumLowStorageRK3<<<grid0,block0,0,s[3]>>>(d_w, d_rhsw1, d_rhsw2, dtC, 1);
		sumLowStorageRK3<<<grid0,block0,0,s[4]>>>(d_e, d_rhse1, d_rhse2, dtC, 1);
		cudaStreamSynchronize(s[0]);
		deviceDiv<<<grid0,block0,0,s[1]>>>(d_u,d_u,d_r);
		deviceDiv<<<grid0,block0,0,s[2]>>>(d_v,d_v,d_r);
		deviceDiv<<<grid0,block0,0,s[3]>>>(d_w,d_w,d_r);
		cudaDeviceSynchronize();

		if(multiGPU) {  //To initiate slowly the routines so that we have time to initiate the memory transfer
			deviceCpy<<<grid0,block0,0,s[0]>>>(d_r,d_r);
			deviceCpy<<<grid0,block0,0,s[1]>>>(d_u,d_u);
			deviceCpy<<<grid0,block0,0,s[2]>>>(d_v,d_v);
			deviceCpy<<<grid0,block0,0,s[3]>>>(d_w,d_w); }

		//runge kutta step 3
		calcRHS(d_rhsr1,d_rhsu1,d_rhsv1,d_rhsw1,d_rhse1,rk);
		deviceMul<<<grid0,block0,0,s[1]>>>(d_u,d_u,d_r);
		deviceMul<<<grid0,block0,0,s[2]>>>(d_v,d_v,d_r);
		deviceMul<<<grid0,block0,0,s[3]>>>(d_w,d_w,d_r);
		sumLowStorageRK3<<<grid0,block0,0,s[0]>>>(d_r, d_rhsr2, d_rhsr1, dtC, 2);
		sumLowStorageRK3<<<grid0,block0,0,s[1]>>>(d_u, d_rhsu2, d_rhsu1, dtC, 2);
		sumLowStorageRK3<<<grid0,block0,0,s[2]>>>(d_v, d_rhsv2, d_rhsv1, dtC, 2);
		sumLowStorageRK3<<<grid0,block0,0,s[3]>>>(d_w, d_rhsw2, d_rhsw1, dtC, 2);
		sumLowStorageRK3<<<grid0,block0,0,s[4]>>>(d_e, d_rhse2, d_rhse1, dtC, 2);
		cudaStreamSynchronize(s[0]);
		deviceDiv<<<grid0,block0,0,s[1]>>>(d_u,d_u,d_r);
		deviceDiv<<<grid0,block0,0,s[2]>>>(d_v,d_v,d_r);
		deviceDiv<<<grid0,block0,0,s[3]>>>(d_w,d_w,d_r);
		cudaDeviceSynchronize();
	}
}

__global__ void eulerSum(myprec *a, myprec *b, myprec *c, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = (b[id.g] + c[id.g]*(*dt)/2.0);
}

__global__ void eulerSumR(myprec *a, myprec *b, myprec *c, myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = (b[id.g] + c[id.g]*(*dt)/2.0)/r[id.g];
}

__global__ void eulerSum3(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = b[id.g] + (2*c2[id.g] - c1[id.g])*(*dt);
}

__global__ void eulerSum3R(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = ( b[id.g] + (2*c2[id.g] - c1[id.g])*(*dt) )/r[id.g];
}

__global__ void rk3final(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a1[id.g] = a2[id.g] + (*dt)*( b[id.g] + 4*c[id.g] + d[id.g])/6.;
}

__global__ void rk3finalR(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a1[id.g] = ( a2[id.g] + (*dt)*( b[id.g] + 4*c[id.g] + d[id.g] )/6. )/ r[id.g];
}

__global__ void calcState(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam, int bc) {

	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int gl = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	if(bc==1) gl += mx*my*mz;

    myprec cvInv = (gam - 1.0)/Rgas;

    myprec invrho = 1.0/rho[gl];

    myprec en = ret[gl]*invrho - 0.5*(uvel[gl]*uvel[gl] + vvel[gl]*vvel[gl] + wvel[gl]*wvel[gl]);
    tem[gl]   = cvInv*en;
    pre[gl]   = rho[gl]*Rgas*tem[gl];
    ht[gl]    = (ret[gl] + pre[gl])*invrho;

    myprec suth = pow(tem[gl],viscexp);
    mu[gl]      = suth/Re;
    lam[gl]     = suth/Re/Pr/Ec;
    __syncthreads();

}

__global__ void sumLowStorageRK3(myprec *var, myprec *rhs1, myprec *rhs2, myprec *dt, int step) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	var[id.g] = var[id.g] + (*dt)*(alpha[step]*rhs1[id.g] + beta[step]*rhs2[id.g]);
}

void calcTimeStepPressGrad(int istep, myprec *dtC, myprec *dpdz, myprec *h_dt, myprec *h_dpdz, Communicator rk) {

	cudaSetDevice(rk.nodeRank);
	calcTimeStep(dtC,d_r,d_u,d_v,d_w,d_e,d_m,rk);
	cudaMemcpy(h_dt  , dtC , sizeof(myprec), cudaMemcpyDeviceToHost);
	allReduceToMin(h_dt,1);
	mpiBarrier();
	cudaMemcpy(dtC , h_dt  , sizeof(myprec), cudaMemcpyHostToDevice);
	if(forcing) {
		calcPressureGrad(dpdz,d_r,d_w,rk);
		cudaMemcpy(h_dpdz, dpdz, sizeof(myprec), cudaMemcpyDeviceToHost);
		allReduceArray(h_dpdz,1);
		mpiBarrier();
		cudaMemcpy(dpdz, h_dpdz, sizeof(myprec), cudaMemcpyHostToDevice);
	}
	if(rk.rank==0) printf("step number %d with %le %le\n",istep,*h_dt,*h_dpdz);
}

void solverWrapper(Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	int start;
    myprec *dpar1, *dpar2, *dtime;
    myprec *hpar1 = new myprec[nsteps];
    myprec *hpar2 = new myprec[nsteps];
    myprec *htime = new myprec[nsteps];

    checkCuda( cudaMalloc((void**)&dpar1, nsteps*sizeof(myprec)) );
    checkCuda( cudaMalloc((void**)&dpar2, nsteps*sizeof(myprec)) );
    checkCuda( cudaMalloc((void**)&dtime, nsteps*sizeof(myprec)) );

    FILE *fp;

    //check the memory usage of the GPU
    checkGpuMem(rk);

    if(restartFile<0) {
    	start=0;
    } else {
    	start=restartFile;
    }

    if(rk.rank==0) fp = fopen("solution.txt","w+");
    for(int file = start+1; file<nfiles+start+1; file++) {

    	if(lowStorage) {
        	runSimulationLowStorage(dpar1,dpar2,dtime,rk);  //running the simulation on the GPU
    	} else {
    		runSimulation(dpar1,dpar2,dtime,rk);  //running the simulation on the GPU
    	}

    	copyField(1,rk);					  //copying back partial results to CPU

    	writeField(file,rk);

    	cudaDeviceSynchronize();

    	checkCuda( cudaMemcpy(htime, dtime, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );
    	checkCuda( cudaMemcpy(hpar1, dpar1, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );
    	checkCuda( cudaMemcpy(hpar2, dpar2, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );

    	calcAvgChan(rk);

    	if(rk.rank==0) {
    		printf("file number: %d  \t step: %d  \t time: %lf  \t kin: %le  \t energy: %le\n",file,file*nsteps,htime[nsteps-1],hpar1[nsteps-1],hpar2[nsteps-1]);
    		for(int t=0; t<nsteps-1; t+=checkCFLcondition)
    			fprintf(fp,"%d %lf %lf %lf %lf\n",file*(t+1),htime[t],hpar1[t],hpar2[t],htime[t+1]-htime[t]);
    	}
    	mpiBarrier();
    }
    if(rk.rank==0) fclose(fp);

    clearSolver(rk);
    cudaDeviceReset();
}




