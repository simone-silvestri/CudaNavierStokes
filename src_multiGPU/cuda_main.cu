#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_main.h"
#include "cuda_math.h"
#include "comm.h"

void runSimulation(myprec *par1, myprec *par2, myprec *time, Communicator rk) {

	myprec h_dt,h_dpdz;

	/* allocating temporary arrays and streams */
	void (*RHSDeviceDir[3])(myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*,
							myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*);

	RHSDeviceDir[0] = RHSDeviceSharedFlxX;
	RHSDeviceDir[1] = RHSDeviceSharedFlxY_new;
	RHSDeviceDir[2] = RHSDeviceSharedFlxZ_new;

	cudaStream_t s[3];
    for (int i=0; i<3; i++) {
    	checkCuda( cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking) );
    }

    for (int istep = 0; istep < nsteps; istep++) {

    	if(multiGPU) updateHaloFive(d_r,d_u,d_v,d_w,d_e,rk); cudaDeviceSynchronize();
    	calcState<<<gridBC,blockBC>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	cudaDeviceSynchronize();

    	if(istep%checkCFLcondition==0) {
    		calcTimeStepPressGrad(istep,dtC,dpdz,&h_dt,&h_dpdz,rk);
    	}
    	if(istep>0)  deviceSumOne<<<1,1>>>(&time[istep],&time[istep-1] ,dtC);
    	if(istep==0) deviceSumOne<<<1,1>>>(&time[istep],&time[nsteps-1],dtC);
    	if(istep%checkBulk==0) calcBulk(&par1[istep],&par2[istep],d_r,d_w,d_e);

    	deviceMul<<<grid0,block0,0,s[0]>>>(d_uO,d_r,d_u);
    	deviceMul<<<grid0,block0,0,s[1]>>>(d_vO,d_r,d_v);
    	deviceMul<<<grid0,block0,0,s[2]>>>(d_wO,d_r,d_w);
    	deviceCpy<<<grid0,block0>>>(d_rO,d_r);
    	deviceCpy<<<grid0,block0>>>(d_eO,d_e);

    	/* rk step 1 */
    	cudaDeviceSynchronize();
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w);
    	cudaDeviceSynchronize();

    	calcDil<<<grid0,block0>>>(d_dil);
    	cudaDeviceSynchronize();
    	if(multiGPU) updateHalo(d_dil,rk); cudaDeviceSynchronize();

#if useStreams
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr1[d],d_rhsu1[d],d_rhsv1[d],d_rhsw1[d],d_rhse1[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#else
		RHSDeviceDir[0]<<<d_grid[0],d_block[0]>>>(d_rhsr1[0],d_rhsu1[0],d_rhsv1[0],d_rhsw1[0],d_rhse1[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceDir[1]<<<d_grid[1],d_block[1]>>>(d_rhsr1[0],d_rhsu1[0],d_rhsv1[0],d_rhsw1[0],d_rhse1[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceDir[2]<<<d_grid[2],d_block[2]>>>(d_rhsr1[0],d_rhsu1[0],d_rhsv1[0],d_rhsw1[0],d_rhse1[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#endif
    	cudaDeviceSynchronize();
    	for (int d=0; d<fin; d++) {
    		eulerSum<<<grid0,block0>>>(d_r,d_rO,d_rhsr1[d],dtC,d);
    		eulerSum<<<grid0,block0>>>(d_e,d_eO,d_rhse1[d],dtC,d);    	}
		cudaDeviceSynchronize();
		for (int d=0; d<fin; d++) {
    		eulerSumR<<<grid0,block0,0,s[0]>>>(d_u,d_uO,d_rhsu1[d],d_r,dtC,d);
    		eulerSumR<<<grid0,block0,0,s[1]>>>(d_v,d_vO,d_rhsv1[d],d_r,dtC,d);
    		eulerSumR<<<grid0,block0,0,s[2]>>>(d_w,d_wO,d_rhsw1[d],d_r,dtC,d);    	}
    	cudaDeviceSynchronize();

    	//rk step 2
    	if(multiGPU) updateHaloFive(d_r,d_u,d_v,d_w,d_e,rk); cudaDeviceSynchronize();
    	calcState<<<gridBC,blockBC>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w);
		cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(d_dil);
    	cudaDeviceSynchronize();
    	if(multiGPU) updateHalo(d_dil,rk); cudaDeviceSynchronize();
#if useStreams
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr2[d],d_rhsu2[d],d_rhsv2[d],d_rhsw2[d],d_rhse2[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#else
		RHSDeviceDir[0]<<<d_grid[0],d_block[0]>>>(d_rhsr2[0],d_rhsu2[0],d_rhsv2[0],d_rhsw2[0],d_rhse2[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceDir[1]<<<d_grid[1],d_block[1]>>>(d_rhsr2[0],d_rhsu2[0],d_rhsv2[0],d_rhsw2[0],d_rhse2[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceDir[2]<<<d_grid[2],d_block[2]>>>(d_rhsr2[0],d_rhsu2[0],d_rhsv2[0],d_rhsw2[0],d_rhse2[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#endif
		cudaDeviceSynchronize();
		for (int d=0; d<fin; d++) {
			eulerSum3<<<grid0,block0>>>(d_r,d_rO,d_rhsr1[d],d_rhsr2[d],dtC,d);
			eulerSum3<<<grid0,block0>>>(d_e,d_eO,d_rhse1[d],d_rhse2[d],dtC,d);   	}
		cudaDeviceSynchronize();
		for (int d=0; d<fin; d++) {
			eulerSum3R<<<grid0,block0,0,s[0]>>>(d_u,d_uO,d_rhsu1[d],d_rhsu2[d],d_r,dtC,d);
			eulerSum3R<<<grid0,block0,0,s[1]>>>(d_v,d_vO,d_rhsv1[d],d_rhsv2[d],d_r,dtC,d);
			eulerSum3R<<<grid0,block0,0,s[2]>>>(d_w,d_wO,d_rhsw1[d],d_rhsw2[d],d_r,dtC,d); }
    	cudaDeviceSynchronize();

    	//rk step 3
    	if(multiGPU) updateHaloFive(d_r,d_u,d_v,d_w,d_e,rk); cudaDeviceSynchronize();
    	calcState<<<gridBC,blockBC>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(d_dil);
    	cudaDeviceSynchronize();
    	if(multiGPU) updateHalo(d_dil,rk); cudaDeviceSynchronize();
#if useStreams
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr3[d],d_rhsu3[d],d_rhsv3[d],d_rhsw3[d],d_rhse3[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#else
		RHSDeviceDir[0]<<<d_grid[0],d_block[0]>>>(d_rhsr3[0],d_rhsu3[0],d_rhsv3[0],d_rhsw3[0],d_rhse3[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceDir[1]<<<d_grid[1],d_block[1]>>>(d_rhsr3[0],d_rhsu3[0],d_rhsv3[0],d_rhsw3[0],d_rhse3[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceDir[2]<<<d_grid[2],d_block[2]>>>(d_rhsr3[0],d_rhsu3[0],d_rhsv3[0],d_rhsw3[0],d_rhse3[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#endif
    	cudaDeviceSynchronize();
    	for (int d=0; d<fin; d++) {
    		rk3final<<<grid0,block0>>>(d_r,d_rO,d_rhsr1[d],d_rhsr2[d],d_rhsr3[d],dtC,d);
    		rk3final<<<grid0,block0>>>(d_e,d_eO,d_rhse1[d],d_rhse2[d],d_rhse3[d],dtC,d); 	}
		cudaDeviceSynchronize();
		for (int d=0; d<fin; d++) {
    		rk3finalR<<<grid0,block0,0,s[0]>>>(d_u,d_uO,d_rhsu1[d],d_rhsu2[d],d_rhsu3[d],d_r,dtC,d);
    		rk3finalR<<<grid0,block0,0,s[1]>>>(d_v,d_vO,d_rhsv1[d],d_rhsv2[d],d_rhsv3[d],d_r,dtC,d);
    		rk3finalR<<<grid0,block0,0,s[2]>>>(d_w,d_wO,d_rhsw1[d],d_rhsw2[d],d_rhsw3[d],d_r,dtC,d); }
    	cudaDeviceSynchronize();

	}

	for (int i=0; i<3; i++) {
		checkCuda( cudaStreamDestroy(s[i]) );
	}
}

__global__ void eulerSum(myprec *a, myprec *b, myprec *c, myprec *dt, int i) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	if(i==0) {
	    a[id.g] = (b[id.g] + c[id.g]*(*dt)/2.0);
	} else {
	    a[id.g] += ( c[id.g]*(*dt)/2.0 );
	}
}

__global__ void eulerSumR(myprec *a, myprec *b, myprec *c, myprec *r, myprec *dt, int i) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	if(i==0) {
	    a[id.g] = (b[id.g] + c[id.g]*(*dt)/2.0)/r[id.g];
	} else {
	    a[id.g] += ( c[id.g]*(*dt)/2.0 )/r[id.g];
	}
}

__global__ void eulerSum3(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *dt, int i) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	if(i==0) {
		a[id.g] = b[id.g] + (2*c2[id.g] - c1[id.g])*(*dt);
	} else {
		a[id.g] +=  ( 2*c2[id.g] - c1[id.g] )*(*dt);
	}
}

__global__ void eulerSum3R(myprec *a, myprec *b, myprec *c1, myprec *c2, myprec *r, myprec *dt, int i) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	if(i==0 ) {
		a[id.g] = ( b[id.g] + (2*c2[id.g] - c1[id.g])*(*dt) )/r[id.g];
	} else {
		a[id.g] +=  ( 2*c2[id.g] - c1[id.g] )*(*dt) / r[id.g];
	}
}

__global__ void rk3final(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *dt, int i) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	if(i==0) {
		a1[id.g] = a2[id.g] + (*dt)*( b[id.g] + 4*c[id.g] + d[id.g])/6.;
	} else {
		a1[id.g] +=  (*dt)*( b[id.g] + 4*c[id.g] + d[id.g] )/6. ;
	}
}

__global__ void rk3finalR(myprec *a1, myprec *a2, myprec *b, myprec *c, myprec *d, myprec *r, myprec *dt, int i) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	if(i==0) {
		a1[id.g] = ( a2[id.g] + (*dt)*( b[id.g] + 4*c[id.g] + d[id.g] )/6. )/ r[id.g];
	} else {
		a1[id.g] += ( (*dt)*( b[id.g] + 4*c[id.g] + d[id.g] )/6. )/ r[id.g];
	}
}

__global__ void calcState(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam) {

	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int gt = blockNumInGrid * threadsPerBlock + threadNumInBlock;

    myprec cvInv = (gamma - 1.0)/Rgas;

    myprec invrho = 1.0/rho[gt];

    myprec en = ret[gt]*invrho - 0.5*(uvel[gt]*uvel[gt] + vvel[gt]*vvel[gt] + wvel[gt]*wvel[gt]);
    tem[gt]   = cvInv*en;
    pre[gt]   = rho[gt]*Rgas*tem[gt];
    ht[gt]    = (ret[gt] + pre[gt])*invrho;


    if(pre[gt] == 0)
    {
    	printf("DEbug!!\n");
    }

    myprec suth = pow(tem[gt],viscexp);
    mu[gt]      = suth/Re;
    lam[gt]     = suth/Re/Pr/Ec;
    __syncthreads();

}

void calcTimeStepPressGrad(int istep, myprec *dtC, myprec *dpdz, myprec *h_dt, myprec *h_dpdz, Communicator rk) {
	calcTimeStep(dtC,d_r,d_u,d_v,d_w,d_e,d_m);
	cudaMemcpy(h_dt  , dtC , sizeof(myprec), cudaMemcpyDeviceToHost);
	allReduceToMin(h_dt,1);
	mpiBarrier();
	cudaMemcpy(dtC , h_dt  , sizeof(myprec), cudaMemcpyHostToDevice);
    if(forcing) {
    	calcPressureGrad(dpdz,d_r,d_w);
    	cudaMemcpy(h_dpdz, dpdz, sizeof(myprec), cudaMemcpyDeviceToHost);
    	allReduceArray(h_dpdz,1);
    	mpiBarrier();
    	cudaMemcpy(dpdz, h_dpdz, sizeof(myprec), cudaMemcpyHostToDevice);
    }
	if(rk.rank==0) printf("step number %d with %lf %lf\n",istep,*h_dt,*h_dpdz);
}

void solverWrapper(Communicator rk) {

	cudaSetDevice(rk.nodeRank);

    myprec *dpar1, *dpar2, *dtime;
    myprec *hpar1 = new myprec[nsteps];
    myprec *hpar2 = new myprec[nsteps];
    myprec *htime = new myprec[nsteps];

    checkCuda( cudaMalloc((void**)&dpar1, nsteps*sizeof(myprec)) );
    checkCuda( cudaMalloc((void**)&dpar2, nsteps*sizeof(myprec)) );
    checkCuda( cudaMalloc((void**)&dtime, nsteps*sizeof(myprec)) );

    // Increase GPU default limits to accomodate the computations
    size_t rsize = 1024ULL*1024ULL*1024ULL*8ULL;  // allocate 10GB of HEAP (dynamic) memory size
    cudaDeviceSetLimit(cudaLimitMallocHeapSize , rsize);
    FILE *fp;
    if(rk.rank==0) fp = fopen("solution.txt","w+");
    for(int file = 1; file<nfiles+1; file++) {

    	runSimulation(dpar1,dpar2,dtime,rk);  //running the simulation on the GPU
    	copyField(1);			  //copying back partial results to CPU

    	writeField(2,rk);

    	cudaDeviceSynchronize();

    	checkCuda( cudaMemcpy(htime, dtime, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );
    	checkCuda( cudaMemcpy(hpar1, dpar1, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );
    	checkCuda( cudaMemcpy(hpar2, dpar2, nsteps*sizeof(myprec) , cudaMemcpyDeviceToHost) );

    	calcAvgChan(rk);

    	if(rk.rank==0) {
    		printf("file number: %d  \t step: %d  \t time: %lf  \t kin: %lf  \t dpdz: %lf\n",file,file*nsteps,htime[nsteps-1],hpar1[nsteps-1],hpar2[nsteps-1]);
    		for(int t=0; t<nsteps-1; t++)
    			fprintf(fp,"%lf %lf %lf %lf\n",htime[t],hpar1[t],hpar2[t],htime[t+1]-htime[t]);
    	}
    	mpiBarrier();
    }
    if(rk.rank==0) fclose(fp);

    clearSolver();
    cudaDeviceReset();
}




