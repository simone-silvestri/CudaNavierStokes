#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_main.h"
#include "cuda_math.h"

__global__ void wraySum(myprec *a, myprec *c1[3], myprec *c2[3], myprec alpha, myprec beta, myprec *dt);
__global__ void wraySumR(myprec *a, myprec *b, myprec *c1[3], myprec *c2[3], myprec alpha, myprec beta, myprec *r, myprec *dt);

#if (capability>capabilityMin)
__global__ void runDevice(myprec *kin, myprec *enst, myprec *time) {

	dtC = d_dt;

	/* allocating temporary arrays and streams */
	void (*RHSDeviceDir[3])(myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*,
							myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec**, myprec*, myprec);

	RHSDeviceDir[0] = RHSDeviceSharedFlxX;
	RHSDeviceDir[1] = RHSDeviceSharedFlxY;
	RHSDeviceDir[2] = RHSDeviceSharedFlxZ;

	__syncthreads();

	cudaStream_t s[3];
    for (int i=0; i<3; i++) {
    	checkCudaDev( cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking) );
    }

	initSolver();
	initStress();

    for (int istep = 0; istep < nsteps; istep++) {

    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	cudaDeviceSynchronize();

    	if(istep%checkCFLcondition==0) {
    		calcTimeStep(&dtC,d_r,d_u,d_v,d_w,d_e,d_m);
    		if(forcing)  calcPressureGrad(&dpdz,d_w);
    	}

    	dt2 = dtC/2.;
    	if(istep==0) {
    		time[istep] = time[nsteps-1] + dtC;
    	} else{
    		time[istep] = time[istep-1] + dtC; }

    	deviceMul<<<grid0,block0>>>(d_uO,d_r,d_u);
    	deviceMul<<<grid0,block0>>>(d_vO,d_r,d_v);
    	deviceMul<<<grid0,block0>>>(d_wO,d_r,d_w);

    	//wray step 1
    	cudaDeviceSynchronize();
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w,gij);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w,gij);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w,gij);
    	cudaDeviceSynchronize();

    	if(istep%checkCFLcondition==0) {
    		calcIntegrals(d_r,d_u,d_v,d_w,gij,&kin[istep],&enst[istep]);
    		enst[istep] = dpdz;
    	}
    	cudaDeviceSynchronize();

    	calcDil<<<grid0,block0>>>(gij,d_dil);
    	cudaDeviceSynchronize();

    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr1[d],d_rhsu1[d],d_rhsv1[d],d_rhsw1[d],d_rhse1[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	wraySum<<<grid0,block0>>>(d_r,d_rhsr1,d_rhsr1,0.0,8.0/15.0,&dtC);
    	cudaDeviceSynchronize();
    	wraySum<<<grid0,block0>>>(d_e,d_rhse1,d_rhse1,0.0,8.0/15.0,&dtC);
    	wraySumR<<<grid0,block0>>>(d_u,d_uO,d_rhsu1,d_rhsu1,0.0,8.0/15.0,d_r,&dtC);
    	wraySumR<<<grid0,block0>>>(d_v,d_vO,d_rhsv1,d_rhsv1,0.0,8.0/15.0,d_r,&dtC);
    	wraySumR<<<grid0,block0>>>(d_w,d_wO,d_rhsw1,d_rhsw1,0.0,8.0/15.0,d_r,&dtC);
    	cudaDeviceSynchronize();

    	//wray step 2
    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w,gij);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w,gij);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w,gij);
		cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(gij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr2[d],d_rhsu2[d],d_rhsv2[d],d_rhsw2[d],d_rhse2[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	wraySum<<<grid0,block0>>>(d_r,d_rhsr1,d_rhsr2,17.0/60.0,5.0/12.0,&dtC);
    	cudaDeviceSynchronize();
    	wraySum<<<grid0,block0>>>(d_e,d_rhse1,d_rhse2,17.0/60.0,5.0/12.0,&dtC);
    	wraySumR<<<grid0,block0>>>(d_u,d_uO,d_rhsu1,d_rhsu2,17.0/60.0,5.0/12.0,d_r,&dtC);
    	wraySumR<<<grid0,block0>>>(d_v,d_vO,d_rhsv1,d_rhsv2,17.0/60.0,5.0/12.0,d_r,&dtC);
    	wraySumR<<<grid0,block0>>>(d_w,d_wO,d_rhsw1,d_rhsw2,17.0/60.0,5.0/12.0,d_r,&dtC);
    	cudaDeviceSynchronize();

    	//wray step 3
    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w,gij);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w,gij);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w,gij);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(gij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr1[d],d_rhsu1[d],d_rhsv1[d],d_rhsw1[d],d_rhse1[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	wraySum<<<grid0,block0>>>(d_r,d_rhsr2,d_rhsr1,-5.0/12.0,3.0/4.0,&dtC);
    	cudaDeviceSynchronize();
    	wraySum<<<grid0,block0>>>(d_e,d_rhse2,d_rhse1,-5.0/12.0,3.0/4.0,&dtC);
    	wraySumR<<<grid0,block0>>>(d_u,d_uO,d_rhsu2,d_rhsu1,-5.0/12.0,3.0/4.0,d_r,&dtC);
    	wraySumR<<<grid0,block0>>>(d_v,d_vO,d_rhsv2,d_rhsv1,-5.0/12.0,3.0/4.0,d_r,&dtC);
    	wraySumR<<<grid0,block0>>>(d_w,d_wO,d_rhsw2,d_rhsw1,-5.0/12.0,3.0/4.0,d_r,&dtC);
    	cudaDeviceSynchronize();

	}
    __syncthreads();

	for (int i=0; i<3; i++) {
		checkCudaDev( cudaStreamDestroy(s[i]) );
	}

	clearSolver();
	clearStress();
}
#else
__global__ void runDevice(myprec *kin, myprec *enst, myprec *time) {

	dtC = d_dt;

	dim3 gr[5],bl[5],gr0,bl0;

	gr[0] = dim3(d_grid[0],d_grid[1],1);
	gr[1] = dim3(d_grid[4],d_grid[5],1);
	gr[2] = dim3(d_grid[8],d_grid[9],1);
	gr[3] = dim3(d_grid[2],d_grid[3],1);
	gr[4] = dim3(d_grid[6],d_grid[7],1);

	bl[0] = dim3(d_block[0],d_block[1],1);
	bl[1] = dim3(d_block[4],d_block[5],1);
	bl[2] = dim3(d_block[8],d_block[9],1);
	bl[3] = dim3(d_block[2],d_block[3],1);
	bl[4] = dim3(d_block[6],d_block[7],1);

	gr0 = dim3(grid0[0],grid0[1],1); bl0 = dim3(block0[0],block0[1],1);

	/* allocating temporary arrays and streams */
	void (*RHSDeviceDir[3])(myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*,
							myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec**, myprec*, myprec*);


	RHSDeviceDir[0] = RHSDeviceSharedFlxX;
	RHSDeviceDir[1] = RHSDeviceSharedFlxY;
	RHSDeviceDir[2] = RHSDeviceSharedFlxZ;

	__syncthreads();

	cudaStream_t s[3];
    for (int i=0; i<3; i++) {
    	checkCudaDev( cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking) );
    }

    initSolver();
    initStress();

    for (int istep = 0; istep < nsteps; istep++) {

    	calcState<<<gr0,bl0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	cudaDeviceSynchronize();

    	if(istep%checkCFLcondition==0) {
    		calcTimeStep(&dtC,d_r,d_u,d_v,d_w,d_e,d_m);
    		if(forcing) calcPressureGrad(&dpdz,d_w);
    	}
    	dt2 = dtC/2.;
    	if(istep==0) {
    		time[istep] = time[nsteps-1] + dtC;
    	} else{
    		time[istep] = time[istep-1] + dtC; }

    	deviceMul<<<gr0,bl0>>>(d_uO,d_r,d_u);
    	deviceMul<<<gr0,bl0>>>(d_vO,d_r,d_v);
    	deviceMul<<<gr0,bl0>>>(d_wO,d_r,d_w);
    	deviceCpy<<<gr0,bl0>>>(d_rO,d_r);
    	deviceCpy<<<gr0,bl0>>>(d_eO,d_e);

    	/* rk step 1 */
    	cudaDeviceSynchronize();
    	calcStressX<<<gr[0],bl[0],0,s[0]>>>(d_u,d_v,d_w,gij);
    	calcStressY<<<gr[3],bl[3],0,s[1]>>>(d_u,d_v,d_w,gij);
    	calcStressZ<<<gr[4],bl[4],0,s[2]>>>(d_u,d_v,d_w,gij);
    	cudaDeviceSynchronize();

    	if(istep%checkCFLcondition==0) {
    		calcIntegrals(d_r,d_u,d_v,d_w,gij,&kin[istep],&enst[istep]);
    		enst[istep] = dpdz;
    	}
    	cudaDeviceSynchronize();

    	calcDil<<<gr0,bl0>>>(gij,d_dil);
    	cudaDeviceSynchronize();

    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<gr[d],bl[d],0,s[d]>>>(d_rhsr1[d],d_rhsu1[d],d_rhsv1[d],d_rhsw1[d],d_rhse1[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	wraySum<<<gr0,bl0>>>(d_r,d_rhsr1,d_rhsr1,0.0,8.0/15.0,&dtC);
    	cudaDeviceSynchronize();
    	wraySum<<<gr0,bl0>>>(d_e,d_rhse1,d_rhse1,0.0,8.0/15.0,&dtC);
    	wraySumR<<<gr0,bl0>>>(d_u,d_uO,d_rhsu1,d_rhsu1,0.0,8.0/15.0,d_r,&dtC);
    	wraySumR<<<gr0,bl0>>>(d_v,d_vO,d_rhsv1,d_rhsv1,0.0,8.0/15.0,d_r,&dtC);
    	wraySumR<<<gr0,bl0>>>(d_w,d_wO,d_rhsw1,d_rhsw1,0.0,8.0/15.0,d_r,&dtC);
    	cudaDeviceSynchronize();

    	//rk step 2
    	calcState<<<gr0,bl0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	calcStressX<<<gr[0],bl[0],0,s[0]>>>(d_u,d_v,d_w,gij);
    	calcStressY<<<gr[3],bl[3],0,s[1]>>>(d_u,d_v,d_w,gij);
    	calcStressZ<<<gr[4],bl[4],0,s[2]>>>(d_u,d_v,d_w,gij);
    	cudaDeviceSynchronize();
    	calcDil<<<gr0,bl0>>>(gij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<gr[d],bl[d],0,s[d]>>>(d_rhsr2[d],d_rhsu2[d],d_rhsv2[d],d_rhsw2[d],d_rhse2[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	wraySum<<<gr0,bl0>>>(d_r,d_rhsr1,d_rhsr2,17.0/60.0,5.0/12.0,&dtC);
    	cudaDeviceSynchronize();
    	wraySum<<<gr0,bl0>>>(d_e,d_rhse1,d_rhse2,17.0/60.0,5.0/12.0,&dtC);
    	wraySumR<<<gr0,bl0>>>(d_u,d_uO,d_rhsu1,d_rhsu2,17.0/60.0,5.0/12.0,d_r,&dtC);
    	wraySumR<<<gr0,bl0>>>(d_v,d_vO,d_rhsv1,d_rhsv2,17.0/60.0,5.0/12.0,d_r,&dtC);
    	wraySumR<<<gr0,bl0>>>(d_w,d_wO,d_rhsw1,d_rhsw2,17.0/60.0,5.0/12.0,d_r,&dtC);
    	cudaDeviceSynchronize();


    	//rk step 3
    	calcState<<<gr0,bl0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	calcStressX<<<gr[0],bl[0],0,s[0]>>>(d_u,d_v,d_w,gij);
    	calcStressY<<<gr[3],bl[3],0,s[1]>>>(d_u,d_v,d_w,gij);
    	calcStressZ<<<gr[4],bl[4],0,s[2]>>>(d_u,d_v,d_w,gij);
    	cudaDeviceSynchronize();
    	calcDil<<<gr0,bl0>>>(gij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<gr[d],bl[d],0,s[d]>>>(d_rhsr3[d],d_rhsu3[d],d_rhsv3[d],d_rhsw3[d],d_rhse3[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	wraySum<<<gr0,bl0>>>(d_r,d_rhsr2,d_rhsr1,-5.0/12.0,3.0/4.0,&dtC);
    	cudaDeviceSynchronize();
    	wraySum<<<gr0,bl0>>>(d_e,d_rhse2,d_rhse1,-5.0/12.0,3.0/4.0,&dtC);
    	wraySumR<<<gr0,bl0>>>(d_u,d_uO,d_rhsu2,d_rhsu1,-5.0/12.0,3.0/4.0,d_r,&dtC);
    	wraySumR<<<gr0,bl0>>>(d_v,d_vO,d_rhsv2,d_rhsv1,-5.0/12.0,3.0/4.0,d_r,&dtC);
    	wraySumR<<<gr0,bl0>>>(d_w,d_wO,d_rhsw2,d_rhsw1,-5.0/12.0,3.0/4.0,d_r,&dtC);
    	cudaDeviceSynchronize();

	}
    __syncthreads();

	for (int i=0; i<3; i++) {
		checkCudaDev( cudaStreamDestroy(s[i]) );
	}

	clearSolver();
	clearStress();
}
#endif

__global__ void wraySum(myprec *a, myprec *c1[3], myprec *c2[3], myprec alpha, myprec beta, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = a[id.g] + alpha     * ( c1[0][id.g] + c1[1][id.g] + c1[2][id.g] )*(*dt)
					  + beta      * ( c2[0][id.g] + c2[1][id.g] + c2[2][id.g] )*(*dt);
}

__global__ void wraySumR(myprec *a, myprec *b, myprec *c1[3], myprec *c2[3], myprec alpha, myprec beta, myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	b[id.g] =  b[id.g] + alpha     * ( c1[0][id.g] + c1[1][id.g] + c1[2][id.g] )*(*dt)
				       + beta      * ( c2[0][id.g] + c2[1][id.g] + c2[2][id.g] )*(*dt) ;
	a[id.g] = b[id.g]/r[id.g];
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

    myprec suth = pow(tem[gt],viscexp);
    mu[gt]      = suth/Re;
    lam[gt]     = suth/Re/Pr/Ec;
    __syncthreads();

}

__device__ void initSolver() {

    for (int i=0; i<3; i++) {
    	checkCudaDev( cudaMalloc((void**)&d_rhsr1[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsu1[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsv1[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsw1[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhse1[i],mx*my*mz*sizeof(myprec)) );

    	checkCudaDev( cudaMalloc((void**)&d_rhsr2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsu2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsv2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsw2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhse2[i],mx*my*mz*sizeof(myprec)) );
    }

	checkCudaDev( cudaMalloc((void**)&d_h,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_t,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_p,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_m,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_l,mx*my*mz*sizeof(myprec)) );

	checkCudaDev( cudaMalloc((void**)&d_uO,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_vO,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_wO,mx*my*mz*sizeof(myprec)) );

	checkCudaDev( cudaMalloc((void**)&d_dil,mx*my*mz*sizeof(myprec)) );
	for (int i=0; i<9; i++)
    	checkCudaDev( cudaMalloc((void**)&gij[i],mx*my*mz*sizeof(myprec)) );

}

__device__ void clearSolver() {

	for (int i=0; i<3; i++) {
		checkCudaDev( cudaFree(d_rhsr1[i]) );
		checkCudaDev( cudaFree(d_rhsu1[i]) );
		checkCudaDev( cudaFree(d_rhsv1[i]) );
		checkCudaDev( cudaFree(d_rhsw1[i]) );
		checkCudaDev( cudaFree(d_rhse1[i]) );

		checkCudaDev( cudaFree(d_rhsr2[i]) );
		checkCudaDev( cudaFree(d_rhsu2[i]) );
		checkCudaDev( cudaFree(d_rhsv2[i]) );
		checkCudaDev( cudaFree(d_rhsw2[i]) );
		checkCudaDev( cudaFree(d_rhse2[i]) );

	}
	checkCudaDev( cudaFree(d_h) );
	checkCudaDev( cudaFree(d_t) );
	checkCudaDev( cudaFree(d_p) );
	checkCudaDev( cudaFree(d_m) );
	checkCudaDev( cudaFree(d_l) );

	checkCudaDev( cudaFree(d_uO) );
	checkCudaDev( cudaFree(d_vO) );
	checkCudaDev( cudaFree(d_wO) );

	checkCudaDev( cudaFree(d_dil) );
	for (int i=0; i<9; i++)
    	checkCudaDev( cudaFree(gij[i]) );

}
