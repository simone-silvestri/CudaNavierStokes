#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_main.h"
#include "cuda_math.h"

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
    		if(forcing)  calcPressureGrad(&dpdz,d_r,d_w);
    	}

    	dt2 = dtC/2.;
    	if(istep==0) {
    		time[istep] = time[nsteps-1] + dtC;
    	} else{
    		time[istep] = time[istep-1] + dtC; }

    	deviceMul<<<grid0,block0>>>(d_uO,d_r,d_u);
    	deviceMul<<<grid0,block0>>>(d_vO,d_r,d_v);
    	deviceMul<<<grid0,block0>>>(d_wO,d_r,d_w);
    	deviceCpy<<<grid0,block0>>>(d_rO,d_r);
    	deviceCpy<<<grid0,block0>>>(d_eO,d_e);

    	/* rk step 1 */
    	cudaDeviceSynchronize();
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w,sij);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w,sij);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();

    	if(istep%checkCFLcondition==0) {
    		calcIntegrals(d_r,d_u,d_v,d_w,sij,&kin[istep],&enst[istep]);
    		enst[istep] = dpdz;
    	}
    	cudaDeviceSynchronize();

    	calcDil<<<grid0,block0>>>(sij,d_dil);
    	cudaDeviceSynchronize();

    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr1[d],d_rhsu1[d],d_rhsv1[d],d_rhsw1[d],d_rhse1[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	eulerSum<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,&dt2);
    	cudaDeviceSynchronize();
    	eulerSum<<<grid0,block0>>>(d_e,d_eO,d_rhse1,&dt2);
    	eulerSumR<<<grid0,block0>>>(d_u,d_uO,d_rhsu1,d_r,&dt2);
    	eulerSumR<<<grid0,block0>>>(d_v,d_vO,d_rhsv1,d_r,&dt2);
    	eulerSumR<<<grid0,block0>>>(d_w,d_wO,d_rhsw1,d_r,&dt2);
    	cudaDeviceSynchronize();

    	//rk step 2
    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w,sij);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w,sij);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w,sij);
		cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(sij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr2[d],d_rhsu2[d],d_rhsv2[d],d_rhsw2[d],d_rhse2[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil,dpdz);
    	cudaDeviceSynchronize();
#if rk==4
    	eulerSum<<<grid0,block0>>>(d_r,d_rO,d_rhsr2,&dt2);
    	cudaDeviceSynchronize();
    	eulerSum<<<grid0,block0>>>(d_e,d_eO,d_rhse2,&dt2);
    	eulerSumR<<<grid0,block0>>>(d_u,d_uO,d_rhsu2,d_r,&dt2);
    	eulerSumR<<<grid0,block0>>>(d_v,d_vO,d_rhsv2,d_r,&dt2);
    	eulerSumR<<<grid0,block0>>>(d_w,d_wO,d_rhsw2,d_r,&dt2);
#elif rk==3
    	eulerSum3<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,&dtC);
    	cudaDeviceSynchronize();
    	eulerSum3<<<grid0,block0>>>(d_e,d_eO,d_rhse1,d_rhse2,&dtC);
    	eulerSum3R<<<grid0,block0>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_r,&dtC);
    	eulerSum3R<<<grid0,block0>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_r,&dtC);
    	eulerSum3R<<<grid0,block0>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_r,&dtC);
#endif
    	cudaDeviceSynchronize();

    	//rk step 3
    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w,sij);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w,sij);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(sij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr3[d],d_rhsu3[d],d_rhsv3[d],d_rhsw3[d],d_rhse3[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil,dpdz);
    	cudaDeviceSynchronize();
#if rk==4
    	eulerSum<<<grid0,block0>>>(d_r,d_rO,d_rhsr3,&dtC);
    	cudaDeviceSynchronize();
    	eulerSum<<<grid0,block0>>>(d_e,d_eO,d_rhse3,&dtC);
    	eulerSumR<<<grid0,block0>>>(d_u,d_uO,d_rhsu3,d_r,&dtC);
    	eulerSumR<<<grid0,block0>>>(d_v,d_vO,d_rhsv3,d_r,&dtC);
    	eulerSumR<<<grid0,block0>>>(d_w,d_wO,d_rhsw3,d_r,&dtC);
    	cudaDeviceSynchronize();

    	//rk step 4
    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w,sij);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w,sij);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(sij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr4[d],d_rhsu4[d],d_rhsv4[d],d_rhsw4[d],d_rhse4[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	rk4final<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,d_rhsr3,d_rhsr4,&dtC);
    	cudaDeviceSynchronize();
    	rk4final<<<grid0,block0>>>(d_e,d_eO,d_rhse1,d_rhse2,d_rhse3,d_rhse4,&dtC);
    	rk4finalR<<<grid0,block0>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_rhsu3,d_rhsu4,d_r,&dtC);
    	rk4finalR<<<grid0,block0>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_rhsv3,d_rhsv4,d_r,&dtC);
    	rk4finalR<<<grid0,block0>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_rhsw3,d_rhsw4,d_r,&dtC);
#elif rk==3
    	rk3final<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,d_rhsr3,&dtC);
    	cudaDeviceSynchronize();
    	rk3final<<<grid0,block0>>>(d_e,d_eO,d_rhse1,d_rhse2,d_rhse3,&dtC);
    	rk3finalR<<<grid0,block0>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_rhsu3,d_r,&dtC);
    	rk3finalR<<<grid0,block0>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_rhsv3,d_r,&dtC);
    	rk3finalR<<<grid0,block0>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_rhsw3,d_r,&dtC);
#endif
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
    		if(forcing)  calcPressureGrad(&dpdz,d_r,d_w);
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
    	calcStressX<<<gr[0],bl[0],0,s[0]>>>(d_u,d_v,d_w,sij);
    	calcStressY<<<gr[3],bl[3],0,s[1]>>>(d_u,d_v,d_w,sij);
    	calcStressZ<<<gr[4],bl[4],0,s[2]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();

    	if(istep%checkCFLcondition==0) {
    		calcIntegrals(d_r,d_u,d_v,d_w,sij,&kin[istep],&enst[istep]);
    		enst[istep] = dpdz;
    	}
    	cudaDeviceSynchronize();

    	calcDil<<<gr0,bl0>>>(sij,d_dil);
    	cudaDeviceSynchronize();

    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<gr[d],bl[d],0,s[d]>>>(d_rhsr1[d],d_rhsu1[d],d_rhsv1[d],d_rhsw1[d],d_rhse1[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	eulerSum<<<gr0,bl0>>>(d_r,d_rO,d_rhsr1,&dt2);
    	cudaDeviceSynchronize();
    	eulerSum<<<gr0,bl0>>>(d_e,d_eO,d_rhse1,&dt2);
    	eulerSumR<<<gr0,bl0>>>(d_u,d_uO,d_rhsu1,d_r,&dt2);
    	eulerSumR<<<gr0,bl0>>>(d_v,d_vO,d_rhsv1,d_r,&dt2);
    	eulerSumR<<<gr0,bl0>>>(d_w,d_wO,d_rhsw1,d_r,&dt2);
    	cudaDeviceSynchronize();

    	//rk step 2
    	calcState<<<gr0,bl0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	calcStressX<<<gr[0],bl[0],0,s[0]>>>(d_u,d_v,d_w,sij);
    	calcStressY<<<gr[3],bl[3],0,s[1]>>>(d_u,d_v,d_w,sij);
    	calcStressZ<<<gr[4],bl[4],0,s[2]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();
    	calcDil<<<gr0,bl0>>>(sij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<gr[d],bl[d],0,s[d]>>>(d_rhsr2[d],d_rhsu2[d],d_rhsv2[d],d_rhsw2[d],d_rhse2[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil,dpdz);
    	cudaDeviceSynchronize();
#if rk==4
    	eulerSum<<<gr0,bl0>>>(d_r,d_rO,d_rhsr2,&dt2);
    	cudaDeviceSynchronize();
    	eulerSum<<<gr0,bl0>>>(d_e,d_eO,d_rhse2,&dt2);
    	eulerSumR<<<gr0,bl0>>>(d_u,d_uO,d_rhsu2,d_r,&dt2);
    	eulerSumR<<<gr0,bl0>>>(d_v,d_vO,d_rhsv2,d_r,&dt2);
    	eulerSumR<<<gr0,bl0>>>(d_w,d_wO,d_rhsw2,d_r,&dt2);
#elif rk==3
    	eulerSum3<<<gr0,bl0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,&dtC);
    	cudaDeviceSynchronize();
    	eulerSum3<<<gr0,bl0>>>(d_e,d_eO,d_rhse1,d_rhse2,&dtC);
    	eulerSum3R<<<gr0,bl0>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_r,&dtC);
    	eulerSum3R<<<gr0,bl0>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_r,&dtC);
    	eulerSum3R<<<gr0,bl0>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_r,&dtC);
#endif
    	cudaDeviceSynchronize();


    	//rk step 3
    	calcState<<<gr0,bl0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	calcStressX<<<gr[0],bl[0],0,s[0]>>>(d_u,d_v,d_w,sij);
    	calcStressY<<<gr[3],bl[3],0,s[1]>>>(d_u,d_v,d_w,sij);
    	calcStressZ<<<gr[4],bl[4],0,s[2]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();
    	calcDil<<<gr0,bl0>>>(sij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<gr[d],bl[d],0,s[d]>>>(d_rhsr3[d],d_rhsu3[d],d_rhsv3[d],d_rhsw3[d],d_rhse3[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil,dpdz);
    	cudaDeviceSynchronize();
#if rk==4
    	eulerSum<<<gr0,bl0>>>(d_r,d_rO,d_rhsr3,&dtC);
    	cudaDeviceSynchronize();
    	eulerSum<<<gr0,bl0>>>(d_e,d_eO,d_rhse3,&dtC);
    	eulerSumR<<<gr0,bl0>>>(d_u,d_uO,d_rhsu3,d_r,&dtC);
    	eulerSumR<<<gr0,bl0>>>(d_v,d_vO,d_rhsv3,d_r,&dtC);
    	eulerSumR<<<gr0,bl0>>>(d_w,d_wO,d_rhsw3,d_r,&dtC);
    	cudaDeviceSynchronize();

    	//rk step 4
    	calcState<<<gr0,bl0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	calcStressX<<<gr[0],bl[0],0,s[0]>>>(d_u,d_v,d_w,sij);
    	calcStressY<<<gr[3],bl[3],0,s[1]>>>(d_u,d_v,d_w,sij);
    	calcStressZ<<<gr[4],bl[4],0,s[2]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();
    	calcDil<<<gr0,bl0>>>(sij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<gr[d],bl[d],0,s[d]>>>(d_rhsr4[d],d_rhsu4[d],d_rhsv4[d],d_rhsw4[d],d_rhse4[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil,dpdz);
    	cudaDeviceSynchronize();
    	rk4final<<<gr0,bl0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,d_rhsr3,d_rhsr4,&dtC);
    	cudaDeviceSynchronize();
    	rk4final<<<gr0,bl0>>>(d_e,d_eO,d_rhse1,d_rhse2,d_rhse3,d_rhse4,&dtC);
    	rk4finalR<<<gr0,bl0>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_rhsu3,d_rhsu4,d_r,&dtC);
    	rk4finalR<<<gr0,bl0>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_rhsv3,d_rhsv4,d_r,&dtC);
    	rk4finalR<<<gr0,bl0>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_rhsw3,d_rhsw4,d_r,&dtC);
#elif rk==3
    	rk3final<<<gr0,bl0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,d_rhsr3,&dtC);
    	cudaDeviceSynchronize();
    	rk3final<<<gr0,bl0>>>(d_e,d_eO,d_rhse1,d_rhse2,d_rhse3,&dtC);
    	rk3finalR<<<gr0,bl0>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_rhsu3,d_r,&dtC);
    	rk3finalR<<<gr0,bl0>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_rhsv3,d_r,&dtC);
    	rk3finalR<<<gr0,bl0>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_rhsw3,d_r,&dtC);
#endif
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

__global__ void eulerSum(myprec *a, myprec *b, myprec *c[3], myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g] + ( c[0][id.g] + c[1][id.g] + c[2][id.g] )*(*dt);
}

__global__ void eulerSumR(myprec *a, myprec *b, myprec *c[3], myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] =  ( b[id.g] +  ( c[0][id.g] + c[1][id.g] + c[2][id.g] ) *(*dt) ) /r[id.g];
}

__global__ void eulerSum3(myprec *a, myprec *b, myprec *c1[3], myprec *c2[3], myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g] -     ( c1[0][id.g] + c1[1][id.g] + c1[2][id.g] )*(*dt)
					  + 2 * ( c2[0][id.g] + c2[1][id.g] + c2[2][id.g] )*(*dt);
}

__global__ void eulerSum3R(myprec *a, myprec *b, myprec *c1[3], myprec *c2[3], myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = ( b[id.g] -     ( c1[0][id.g] + c1[1][id.g] + c1[2][id.g] )*(*dt)
					    + 2 * ( c2[0][id.g] + c2[1][id.g] + c2[2][id.g] )*(*dt) )/ r[id.g];
}

__global__ void rk4final(myprec *a1, myprec *a2, myprec *b[3], myprec *c[3], myprec *d[3], myprec *e[3], myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a1[id.g] = a2[id.g];
	for (int it=0; it<3; it++)
		a1[id.g] = a1[id.g] + (*dt)*( b[it][id.g] + 2*c[it][id.g] + 2*d[it][id.g] + e[it][id.g])/6.;
}

__global__ void rk4finalR(myprec *a1, myprec *a2, myprec *b[3], myprec *c[3], myprec *d[3], myprec *e[3], myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a1[id.g] =  a2[id.g]/r[id.g];
	for (int it=0; it<3; it++)
		a1[id.g] +=  (*dt)*( b[it][id.g] + 2*c[it][id.g] + 2*d[it][id.g] + e[it][id.g])/6./ r[id.g];
}

__global__ void rk3final(myprec *a1, myprec *a2, myprec *b[3], myprec *c[3], myprec *d[3], myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a1[id.g] = a2[id.g];
	for (int it=0; it<3; it++)
		a1[id.g] = a1[id.g] + (*dt)*( b[it][id.g] + 4*c[it][id.g] + d[it][id.g])/6.;
}

__global__ void rk3finalR(myprec *a1, myprec *a2, myprec *b[3], myprec *c[3], myprec *d[3], myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a1[id.g] = a2[id.g]/r[id.g];
	for (int it=0; it<3; it++)
		a1[id.g] += (*dt)*( b[it][id.g] + 4*c[it][id.g] + d[it][id.g])/6. / r[id.g];
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

    	checkCudaDev( cudaMalloc((void**)&d_rhsr3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsu3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsv3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsw3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhse3[i],mx*my*mz*sizeof(myprec)) );
#if rk == 4
    	checkCudaDev( cudaMalloc((void**)&d_rhsr4[i],mx*my*mz*sizeof(myprec)) );
		checkCudaDev( cudaMalloc((void**)&d_rhsu4[i],mx*my*mz*sizeof(myprec)) );
		checkCudaDev( cudaMalloc((void**)&d_rhsv4[i],mx*my*mz*sizeof(myprec)) );
		checkCudaDev( cudaMalloc((void**)&d_rhsw4[i],mx*my*mz*sizeof(myprec)) );
		checkCudaDev( cudaMalloc((void**)&d_rhse4[i],mx*my*mz*sizeof(myprec)) );
#endif
    }

	checkCudaDev( cudaMalloc((void**)&d_h,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_t,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_p,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_m,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_l,mx*my*mz*sizeof(myprec)) );

	checkCudaDev( cudaMalloc((void**)&d_rO,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_eO,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_uO,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_vO,mx*my*mz*sizeof(myprec)) );
	checkCudaDev( cudaMalloc((void**)&d_wO,mx*my*mz*sizeof(myprec)) );

	checkCudaDev( cudaMalloc((void**)&d_dil,mx*my*mz*sizeof(myprec)) );
	for (int i=0; i<9; i++)
    	checkCudaDev( cudaMalloc((void**)&sij[i],mx*my*mz*sizeof(myprec)) );

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

		checkCudaDev( cudaFree(d_rhsr3[i]) );
		checkCudaDev( cudaFree(d_rhsu3[i]) );
		checkCudaDev( cudaFree(d_rhsv3[i]) );
		checkCudaDev( cudaFree(d_rhsw3[i]) );
		checkCudaDev( cudaFree(d_rhse3[i]) );
#if rk==4
		checkCudaDev( cudaFree(d_rhsr4[i]) );
		checkCudaDev( cudaFree(d_rhsu4[i]) );
		checkCudaDev( cudaFree(d_rhsv4[i]) );
		checkCudaDev( cudaFree(d_rhsw4[i]) );
		checkCudaDev( cudaFree(d_rhse4[i]) );
#endif
	}
	checkCudaDev( cudaFree(d_h) );
	checkCudaDev( cudaFree(d_t) );
	checkCudaDev( cudaFree(d_p) );
	checkCudaDev( cudaFree(d_m) );
	checkCudaDev( cudaFree(d_l) );

	checkCudaDev( cudaFree(d_rO) );
	checkCudaDev( cudaFree(d_eO) );
	checkCudaDev( cudaFree(d_uO) );
	checkCudaDev( cudaFree(d_vO) );
	checkCudaDev( cudaFree(d_wO) );

	checkCudaDev( cudaFree(d_dil) );
	for (int i=0; i<9; i++)
    	checkCudaDev( cudaFree(sij[i]) );

}
