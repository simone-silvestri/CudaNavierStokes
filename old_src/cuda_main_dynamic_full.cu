#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_main.h"

__global__ void runDevice(myprec *kin, myprec *enst, myprec *time) {

	dtC = d_dt;

	/* allocating temporary arrays and streams */
	void (*RHSDeviceDir[3])(myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec**, myprec *);
	void (*calcStresDir[3])(myprec*, myprec*, myprec*, myprec**);
	calcStresDir[0] = calcStressX;
	calcStresDir[1] = calcStressY;
	calcStresDir[2] = calcStressZ;

	RHSDeviceDir[0] = RHSDeviceSharedFlxX;
#if lPencilY == 1
	RHSDeviceDir[1] = RHSDeviceFullYL;
#else
	RHSDeviceDir[1] = RHSDeviceY;
#endif
#if lPencilZ == 1
	RHSDeviceDir[2] = RHSDeviceFullZL;
#else
	RHSDeviceDir[2] = RHSDeviceZ;
#endif

	__syncthreads();

	cudaStream_t s[3];
    for (int i=0; i<3; i++) {
    	checkCudaDev( cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking) );
    }

    initSolver();

    for (int istep = 0; istep < nsteps; istep++) {

    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	cudaDeviceSynchronize();
//    	if (istep%100==0) {
//    		calcTimeStep<<<grid0,block0>>>(dttemp,d_r,d_u,d_v,d_w,d_e,d_m);
//    		for (int it=0; it<block0.x*block0.y; it++)
//    			dtC = MIN(dtC,dttemp[it]);
//    	}

    	dt2 = dtC/2.;


    	/* rk step 1 */
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		calcStresDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_u,d_v,d_w,gij);
    	cudaDeviceSynchronize();

    	/* Only if you want a bulk solution changing in time!!! */
    	if(istep%250==0) {
    		calcIntegrals<<<1,1>>>(d_r,d_u,d_v,d_w,gij,&kin[istep],&enst[istep]);
    	}
    	if(istep > 0) 	time[istep] = time[istep-1] + dtC;
    	/* This will take some time to execute so take it away if not needed!! */


    	calcDil<<<grid0,block0>>>(gij,d_dil);
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr1[d],d_rhsu1[d],d_rhsv1[d],d_rhsw1[d],d_rhse1[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij,d_dil);
    	cudaDeviceSynchronize();
    	eulerSum<<<grid0,block0>>>(d_tr,d_r,d_rhsr1[0],d_rhsr1[1],d_rhsr1[2],&dt2);
    	eulerSum<<<grid0,block0>>>(d_tu,d_u,d_rhsu1[0],d_rhsu1[1],d_rhsu1[2],&dt2);
    	eulerSum<<<grid0,block0>>>(d_tv,d_v,d_rhsv1[0],d_rhsv1[1],d_rhsv1[2],&dt2);
    	eulerSum<<<grid0,block0>>>(d_tw,d_w,d_rhsw1[0],d_rhsw1[1],d_rhsw1[2],&dt2);
    	eulerSum<<<grid0,block0>>>(d_te,d_e,d_rhse1[0],d_rhse1[1],d_rhse1[2],&dt2);
    	cudaDeviceSynchronize();

    	//rk step 2
    	calcState<<<grid0,block0>>>(d_tr,d_tu,d_tv,d_tw,d_te,d_h,d_t,d_p,d_m,d_l);
    	for (int d = 0; d < 3; d++)
    		calcStresDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_tu,d_tv,d_tw,gij);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(gij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr2[d],d_rhsu2[d],d_rhsv2[d],d_rhsw2[d],d_rhse2[d],d_tr,d_tu,d_tv,d_tw,d_h,d_t,d_p,d_m,d_l,gij,d_dil);
    	cudaDeviceSynchronize();
    	eulerSum<<<grid0,block0>>>(d_tr,d_r,d_rhsr2[0],d_rhsr2[1],d_rhsr2[2],&dt2);
    	eulerSum<<<grid0,block0>>>(d_tu,d_u,d_rhsu2[0],d_rhsu2[1],d_rhsu2[2],&dt2);
    	eulerSum<<<grid0,block0>>>(d_tv,d_v,d_rhsv2[0],d_rhsv2[1],d_rhsv2[2],&dt2);
    	eulerSum<<<grid0,block0>>>(d_tw,d_w,d_rhsw2[0],d_rhsw2[1],d_rhsw2[2],&dt2);
    	eulerSum<<<grid0,block0>>>(d_te,d_e,d_rhse2[0],d_rhse2[1],d_rhse2[2],&dt2);
    	cudaDeviceSynchronize();


    	//rk step 3
    	calcState<<<grid0,block0>>>(d_tr,d_tu,d_tv,d_tw,d_te,d_h,d_t,d_p,d_m,d_l);
    	for (int d = 0; d < 3; d++)
    		calcStresDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_tu,d_tv,d_tw,gij);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(gij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr3[d],d_rhsu3[d],d_rhsv3[d],d_rhsw3[d],d_rhse3[d],d_tr,d_tu,d_tv,d_tw,d_h,d_t,d_p,d_m,d_l,gij,d_dil);
    	cudaDeviceSynchronize();
    	eulerSum<<<grid0,block0>>>(d_tr,d_r,d_rhsr3[0],d_rhsr3[1],d_rhsr3[2],&dtC);
    	eulerSum<<<grid0,block0>>>(d_tu,d_u,d_rhsu3[0],d_rhsu3[1],d_rhsu3[2],&dtC);
    	eulerSum<<<grid0,block0>>>(d_tv,d_v,d_rhsv3[0],d_rhsv3[1],d_rhsv3[2],&dtC);
    	eulerSum<<<grid0,block0>>>(d_tw,d_w,d_rhsw3[0],d_rhsw3[1],d_rhsw3[2],&dtC);
    	eulerSum<<<grid0,block0>>>(d_te,d_e,d_rhse3[0],d_rhse3[1],d_rhse3[2],&dtC);
    	cudaDeviceSynchronize();

    	//rk step 4
    	calcState<<<grid0,block0>>>(d_tr,d_tu,d_tv,d_tw,d_te,d_h,d_t,d_p,d_m,d_l);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		calcStresDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_tu,d_tv,d_tw,gij);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(gij,d_dil);
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr4[d],d_rhsu4[d],d_rhsv4[d],d_rhsw4[d],d_rhse4[d],d_tr,d_tu,d_tv,d_tw,d_h,d_t,d_p,d_m,d_l,gij,d_dil);
    	cudaDeviceSynchronize();
    	rk4final<<<grid0,block0>>>(d_r  ,d_rhsr1[0],d_rhsr2[0],d_rhsr3[0],d_rhsr4[0],
											 d_rhsr1[1],d_rhsr2[1],d_rhsr3[1],d_rhsr4[1],
											 d_rhsr1[2],d_rhsr2[2],d_rhsr3[2],d_rhsr4[2],&dtC);
    	rk4final<<<grid0,block0>>>(d_u  ,d_rhsu1[0],d_rhsu2[0],d_rhsu3[0],d_rhsu4[0],
											 d_rhsu1[1],d_rhsu2[1],d_rhsu3[1],d_rhsu4[1],
											 d_rhsu1[2],d_rhsu2[2],d_rhsu3[2],d_rhsu4[2],&dtC);
    	rk4final<<<grid0,block0>>>(d_v  ,d_rhsv1[0],d_rhsv2[0],d_rhsv3[0],d_rhsv4[0],
											 d_rhsv1[1],d_rhsv2[1],d_rhsv3[1],d_rhsv4[1],
											 d_rhsv1[2],d_rhsv2[2],d_rhsv3[2],d_rhsv4[2],&dtC);
    	rk4final<<<grid0,block0>>>(d_w  ,d_rhsw1[0],d_rhsw2[0],d_rhsw3[0],d_rhsw4[0],
											 d_rhsw1[1],d_rhsw2[1],d_rhsw3[1],d_rhsw4[1],
											 d_rhsw1[2],d_rhsw2[2],d_rhsw3[2],d_rhsw4[2],&dtC);
    	rk4final<<<grid0,block0>>>(d_e  ,d_rhse1[0],d_rhse2[0],d_rhse3[0],d_rhse4[0],
											 d_rhse1[1],d_rhse2[1],d_rhse3[1],d_rhse4[1],
											 d_rhse1[2],d_rhse2[2],d_rhse3[2],d_rhse4[2],&dtC);
    	cudaDeviceSynchronize();

	}

    __syncthreads();

	for (int i=0; i<3; i++) {
		checkCudaDev( cudaStreamDestroy(s[i]) );
	}
    clearSolver();

}

__global__ void eulerSum(myprec *a, myprec *b, myprec *cx, myprec *cy, myprec *cz, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g] + ( cx[id.g] + cy[id.g] + cz[id.g] )*(*dt);
}

__global__ void rk4final(myprec *a, myprec *bx, myprec *cx, myprec *dx, myprec *ex,	myprec *by, myprec *cy, myprec *dy, myprec *ey,	myprec *bz, myprec *cz, myprec *dz, myprec *ez, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = a[id.g] + (*dt)*( bx[id.g] + 2*cx[id.g] + 2*dx[id.g] + ex[id.g] +
								by[id.g] + 2*cy[id.g] + 2*dy[id.g] + ey[id.g] +
								bz[id.g] + 2*cz[id.g] + 2*dz[id.g] + ez[id.g])/6.;
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

    myprec suth = pow(tem[gt],0.75);
    mu[gt]      = suth/Re;
    lam[gt]     = suth/Re/Pr/Ec;
    __syncthreads();

}

__global__ void calcTimeStep(myprec *temporary, myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *mu) {

	__shared__ myprec temp[mx];

	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int gt = blockNumInGrid * threadsPerBlock + threadNumInBlock;

    myprec dtConvInv = 0.0;
    myprec dtViscInv = 0.0;


    myprec ien = ret[gt]/rho[gt] - 0.5*(uvel[gt]*uvel[gt] + vvel[gt]*vvel[gt] + wvel[gt]*wvel[gt]);
    myprec sos = pow(gamma*(gamma-1)*ien,0.5);

    dtConvInv =  MAX( (abs(uvel[gt]) + sos)*d_dx, MAX( (abs(vvel[gt]) + sos)*d_dy, (abs(wvel[gt]) + sos)*d_dz) );
    dtViscInv =  MAX( mu[gt]*d_d2x, MAX( mu[gt]*d_d2y, mu[gt]*d_d2z) );

    temp[threadNumInBlock] = CFL/MAX(dtConvInv, dtViscInv);
    __syncthreads();

    if(threadNumInBlock == 0) {
    	temporary[blockNumInGrid] = 1000.0;
    	for (int it = 0; it < blockDim.x*blockDim.y; it++)
    		temporary[blockNumInGrid] = MIN(temporary[blockNumInGrid],temp[it] );
    }

    __syncthreads();
}

__device__ void initSolver() {

    for (int i=0; i<3; i++) {
    	checkCudaDev( cudaMalloc((void**)&d_rhsr1[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsr2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsr3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsr4[i],mx*my*mz*sizeof(myprec)) );
		checkCudaDev( cudaMalloc((void**)&d_rhsu1[i],mx*my*mz*sizeof(myprec)) );
		checkCudaDev( cudaMalloc((void**)&d_rhsu2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsu3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsu4[i],mx*my*mz*sizeof(myprec)) );
		checkCudaDev( cudaMalloc((void**)&d_rhsv1[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsv2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsv3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsv4[i],mx*my*mz*sizeof(myprec)) );
		checkCudaDev( cudaMalloc((void**)&d_rhsw1[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsw2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsw3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhsw4[i],mx*my*mz*sizeof(myprec)) );
		checkCudaDev( cudaMalloc((void**)&d_rhse1[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhse2[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhse3[i],mx*my*mz*sizeof(myprec)) );
    	checkCudaDev( cudaMalloc((void**)&d_rhse4[i],mx*my*mz*sizeof(myprec)) );
    }
	checkCudaDev( cudaMalloc((void**)&dttemp,block0.x*block0.y*sizeof(myprec)) );
    for (int i=0; i<9; i++)
    	checkCudaDev( cudaMalloc((void**)&gij[i],mx*my*mz*sizeof(myprec)) );

}

__device__ void clearSolver() {

	for (int i=0; i<3; i++) {
		checkCudaDev( cudaFree(d_rhsr1[i]) );
		checkCudaDev( cudaFree(d_rhsr2[i]) );
		checkCudaDev( cudaFree(d_rhsr3[i]) );
		checkCudaDev( cudaFree(d_rhsr4[i]) );
		checkCudaDev( cudaFree(d_rhsu1[i]) );
		checkCudaDev( cudaFree(d_rhsu2[i]) );
		checkCudaDev( cudaFree(d_rhsu3[i]) );
		checkCudaDev( cudaFree(d_rhsu4[i]) );
		checkCudaDev( cudaFree(d_rhsv1[i]) );
		checkCudaDev( cudaFree(d_rhsv2[i]) );
		checkCudaDev( cudaFree(d_rhsv3[i]) );
		checkCudaDev( cudaFree(d_rhsv4[i]) );
		checkCudaDev( cudaFree(d_rhsw1[i]) );
		checkCudaDev( cudaFree(d_rhsw2[i]) );
		checkCudaDev( cudaFree(d_rhsw3[i]) );
		checkCudaDev( cudaFree(d_rhsw4[i]) );
		checkCudaDev( cudaFree(d_rhse1[i]) );
		checkCudaDev( cudaFree(d_rhse2[i]) );
		checkCudaDev( cudaFree(d_rhse3[i]) );
		checkCudaDev( cudaFree(d_rhse4[i]) );
	}
	checkCudaDev( cudaFree(dttemp) );
    for (int i=0; i<9; i++)
    	checkCudaDev( cudaFree(gij[i]) );

}
