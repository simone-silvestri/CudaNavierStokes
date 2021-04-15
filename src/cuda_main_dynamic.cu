#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_main.h"
#include "cuda_math.h"

__global__ void runDevice(myprec *kin, myprec *enst, myprec *time) {

	dtC = d_dt;

	/* allocating temporary arrays and streams */
	void (*RHSDeviceDir[5])(myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec**, myprec*);
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
	RHSDeviceDir[3] = FLXDeviceY;
	RHSDeviceDir[4] = FLXDeviceZ;

	__syncthreads();

	cudaStream_t s[5];
    for (int i=0; i<5; i++) {
    	checkCudaDev( cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking) );
    }

    initSolver();
    initRHS();

    for (int istep = 0; istep < nsteps; istep++) {

    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	cudaDeviceSynchronize();
//    	if (istep%100==0) {
//    		calcTimeStep<<<grid0,block0>>>(dttemp,d_r,d_u,d_v,d_w,d_e,d_m);
//    		for (int it=0; it<block0.x*block0.y; it++)
//    			dtC = MIN(dtC,dttemp[it]);
//    	}

    	dt2 = dtC/2.;

    	deviceMul<<<grid0,block0>>>(d_uO,d_r,d_u);
    	deviceMul<<<grid0,block0>>>(d_vO,d_r,d_v);
    	deviceMul<<<grid0,block0>>>(d_wO,d_r,d_w);

    	deviceCpy<<<grid0,block0>>>(d_rO,d_r);
    	deviceCpy<<<grid0,block0>>>(d_eO,d_e);

    	/* rk step 1 */
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		calcStresDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();

    	/* Only if you want a bulk solution changing in time!!! */
    	//if(istep%250==0) {
//    	calcIntegrals<<<1,1>>>(d_r,d_u,d_v,d_w,sij,&kin[istep],&enst[istep]);
    	//}
    	if(istep > 0) 	time[istep] = time[istep-1] + dtC;
    	/* This will take some time to execute so take it away if not needed!! */


    	calcDil<<<grid0,block0>>>(sij,d_dil);
    	for (int d = 0; d < 5; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr1[d],d_rhsu1[d],d_rhsv1[d],d_rhsw1[d],d_rhse1[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil);
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
    	for (int d = 0; d < 3; d++)
    		calcStresDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(sij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 5; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr2[d],d_rhsu2[d],d_rhsv2[d],d_rhsw2[d],d_rhse2[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil);
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
    	for (int d = 0; d < 3; d++)
    		calcStresDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(sij,d_dil);
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 5; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr3[d],d_rhsu3[d],d_rhsv3[d],d_rhsw3[d],d_rhse3[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil);
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
    	cudaDeviceSynchronize();
    	for (int d = 0; d < 3; d++)
    		calcStresDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_u,d_v,d_w,sij);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(sij,d_dil);
    	for (int d = 0; d < 5; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr4[d],d_rhsu4[d],d_rhsv4[d],d_rhsw4[d],d_rhse4[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,sij,d_dil);
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

	for (int i=0; i<5; i++) {
		checkCudaDev( cudaStreamDestroy(s[i]) );
	}
    clearSolver();
    clearRHS();
}

__global__ void eulerSum(myprec *a, myprec *b, myprec *c[5], myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g] + ( c[0][id.g] + c[1][id.g] + c[2][id.g] + c[3][id.g] + c[4][id.g] )*(*dt);
}

__global__ void eulerSumR(myprec *a, myprec *b, myprec *c[5], myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] =  ( b[id.g] +  ( c[0][id.g] + c[1][id.g] + c[2][id.g] + c[3][id.g] + c[4][id.g] ) *(*dt) ) /r[id.g];
}

__global__ void eulerSum3(myprec *a, myprec *b, myprec *c1[5], myprec *c2[5], myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = b[id.g] -     ( c1[0][id.g] + c1[1][id.g] + c1[2][id.g] + c1[3][id.g] + c1[4][id.g] )*(*dt)
					  + 2 * ( c2[0][id.g] + c2[1][id.g] + c2[2][id.g] + c2[3][id.g] + c2[4][id.g] )*(*dt);
}

__global__ void eulerSum3R(myprec *a, myprec *b, myprec *c1[5], myprec *c2[5], myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a[id.g] = ( b[id.g] -     ( c1[0][id.g] + c1[1][id.g] + c1[2][id.g] + c1[3][id.g] + c1[4][id.g] )*(*dt)
					    + 2 * ( c2[0][id.g] + c2[1][id.g] + c2[2][id.g] + c2[3][id.g] + c2[4][id.g] )*(*dt) )/ r[id.g];
}

__global__ void rk4final(myprec *a1, myprec *a2, myprec *b[5], myprec *c[5], myprec *d[5], myprec *e[5], myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a1[id.g] = a2[id.g];
	for (int it=0; it<5; it++)
		a1[id.g] = a1[id.g] + (*dt)*( b[it][id.g] + 2*c[it][id.g] + 2*d[it][id.g] + e[it][id.g])/6.;
}

__global__ void rk4finalR(myprec *a1, myprec *a2, myprec *b[5], myprec *c[5], myprec *d[5], myprec *e[5], myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a1[id.g] =  a2[id.g]/r[id.g];
	for (int it=0; it<5; it++)
		a1[id.g] +=  (*dt)*( b[it][id.g] + 2*c[it][id.g] + 2*d[it][id.g] + e[it][id.g])/6./ r[id.g];
}

__global__ void rk3final(myprec *a1, myprec *a2, myprec *b[5], myprec *c[5], myprec *d[5], myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a1[id.g] = a2[id.g];
	for (int it=0; it<5; it++)
		a1[id.g] = a1[id.g] + (*dt)*( b[it][id.g] + 4*c[it][id.g] + d[it][id.g])/6.;
}

__global__ void rk3finalR(myprec *a1, myprec *a2, myprec *b[5], myprec *c[5], myprec *d[5], myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	a1[id.g] = a2[id.g]/r[id.g];
	for (int it=0; it<5; it++)
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

    for (int i=0; i<5; i++) {
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

//	checkCudaDev( cudaMalloc((void**)&dttemp,block0.x*block0.y*sizeof(myprec)) );

	checkCudaDev( cudaMalloc((void**)&d_dil,mx*my*mz*sizeof(myprec)) );
	for (int i=0; i<9; i++)
    	checkCudaDev( cudaMalloc((void**)&sij[i],mx*my*mz*sizeof(myprec)) );

}

__device__ void clearSolver() {

	for (int i=0; i<5; i++) {
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

	//	checkCudaDev( cudaFree(dttemp) );

	checkCudaDev( cudaFree(d_dil) );
	for (int i=0; i<9; i++)
    	checkCudaDev( cudaFree(sij[i]) );

}
