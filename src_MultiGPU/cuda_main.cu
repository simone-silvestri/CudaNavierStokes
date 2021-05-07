#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_main.h"
#include "cuda_math.h"

#if useStreams
	const int fin = 3;
#else
	const int fin = 1;
#endif

void runSimulation(myprec *kin, myprec *enst, myprec *time) {

	/* allocating temporary arrays and streams */
#if useStreams
	void (*RHSDeviceDir[3])(myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*,
							myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*, myprec*);

	RHSDeviceDir[0] = RHSDeviceSharedFlxX;
	RHSDeviceDir[1] = RHSDeviceSharedFlxY;
	RHSDeviceDir[2] = RHSDeviceSharedFlxZ;
#endif

	cudaStream_t s[3];
    for (int i=0; i<3; i++) {
    	checkCuda( cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking) );
    }

	initSolver();
	initStress<<<1,1>>>();

    for (int istep = 0; istep < nsteps; istep++) {


    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
    	cudaDeviceSynchronize();
    	if(istep%100==0) printf("step number %d\n",istep);

    	if(istep%checkCFLcondition==0) {
    		calcTimeStep<<<1,1>>>(dtC,d_r,d_u,d_v,d_w,d_e,d_m);
    		if(forcing)  calcPressureGrad<<<1,1>>>(dpdz,d_r,d_w);
    	}

    	deviceMul<<<grid0,block0>>>(d_uO,d_r,d_u);
    	deviceMul<<<grid0,block0>>>(d_vO,d_r,d_v);
    	deviceMul<<<grid0,block0>>>(d_wO,d_r,d_w);
    	deviceCpy<<<grid0,block0>>>(d_rO,d_r);
    	deviceCpy<<<grid0,block0>>>(d_eO,d_e);

    	/* rk step 1 */
    	cudaDeviceSynchronize();
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w);
    	cudaDeviceSynchronize();

    	if(istep%checkCFLcondition==0) {
    		calcIntegrals<<<1,1>>>(d_r,d_u,d_v,d_w,&kin[istep],&enst[istep]);
    	}
    	cudaDeviceSynchronize();

    	calcDil<<<grid0,block0>>>(d_dil);
    	cudaDeviceSynchronize();

#if useStreams
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr1[d],d_rhsu1[d],d_rhsv1[d],d_rhsw1[d],d_rhse1[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#else
		RHSDeviceSharedFlxX<<<d_grid[0],d_block[0]>>>(d_rhsr1[0],d_rhsu1[0],d_rhsv1[0],d_rhsw1[0],d_rhse1[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceSharedFlxY<<<d_grid[1],d_block[1]>>>(d_rhsr1[0],d_rhsu1[0],d_rhsv1[0],d_rhsw1[0],d_rhse1[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceSharedFlxZ<<<d_grid[2],d_block[2]>>>(d_rhsr1[0],d_rhsu1[0],d_rhsv1[0],d_rhsw1[0],d_rhse1[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#endif
    	cudaDeviceSynchronize();
    	eulerSum<<<grid0,block0>>>(d_r,d_rO,d_rhsr1[0],dtC);
    	cudaDeviceSynchronize();
    	eulerSum<<<grid0,block0>>>(d_e,d_eO,d_rhse1[0],dtC);
    	eulerSumR<<<grid0,block0>>>(d_u,d_uO,d_rhsu1[0],d_r,dtC);
    	eulerSumR<<<grid0,block0>>>(d_v,d_vO,d_rhsv1[0],d_r,dtC);
    	eulerSumR<<<grid0,block0>>>(d_w,d_wO,d_rhsw1[0],d_r,dtC);
    	cudaDeviceSynchronize();

    	//rk step 2
    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w);
		cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(d_dil);
    	cudaDeviceSynchronize();
#if useStreams
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr2[d],d_rhsu2[d],d_rhsv2[d],d_rhsw2[d],d_rhse2[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#else
		RHSDeviceSharedFlxX<<<d_grid[0],d_block[0]>>>(d_rhsr2[0],d_rhsu2[0],d_rhsv2[0],d_rhsw2[0],d_rhse2[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceSharedFlxY<<<d_grid[1],d_block[1]>>>(d_rhsr2[0],d_rhsu2[0],d_rhsv2[0],d_rhsw2[0],d_rhse2[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceSharedFlxZ<<<d_grid[2],d_block[2]>>>(d_rhsr2[0],d_rhsu2[0],d_rhsv2[0],d_rhsw2[0],d_rhse2[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#endif
    	cudaDeviceSynchronize();

    	eulerSum3<<<grid0,block0>>>(d_r,d_rO,d_rhsr1[0],d_rhsr2[0],dtC);
    	cudaDeviceSynchronize();
    	eulerSum3<<<grid0,block0>>>(d_e,d_eO,d_rhse1[0],d_rhse2[0],dtC);
    	eulerSum3R<<<grid0,block0>>>(d_u,d_uO,d_rhsu1[0],d_rhsu2[0],d_r,dtC);
    	eulerSum3R<<<grid0,block0>>>(d_v,d_vO,d_rhsv1[0],d_rhsv2[0],d_r,dtC);
    	eulerSum3R<<<grid0,block0>>>(d_w,d_wO,d_rhsw1[0],d_rhsw2[0],d_r,dtC);
    	cudaDeviceSynchronize();

    	//rk step 3
    	calcState<<<grid0,block0>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l);
		calcStressX<<<d_grid[0],d_block[0],0,s[0]>>>(d_u,d_v,d_w);
		calcStressY<<<d_grid[3],d_block[3],0,s[1]>>>(d_u,d_v,d_w);
		calcStressZ<<<d_grid[4],d_block[4],0,s[2]>>>(d_u,d_v,d_w);
    	cudaDeviceSynchronize();
    	calcDil<<<grid0,block0>>>(d_dil);
    	cudaDeviceSynchronize();
#if useStreams
    	for (int d = 0; d < 3; d++)
    		RHSDeviceDir[d]<<<d_grid[d],d_block[d],0,s[d]>>>(d_rhsr3[d],d_rhsu3[d],d_rhsv3[d],d_rhsw3[d],d_rhse3[d],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#else
		RHSDeviceSharedFlxX<<<d_grid[0],d_block[0]>>>(d_rhsr3[0],d_rhsu3[0],d_rhsv3[0],d_rhsw3[0],d_rhse3[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceSharedFlxY<<<d_grid[1],d_block[1]>>>(d_rhsr3[0],d_rhsu3[0],d_rhsv3[0],d_rhsw3[0],d_rhse3[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
		RHSDeviceSharedFlxZ<<<d_grid[2],d_block[2]>>>(d_rhsr3[0],d_rhsu3[0],d_rhsv3[0],d_rhsw3[0],d_rhse3[0],d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,d_dil,dpdz);
#endif
    	cudaDeviceSynchronize();
    	for (int d=0; d<fin; d++) {
    		rk3final<<<grid0,block0>>>(d_r,d_rO,d_rhsr1[d],d_rhsr2[d],d_rhsr3[d],dtC);
    		cudaDeviceSynchronize();
    		rk3final<<<grid0,block0>>>(d_e,d_eO,d_rhse1[d],d_rhse2[d],d_rhse3[d],dtC);
    		rk3finalR<<<grid0,block0>>>(d_u,d_uO,d_rhsu1[d],d_rhsu2[d],d_rhsu3[d],d_r,dtC);
    		rk3finalR<<<grid0,block0>>>(d_v,d_vO,d_rhsv1[d],d_rhsv2[d],d_rhsv3[d],d_r,dtC);
    		rk3finalR<<<grid0,block0>>>(d_w,d_wO,d_rhsw1[d],d_rhsw2[d],d_rhsw3[d],d_r,dtC);}
    	cudaDeviceSynchronize();

	}

	for (int i=0; i<3; i++) {
		checkCuda( cudaStreamDestroy(s[i]) );
	}

	clearSolver();
	clearStress<<<1,1>>>();
}

__global__ void eulerSum(myprec *a, myprec *b, myprec *c, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	a[id.g] = b[id.g] + ( c[id.g] )*(*dt)/2.0;
}

__global__ void eulerSumR(myprec *a, myprec *b, myprec *c, myprec *r, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
    a[id.g] = (b[id.g] + (c[id.g])*(*dt)/2.0)/r[id.g];
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
	a1[id.g] += ( a2[id.g] + (*dt)*( b[id.g] + 4*c[id.g] + d[id.g])/6. )/ r[id.g];
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

void initSolver() {

    for (int i=0; i<fin; i++) {
    	checkCuda( cudaMalloc((void**)&d_rhsr1[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsu1[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsv1[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsw1[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhse1[i],mx*my*mz*sizeof(myprec)) );

    	checkCuda( cudaMalloc((void**)&d_rhsr2[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsu2[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsv2[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsw2[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhse2[i],mx*my*mz*sizeof(myprec)) );

    	checkCuda( cudaMalloc((void**)&d_rhsr3[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsu3[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsv3[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhsw3[i],mx*my*mz*sizeof(myprec)) );
    	checkCuda( cudaMalloc((void**)&d_rhse3[i],mx*my*mz*sizeof(myprec)) );
    }

	checkCuda( cudaMalloc((void**)&d_h,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_t,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_p,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_m,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_l,mx*my*mz*sizeof(myprec)) );

	checkCuda( cudaMalloc((void**)&d_rO,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_eO,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_uO,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_vO,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_wO,mx*my*mz*sizeof(myprec)) );

	checkCuda( cudaMalloc((void**)&d_dil,mx*my*mz*sizeof(myprec)) );

}

void clearSolver() {

	for(int i=0; i<fin; i++) {
		checkCuda( cudaFree(d_rhsr1[i]) );
		checkCuda( cudaFree(d_rhsu1[i]) );
		checkCuda( cudaFree(d_rhsv1[i]) );
		checkCuda( cudaFree(d_rhsw1[i]) );
		checkCuda( cudaFree(d_rhse1[i]) );

		checkCuda( cudaFree(d_rhsr2[i]) );
		checkCuda( cudaFree(d_rhsu2[i]) );
		checkCuda( cudaFree(d_rhsv2[i]) );
		checkCuda( cudaFree(d_rhsw2[i]) );
		checkCuda( cudaFree(d_rhse2[i]) );

		checkCuda( cudaFree(d_rhsr3[i]) );
		checkCuda( cudaFree(d_rhsu3[i]) );
		checkCuda( cudaFree(d_rhsv3[i]) );
		checkCuda( cudaFree(d_rhsw3[i]) );
		checkCuda( cudaFree(d_rhse3[i]) );
	}
	checkCuda( cudaFree(d_h) );
	checkCuda( cudaFree(d_t) );
	checkCuda( cudaFree(d_p) );
	checkCuda( cudaFree(d_m) );
	checkCuda( cudaFree(d_l) );

	checkCuda( cudaFree(d_rO) );
	checkCuda( cudaFree(d_eO) );
	checkCuda( cudaFree(d_uO) );
	checkCuda( cudaFree(d_vO) );
	checkCuda( cudaFree(d_wO) );

	checkCuda( cudaFree(d_dil) );

}
