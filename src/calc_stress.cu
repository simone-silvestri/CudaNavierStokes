
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_math.h"
#include "boundary_condition_x.h"
#include "boundary_condition_y.h"
#include "boundary_condition_z.h"
#include "comm.h"
#include "sponge.h"

__global__ void deviceAdvanceTime(myprec *dt) {
	time_on_GPU += *dt;
}

__global__ void deviceCalcPress(myprec *a, myprec *b, myprec *c) {
	*a = 0.99*(*b) - 0.5*( *a/(*c) - 1 ); //*a/(*c)-1 //a here is rw_bulk // changed from rw_bulk/r_bulk to rw_bulk
}


__global__ void derVelY(myprec *u, myprec *v, myprec *w, myprec *dudy, myprec *dvdy, myprec *dwdy) {

       Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
       int jNum = blockIdx.z;

       id.mkidYFlx(jNum);

       int tid = id.tix + id.bdx*id.tiy;

       int si = id.tiy + stencilSize;       // local i for shared memory access + halo offset
       int sj = id.tix;                   // local j for shared memory access
       int si1 = tid%id.bdy +  stencilSize;       // local i for shared memory access + halo offset
       int sj1 = tid/id.bdy;

       Indices id1(sj1,si1-stencilSize, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
       id1.mkidYFlx(jNum);

       myprec wrk1=0;
       myprec wrk2=0;

      __shared__ myprec s_u[mx/nDivX][my/nDivY+stencilSize*2];
      __shared__ myprec s_v[mx/nDivX][my/nDivY+stencilSize*2];
      s_u[sj][si] = u[id.g];
      s_v[sj][si] = v[id.g];
      __syncthreads();


	if (id.tiy < stencilSize){
		if (nDivY ==1){
			if(pRow>1) {
				haloBCyTop(s_u[sj],u,si,id); haloBCyTop(s_v[sj],v,si,id);

				haloBCyBot(s_u[sj],u,si,id); haloBCyBot(s_v[sj],v,si,id);
			} else{
  			TopBCyNumber2(s_u[sj],u,id,si,my); TopBCyNumber2(s_v[sj],v,id,si,my); 
  			BotBCyNumber2(s_u[sj],u,id,si,my); BotBCyNumber2(s_v[sj],v,id,si,my); 
			}
		} else {
			if (jNum ==0){
				TopBCyCpy(s_u[sj],u,si,id); TopBCyCpy(s_v[sj],v,si,id); 

				if(pRow>1) {

					haloBCyBot(s_u[sj],u,si,id); haloBCyBot(s_v[sj],v,si,id);
				} else{
  			BotBCyNumber2(s_u[sj],u,id,si,my); BotBCyNumber2(s_v[sj],v,id,si,my); 
				}
			} else if (jNum == nDivY-1){
				BotBCyCpy(s_u[sj],u,si,id); BotBCyCpy(s_v[sj],v,si,id); 
				if(pRow>1) {
					haloBCyTop(s_u[sj],u,si,id); haloBCyTop(s_v[sj],v,si,id); 

				} else {
  			TopBCyNumber2(s_u[sj],u,id,si,my); TopBCyNumber2(s_v[sj],v,id,si,my); 
				}
			} else {
				TopBCyCpy(s_u[sj],u,si,id); TopBCyCpy(s_v[sj],v,si,id); 

				BotBCyCpy(s_u[sj],u,si,id); BotBCyCpy(s_v[sj],v,si,id); 
			}
		}
	}
      __syncthreads();
// COMP Tile sj1 and si1
      derDevSharedV1y(&wrk1,s_u[sj1],si1);
      derDevSharedV1y(&wrk2,s_v[sj1],si1);
      //derDevSharedV1y(&wrk3,s_w[sj1],si1);
      __syncthreads();      
      s_u[sj1][si1] = wrk1;
      s_v[sj1][si1] = wrk2;
     // s_w[sj1][si1] = wrk3;
      __syncthreads();       
//MEM tile sj si      
      dudy[id.g] = s_u[sj][si]; 
      dvdy[id.g] = s_v[sj][si];
      //dwdy[id.g] = s_w[sj][si];
      __syncthreads();

/*	derDevV1yL(dudy,u,id);
	derDevV1yL(dvdy,v,id);
	//derDevV1yL(dwdy,w,id);*/
}

__global__ void derVelZ(myprec *u, myprec *v, myprec *w, myprec *dudz, myprec *dvdz, myprec *dwdz, recycle rec, Communicator rk) {

       Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
       int kNum = blockIdx.z;

       id.mkidZFlx(kNum);

       int tid = id.tix + id.bdx*id.tiy;

       int si = id.tiy + stencilSize;       // local i for shared memory access + halo offset
       int sj = id.tix;                   // local j for shared memory access
       int si1 = tid%id.bdy +  stencilSize;       // local i for shared memory access + halo offset
       int sj1 = tid/id.bdy;

       Indices id1(sj1,si1-stencilSize, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
       id1.mkidZFlx(kNum);

       myprec wrk1=0;
       myprec wrk2=0;
       myprec wrk3=0;

      __shared__ myprec s_u[mx/nDivX][mz/nDivZ+stencilSize*2];
      __shared__ myprec s_v[mx/nDivX][mz/nDivZ+stencilSize*2];
      __shared__ myprec s_w[mx/nDivX][mz/nDivZ+stencilSize*2];
      s_u[sj][si] = u[id.g];
      s_v[sj][si] = v[id.g];
      s_w[sj][si] = w[id.g];
      __syncthreads();



	if (id.tiy < stencilSize){
		if (nDivZ ==1){
			if(pCol > 1) { // because even if its multi gpu but pcol == 1 then we can use directly TopBCzNumber 1 and Bot..
				if (inletbc == 1 || outletbc ==1 ){ // if periodic
					haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);

					haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);
				} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
					if (rk.km == pCol-2){ // last block
						haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);

    		  TopBCzderVel(s_u[sj],s_v[sj],s_w[sj],u,v,w,id,si,mx,outletbc);
					} else if (rk.kp == 1) {// first block
						haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);

    		  BotBCzderVel(s_u[sj],s_v[sj],s_w[sj],u,v,w,id,si,mx,inletbc, rec);
					} else { // all internal blocks
						haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);

						haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);
					}
				}

			} else{
    		  TopBCzderVel(s_u[sj],s_v[sj],s_w[sj],u,v,w,id,si,mx,outletbc);
    		  BotBCzderVel(s_u[sj],s_v[sj],s_w[sj],u,v,w,id,si,mx,inletbc, rec);
			}
		} else {
			if (kNum ==0){
				TopBCzCpy(s_u[sj],u,si,id); TopBCzCpy(s_v[sj],v,si,id); TopBCzCpy(s_w[sj],w,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.kp == 1){ // first block
    		  BotBCzderVel(s_u[sj],s_v[sj],s_w[sj],u,v,w,id,si,mx,inletbc, rec);
						} else { // all other blocks
							haloBCzBot(s_u[sj],u,si,id); haloBCzBot(s_v[sj],v,si,id); haloBCzBot(s_w[sj],w,si,id);
						}
					}

				} else{
    		  BotBCzderVel(s_u[sj],s_v[sj],s_w[sj],u,v,w,id,si,mx,inletbc, rec);
				}
			} else if (kNum == nDivZ-1){
				BotBCzCpy(s_u[sj],u,si,id); BotBCzCpy(s_v[sj],v,si,id); BotBCzCpy(s_w[sj],w,si,id);
				if(pCol > 1) {
					if (inletbc == 1 || outletbc ==1 ){ // if periodic
						haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);
					} else { // bc is something else then periodic. Because periodic is taken care of by the halo exchange(neighbour of last is first)
						if (rk.km == pCol -2 ){ // last block
    		  TopBCzderVel(s_u[sj],s_v[sj],s_w[sj],u,v,w,id,si,mx,outletbc);
						} else { // all other blocks
							haloBCzTop(s_u[sj],u,si,id); haloBCzTop(s_v[sj],v,si,id); haloBCzTop(s_w[sj],w,si,id);
						}
					}
				} else{
    		  TopBCzderVel(s_u[sj],s_v[sj],s_w[sj],u,v,w,id,si,mx,outletbc);
				}
			} else {
				TopBCzCpy(s_u[sj],u,si,id); TopBCzCpy(s_v[sj],v,si,id); TopBCzCpy(s_w[sj],w,si,id);

				BotBCzCpy(s_u[sj],u,si,id); BotBCzCpy(s_v[sj],v,si,id); BotBCzCpy(s_w[sj],w,si,id);
			}
		}
	}

      __syncthreads();

// COMP Tile sj1 and si1
      derDevSharedV1z(&wrk1,s_u[sj1],si1);
      derDevSharedV1z(&wrk2,s_v[sj1],si1);
      derDevSharedV1z(&wrk3,s_w[sj1],si1);
      __syncthreads();      
      s_u[sj1][si1] = wrk1;
      s_v[sj1][si1] = wrk2;
      s_w[sj1][si1] = wrk3;
      __syncthreads();       
//MEM tile sj si      
      dudz[id.g] = s_u[sj][si]; 
      dvdz[id.g] = s_v[sj][si];
      dwdz[id.g] = s_w[sj][si];
      __syncthreads();
	/*derDevV1zL(dudz,u,uInit,id,kNum);
	derDevV1zL(dvdz,v,vInit,id,kNum);
	derDevV1zL(dwdz,w,wInit,id,kNum);*/
}

__global__ void derVelYBC(myprec *u, myprec *v, myprec *w, myprec *dudy, myprec *dvdy, myprec *dwdy, int direction) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	id.mkidYBC(direction);

	derDev1yBC(dudy,u,id,direction);
	derDev1yBC(dvdy,v,id,direction);
	derDev1yBC(dwdy,w,id,direction);
}

__global__ void derVelZBC(myprec *u, myprec *v, myprec *w, myprec *dudz, myprec *dvdz, myprec *dwdz, int direction) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	id.mkidZBC(direction);

	derDev1zBC(dudz,u,id,direction);
	derDev1zBC(dvdz,v,id,direction);
	derDev1zBC(dwdz,w,id,direction);
}

__global__ void calcDil(myprec *dil, myprec *dudx, myprec *dvdy, myprec *dwdz) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();

	//Stress goes with RHS old

	dil[id.g] = dudx[id.g] + dvdy[id.g] + dwdz[id.g];

}

void calcPressureGrad(myprec *dpdz, myprec *r, myprec *w, Communicator rk) {

    cudaSetDevice(rk.nodeRank);

	myprec *workA, *dpdz_prev, *rbulk;
	dim3 gr0  = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);

	checkCuda( cudaMalloc((void**)&workA,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&dpdz_prev,     sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&rbulk    ,     sizeof(myprec)) );

	deviceCpyOne<<<1,1>>>(dpdz_prev,dpdz);
	hostVolumeIntegral(rbulk,r,rk);
	deviceMul<<<gr0,bl0>>>(workA,r,w);
	hostVolumeIntegral(dpdz,workA,rk);

	deviceCalcPress<<<1,1>>>(dpdz,dpdz_prev,rbulk); //I changed here from rbulk to workA. // i CHANGED TO 1

	checkCuda( cudaFree(workA) );
	checkCuda( cudaFree(dpdz_prev) );
	checkCuda( cudaFree(rbulk) );
}

void calcTimeStep(myprec *dt, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu, Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	myprec *workA;

	dim3 gr0 = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);
    checkCuda( cudaMalloc((void**)&workA,mx*my*mz*sizeof(myprec)) );
    deviceCalcDt<<<gr0,bl0>>>(workA,r,u,v,w,e,mu);

	cudaDeviceSynchronize();
	hostReduceToMin(dt,workA,rk);

	checkCuda( cudaFree(workA) );
	cudaDeviceSynchronize();
}

__global__ void deviceCalcDt(myprec *wrkArray, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *mu) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	myprec dtConvInv = 0.0;
    myprec dtViscInv = 0.0;

    myprec ien = e[id.g]/r[id.g] - 0.5*(u[id.g]*u[id.g] + v[id.g]*v[id.g] + w[id.g]*w[id.g]);
    myprec sos = pow(abs(gam*(gam-1)*ien),0.5);

    myprec dx,d2x;
    dx = d_dxv[id.i];
    d2x = dx*dx;

    myprec nu = mu[id.g]/r[id.g];

    dtConvInv =  MAX( (abs(u[id.g]) + sos)/dx, MAX( (abs(v[id.g]) + sos)*d_dy, (abs(w[id.g]) + sos)*d_dz) );
    dtViscInv =  MAX( nu/d2x, MAX( nu*d_d2y, nu*d_d2z) );

    wrkArray[id.g] = CFL/MAX(dtConvInv, dtViscInv);
    __syncthreads();

}

void calcBulk(myprec *par1, myprec *par2, myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *dtC, myprec *dpdz, int file, int istep, Communicator rk) {

	cudaSetDevice(rk.nodeRank);

	myprec *workA, *rbulk;
	myprec *hostWork = (myprec*)malloc(sizeof(myprec));

	dim3 gr0  = dim3(my / sPencils, mz, 1);
	dim3 bl0 = dim3(mx, sPencils, 1);


	checkCuda( cudaMalloc((void**)&workA,mx*my*mz*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&rbulk    ,     sizeof(myprec)) );

	if(forcing) {
		hostVolumeIntegral(rbulk,r,rk);
		checkCuda( cudaMemcpy(hostWork, rbulk, sizeof(myprec), cudaMemcpyDeviceToHost) );
		allReduceSum(hostWork,1);
		checkCuda( cudaMemcpy(rbulk, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );

		deviceMul<<<gr0,bl0>>>(workA,r,w);
		hostVolumeIntegral(par1,workA,rk);
		checkCuda( cudaMemcpy(hostWork, par1, sizeof(myprec), cudaMemcpyDeviceToHost) );
		allReduceSum(hostWork,1);
		checkCuda( cudaMemcpy(par1, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );
		deviceDivOne<<<1,1>>>(par1,par1,rbulk);

		hostVolumeIntegral(par2,e,rk);
		checkCuda( cudaMemcpy(hostWork, par2, sizeof(myprec), cudaMemcpyDeviceToHost) );
		allReduceSum(hostWork,1);
		checkCuda( cudaMemcpy(par2, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );
	} else {
		deviceSca<<<gr0,bl0>>>(workA,u,v,w,u,v,w);
		hostVolumeAverage(par1,workA,rk);
		checkCuda( cudaMemcpy(hostWork, par1, sizeof(myprec), cudaMemcpyDeviceToHost) );
		allReduceSum(hostWork,1);
		checkCuda( cudaMemcpy(par1, hostWork, sizeof(myprec), cudaMemcpyHostToDevice) );
	}
	free(hostWork);
	checkCuda( cudaFree(workA) );
	checkCuda( cudaFree(rbulk) );

	if(rk.rank==0) printf("step number %d with %le %le\n",nsteps*(file-1) + istep ,*dtC,*dpdz);


}
