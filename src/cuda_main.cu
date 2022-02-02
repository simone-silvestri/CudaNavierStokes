#include "globals.h"
#include "cuda_functions.h"
#include "cuda_globals.h"
#include "cuda_main.h"
#include "cuda_math.h"
#include "comm.h"
#include "sponge.h"

__device__ __constant__ myprec alpha[] = {0., -17./60., -5./12.};
__device__ __constant__ myprec beta[]  = {8./15.,   5./12.,  3./4. };

cudaStream_t s[8+nDivZ];

inline void calcRHS(myprec *rhsr, myprec *rhsu, myprec *rhsv, myprec *rhsw, myprec *rhse, Communicator rk, recycle rec) {
	if (topbc == 3) {
		BCwallCenteredTop<<<gridBCw,blockBCw,0,s[0]>>>(d_r,d_u,d_v,d_w,d_e); // The values from RK3 step for the boundary nodes will be not accurate as we already have BCspecified for them and we do not solvecharacteristic equations for them (except for rho).
		// So u[0] may not be 0 after RK3. Thus, we ensure the boundary values are accurate using this step. rho will be accurate at
		// boundary as we solve the NS eqn for rho accurately at the wall (reflecting BC).
	}
	if (bottombc == 3) {
		BCwallCenteredBot<<<gridBCw,blockBCw,0,s[0]>>>(d_r,d_u,d_v,d_w,d_e);

	}
	if(multiGPU) deviceBlocker<<<grid0,block0,0,s[0]>>>();   //in order to hide the halo update with deviceRHSX (on stream s[0])
	calcState<<<grid0,block0,0,s[0]>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l,0); //here 0 means interior points
	//    if(multiGPU) deviceBlocker<<<grid0,block0,0,s[0]>>>();   //in order to hide the halo update with deviceRHSX (on stream s[0])
	deviceRHSX<<<d_gridx,d_blockx,0,s[0]>>>(rhsr,rhsu,rhsv,rhsw,rhse,d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij[0],gij[1],gij[2],gij[3],gij[6],gij[4],gij[8],d_dil,dpdz);

	if(multiGPU) {
		updateHaloFive(d_r,d_u,d_v,d_w,d_e,rk);
		cudaDeviceSynchronize();
		calcState<<<gridHalo,blockHalo,0,s[4]>>>(d_r,d_u,d_v,d_w,d_e,d_h,d_t,d_p,d_m,d_l,1); //here 1 means halo points
		//        derVelYBC<<<gridHaloY,blockHaloY,0,s[0]>>>(d_u,d_v,d_w,gij[3],gij[4],gij[5],0);  //here 0 means lower boundary (0-index)
		//        derVelZBC<<<gridHaloZ,blockHaloZ,0,s[1]>>>(d_u,d_v,d_w,gij[6],gij[7],gij[8],0);	//here 0 means lower boundary (0-index)
		//        derVelYBC<<<gridHaloY,blockHaloY,0,s[2]>>>(d_u,d_v,d_w,gij[3],gij[4],gij[5],1);	//here 1 means upper boundary (my-index)
		//        derVelZBC<<<gridHaloZ,blockHaloZ,0,s[3]>>>(d_u,d_v,d_w,gij[6],gij[7],gij[8],1);	//here 1 means upper boundary (mz-index)
	}

	if(multiGPU) updateHalo(d_dil,rk);
	cudaDeviceSynchronize();
	deviceRHSY<<<d_grid[1],d_block[1]>>>(rhsr,rhsu,rhsv,rhsw,rhse,d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij[1],gij[3],gij[4],gij[5],gij[7],d_dil,dpdz);
	cudaDeviceSynchronize();
	deviceRHSZ<<<d_grid[2],d_block[2],0,s[8]>>>(rhsr,rhsu,rhsv,rhsw,rhse,d_r,d_u,d_v,d_w,d_h,d_t,d_p,d_m,d_l,gij[2],gij[5],gij[6],gij[7],gij[8],d_dil,dpdz, rk,rec);
	cudaDeviceSynchronize();
	//if(boundaryLayer) addSponge<<<d_grid[0],d_block[0]>>>(rhsr,rhsu,rhsv,rhsw,rhse,d_r,d_u,d_v,d_w,d_e);
	cudaDeviceSynchronize();
}

void runSimulation(int file, myprec *par1, myprec *par2, myprec *time, Communicator rk) {

	cudaSetDevice(rk.nodeRank);
	myprec h_dt,h_dpdz;

	for (int istep = 0; istep < nsteps; istep++) {
		if (topbc == 3) {
			BCwallCenteredTop<<<gridBCw,blockBCw,0,s[0]>>>(d_r,d_u,d_v,d_w,d_e); // The values from RK3 step for the boundary nodes will be not accurate as we already have BCspecified for them and we do not solvecharacteristic equations for them (except for rho).
			// So u[0] may not be 0 after RK3. Thus, we ensure the boundary values are accurate using this step. rho will be accurate at
			// boundary as we solve the NS eqn for rho accurately at the wall (reflecting BC).
		}
		if (bottombc == 3) {
			BCwallCenteredBot<<<gridBCw,blockBCw,0,s[0]>>>(d_r,d_u,d_v,d_w,d_e);

		}
		cudaDeviceSynchronize();

		if(istep%checkCFLcondition==0) calcTimeStepPressGrad(istep,dtC,dpdz,&h_dt,&h_dpdz,rk);
		if(istep>0)  deviceSumOne<<<1,1>>>(&time[istep],&time[istep-1],dtC);
		if(istep==0) deviceSumOne<<<1,1>>>(&time[istep],&time[nsteps-1],dtC);
		deviceAdvanceTime<<<1,1>>>(dtC);
		if(istep%checkBulk==0) calcBulk(&par1[istep],&par2[istep],d_r,d_u,d_v,d_w,d_e,&h_dt,&h_dpdz,file, istep, rk);

		int notanum=0;
                if (h_dt != h_dt) notanum = 1;
                if (notanum>0){
                    if(rk.rank==0) printf("Encountered a NAN in dt\n");
                    exit(1);
                }

		//Recycling-Rescaling
		if (inletbc == 5) {
			if (( rk.kstart <= recstn) && ( rk.kend >= recstn) ) { // this section will be executed only by the processors that includes recycling stn.
				if (abs(recstn - rk.kstart) < stencilSize-1) {
					printf("The Recycling station is too close to the processor left boundary.");
					exit(1);
				}
				if(istep%checkMeanRec==0) {
					calcMeanRec(d_r, d_u, d_w, d_t, rm, um, wm, tm, a_inpl, b_inpl, idxm, idxp, delta_rec, delta_in,rk);
				}
				cudaDeviceSynchronize();
				Recycle_Rescale(d_r, d_u, d_v, d_w, d_t, rm, um, wm, tm, rm_in, um_in, wm_in, tm_in, recy_r,
						recy_u, recy_v, recy_w, recy_t, a_inpl, b_inpl, idxm, idxp, delta_rec, delta_in,rk);
				cudaDeviceSynchronize();
			}
			if (pCol > 1) {
				TransferToInlet(recy_r, recy_u, recy_v, recy_w, recy_t, rk);
				cudaDeviceSynchronize();
			}

			if (pRow==1 && spanshift) {
				Spanwiseshift(recy_r, recy_u, recy_v, recy_w, recy_t, rk);
				cudaDeviceSynchronize();
			}




			dim3 gridcalcstateRR;
			gridcalcstateRR = dim3(my, stencilSize, 1);
			calcStateRR<<<gridcalcstateRR, mx_tot>>>(recy_r, recy_u, recy_v, recy_w, recy_e, recy_h, recy_t, recy_p, recy_m, recy_l);
		}
		cudaDeviceSynchronize();

			if ((istep%100==0)){
				/*myprec *h_wminl = (myprec*)malloc(mx*sizeof(myprec));
				myprec *wminl;
				checkCuda( cudaMalloc( (void**)&wminl, mx_tot*sizeof(myprec) ) );
				int gridYAvg = mx_tot;
				int blockYAvg = my;
				YAvg<<<gridYAvg,blockYAvg,blockYAvg*2*sizeof(myprec),s[0]>>>(recy_w, wminl, stencilSize-1, 0);
			        cudaDeviceSynchronize();	
				checkCuda( cudaMemcpy(h_wminl, wminl, mx_tot*sizeof(myprec), cudaMemcpyDeviceToHost) );
				allReduceSumYavg(h_wminl,mx_tot);*/

				myprec *h_delre = (myprec*)malloc(stencilSize*sizeof(myprec));
				checkCuda( cudaMemcpy(h_delre, delta_rec, stencilSize*sizeof(myprec), cudaMemcpyDeviceToHost) );
			
				myprec *h_wm = (myprec*)malloc(stencilSize*mx*sizeof(myprec));
				checkCuda( cudaMemcpy(h_wm, wm, mx*stencilSize*sizeof(myprec), cudaMemcpyDeviceToHost) );
        
				myprec deltastar=0;
				int kn = stencilSize-1;
                                myprec Uinv = 1/h_wm[mx-1 + kn*mx];
                                for (int i=0; i<mx; i++){
                                        myprec dx = Lx/(mx-1)/xp[i]/h_delre[kn] ;
                                        if (i==0 || i== mx-1){
                                               dx = 0.5*dx;
                                        }
                                        deltastar += (1-h_wm[i+ kn*mx]*Uinv)*dx;
                                }
                                if(rk.rank==5){
                                        FILE *fdel = fopen("delstar.txt","a+");
                                        fprintf(fdel,"%lf\n",deltastar);
                                        fclose(fdel);
                                }

                                free(h_wm); 
			        free(h_delre);
			}
                mpiBarrier();

		if(rk.rank==5 && istep==nsteps-1){
			myprec *recyt = (myprec*)malloc(stencilSize*my*mx_tot*sizeof(myprec));
			myprec *tt = (myprec*)malloc(mz*my*mx_tot*sizeof(myprec));
			myprec *din = (myprec*)malloc(stencilSize*sizeof(myprec));
			myprec *dre = (myprec*)malloc(stencilSize*sizeof(myprec));
			myprec *tmin = (myprec*)malloc(stencilSize*mx*sizeof(myprec));
			myprec *tmre = (myprec*)malloc(stencilSize*mx*sizeof(myprec));



			checkCuda( cudaMemcpy(recyt, recy_w, stencilSize*my*mx_tot*sizeof(myprec), cudaMemcpyDeviceToHost) );
			checkCuda( cudaMemcpy(tt, d_w, mz*my*mx_tot*sizeof(myprec), cudaMemcpyDeviceToHost) );
			checkCuda( cudaMemcpy(din, delta_in, stencilSize*sizeof(myprec), cudaMemcpyDeviceToHost) );
			checkCuda( cudaMemcpy(dre, delta_rec, stencilSize*sizeof(myprec), cudaMemcpyDeviceToHost) );
			checkCuda( cudaMemcpy(tmre, wm, stencilSize*mx*sizeof(myprec), cudaMemcpyDeviceToHost) );
			checkCuda( cudaMemcpy(tmin, wm_in, stencilSize*mx*sizeof(myprec), cudaMemcpyDeviceToHost) );


			FILE *fp = fopen("Rec1.txt","w+");
			int j = my/2;
			int kt = recstn-rk.kstart;
			for (int i=0; i<mx; i++)
				fprintf(fp,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",x[i],x[i]/din[2],x[i]/dre[2],recyt[i+j*mx+2*mx*my],tmin[i+2*mx], tt[i+j*mx+kt*mx*my], tmre[i+2*mx]);
			fclose(fp);

			free(recyt); free(tt); free(din); free(dre); free(tmin); free(tmre);

		}



		recycle rec(recy_r, recy_u, recy_v, recy_w, recy_e, recy_h, recy_t, recy_p, recy_m, recy_l);
		cudaDeviceSynchronize();

		deviceMul<<<grid0,block0,0,s[0]>>>(d_uO,d_r,d_u);
		deviceMul<<<grid0,block0,0,s[1]>>>(d_vO,d_r,d_v);
		deviceMul<<<grid0,block0,0,s[2]>>>(d_wO,d_r,d_w);
		deviceCpy<<<grid0,block0,0,s[3]>>>(d_rO,d_r);
		deviceCpy<<<grid0,block0,0,s[4]>>>(d_eO,d_e);

		//Starting the Runge-Kutta Steps

		//runge kutta step 1
		calcRHS(d_rhsr1,d_rhsu1,d_rhsv1,d_rhsw1,d_rhse1,rk,rec);

		//eulerSum<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,dtC);
		//eulerSum<<<grid0,block0>>>(d_e,d_eO,d_rhse1,dtC);

		eulerSumAll<<<grid0,block0>>>(d_r, d_rO, d_u, d_uO, d_v, d_vO, d_w, d_wO, d_e, d_eO,
				d_rhsr1, d_rhsu1, d_rhsv1, d_rhsw1, d_rhse1,dtC);
		//cudaDeviceSynchronize();
		//eulerSumR<<<grid0,block0,0,s[0]>>>(d_u,d_uO,d_rhsu1,d_r,dtC);
		//eulerSumR<<<grid0,block0,0,s[1]>>>(d_v,d_vO,d_rhsv1,d_r,dtC);
		//eulerSumR<<<grid0,block0,0,s[2]>>>(d_w,d_wO,d_rhsw1,d_r,dtC);
		cudaDeviceSynchronize();

		/*      if(multiGPU) {  //To initiate slowly the routines so that we have time to initiate the memory transfer
            deviceCpy<<<grid0,block0,0,s[0]>>>(d_r,d_r);
            deviceCpy<<<grid0,block0,0,s[1]>>>(d_u,d_u);
            deviceCpy<<<grid0,block0,0,s[2]>>>(d_v,d_v);
            deviceCpy<<<grid0,block0,0,s[3]>>>(d_w,d_w);
        }*/

		//runge kutta step 2
		calcRHS(d_rhsr2,d_rhsu2,d_rhsv2,d_rhsw2,d_rhse2,rk,rec);
		//eulerSum3<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,dtC);
		//eulerSum3<<<grid0,block0>>>(d_e,d_eO,d_rhse1,d_rhse2,dtC);
		eulerSumAll2<<<grid0,block0>>>(d_r, d_rO, d_u, d_uO, d_v, d_vO, d_w, d_wO, d_e, d_eO,
				d_rhsr1, d_rhsu1, d_rhsv1, d_rhsw1, d_rhse1,d_rhsr2, d_rhsu2, d_rhsv2, d_rhsw2, d_rhse2, dtC);
		//cudaDeviceSynchronize();
		//eulerSum3R<<<grid0,block0,0,s[0]>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_r,dtC);
		//eulerSum3R<<<grid0,block0,0,s[1]>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_r,dtC);
		//eulerSum3R<<<grid0,block0,0,s[2]>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_r,dtC);
		cudaDeviceSynchronize();

		/* if(multiGPU) {  //To initiate slowly the routines so that we have time to initiate the memory transfer
            deviceCpy<<<grid0,block0,0,s[0]>>>(d_r,d_r);
            deviceCpy<<<grid0,block0,0,s[1]>>>(d_u,d_u);
            deviceCpy<<<grid0,block0,0,s[2]>>>(d_v,d_v);
            deviceCpy<<<grid0,block0,0,s[3]>>>(d_w,d_w);
        }*/

		//runge kutta step 3
		calcRHS(d_rhsr3,d_rhsu3,d_rhsv3,d_rhsw3,d_rhse3,rk,rec);
		//		rk3final<<<grid0,block0>>>(d_r,d_rO,d_rhsr1,d_rhsr2,d_rhsr3,dtC);
		//		rk3final<<<grid0,block0>>>(d_e,d_eO,d_rhse1,d_rhse2,d_rhse3,dtC);
		eulerSumAll3<<<grid0,block0>>>(d_r, d_rO, d_u, d_uO, d_v, d_vO, d_w, d_wO, d_e, d_eO,
				d_rhsr1, d_rhsu1, d_rhsv1, d_rhsw1, d_rhse1,d_rhsr2, d_rhsu2, d_rhsv2, d_rhsw2, d_rhse2,
				d_rhsr3, d_rhsu3, d_rhsv3, d_rhsw3, d_rhse3,dtC);

		//		cudaDeviceSynchronize();
		//		rk3finalR<<<grid0,block0,0,s[0]>>>(d_u,d_uO,d_rhsu1,d_rhsu2,d_rhsu3,d_r,dtC);
		//		rk3finalR<<<grid0,block0,0,s[1]>>>(d_v,d_vO,d_rhsv1,d_rhsv2,d_rhsv3,d_r,dtC);
		//		rk3finalR<<<grid0,block0,0,s[2]>>>(d_w,d_wO,d_rhsw1,d_rhsw2,d_rhsw3,d_r,dtC);
		cudaDeviceSynchronize();
	}
}

void runSimulationLowStorage(myprec *par1, myprec *par2, myprec *time, Communicator rk) {

	/*    cudaSetDevice(rk.nodeRank);

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
            deviceCpy<<<grid0,block0,0,s[3]>>>(d_w,d_w);
        }


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
            deviceCpy<<<grid0,block0,0,s[3]>>>(d_w,d_w);
        }

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
    }*/
}

__global__ void eulerSumAll(myprec *r, myprec *r0,myprec *u, myprec *u0, myprec *v, myprec *v0,myprec *w,
		myprec *w0, myprec *e, myprec *e0, myprec *rhsr1, myprec *rhsu1, myprec *rhsv1, myprec *rhsw1, myprec *rhse1
		,myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	myprec tmp1 = 0;
	tmp1 = (r0[id.g] + rhsr1[id.g]*(*dt)/2.0);
	r[id.g] = tmp1;
	u[id.g] = (u0[id.g] + rhsu1[id.g]*(*dt)/2.0)/tmp1;
	v[id.g] = (v0[id.g] + rhsv1[id.g]*(*dt)/2.0)/tmp1;
	w[id.g] = (w0[id.g] + rhsw1[id.g]*(*dt)/2.0)/tmp1;
	e[id.g] = (e0[id.g] + rhse1[id.g]*(*dt)/2.0);

}


__global__ void eulerSumAll2(myprec *r, myprec *r0,myprec *u, myprec *u0, myprec *v, myprec *v0,myprec *w,
		myprec *w0, myprec *e, myprec *e0, myprec *rhsr1, myprec *rhsu1, myprec *rhsv1, myprec *rhsw1, myprec *rhse1,
		myprec *rhsr2, myprec *rhsu2, myprec *rhsv2, myprec *rhsw2, myprec *rhse2, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	myprec tmp1 = 0;
	tmp1 =    ( r0[id.g] + (2*rhsr2[id.g] - rhsr1[id.g])*(*dt) );
	r[id.g] = tmp1;
	u[id.g] = ( u0[id.g] + (2*rhsu2[id.g] - rhsu1[id.g])*(*dt) )/tmp1;
	v[id.g] = ( v0[id.g] + (2*rhsv2[id.g] - rhsv1[id.g])*(*dt) )/tmp1;
	w[id.g] = ( w0[id.g] + (2*rhsw2[id.g] - rhsw1[id.g])*(*dt) )/tmp1;
	e[id.g] = ( e0[id.g] + (2*rhse2[id.g] - rhse1[id.g])*(*dt) );

}


__global__ void eulerSumAll3(myprec *r, myprec *r0,myprec *u, myprec *u0, myprec *v, myprec *v0,myprec *w,
		myprec *w0, myprec *e, myprec *e0, myprec *rhsr1, myprec *rhsu1, myprec *rhsv1, myprec *rhsw1, myprec *rhse1,
		myprec *rhsr2, myprec *rhsu2, myprec *rhsv2, myprec *rhsw2, myprec *rhse2,
		myprec *rhsr3, myprec *rhsu3, myprec *rhsv3, myprec *rhsw3, myprec *rhse3, myprec *dt) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	myprec tmp1 = 0;
	tmp1    =    r0[id.g] + (*dt)*( rhsr1[id.g] + 4*rhsr2[id.g] + rhsr3[id.g])/6.;
	r[id.g] = tmp1;
	u[id.g] = (  u0[id.g] + (*dt)*( rhsu1[id.g] + 4*rhsu2[id.g] + rhsu3[id.g])/6.)/tmp1;
	v[id.g] = (  v0[id.g] + (*dt)*( rhsv1[id.g] + 4*rhsv2[id.g] + rhsv3[id.g])/6.)/tmp1;
	w[id.g] = (  w0[id.g] + (*dt)*( rhsw1[id.g] + 4*rhsw2[id.g] + rhsw3[id.g])/6.)/tmp1;
	e[id.g] = (  e0[id.g] + (*dt)*( rhse1[id.g] + 4*rhse2[id.g] + rhse3[id.g])/6.);

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


	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidX();
	int gl = id.g;
	if(bc==1) {
		int threadsPerBlock  = blockDim.x * blockDim.y;
		int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
		int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
		gl = blockNumInGrid * threadsPerBlock + threadNumInBlock;
		gl += mx*my*mz;
	}
	myprec cvInv = (gam - 1.0)/Rgas;
	myprec r = rho[gl];
	myprec u = uvel[gl];
	myprec v = vvel[gl];
	myprec w = wvel[gl];
	myprec etot = ret[gl];

	myprec invrho = 1.0/r;
	myprec en = etot*invrho - 0.5*(u*u + v*v + w*w);

	myprec  t   = cvInv*en;
	myprec  p   = r*Rgas*t;
	tem[gl]     = t;
	pre[gl]     = p;

	ht[gl]      = (etot + p)*invrho;

	myprec suth = pow(t,viscexp);
	mu[gl]      = suth/Re;
	lam[gl]     = suth/Re/Pr/Ec;

	__syncthreads();

}

__global__ void calcStateRR(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret, myprec *ht, myprec *tem, myprec *pre, myprec *mu, myprec *lam) {


	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidBCRR(blockIdx.y);
	int gl = id.g;

	myprec cv = Rgas/(gam - 1.0) ;
	myprec r  = rho[gl];
	myprec u  = uvel[gl];
	myprec v  = vvel[gl];
	myprec w  = wvel[gl];
	myprec t  = tem[gl];

	myprec en = cv*t ;

	myprec  etot   = r*( en + 0.5*(u*u + v*v + w*w) ) ;
	myprec  p      = r*Rgas*t;
	ret[gl]        = etot;
	pre[gl]        = p;

	ht[gl]         = (etot + p)/r;

	myprec suth = pow(t,viscexp);
	mu[gl]      = suth/Re;
	lam[gl]     = suth/Re/Pr/Ec;

	__syncthreads();

}

__global__ void BCwallCenteredTop(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret) {


	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidBCwallTop();


	myprec u = 0.0;
	myprec v = 0.0;
	myprec w = 0.0;
	myprec t = TwallTop;

	uvel[id.g] = u;
	vvel[id.g] = v;
	wvel[id.g] = w;

	myprec cv = Rgas/(gam - 1.0);
	myprec r = rho[id.g];

	myprec en = cv*t;

	myprec etot = r*( en + 0.5*(u*u + v*v + w*w) );

	ret[id.g] = etot;

	__syncthreads();

}
__global__ void BCwallCenteredBot(myprec *rho, myprec *uvel, myprec *vvel, myprec *wvel, myprec *ret) {


	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidBCwallBot();


	myprec u = 0.0;
	myprec v = 0.0;
	myprec w = 0.0;
	myprec t = TwallBot;

	uvel[id.g] = u;
	vvel[id.g] = v;
	wvel[id.g] = w;

	myprec cv = Rgas/(gam - 1.0);
	myprec r = rho[id.g];

	myprec en = cv*t;

	myprec etot = r*( en + 0.5*(u*u + v*v + w*w) );

	ret[id.g] = etot;

	__syncthreads();

}


__global__ void sumLowStorageRK3(myprec *var, myprec *rhs1, myprec *rhs2, myprec *dt, int step) {
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	var[id.g] = var[id.g] + (*dt)*(alpha[step]*rhs1[id.g] + beta[step]*rhs2[id.g]);
}

void calcTimeStepPressGrad(int istep, myprec *dtC, myprec *dpdz, myprec *h_dt, myprec *h_dpdz, Communicator rk) {

	cudaSetDevice(rk.nodeRank);
	calcTimeStep(dtC,d_r,d_u,d_v,d_w,d_e,d_m,rk);
	cudaMemcpy(h_dt, dtC, sizeof(myprec), cudaMemcpyDeviceToHost);
	allReduceToMin(h_dt,1);
	mpiBarrier();
	cudaMemcpy(dtC, h_dt, sizeof(myprec), cudaMemcpyHostToDevice);
	if(forcing) {
		calcPressureGrad(dpdz,d_r,d_w,rk); //Changed here dpdz_new = dpdz_old - 0.5*(rw_bulk - 1 )
		cudaMemcpy(h_dpdz, dpdz, sizeof(myprec), cudaMemcpyDeviceToHost);
		allReduceArray(h_dpdz,1);
		mpiBarrier();
		cudaMemcpy(dpdz, h_dpdz, sizeof(myprec), cudaMemcpyHostToDevice);
	}
	//if(rk.rank==0) printf("step number %d with %le %le\n",nsteps*(file) + istep ,*h_dt,*h_dpdz);
}

void calcMeanRec(myprec *r, myprec *u,myprec *w, myprec *t, myprec *rm, myprec *um, myprec *wm, myprec *tm, myprec *a, myprec *b, int *idxm, int *idxp, myprec *delta_rec, myprec *delta_in,Communicator rk) {

	myprec *hostYmean = (myprec*)malloc(stencilSize*mx_tot*sizeof(myprec));

	int gridYAvg = mx_tot;
	int blockYAvg = my;

	for (int k = 0; k < stencilSize; k++){
	int krec = recstn - rk.kstart - (stencilSize-1) + k; // stencilSize-1 is there becase we want k = 0 coreesponding to plane recystn-2, k=1 correspond to plane recycstn-1 and k=2 to recycstn
	YAvg<<<gridYAvg,blockYAvg,blockYAvg*2*sizeof(myprec),s[0]>>>(r, rm, krec, k);// *2 because the next power of 2 shall be <= 2*my. Eg: 139.. next power = 256 < 278
	YAvg<<<gridYAvg,blockYAvg,blockYAvg*2*sizeof(myprec),s[0]>>>(u, um, krec, k); //wall normal velocity
	YAvg<<<gridYAvg,blockYAvg,blockYAvg*2*sizeof(myprec),s[0]>>>(w, wm, krec, k); //streamwise velocity
	YAvg<<<gridYAvg,blockYAvg,blockYAvg*2*sizeof(myprec),s[0]>>>(t, tm, krec, k);
	cudaDeviceSynchronize();
	}

	checkCuda( cudaMemcpy(hostYmean, rm, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyDeviceToHost) );
	allReduceSumYavg(hostYmean,stencilSize*mx_tot); // check if mpi_barrier is needed with comm_col communicator?
	checkCuda( cudaMemcpy(rm, hostYmean, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyHostToDevice) );

	checkCuda( cudaMemcpy(hostYmean, um, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyDeviceToHost) );
	allReduceSumYavg(hostYmean,stencilSize*mx_tot);
	checkCuda( cudaMemcpy(um, hostYmean, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyHostToDevice) );

	checkCuda( cudaMemcpy(hostYmean, wm, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyDeviceToHost) );
	allReduceSumYavg(hostYmean,stencilSize*mx_tot);
	checkCuda( cudaMemcpy(wm, hostYmean, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyHostToDevice) );

	checkCuda( cudaMemcpy(hostYmean, tm, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyDeviceToHost) );
	allReduceSumYavg(hostYmean,stencilSize*mx_tot);
	checkCuda( cudaMemcpy(tm, hostYmean, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyHostToDevice) );

	deltaRR<<<1, stencilSize,0,s[0]>>>(wm, delta_rec);
	interpRR<<<stencilSize, mx_tot,0,s[0]>>>(a, b, idxm, idxp, delta_rec, delta_in) ;

	free(hostYmean);
}

void Recycle_Rescale(myprec *r, myprec *u, myprec *v, myprec *w, myprec *t, myprec *rm, myprec *um, myprec *wm, myprec *tm,
		myprec *rm_in, myprec *um_in, myprec *wm_in, myprec *tm_in, myprec *recy_r, myprec *recy_u,
		myprec *recy_v, myprec *recy_w, myprec *recy_t,myprec *a, myprec *b, int *idxm, int *idxp, myprec *delta_rec, myprec *delta_in, Communicator rk) {

	dim3 gridRR;
	gridRR = dim3(my,stencilSize,1);
	int blockRR = mx_tot;

	Recy_Resc<<<gridRR,blockRR,0,s[0]>>>(r, u, v, w, t, rm, um, wm, tm, rm_in, um_in, wm_in, tm_in, recy_r, recy_u, recy_v, recy_w, recy_t,
			a, b, idxm, idxp, delta_rec, delta_in, rk);


}


void InletMeanUpdate(myprec *rm_in, myprec *um_in, myprec *wm_in, myprec *tm_in, myprec *delta_in, Communicator rk) {

	myprec *um_inRR      = (myprec*)malloc(stencilSize*mx_tot*sizeof(myprec));
	myprec *wm_inRR      = (myprec*)malloc(stencilSize*mx_tot*sizeof(myprec));
	myprec *tm_inRR      = (myprec*)malloc(stencilSize*mx_tot*sizeof(myprec));
	myprec *rm_inRR      = (myprec*)malloc(stencilSize*mx_tot*sizeof(myprec));
	myprec *delta99_inRR = (myprec*)malloc(stencilSize*sizeof(myprec));

	readFileInitBL_inRR('r',rm_inRR,rk);	
	readFileInitBL_inRR('u',um_inRR,rk);
	readFileInitBL_inRR('w',wm_inRR,rk);
	readFileInitBL_inRR('t',tm_inRR,rk);
	readFileInitBL1D_inRR('d',delta99_inRR,rk); 

	checkCuda( cudaMemcpy(rm_in, rm_inRR, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(um_in, um_inRR, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(wm_in, wm_inRR, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(tm_in, tm_inRR, stencilSize*mx_tot*sizeof(myprec), cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(delta_in, delta99_inRR, stencilSize*sizeof(myprec), cudaMemcpyHostToDevice) );

	free(um_inRR);
	free(wm_inRR);
	free(tm_inRR);
	free(rm_inRR);
	free(delta99_inRR);
}

void Spanwiseshift(myprec *recy_r,myprec *recy_u,myprec *recy_v,myprec *recy_w,myprec *recy_t, Communicator rk){

	dim3 grid;
	grid = dim3(mx_tot, stencilSize, 1);
      
	spanshifting<<<grid, my,0,s[0]>>>(recy_r);
	spanshifting<<<grid, my,0,s[0]>>>(recy_u);
	spanshifting<<<grid, my,0,s[0]>>>(recy_v);
	spanshifting<<<grid, my,0,s[0]>>>(recy_w);
	spanshifting<<<grid, my,0,s[0]>>>(recy_t);

}

__global__ void spanshifting(myprec *var){

	int k = blockIdx.y;
	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidSpanShift(k,0);
	__shared__ myprec s_var[my];
        
        s_var[id.tix] = var[id.g];
        __syncthreads();
	int shift = my/2;
	Indices id1(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id1.mkidSpanShift(k,shift);
        var[id1.g] = s_var[id.tix] ;
        __syncthreads();

}


__global__ void Recy_Resc(myprec *r, myprec *u, myprec *v, myprec *w, myprec *t, myprec *rm, myprec *um, myprec *wm, myprec *tm,
		myprec *rm_in, myprec *um_in, myprec *wm_in, myprec *tm_in, myprec *recy_r, myprec *recy_u,
		myprec *recy_v, myprec *recy_w, myprec *recy_t,myprec *a, myprec *b, int *idxm, int *idxp, myprec *delta_rec, myprec *delta_in, Communicator rk) {

	int k = blockIdx.y;
	int krec = (recstn-stencilSize+1) - rk.kstart + k; // stencilSize+1 is there becase we want k = 0 coreesponding to plane recystn-2, k=1 correspond to plane recycstn-1 and k=2 to recycstn

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidBCRR(krec);

	Indices id1(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id1.mkidBCRR(k);

	int id_mean = id.i + mx_tot*k ; //

	int idm = idxm[id_mean];
	int idp = idxp[id_mean];

	myprec delratio = delta_rec[k]/delta_in[k] ;
	myprec beta_rec = pow(abs(delratio),0.1); // (delta_rec / delta_in)^(1/10) -- delta_in = 1;

	__shared__ myprec s_fluc[mx_tot];
	__shared__ myprec s_mean[mx_tot];

/*	s_mean[id.i] = um[id_mean] ;
        __syncthreads();
        um_in[id_mean] = 1*( a[id_mean] * s_mean[idm] + (1.0-a[id_mean])* s_mean[idp] ) ;// not using b as I dont want um_in=0 which will happen when a=0&b=0.
*/

	__syncthreads();
	s_fluc[id.i] = r[id.g] - rm[id_mean] ;
	__syncthreads();
	myprec flucinlet = 1*( a[id_mean] * s_fluc[idm] + b[id_mean] * s_fluc[idp] ) ;
	recy_r[id1.g] = rm_in[id_mean] + flucinlet;

	__syncthreads();
	s_fluc[id.i] = u[id.g] - um[id_mean] ;
	__syncthreads();
	flucinlet = beta_rec*( a[id_mean] * s_fluc[idm] + b[id_mean] * s_fluc[idp] ) ;
	recy_u[id1.g] = um_in[id_mean] + flucinlet;

	__syncthreads();
	s_fluc[id.i] = v[id.g] - 0 ;
	__syncthreads();
	flucinlet = beta_rec*( a[id_mean] * s_fluc[idm] + b[id_mean] * s_fluc[idp] ) ;
	recy_v[id1.g] = 0 + flucinlet;

	__syncthreads();
	s_fluc[id.i] = w[id.g] - wm[id_mean] ;
	__syncthreads();
	flucinlet = beta_rec*( a[id_mean] * s_fluc[idm] + b[id_mean] * s_fluc[idp] ) ;
	recy_w[id1.g] = wm_in[id_mean] + flucinlet;

	__syncthreads();
	s_fluc[id.i] = t[id.g] - tm[id_mean] ;
	__syncthreads();
	flucinlet = 1*( a[id_mean] * s_fluc[idm] + b[id_mean] * s_fluc[idp] ) ;
	recy_t[id1.g] = tm_in[id_mean] + flucinlet;

}


__global__ void YAvg(myprec *f, myprec *fm, int krec, int kRR) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	id.mkidBCRecycAvg(krec);

	unsigned int n = id.bdx;
	n = findNextPowerOf2(n); // to perform the reduction fast we need an array that is of the power of 2

	extern __shared__ myprec s_avg[] ;

	for(int i=id.tix; i<n; i+=id.bdx) {
		if (i < id.bdx) {
			s_avg[i] = f[id.g]; //actual domain values
		} else {
			s_avg[i] = 0; // fillers to fill the array as n>id.bdx to be a power of 2.
		}
	}
	__syncthreads();

	for (int size = n/2; size>0; size/=2) {
		if (id.tix<size)
			s_avg[id.tix] += s_avg[id.tix+size];
		__syncthreads();
	}
	if (id.tix == 0) {
		fm[id.bix + kRR*mx_tot] = s_avg[0]/my_tot; // ultimately I will add results of all the processors in Y and hence my_tot
	}
	__syncthreads();

}

__global__ void deltaRR(myprec *wm, myprec *delta) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	int khi = mx_tot-1;
	int klo = 0;
	int k;
	myprec winf = 0.99*wm[mx_tot-1 + id.tix*mx_tot]; 
	while (khi-klo > 1) {
		k = (khi+klo)/2;
		if(wm[k + id.tix*mx_tot] > winf) {
			khi=k;
		} else {
			klo=k;
		}
	}


	myprec a = (wm[khi+id.tix*mx_tot] - winf)/(wm[khi+id.tix*mx_tot] - wm[klo+id.tix*mx_tot]);

	delta[id.tix] = d_x[khi] - a*(d_x[khi] - d_x[klo]) ;
}

__global__ void interpRR(myprec *a, myprec *b, int *idxm, int *idxp, myprec *delta_rec, myprec *delta_in) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
	int idii = id.tix + mx_tot*id.bix ;

	__shared__ myprec s_x[mx_tot];
	s_x[id.tix] = d_x[id.tix]/delta_rec[id.bix];
	__syncthreads();

	int khi = mx_tot-1;
	int klo = 0;
	int k;

	myprec xi = d_x[id.tix]/delta_in[id.bix]; //inlet x/delta

	// In the following lines of code we will find that what indices on the recycling station corresponds to the x/delta of the inlet.
	// This will help in interpolating the data at the recycling station so as to be recycled to the inlet.
	// Suppose we have x/delta_in = 0.35. But at the recycling end we have x/delta_re = 0.32 and 0.38. Thus, first using the while loop
	// we compute the khi and klo correcponding to x=0.32 and x=0.38. Then we compute the weights as b=(0.38-0.35)/(0.38-0.32) & a=1-b;
	// a and b are the weights which will be used for interpolation.
	while (khi-klo > 1) {
		k = (khi+klo)/2;
		if( s_x[k] > xi) { // compare with the recycling station x/delta;
			khi=k;
		} else {
			klo=k;
		}
	}
	if (xi <= s_x[id.bdx-1]) { //id.bdx implies last value
		idxp[idii] = khi ;
		idxm[idii] = klo ;

		a[idii]    = (s_x[khi] - xi)/(s_x[khi] - s_x[klo]) ;
		b[idii]    = 1.0 - a[idii];
	} else {
		idxp[idii] = khi ;
		idxm[idii] = klo ;

		a[idii]    = 0.0 ;
		b[idii]    = 0.0; // This implies that for all the cells at the inlet that are beyond the x/delta_re|max,
		//we do the interpolation such that u[x/deltain> x/delta_remax] = beta_rec * (b * u[idp]) where idp is khi.
		//The while loop should ensure that the khi is maintained to its set value of mx_tot-1.
	}
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
	if (inletbc == 5) {
		InletMeanUpdate(rm_in, um_in, wm_in, tm_in, delta_in, rk); // The means at the inflow bc (ghost cells of inflow) will remain constant for the entire simulation and hence it is set once outside the loop
	}

        if(rk.rank==5){
	FILE *fdel = fopen("delstar.txt","w+");
	fclose(fdel);}

	if(restartFile<0) {
		start=0;
	} else {
		start=restartFile;
	}
	for(int file = start+1; file<nfiles+start+1; file++) {

		if(rk.rank==0) fp = fopen("solution.txt","w+");

		/////Time----------------------------------------------------------------------------------
		float tm;
		cudaEvent_t start1, stop;
		checkCuda( cudaEventCreate(&start1) );
		checkCuda( cudaEventCreate(&stop) );
		checkCuda( cudaEventRecord(start1, 0) );

		if(lowStorage) {
			runSimulationLowStorage(dpar1,dpar2,dtime,rk);  //running the simulation on the GPU
		} else {
			runSimulation(file, dpar1,dpar2,dtime,rk);  //running the simulation on the GPU
		}

		checkCuda( cudaEventRecord(stop, 0) );
		checkCuda( cudaEventSynchronize(stop) );
		checkCuda( cudaEventElapsedTime(&tm, start1, stop) );
		printf("Time for runSimulation  %3.4f s \n", tm/nsteps/1000);
		// EndTime---------------------------------------------------------------------------------

		copyField(1,rk);					  //copying back partial results to CPU

		writeField(file,rk);

		cudaDeviceSynchronize();

		checkCuda( cudaMemcpy(htime, dtime, nsteps*sizeof(myprec), cudaMemcpyDeviceToHost) );
		checkCuda( cudaMemcpy(hpar1, dpar1, nsteps*sizeof(myprec), cudaMemcpyDeviceToHost) );
		checkCuda( cudaMemcpy(hpar2, dpar2, nsteps*sizeof(myprec), cudaMemcpyDeviceToHost) );

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




