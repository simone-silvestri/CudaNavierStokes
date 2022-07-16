/*
 * comm.h
 *
 *  Created on: May 11, 2021
 *      Author: simone
 */

#ifndef COMM_H_
#define COMM_H_
#include "globals.h"

void splitComm(Communicator *rk, int myRank);
void updateHalo(myprec *var, Communicator rk);
void updateHaloTest(myprec *var, Communicator rk);
void updateHaloTestFive(myprec *dr, myprec *du, myprec *dv, myprec *dw, myprec *de, Communicator rk);
void updateHaloFive(myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, Communicator rk);
void readFileMPI(char filename, int timestep, double *var, Communicator rk);
void saveFileMPI(char filename, int timestep, double *var, Communicator rk);
void reduceArray(int rcvCore, myprec *sendArr, int sizeArr, Communicator rk);
void allReduceArray(myprec *sendArr, int sizeArr);
void allReduceArrayDouble(double *sendArr, int sizeArr);
void allReduceToMin(myprec *sendArr, int sizeArr);
void allReduceSum(myprec *sendArr, int sizeArr);
void allReduceSumYavg(myprec *sendArr, int sizeArr);
void broadCastArray(int bcstCore, double *sendArr, int sizeArr, Communicator rk);
void mpiBarrier();
void TransferToInlet(myprec *recy_r, myprec *recy_u,myprec *recy_v, myprec *recy_w, myprec *recy_t, Communicator rk) ;
void readFileInitBL(char filename, double *var, Communicator rk);
void readFileInitBL1D(char filename, double *var, Communicator rk);
void readFileInitBL_inRR(char filename, double *var, Communicator rk);
void readFileInitBL1D_inRR(char filename, double *var, Communicator rk);
void readFileMPIInit(char filename, int timestep,  double *r,  double *u,  double *v,  double *w,  double *ret, Communicator rk);

#endif /* COMM_H_ */
