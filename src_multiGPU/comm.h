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
void updateHaloFive(myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, Communicator rk);
void readFileMPI(char filename, int timestep, double *var, Communicator rk);
void saveFileMPI(char filename, int timestep, double *var, Communicator rk);
void reduceArray(int rcvCore, double *sendArr, int sizeArr, Communicator rk);
void allReduceArray(double *sendArr, int sizeArr);
void allReduceToMin(double *sendArr, int sizeArr);
void allReduceSum(double *sendArr, int sizeArr);
void broadCastArray(int bcstCore, double *sendArr, int sizeArr, Communicator rk);
void mpiBarrier();

#endif /* COMM_H_ */
