/*
 * comm.h
 *
 *  Created on: May 11, 2021
 *      Author: simone
 */

#ifndef COMM_H_
#define COMM_H_
#include "globals.h"

class Boundaries {
public:
	myprec *jm,*jp,*km,*kp;
	Boundaries(int _mx, int _my, int _mz, int _stencil) {
		jm = new myprec[_stencil*_mz*_mx];
		jp = new myprec[_stencil*_mz*_mx];
		km = new myprec[_stencil*_my*_mx];
		kp = new myprec[_stencil*_my*_mx];
	}
};

void splitComm(Communicator *rk);
void updateHalo(Boundaries *bc, myprec *var, Communicator rk);
void testHalo(Communicator rk);
void readFileMPI(char filename, int timestep, myprec *var, Communicator rk);
void saveFileMPI(char filename, int timestep, myprec *var, Communicator rk);

#endif /* COMM_H_ */
