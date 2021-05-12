#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdlib.h>

#define MAX(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a >= _b ? _a : _b; })
#define MIN(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a <= _b ? _a : _b; })

class Communicator {
public:
	int rank, jp, jm, kp, km;
	int jstart,jend,kstart,kend;
	void myRank(int rk) {
		rank = rk;
	}
	void printGrid() {
		printf("My Rank is %d with neighbours %d %d %d %d\n",rank,jp,jm,kp,km);
		printf("My Rank is %d with limits     %d %d %d %d\n",rank,jstart,jend,kstart,kend);
	}
};

void writeField(int timestep, Communicator rk);
void initField(int timestep, Communicator rk);
void printRes(Communicator rk);
extern void initCHIT(Communicator rk);
extern void initGrid(Communicator rk);
extern void initChannel(Communicator rk);
extern void calcdt();
void calcAvgChan(Communicator rk);

#endif
