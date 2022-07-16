
#ifndef GLOBALS_H_
#define GLOBALS_H_

#define myprec double
#define MPI_myprec MPI_DOUBLE

#include "math.h"
#include "main.h"
#include <stdlib.h>
#include <stdio.h>

//Remember: Run always the code with num-tasks-per-node = num-GPUs-per-node. Otherwise it will not work!
#define pRow 1
#define pCol 1

#define GPUperNode 4

//Remember : viscous stencil should ALWAYS be smaller than the advective stencil!!! (otherwise errors in how you load global into shared memory)
#define stencilSize 3  // the order is double the stencilSize (advective fluxes stencil)
#define stencilVisc 2  // the order is double the stencilVisc (viscous fluxes stencil)

#define boundaryLayer (true)
#define channel (false)
#define misc (false)

#define Lx       (12.0)
#define Ly       (10.0)
#define Lz       (52.0)
#define mx_tot   240
#define my_tot   60 // 246
#define mz_tot   2048
#define nsteps   10
#define nfiles   1
#define CFL      0.75f
#define restartFile -1
#define spanshift (true)

#define recstn   1500  

#define lowStorage    (false)
#define perturbed     (false)

#define checkCFLcondition 100
#define checkBulk         100
#define checkMeanRec      1

#define Re       16609
#define Pr       0.7
#define Ma       1.9
#define viscexp  0.75
#define gam      1.4
#define Ec       ((gam - 1.f)*Ma*Ma)
#define Rgas     (1.f/(gam*Ma*Ma))
#define pinf     (Rgas)
#define rfac        0.89
#define Trat     1
#define Tr       (1 + 0.5*(gam-1)*rfac*Ma*Ma)

#if boundaryLayer

#define forcing       (false)
#define periodicX     (false)
#define nonUniformX   (true)

//1 = periodic ; 2 = wall_staggered; 3 = wall_centered; 4= 0th order extrapolation 5 = Inflow
#define inletbc   (5)
#define outletbc  (4)
#define bottombc  (3)
#define topbc     (4)

#define gridtype  (0) // 0 = BL; 1 = channel_centered; 2 = channel staggered, no wall ,etc.

const myprec TwallTop = 0.0; //Holds no relevance for BL
const myprec TwallBot = Trat*Tr;

#elif channel

#define forcing       (true)
#define periodicX     (false)
#define nonUniformX   (true)

//1 = periodic ; 2 = wall_staggered; 3 = wall_centered; 4= 0th order extrapolation 5 = Inflow
#define inletbc   (1)
#define outletbc  (1)
#define bottombc  (2)
#define topbc     (2)

#define gridtype  (2) // 0 = BL; 1 = channel_centered; 2 = channel staggered, no wall ,etc.

const myprec TwallTop = 1.0;
const myprec TwallBot = 1.0;

#elif misc

#endif

const double stretch  = 5.0;

#define mx (mx_tot)
#define my (my_tot/pRow)
#define mz (mz_tot/pCol)

#define nX 	(1) 	// never put larger than 8 as there will be no streams to accomodate the kernels (max 8) (actually check the machine limitations)

#define nDivX 	(15) 	// never put larger than 8 as there will be no streams to accomodate the kernels (max 8) (actually check the machine limitations)
#define nDivY 	(16) 	// never put larger than 8 as there will be no streams to accomodate the kernels (max 8) (actually check the machine limitations)
#define nDivZ 	(64) 	// never put larger than 8 as there will be no streams to accomodate the kernels (max 8) (actually check the machine limitations)

#define idx(i,j,k) \
		({ ( k )*mx*my +( j )*mx + ( i ); })

#if pRow*pCol>1
const int multiGPU = 1;
#else
const int multiGPU = 0;
#endif

#if stencilSize==1
const double coeffF[] = {-1.0/2.0};
const double coeffS[] = {1.0, -2.0};
#elif stencilSize==2
const double coeffF[] = { 1.0/12.0, -2.0/3.0};
const double coeffS[] = {-1.0/12.0,  4.0/3.0, -5.0/2.0};
#elif stencilSize==3
const double coeffF[] = {-1.0/60.0,  3.0/20.0, -3.0/4.0};
const double coeffS[] = { 1.0/90.0, -3.0/20.0,  3.0/2.0, -49.0/18.0};
#elif stencilSize==4
const double coeffF[] = { 1.0/280.0, -4.0/105.0,  1.0/5.0, -4.0/5.0};
const double coeffS[] = {-1.0/560.0,  8.0/315.0, -1.0/5.0,  8.0/5.0,  -205.0/72.0};
#endif

#if stencilVisc==1
const double coeffVF[] = {-1.0/2.0};
const double coeffVS[] = {1.0, -2.0};
#elif stencilVisc==2
const double coeffVF[] = { 1.0/12.0, -2.0/3.0};
const double coeffVS[] = {-1.0/12.0,  4.0/3.0, -5.0/2.0};
#elif stencilVisc==3
const double coeffVF[] = {-1.0/60.0,  3.0/20.0, -3.0/4.0};
const double coeffVS[] = { 1.0/90.0, -3.0/20.0,  3.0/2.0, -49.0/18.0};
#elif stencilVisc==4
const double coeffVF[] = { 1.0/280.0, -4.0/105.0,  1.0/5.0, -4.0/5.0};
const double coeffVS[] = {-1.0/560.0,  8.0/315.0, -1.0/5.0,  8.0/5.0,  -205.0/72.0};
#endif

extern double dt, h_dpdz;

extern double dx,x[mx],xp[mx],xpp[mx],y[my_tot],z[mz_tot];

extern double r[mx*my*mz];
extern double u[mx*my*mz];
extern double v[mx*my*mz];
extern double w[mx*my*mz];
extern double e[mx*my*mz];

#endif
