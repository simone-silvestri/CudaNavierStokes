
#ifndef GLOBALS_H_
#define GLOBALS_H_

#include "math.h"
#include "main.h"
#include <stdlib.h>
#include <stdio.h>

#define stencilSize 4  // the order is double the stencilSize (advective fluxes stencil)
#define stencilVisc 4  // the order is double the stencilSize (viscous fluxes stencil)

#define Lx       (2*M_PI)
#define Ly       (2*M_PI)
#define Lz       (2*M_PI)
#define mx       128
#define my       128
#define mz       128
#define nsteps   1001
#define nfiles   1
#define CFL      0.3f
#define rk       3             // rk = 3 is the runge-kutta 3 method while rk = 4 is runge-kutta 4 method and rk = 2 is the Adam's Bashforth method

#define Re       1600.f
#define Pr       1.f
#define gamma    1.4f
#define Ma       0.1f
#define Ec       ((gamma - 1.f)*Ma*Ma)
#define Rgas     (1.f/(gamma*Ma*Ma))
#define viscexp  0.7
#define lamexp   0.7

#define checkCFLcondition 50
#define calculateBulks    50


const bool periodic = true;


#define idx(i,j,k) \
		({ ( k )*mx*my +( j )*mx + ( i ); }) 

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

extern double dt;

extern double x[mx],y[my],z[mz];

extern double r[mx*my*mz];
extern double u[mx*my*mz];
extern double v[mx*my*mz];
extern double w[mx*my*mz];
extern double e[mx*my*mz];

#endif
