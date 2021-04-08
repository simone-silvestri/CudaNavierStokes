
#ifndef GLOBALS_H_
#define GLOBALS_H_

#include "math.h"
#include "main.h"
#include <stdlib.h>
#include <stdio.h>

#define stencilSize 4  //the order is double the stencilSize 

#define Lx       (2*M_PI)
#define Ly       (2*M_PI)
#define Lz       (2*M_PI)
#define mx       64
#define my       64
#define mz       64
#define nsteps   100
#define CFL      0.3f

#define Re       395.f
#define Pr       1.f
#define gamma    1.4f
#define Ma       0.1f
#define Ec       ((gamma - 1.f)*Ma*Ma)
#define Rgas     (1.f/(gamma*Ma*Ma))     //1.0/(gam*Ma**2.0)

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

extern double dt;

extern double x[mx],y[my],z[mz];

extern double r[mx*my*mz];
extern double u[mx*my*mz];
extern double v[mx*my*mz];
extern double w[mx*my*mz];
extern double e[mx*my*mz];

extern double rhs1[mx*my*mz]; 
extern double rhs2[mx*my*mz];
extern double rhs3[mx*my*mz];
extern double rhs4[mx*my*mz];
extern double temp[mx*my*mz];

#endif
