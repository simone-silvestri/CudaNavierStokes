
#ifndef GLOBALS_H_
#define GLOBALS_H_

#include "main.h"
#include <stdlib.h>
#include <stdio.h>

#define stencilSize 4  //the order is double the stencilSize 

#define L        1.f
#define mx       256 
#define my       128
#define mz       128 
#define nsteps   100
#define U        1.f
#define CFL      1.f
const bool periodic = true;

#define idx(i,j,k) \
		({ k*mx*my+j*mx+i; }) 

#if stencilSize==1
const double coeffS[] = {-1.0/2.0};
#elif stencilSize==2
const double coeffS[] = {1.0/12.0, -2.0/3.0};
#elif stencilSize==3
const double coeffS[] = {-1.0/60.0, 3.0/20.0, -2.0/4.0};
#elif stencilSize==4
const double coeffS[] = {1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0};
#endif

extern double dt;

extern double x[mx],phi[mx*my*mz];

extern double rhs1[mx*my*mz]; 
extern double rhs2[mx*my*mz];
extern double rhs3[mx*my*mz];
extern double rhs4[mx*my*mz];
extern double temp[mx*my*mz];

extern void initProfile();
#endif
