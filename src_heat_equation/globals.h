
#ifndef GLOBALS_H_
#define GLOBALS_H_

#include "main.h"
#include <stdlib.h>
#include <stdio.h>

#define stencilSize 4  //the order is double the stencilSize 

#define Lx       1.f
#define Ly       0.5f
#define Lz       0.5f
#define mx       128
#define my       128
#define mz       256
#define nsteps   1000
#define U        1.f
#define visc     0.1f
#define CFL      1.f

#define parentGrid  0

const bool periodic = true;


#if parentGrid == 0
const double Ux = U;
const double Uy = 0.0;
const double Uz = U; 
#elif parentGrid == 1
const double Ux = 0.0; 
const double Uy = U; 
const double Uz = 0.0; 
#else
const double Ux = U;
const double Uy = 0.0; 
const double Uz = U; 
#endif


#define idx(i,j,k) \
		({ ( k )*mx*my +( j )*mx + ( i ); }) 

#if stencilSize==1
const double coeffF[] = {-1.0/2.0};
const double coeffS[] = {1.0, -2.0};
#elif stencilSize==2
const double coeffF[] = { 1.0/12.0, -2.0/3.0};
const double coeffS[] = {-1.0/12.0,  4.0/3.0, -5.0/2.0};
#elif stencilSize==3
const double coeffF[] = {-1.0/60.0,  3.0/20.0, -2.0/4.0};
const double coeffS[] = { 1.0/90.0, -3.0/20.0,  3.0/2.0, -49.0/18.0};
#elif stencilSize==4
const double coeffF[] = { 1.0/280.0, -4.0/105.0,  1.0/5.0, -4.0/5.0};
const double coeffS[] = {-1.0/560.0,  8.0/315.0, -1.0/5.0,  8.0/5.0,  -205.0/72.0};
#endif

extern double dt;

extern double x[mx],y[my],z[mz],phi[mx*my*mz];

extern double rhs1[mx*my*mz]; 
extern double rhs2[mx*my*mz];
extern double rhs3[mx*my*mz];
extern double rhs4[mx*my*mz];
extern double temp[mx*my*mz];

extern void initProfile();
extern void calcdt();
#endif
