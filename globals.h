
#ifndef GLOBALS_H_
#define GLOBALS_H_

#include "main.h"

#define stencilSize 4  //the order is double the stencilSize 
const int L = 1;
const int N = 100;
const int nsteps = 2950;
const double U = 1.0;
const double CFL = 1.0;
const bool periodic = true;

#if stencilSize==1
const double coeffS[] = {-1.0/2.0};
#elif stencilSize==2
const double coeffS[] = {1.0/12.0, -2.0/3.0};
#elif stencilSize==3
const double coeffS[] = {-1.0/60.0, 3.0/20.0, -2.0/4.0};
#elif stencilSize==4
const double coeffS[] = {1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0};
#endif

double *coeff;

double dt;
double x[N],phi[N];

double rhs1[N]; 
double rhs2[N];
double rhs3[N];
double rhs4[N];
double temp[N];


int colorMap = 0;

#endif
