#ifndef MAIN_H_
#define MAIN_H_

#define MAX(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a >= _b ? _a : _b; })
#define MIN(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a <= _b ? _a : _b; })

void higherOrder_FD(double *dydx, double *y, double *g);
void RHS(double *rhs, double *var, double *g);
void rk4(); 
void run();

#endif
