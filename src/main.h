#ifndef MAIN_H_
#define MAIN_H_

#define MAX(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a >= _b ? _a : _b; })
#define MIN(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a <= _b ? _a : _b; })

extern void writeFields(int timestep);
extern void initProfile();
extern void calcdt();

#endif
