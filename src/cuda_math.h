/*
 * cuda_math.h
 *
 *  Created on: Apr 15, 2021
 *      Author: simone
 */

#ifndef CUDA_MATH_H_
#define CUDA_MATH_H_


__global__ void deviceCpy(myprec *a, myprec *b);
__global__ void deviceSum(myprec *a, myprec *b, myprec *c);
__global__ void deviceSub(myprec *a, myprec *b, myprec *c);
__global__ void deviceMul(myprec *a, myprec *b, myprec *c);
__global__ void deviceSca(myprec *a, myprec *bx, myprec *by, myprec *bz, myprec *cx, myprec *cy, myprec *cz);
__global__ void reduceToOne(myprec *gout, myprec *var);


#endif /* CUDA_MATH_H_ */
