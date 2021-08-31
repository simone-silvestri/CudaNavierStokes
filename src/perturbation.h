/*
 * perturbation.h
 *
 *  Created on: Aug 25, 2021
 *      Author: simone
 */

#ifndef PERTURBATION_H_
#define PERTURBATION_H_


//const int kC = (int) (mz/Lz*(spInlLen + Lz/20.0));
//const int LP = 2*(kC - (int) (mz/Lz*spInlLen) - 5);
const int kC = 75;
const int LP = 24;

const myprec lambda = Ly/(2.0*M_PI);
const myprec amp1   = 2e-4;
const myprec amp2   = 3e-5;
const myprec omega1 = Re*121.e-6;
const myprec omega2 = 10.0;

extern __device__ __forceinline__ void PerturbUvel(myprec *s_u, Indices id, int si);

__device__ __forceinline__ __attribute__((always_inline)) void PerturbUvel(myprec *s_u, Indices id, int si) {

	int kSt = kC - LP/2;
	int kEn = kC + LP/2;

	int ktot = id.k + rkGPU.kstart;

	int alpha, beta, kappa;

	if (ktot >= kSt && ktot<= kEn) {
		if(ktot<kC) {
			kappa = 1;
			alpha = ktot - kSt;
			beta  = kC   - kSt;
		} else {
			kappa = - 1;
			alpha = kEn - ktot;
			beta  = kEn - kC  ;
		}

		myprec ksi = alpha*1.0/beta;

		myprec g = (15.1875*ksi*ksi*ksi*ksi*ksi) - (35.4375*ksi*ksi*ksi*ksi) + (20.25*ksi*ksi*ksi);

		myprec y_glob = (id.j + rkGPU.jstart)/d_dy;

		s_u[si-stencilSize] = amp1*kappa*g*sin(omega1*time_on_GPU) + amp2*kappa*g*sin(omega2*time_on_GPU)*cos(y_glob/lambda);
	}
}

#endif /* PERTURBATION_H_ */
