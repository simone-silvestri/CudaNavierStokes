#ifndef SPONGE_H_
#define SPONGE_H_

extern myprec *d_spongeX, *d_spongeZ;
extern myprec *d_rref, *d_uref, *d_wref, *d_eref;

__global__ void addSponge(myprec *rhsr, myprec *rhsu, myprec *rhsv, myprec *rhsw, myprec *rhse,
						  myprec *r, myprec *u, myprec *v, myprec *w, myprec *e, myprec *spongeX, myprec *spongeZ,
						  myprec *rref, myprec *uref, myprec *wref, myprec *eref);
void calculateSpongePar(myprec *x, myprec *z, Communicator rk);
void calculateRefSponge(myprec *x, myprec *z, Communicator rk);

#endif /* SPONGE_H_ */
