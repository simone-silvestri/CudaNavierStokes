#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include "mpi.h"
#include "../src_multiGPU/globals.h"
#include "../src_multiGPU/comm.h"

using namespace std;

double dt;

double dx,dxv[mx_tot],x[mx_tot],xp[mx_tot],xpp[mx_tot],y[my_tot],z[mz_tot];

double r[mx*my*mz];
double u[mx*my*mz];
double v[mx*my*mz];
double w[mx*my*mz];
double e[mx*my*mz];

double h[mx*my*mz];
double t[mx*my*mz];
double p[mx*my*mz];
double m[mx*my*mz];

class Variables {
public:
	const char *name;
	int N;
	double *r,*uf,*vf,*wf,*u,*v,*w,*e,*h,*hf,*t,*p,*m;
	Variables(int _size, const char *filename) {
		N = _size;
		r = new double[N];
		uf= new double[N];
		vf= new double[N];
		wf= new double[N];
		u = new double[N];
		v = new double[N];
		w = new double[N];
		e = new double[N];
		h = new double[N];
		hf= new double[N];
		t = new double[N];
		p = new double[N];
		m = new double[N];
		name = filename;
		for(int i=0; i<N; i++){
			r[i] = 0;
			u[i] = 0;
			v[i] = 0;
			w[i] = 0;
			uf[i] = 0;
			vf[i] = 0;
			wf[i] = 0;
			e[i] = 0;
			h[i] = 0;
			hf[i] = 0;
			t[i] = 0;
			p[i] = 0;
			m[i] = 0;
		}

	}
	void printFile(int myRank, double Ret, double ut) {
		if(myRank==0) {
			FILE *fp = fopen(name,"w+");
			fprintf(fp,"Reynolds number based on utau %lf with utau %lf\n",Ret,ut);
			fprintf(fp,"1)  y\n");
			fprintf(fp,"2)  rho\n");
			fprintf(fp,"3)  uFavre\n");
			fprintf(fp,"4)  vFavre\n");
			fprintf(fp,"5)  wFavre\n");
			fprintf(fp,"6)  u\n");
			fprintf(fp,"7)  v\n");
			fprintf(fp,"8)  w\n");
			fprintf(fp,"9)  eTotal\n");
			fprintf(fp,"10) hFavre\n");
			fprintf(fp,"11) h\n");
			fprintf(fp,"12) Temperature\n");
			fprintf(fp,"13) Pressure\n");
			fprintf(fp,"13) visc\n");
			fprintf(fp,"---------------------------------------------------------------------------------------------------------------------------------------------------\n");
			for (int i=0; i<N; i++)
				fprintf(fp,"%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\t%le\n",x[i],r[i],uf[i],vf[i],wf[i],u[i],v[i],w[i],e[i],hf[i],h[i],t[i],p[i],m[i]);
			fclose(fp);
		}
	}
	void calcFavre() {
		for (int i=0; i<N; i++) {
			uf[i] /= r[i];
			vf[i] /= r[i];
			wf[i] /= r[i];
			hf[i] /= r[i];
		}
	}
	void allReduceVariables() {
		allReduceArrayDouble(r,N);
		allReduceArrayDouble(u,N);
		allReduceArrayDouble(v,N);
		allReduceArrayDouble(w,N);
		allReduceArrayDouble(e,N);
		allReduceArrayDouble(h,N);
		allReduceArrayDouble(t,N);
		allReduceArrayDouble(p,N);
		allReduceArrayDouble(m,N);
	}
	void allReduceFavre() {
		allReduceArrayDouble(uf,N);
		allReduceArrayDouble(vf,N);
		allReduceArrayDouble(wf,N);
		allReduceArrayDouble(hf,N);
	}
};

const int initfile= 1;
const int endfile = 21;

int denom;

void calcState();
void addMean(Variables *var);
void addFluc(Variables *var, Variables mean);
void calcRet(Communicator rk, double *Retot, double *uttot);

int main(int argc, char** argv) {

	int myRank, nProcs;

    int ierr;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    double begin = MPI_Wtime();

    if(nProcs != pRow*pCol) {
    	if(myRank==0) {
    		printf("Error! -> nProcs different that pRow*pCol\n");
    	}
    	ierr = MPI_Barrier(MPI_COMM_WORLD);
        ierr = MPI_Finalize();
        return 0;
    }

    //Initialize the 2D processor grid
    Communicator rk;
    splitComm(&rk,myRank);
    mpiBarrier();

    //Initialize the computational mesh
    initGrid(rk);

    Variables mean(mx,"mean.txt"), fluc(mx,"fluc.txt"), bulk(1,"bulk.txt");
    int end = endfile;
    int ini = initfile;

    if(argc>=3) {
    	ini = atoi(argv[1]);
    	end = atoi(argv[2]);
    	if(rk.rank==0) printf("Fields to postprocess :  %d -> %d\n",ini,end);
    }

    int numberOfFiles = end - ini + 1;
    denom = numberOfFiles*my*mz;

    mpiBarrier();

    double Ret = 0.0;
    double ut  = 0.0;

    for(int file = ini; file<end+1; file++) {
    	initField(file,rk);
    	calcState();
    	addMean(&mean);
    	addMean(&bulk);
    	calcRet(rk,&Ret,&ut);
    }
    Ret/=numberOfFiles;
    ut /=numberOfFiles;
	mean.allReduceVariables();
	bulk.allReduceVariables();
	mean.calcFavre();
	bulk.calcFavre();
	mean.allReduceFavre();
	bulk.allReduceFavre();
	for(int file = ini; file<end+1; file++) {
    	initField(file,rk);
		calcState();
		addFluc(&fluc,mean);
	}
	fluc.allReduceVariables();
	mean.printFile(rk.rank,Ret,ut);
	fluc.printFile(rk.rank,Ret,ut);
	bulk.printFile(rk.rank,Ret,ut);

	ierr = MPI_Finalize();
	return 0;
}

void addFluc(Variables *var, Variables mean) {
	for (int i=0; i<mx; i++) {
		for (int k=0; k<mz; k++) {
			for (int j=0; j<my; j++) {
				int gt = ( i ) + mx * ( j + my * ( k ) ) ;
				var->r[i] += (r[gt]-mean.r[i])*(r[gt]-mean.r[i])/denom;
				var->u[i] += (u[gt]-mean.u[i])*(u[gt]-mean.u[i])/denom;
				var->v[i] += (v[gt]-mean.v[i])*(v[gt]-mean.v[i])/denom;
				var->w[i] += (w[gt]-mean.w[i])*(w[gt]-mean.w[i])/denom;
				var->uf[i]+= (u[gt]-mean.uf[i])*(u[gt]-mean.uf[i])/denom;
				var->vf[i]+= (v[gt]-mean.vf[i])*(v[gt]-mean.vf[i])/denom;
				var->wf[i]+= (w[gt]-mean.wf[i])*(w[gt]-mean.wf[i])/denom;
				var->e[i] += (e[gt]-mean.e[i])*(e[gt]-mean.e[i])/denom;
				var->h[i] += (h[gt]-mean.h[i])*(h[gt]-mean.h[i])/denom;
				var->hf[i]+= (h[gt]-mean.hf[i])*(h[gt]-mean.hf[i])/denom;
				var->t[i] += (t[gt]-mean.t[i])*(t[gt]-mean.t[i])/denom;
				var->p[i] += (p[gt]-mean.p[i])*(p[gt]-mean.p[i])/denom;
				var->m[i] += (m[gt]-mean.m[i])*(m[gt]-mean.m[i])/denom;
			}
		}
	}
}

void addMean(Variables *var) {
	int idx;
	double denom2;
	for (int i=0; i<mx; i++) {
		if(var->N==1) {
			idx = 0;
		} else {
			idx = i;
		}
		for (int k=0; k<mz; k++) {
			for (int j=0; j<my; j++) {
				int gt = ( i ) + mx * ( j + my * ( k ) ) ;
				if(var->N==1) {
					denom2 = Lx/dxv[i]*denom;
				} else {
					denom2 = (double) denom;
				}
				var->r[idx] += r[gt]/denom2;
				var->u[idx] += u[gt]/denom2;
				var->v[idx] += v[gt]/denom2;
				var->w[idx] += w[gt]/denom2;
				var->uf[idx]+= r[gt]*u[gt]/denom2;
				var->vf[idx]+= r[gt]*v[gt]/denom2;
				var->wf[idx]+= r[gt]*w[gt]/denom2;
				var->e[idx] += e[gt]/denom2;
				var->h[idx] += h[gt]/denom2;
				var->hf[idx]+= r[gt]*h[gt]/denom2;
				var->t[idx] += t[gt]/denom2;
				var->p[idx] += p[gt]/denom2;
				var->m[idx] += m[gt]/denom2;
			}
		}
	}
}

void calcState() {
	for(int k=0; k<mz; k++)
		for(int j=0; j<my; j++)
			for(int i=0; i<mx; i++) {
				int gt = ( i ) + mx * ( j + my * ( k ) ) ;

				double cvInv = (gamma - 1.0)/Rgas;

				double invrho = 1.0/r[gt];

				double en = e[gt]*invrho - 0.5*(u[gt]*u[gt] + v[gt]*v[gt] + w[gt]*w[gt]);
				t[gt]   = cvInv*en;
				p[gt]   = r[gt]*Rgas*t[gt];
				h[gt]    = (e[gt] + p[gt])*invrho;

				double suth = pow(t[gt],viscexp);
				m[gt]      = suth/Re;
			}
}

void calcRet(Communicator rk, double *Retot, double *uttot) {

	double ut  = 0.0;
	double Ret = 0.0;

	for (int k=0; k<mz; k++)
		for (int j=0; j<my; j++) {

			    double rw = 0.5*(p[idx(0,j,k)]+p[idx(mx-1,j,k)])/Rgas;
			    double muw      = 1.0/Re;

			    double   ub[mx+stencilSize*2+2];
			    double dudx[mx+stencilSize*2+2];
			    for (int i=0; i<mx; i++)
			    	ub[i+stencilSize+1] = w[idx(i,j,k)];

			    for (int i=0; i<stencilSize+1; i++) {
			    	ub[i]                  = -w[idx(stencilSize-i,j,k)];
			    	ub[mx+stencilSize+1+i] = -w[idx(mx-i-1       ,j,k)];
			    }

			    for (int j=3; j<mx+stencilSize+2; j++) {
			    	dudx[j] = 0;
			    	for (int i=0; i<stencilSize; i++) {
			    		dudx[j] += coeffF[i]*(ub[j+i-stencilSize]  - ub[j-i+stencilSize])/dx;
			    	}
			    }

			    double dudxavg = abs(dudx[3])+abs(dudx[4])+abs(dudx[mx+stencilSize])+abs(dudx[mx+stencilSize+1]);
			    dudxavg = dudxavg*0.25*xp[0];

			    double uttemp = sqrt(muw*dudxavg/rw);
			    ut  += uttemp;

			    Ret += uttemp*rw/muw;
		}

	Ret = Ret/my/mz;
	ut  = ut/my/mz;
	allReduceArrayDouble(&Ret,1);
	allReduceArrayDouble(&ut,1);
	*Retot += Ret;
	*uttot += ut;
	if(rk.rank==0) {
		printf("The average friction Reynolds number is: \t %lf with friction velocity \t %lf\n",Ret,ut);
	}
}
