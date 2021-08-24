#include "globals.h"
#include "cuda_globals.h"
#include "cuda_functions.h"
#include "comm.h"
#include "main.h"
#include "sponge.h"

const myprec spTopStr = 0.5;
const myprec spTopLen = 2.0;
const myprec spTopExp = 2.0;
const myprec spInlStr = 0.5;
const myprec spInlLen = 10.0;
const myprec spInlExp = 2.0;
const myprec spOutStr = 0.5;
const myprec spOutLen = 10.0;
const myprec spOutExp = 2.0;

__device__ myprec spongeX[mx];
__device__ myprec spongeZ[mz];
__device__ myprec rref[mx*mz];
__device__ myprec uref[mx*mz];
__device__ myprec wref[mx*mz];
__device__ myprec eref[mx*mz];
__device__ myprec href[mx*mz];
__device__ myprec tref[mx*mz];
__device__ myprec pref[mx*mz];
__device__ myprec mref[mx*mz];
__device__ myprec lref[mx*mz];

void spline(myprec x[], myprec y[], int n, myprec yp1, myprec ypn, myprec y2[]);
myprec splint(myprec xa[], myprec ya[], myprec y2a[], int n, myprec x);

__global__ void addSponge(myprec *rhsr, myprec *rhsu, myprec *rhsv, myprec *rhsw, myprec *rhse,
						  myprec *r, myprec *u, myprec *v, myprec *w, myprec *e) {

	Indices id(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

	rhsr[id.g] += (spongeX[id.i] + spongeZ[id.k]) * (rref[idx2(id.i,id.k)] - r[id.g]);
	rhsu[id.g] += (spongeX[id.i] + spongeZ[id.k]) * (uref[idx2(id.i,id.k)] - u[id.g]);
//	rhsv[id.g] += (spongeX[id.i] + spongeZ[id.k]) * (0.0                   - v[id.g]);
	rhsw[id.g] += (spongeX[id.i] + spongeZ[id.k]) * (wref[idx2(id.i,id.k)] - w[id.g]);
	rhse[id.g] += (spongeX[id.i] + spongeZ[id.k]) * (eref[idx2(id.i,id.k)] - e[id.g]);
}

__global__ void copySpongeToDevice(myprec *d_spongeX, myprec *d_spongeZ, myprec *d_rref, myprec *d_uref, myprec *d_wref, myprec *d_eref) {

	int bdx = blockDim.x ;
	int tix = threadIdx.x;
	int bix = blockIdx.x ;

	int gl = tix + bdx * bix;

	if(bix==0)	spongeX[tix] = d_spongeX[tix];
	if(tix==0)	spongeZ[bix] = d_spongeZ[bix];

	rref[gl] = d_rref[gl];
	uref[gl] = d_uref[gl];
	wref[gl] = d_wref[gl];
	eref[gl] = d_eref[gl];

	myprec cvInv = (gamma - 1.0)/Rgas;

    myprec invrho = 1.0/rref[gl];

    myprec en  = eref[gl]*invrho - 0.5*(uref[gl]*uref[gl] + wref[gl]*wref[gl]);
    tref[gl]   = cvInv*en;
    pref[gl]   = rref[gl]*Rgas*tref[gl];
    href[gl]   = (eref[gl] + pref[gl])*invrho;

    myprec suth = pow(tref[gl],viscexp);
    mref[gl]    = suth/Re;
    lref[gl]    = suth/Re/Pr/Ec;
    __syncthreads();
}

void calculateSponge(Communicator rk) {

	myprec *h_spongeX = new myprec[mx];
	myprec *h_spongeZ = new myprec[mz];
	myprec *h_rref = new myprec[mx*mz];
	myprec *h_uref = new myprec[mx*mz];
	myprec *h_wref = new myprec[mx*mz];
	myprec *h_eref = new myprec[mx*mz];
	myprec *d_spongeX, *d_spongeZ;
	myprec *d_rref;
	myprec *d_uref;
	myprec *d_wref;
	myprec *d_eref;

	checkCuda( cudaMalloc((void**)&d_spongeX, mx*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_spongeZ, mz*sizeof(myprec)) );

	checkCuda( cudaMalloc((void**)&d_rref, mz*mx*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_uref, mz*mx*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_wref, mz*mx*sizeof(myprec)) );
	checkCuda( cudaMalloc((void**)&d_eref, mz*mx*sizeof(myprec)) );

	for(int i=0; i<mx; i++) {
		h_spongeX[i] = 0.0;
		if ((spTopLen > 0.0) && (x[i] >= Lx - spTopLen))
			h_spongeX[i] = spTopStr*pow((x[i] - (Lx-spTopLen))/spTopLen , spTopExp);
	}

	for(int k=0; k<mz; k++) {
		h_spongeZ[k] = 0.0;
		myprec fz = z[k+rk.kstart];
		if ((spInlLen > 0.0) && (fz <= spInlLen))
			h_spongeZ[k] = spInlStr*pow( (spInlLen-fz)/spInlLen , spInlExp);

		if ((spOutLen > 0.0) && (fz >= (Lz-spOutLen)))
			h_spongeZ[k] = spOutStr*pow((fz - (Lz-spOutLen))/spOutLen,spOutExp);
	}

    checkCuda( cudaMemcpy(d_spongeX, h_spongeX, mx*sizeof(myprec), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_spongeZ, h_spongeZ, mz*sizeof(myprec), cudaMemcpyHostToDevice) );

    delete [] h_spongeX;
    delete [] h_spongeZ;

	FILE *fp = fopen("blasius1D/xProf.bin","rb");

	fseek(fp, 0, SEEK_END); 		  // seek to end of file
	int size = ftell(fp) / sizeof(double); // get current file pointer
	fseek(fp, 0, SEEK_SET); 		  // seek back to beginning of file

	myprec xIn[size] , rIn[size] , uIn[size] , wIn[size] , eIn[size];
	myprec             r2In[size], u2In[size], w2In[size], e2In[size];
	size_t result;

	result = fread(xIn, sizeof(double), size, fp); fclose(fp);
	fp = fopen("blasius1D/rProf.bin","rb");
	result = fread(rIn, sizeof(double), size, fp); fclose(fp);
	fp = fopen("blasius1D/uProf.bin","rb");
	result = fread(uIn, sizeof(double), size, fp); fclose(fp);
	fp = fopen("blasius1D/wProf.bin","rb");
	result = fread(wIn, sizeof(double), size, fp); fclose(fp);
	fp = fopen("blasius1D/eProf.bin","rb");
	result = fread(eIn, sizeof(double), size, fp);

	spline(xIn, rIn, size, 1e30, 1e30, r2In);
	spline(xIn, uIn, size, 1e30, 1e30, u2In);
	spline(xIn, wIn, size, 1e30, 1e30, w2In);
	spline(xIn, eIn, size, 1e30, 1e30, e2In);
	for (int k=0; k<mz; k++)
		for (int i=0; i<mx; i++) {
			myprec scale = pow( 1 + z[k+rk.kstart]/Re, 0.5 );
			h_rref[idx2(i,k)] = splint(xIn,rIn,r2In,size,x[i]/scale);
			h_uref[idx2(i,k)] = splint(xIn,uIn,u2In,size,x[i]/scale); h_uref[idx2(i,k)] /= (scale*Re);
			h_wref[idx2(i,k)] = splint(xIn,wIn,w2In,size,x[i]/scale);
			h_eref[idx2(i,k)] = splint(xIn,eIn,e2In,size,x[i]/scale);
			h_eref[idx2(i,k)] = h_eref[idx2(i,k)] + 0.5*(h_uref[idx2(i,k)]*h_uref[idx2(i,k)]+h_wref[idx2(i,k)]*h_wref[idx2(i,k)]);
			h_eref[idx2(i,k)]*= h_rref[idx2(i,k)];
		}

	if(restartFile<0) {
		for (int k=0; k<mz; k++)
			for (int j=0; j<my; j++)
				for (int i=0; i<mx; i++) {
					r[idx(i,j,k)] = h_rref[idx2(i,k)];
					u[idx(i,j,k)] = h_uref[idx2(i,k)];
					v[idx(i,j,k)] = 0.0;
					w[idx(i,j,k)] = h_wref[idx2(i,k)];
					e[idx(i,j,k)] = h_eref[idx2(i,k)];
				}
	}
    checkCuda( cudaMemcpy(d_rref, h_rref, mz*mx*sizeof(myprec), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_uref, h_uref, mz*mx*sizeof(myprec), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_wref, h_wref, mz*mx*sizeof(myprec), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_eref, h_eref, mz*mx*sizeof(myprec), cudaMemcpyHostToDevice) );

    delete [] h_rref;
    delete [] h_uref;
    delete [] h_wref;
    delete [] h_eref;

    copySpongeToDevice<<<mx,mz>>>(d_spongeX,d_spongeZ,d_rref,d_uref,d_wref,d_eref);

    checkCuda( cudaFree(d_spongeX) );
    checkCuda( cudaFree(d_spongeZ) );
    checkCuda( cudaFree(d_rref) );
    checkCuda( cudaFree(d_uref) );
    checkCuda( cudaFree(d_wref) );
    checkCuda( cudaFree(d_eref) );
}

myprec *vector12(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
	myprec *v;

	v=(myprec *)malloc((size_t) ((nh-nl+1+1)*sizeof(myprec)));
	if (!v){
		fprintf(stderr,"allocation failure in vector()");
		fprintf(stderr,"...now exiting to system...\n");
		exit(1);
	}
	return v-nl+1;
}

void free_vector(myprec *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
	free((char*) (v+nl-1));
}

void spline(myprec x[], myprec y[], int n, myprec yp1, myprec ypn, myprec y2[])
{
	int i,k;
	myprec p,qn,sig,un,*u;
	u=vector12(1,n-1);
	if (yp1 > 0.99e30)
		y2[1]=u[1]=0.0;
	else {
		y2[1] = -0.5;
		u[1]=(3.0/(x[2]-x[1]))*((y[2]-y[1])/(x[2]-x[1])-yp1);
	}
	for (i=2;i<=n-1;i++) {
		sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		p=sig*y2[i-1]+2.0;
		y2[i]=(sig-1.0)/p;
		u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
		u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
	}
	if (ypn > 0.99e30)
		qn=un=0.0;
	else {
		qn=0.5;
		un=(3.0/(x[n]-x[n-1]))*(ypn-(y[n]-y[n-1])/(x[n]-x[n-1]));
	}
	y2[n]=(un-qn*u[n-1])/(qn*y2[n-1]+1.0);
	for (k=n-1;k>=1;k--)
		y2[k]=y2[k]*y2[k+1]+u[k];
	free_vector(u,1,n-1);
}

myprec splint(myprec xa[], myprec ya[], myprec y2a[], int n, myprec x)
{
	int klo,khi,k;
	myprec h,b,a,y;
	klo=1;
	khi=n;
	while (khi-klo > 1) {
		k=(khi+klo) >> 1;
		if (xa[k] > x) khi=k;
		else klo=k;
	}
	h=xa[khi]-xa[klo];
	if (h == 0.0){
		fprintf(stderr,"Bad xa input to routine splint");
		fprintf(stderr,"...now exiting to system...\n");
		exit(1);
	}
	a=(xa[khi]-x)/h;
	b=(x-xa[klo])/h;
	y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
	return y;
}
