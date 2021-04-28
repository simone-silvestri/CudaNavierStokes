#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "globals.h"
#include "cuda_functions.h"

using namespace std;

double dt;

double dx,x[mx],xp[mx],xpp[mx],y[my],z[mz];

double r[mx*my*mz];
double u[mx*my*mz];
double v[mx*my*mz];
double w[mx*my*mz];
double e[mx*my*mz];

const int filenum = 20;
const int skipfile= 10;

int main(int argc, char** argv) {

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	double um[mx],vm[mx],wm[mx],rm[mx],em[mx];
	double uf[mx],vf[mx],wf[mx],rf[mx],ef[mx];

	initGrid();
	for (int i=0; i<mx; i++) {
		rm[i] = 0.0;
		um[i] = 0.0;
		vm[i] = 0.0;
		wm[i] = 0.0;
		em[i] = 0.0;
		rf[i] = 0.0;
		uf[i] = 0.0;
		vf[i] = 0.0;
		wf[i] = 0.0;
		ef[i] = 0.0;}

	for(int file = skipfile; file<filenum+1; file++) {
		initFile(file);
		cout << "loaded file " << file << "\n";
		for (int i=0; i<mx; i++) {
			for (int k=0; k<mz; k++) {
				for (int j=0; j<my; j++) {
					rm[i] += r[idx(i,j,k)]/my/mz/filenum;
					um[i] += r[idx(i,j,k)]*u[idx(i,j,k)]/my/mz/filenum;
					vm[i] += r[idx(i,j,k)]*v[idx(i,j,k)]/my/mz/filenum;
					wm[i] += r[idx(i,j,k)]*w[idx(i,j,k)]/my/mz/filenum;
					em[i] += e[idx(i,j,k)]/my/mz;
				}
			}
		}
	}
	for (int i=0; i<mx; i++) {
		um[i] /= rm[i];
		vm[i] /= rm[i];
		wm[i] /= rm[i]; }
	for(int file = skipfile; file<filenum+1; file++) {
		initFile(file);
		cout << "loaded file " << file << "\n";
		for (int i=0; i<mx; i++) {
			for (int k=0; k<mz; k++) {
				for (int j=0; j<my; j++) {
					rf[i] += (r[idx(i,j,k)]-rm[i])*(r[idx(i,j,k)]-rm[i])/my/mz/filenum;
					uf[i] += (u[idx(i,j,k)]-um[i])*(u[idx(i,j,k)]-um[i])/my/mz/filenum;
					vf[i] += (v[idx(i,j,k)]-vm[i])*(v[idx(i,j,k)]-vm[i])/my/mz/filenum;
					wf[i] += (w[idx(i,j,k)]-wm[i])*(w[idx(i,j,k)]-wm[i])/my/mz/filenum;
					ef[i] += (e[idx(i,j,k)]-em[i])*(e[idx(i,j,k)]-em[i])/my/mz/filenum;
				}
			}
		}
	}


	FILE *fp = fopen("profpost.txt","w+");
	for (int i=0; i<mx; i++)
		fprintf(fp,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",x[i],rm[i],um[i],vm[i],wm[i],em[i],rf[i],uf[i],vf[i],wf[i],ef[i]);
	fclose(fp);

	return 0;
}


