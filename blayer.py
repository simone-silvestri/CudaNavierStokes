import sys
import numpy as np
import math as m
import os.path
import subprocess
from os import path
sys.path.append('python-utils')

from writexmf import writexmf
from CompNavierStokes import CompNavierStokes

CompNavierStokes(mx=320, my=10, mz=320, \
		 Lx=20,   Ly=2, Lz=100, \
		 perX=False, nUnifX=True, forcing=False, boundaryLayer=True, \
		 Re=300, Ma=0.8, visc=0.75, Pr=0.71, \
		 checkCFL=100,checkBulk=100, \
		 stenA=3, stenV=2, \
		 nsteps=101, nfiles=400, restart=-1) 

## directory='./fields/laminar-BL/'
## if not(path.exists(directory)):
## 	subprocess.call(["mkdir",directory])
## 
## subprocess.call('mv ./fields/*.bin ' + directory, shell=True)
## 
## x = np.fromfile(directory + 'x.bin', dtype='double')
## y = np.fromfile(directory + 'y.bin', dtype='double')
## z = np.fromfile(directory + 'z.bin', dtype='double')
## writexmf(directory + 'supersonic-channel.xmf','double', \
##          x, y, z, \
##          np.arange(1,Nfiles+1,1), 1.0, \
##          ['r',\
##           'u',\
##           'v',\
## 	     'w',\
##           'e'])
