import sys
import numpy as np
import math as m
import os.path
import subprocess
from os import path
sys.path.append('python-utils')

from writexmf import writexmf
from CompNavierStokes import CompNavierStokes

Nfiles=100

CompNavierStokes(mx=120, my=10, mz=120, \
		 Lx=2,   Ly=2*m.pi, Lz=4*m.pi, \
		 perX=False, nUnifX=True, forcing=True, \
		 Re=100, Ma=0.8, visc=0.7, \
		 nsteps=101, nfiles=Nfiles) 

directory='./fields/supersonic-channel/'
if not(path.exists(directory)):
	subprocess.call(["mkdir",directory])

subprocess.call('mv ./fields/*.bin ' + directory, shell=True)

x = np.fromfile(directory + 'x.bin', dtype='double')
y = np.fromfile(directory + 'y.bin', dtype='double')
z = np.fromfile(directory + 'z.bin', dtype='double')
writexmf(directory + 'supersonic-channel.xmf','double', \
         x, y, z, \
         np.arange(1,Nfiles+1,1), 1.0, \
         ['r',\
          'u',\
          'v',\
	  'w',\
          'e'])

