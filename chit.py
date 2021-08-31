import sys
import numpy as np
import math as m
import os.path
import subprocess
from os import path
sys.path.append('python-utils')

from writexmf import writexmf
from CompNavierStokes import CompNavierStokes


Nfiles=10 # we want to save down 10 fields

CompNavierStokes(nsteps=101,nfiles=Nfiles,Lx=2*m.pi,Ly=2*m.pi,Lz=2*m.pi)


directory='./fields/decaying-turbulence/'
if not(path.exists(directory)):
	subprocess.call(["mkdir",directory])

subprocess.call('mv ./fields/*.bin ' + directory, shell=True)

x = np.fromfile(directory + 'x.bin', dtype='double')
y = np.fromfile(directory + 'y.bin', dtype='double')
z = np.fromfile(directory + 'z.bin', dtype='double')
writexmf(directory + 'decaying-turbulence.xmf','double', \
         x, y, z, \
         np.arange(1,Nfiles+1,1), 1.0, \
         ['r',\
          'u',\
          'v',\
	  'w',\
          'e'])


