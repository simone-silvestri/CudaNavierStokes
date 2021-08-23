import sys
import numpy as np
import math as m
import os.path
import subprocess
from os import path
sys.path.append('python-utils')

from writexmf import writexmf
from CompNavierStokes import CompNavierStokes
from CompNavierStokes import initializeSolution

# Parameters that can be specified are:

# Grid Parameters (defaults are shown here)

# mx, my, mz = 128, 128, 128 = total grid size in the x-, y- and z- direction
# Lx, Ly, Lz =   1,   1,   1 = total domain length in the x-, y- and z- direction
# perX       = True          = boolean (true=periodic BC in X, false=wall BC in X)
# nUnifX     = False         = boolean (true=hyperbolic tangent in X, false=uniform grid in X)

# Numerical Parameters

# cfl        = 0.5  = CFL number
# stenA      = 4    = half of order for advective terms (minimum 1 maximum 4) 
# stenV      = 4    = half of order for viscous   terms (minimum 1 maximum 4 <= stenA) 

# Physical Parameters (defaults are shown here)

# Re      = 1600  = Reynolds number
# Ma      = 0.1   = Mach number
# Pr      = 1     = Prandtl number
# visc    = 1     = exponent for viscosity as a function of temperature
# forcing = False = boolean (true=uniform forcing in z- direction, false=no forcing)
#             	             if restart=-1, then the initialization will be taylor green vortices
#			     for forcing false and rollers for forcing true

# Simulation Parameters (defaults are shown)

# nsteps     = 200   = iterations between saving files
# nfiles     = 1     = number of files to save (total simulation nsteps*nfiles)
# restart    = -1    = if negative start from scratch, otherwise number of restart file
# pRow, pCol = 1, 1  = number of GPU used in y- and z- direction (divide my and mz)
# stream     = False = boolean (true=use streams for RHS, false=serial RHS calculation)
# checkCFL   = 10    = number of iteration at which the CFL condition is checked and Dt is changed accordingly
# checkBulk  = 10    = number of iteration at which Bulk values are calculated

# as a first example we do a decaying turbulence simulation. 
# all defaults are set up to have a decaying turbulence simulation
# we just have to set up the amount of required files (100 will do)
# and make sure the domain is 2\pi*2\pi*2\pi

Nfiles=100 # we want to save down 100 fields

CompNavierStokes(nsteps=101,nfiles=Nfiles,Lx=2*m.pi,Ly=2*m.pi,Lz=2*m.pi)

# when the solution is finished let's save the results in a different folder and create a visualization xmf file

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


# Now let's run a supersonic channel with Mach equal to 1.5 and reynolds (bulk) equal to 3000
# this time we need a slighlty bigger grid and to fix also the domain.
# we need to increase the reynolds number and make the grid not uniform as well as include forcing.
# we can also increase the number of steps in between the saving of the files 

CompNavierStokes(mx=160, my=192, mz=192, \
		 Lx=2,   Ly=2*m.pi, Lz=4*m.pi, \
		 perX=False, nUnifX=True, forcing=True, \
		 Re=3000, Ma=1.5, visc=0.7, \
		 nsteps=1000, nfiles=100) 

# Let's again save the results in a different folder and create a visualization xmf file

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
