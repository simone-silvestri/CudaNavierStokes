

import numpy as np
from writexmf import writexmf



precision = 'double'

x = np.linspace(0,2*np.pi,64)
y = np.linspace(0,2*np.pi,64)
z = np.linspace(0,2*np.pi,64)

writexmf("field64.xmf", precision, \
         x, y, z, \
         np.arange(0,3,1), 1.0, \
         ['../fields/r',\
          '../fields/u',\
	  '../fields/v',\
          '../fields/w',\
          '../fields/e'])

