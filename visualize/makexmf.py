

import numpy as np
from writexmf import writexmf



precision = 'double'

x = np.linspace(0,2*np.pi,128)
y = np.linspace(0,2*np.pi,128)
z = np.linspace(0,2*np.pi,128)

writexmf("field.xmf", precision, \
         x, y, z, \
         np.arange(0,2,1), 1.0, \
         ['../fields/r',\
          '../fields/u',\
	  '../fields/v',\
          '../fields/w',\
          '../fields/e'])

