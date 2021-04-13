

import numpy as np
from writexmf import writexmf



precision = 'double'

x = np.linspace(0,2*np.pi,128)
y = np.linspace(0,2*np.pi,128)
z = np.linspace(0,2*np.pi,128)

writexmf("field128.xmf", precision, \
         x, y, z, \
         np.arange(0,121,1), 1.0, \
         ['r',\
          'u',\
	  'v',\
          'w',\
          'e'])

