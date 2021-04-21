

import numpy as np
from writexmf import writexmf



precision = 'double'

x = np.linspace(0,2,128)
y = np.linspace(0,np.pi,128)
z = np.linspace(0,2*np.pi,128)

writexmf("fieldtest.xmf", precision, \
         x, y, z, \
         np.arange(0,201,20), 1.0, \
         ['r',\
	  'u',\
	  'v',\
          'w',\
	  'e'])

