

import numpy as np
from writexmf import writexmf



precision = 'double'

x = np.linspace(0,2*np.pi,192)
y = np.linspace(0,2*np.pi,192)
z = np.linspace(0,2*np.pi,192)

writexmf("field.0000.xmf", precision, \
         x, y, z, \
         np.arange(0,1,1), 1.0, \
         ['r',\
          'u',\
	  'v',\
          'w',\
          'e'])

