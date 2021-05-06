

import numpy as np
from writexmf import writexmf



precision = 'double'

x = np.fromfile('x.bin', dtype=precision)
y = np.fromfile('y.bin', dtype=precision)
z = np.fromfile('z.bin', dtype=precision)

writexmf("fieldtest.xmf", precision, \
         x, y, z, \
         np.arange(0,201,1), 1.0, \
         ['r',\
	  'u',\
	  'v',\
          'w',\
	  'e'])

