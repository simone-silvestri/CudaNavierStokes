

import numpy as np
from writexmf import writexmf



precision = 'double'

x = np.fromfile('x.bin', dtype=precision)
y = np.fromfile('y.bin', dtype=precision)
z = np.fromfile('z.bin', dtype=precision)

writexmf("field.NUMBER.xmf", precision, \
         x, y, z, \
         np.arange(0,1,1), 1.0, \
         ['u',\
	  'v',\
          'w'])
