import sys
sys.path.append("../python-utils")

import numpy as np
from writexmf import writexmf


val1 = input("Enter value1: ")
val2 = input("Enter value2: ")
precision = 'double'

x = np.fromfile('x.bin', dtype=precision)
y = np.fromfile('y.bin', dtype=precision)
z = np.fromfile('z.bin', dtype=precision)

writexmf("field.xmf", precision, \
         x, y, z, \
         np.arange(val1,val2,1), 1.0, \
         ['r',\
	  'u',\
	  'v',\
          'w',\
	  'e'])
