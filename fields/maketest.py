import numpy as np
from writexmf import writexmf


val1 = input("Enter value1: ")
val2 = input("Enter value2: ")
val3 = input("Enter folder: ")
precision = 'double'

x = np.fromfile(val3+'/x.bin', dtype=precision)
y = np.fromfile(val3+'/y.bin', dtype=precision)
z = np.fromfile(val3+'/z.bin', dtype=precision)

writexmf("field"+val3+".xmf", precision, \
         x, y, z, \
         np.arange(val1,val2,1), 1.0, \
         [val3+'/r',\
	  val3+'/u',\
	  val3+'/v',\
          val3+'/w',\
	  val3+'/e'])
