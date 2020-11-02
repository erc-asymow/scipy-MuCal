import numpy as np

a = np.empty((1024*1024*1024*16,), dtype=np.uint8)

a[4*1024**3:12*1024**3] = 1
