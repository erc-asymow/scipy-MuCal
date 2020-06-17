import numpy as np

a = np.arange(20)
a = np.flip(a)

a = np.stack((a,a),axis=-1)

a = np.argpartition(a,(3,5,1,6),axis=0)

print(a)
