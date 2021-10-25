import numpy as np

import matplotlib.pyplot as plt


ntoys = int(1e8)

mean = 1.
sigma0 = 0.01
sigma1 = 0.01

krnd0 = mean + sigma0*np.random.standard_normal((ntoys,))
krnd1 = mean + sigma1*np.random.standard_normal((ntoys,))


mrnd = np.sqrt(1./krnd0/krnd1)

invmsqrnd = krnd0*krnd1

#k0cons

#invrnd = 1./krnd

print(np.mean(krnd0))
print(np.mean(krnd1))
#print(np.mean(invrnd))
#print(np.mean(np.log(krnd)))

print(1.-np.mean(mrnd))
print(1.-np.mean(invmsqrnd))

print(np.std(mrnd))
