import numpy as np

sigm = 0.03
sigc = 90e-6

m = np.random.normal(loc=0., scale = sigm, size = [1000*1000])


mc = (m/sigm/sigm + 0./sigc/sigc)/(1./sigm/sigm + 1./sigc/sigc)
print(mc)

print(np.mean(mc))
print(np.std(mc))
