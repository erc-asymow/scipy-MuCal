import math
import numpy as np

d = np.random.standard_normal(100001)

d.sort()

mediansum = 0.
median2sum = 0.
wsum = 0.
nmeds = []
for i in range(1000):
    n = d.shape[0]
    pois = np.random.poisson(size=n)
    npois = np.sum(pois)
    sumpois = np.cumsum(pois)
    
    nmed = np.searchsorted(sumpois,npois//2)
    nmeds.append(nmed)
    
    #dalt = np.random.choice(d,size=n,replace=True)
    #dalt.sort()
    #print( math.sqrt(n//2) )
    nmed = n/2 + math.sqrt(n/4)*np.random.standard_normal()
    #print(nmed)
    nmed = int(round(nmed))
    #print(nmed)

    #nmed = n//2
    #print(dalt)
    #median = np.mean(dalt)
    #median = dalt[n//2]
    median = d[nmed]
    mediansum += median
    median2sum += median**2
    wsum += 1.
    
print(np.mean(nmeds), np.std(nmeds))
medianerr = math.sqrt(median2sum/wsum - (mediansum/wsum)**2)
print(medianerr)


n = d.shape[0]
nmedup = n/2 + np.sqrt(n/4)
nmeddown = n/2 - np.sqrt(n/4)
nmedup = np.round(nmedup).astype(np.int32)
nmeddown = np.round(nmeddown).astype(np.int32)


medianerrfast = 0.5*(d[nmedup]-d[nmeddown])
print(medianerrfast)
