
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#N = 5

R = np.array([4.1437478, 7.0870599, 7.5626616, 11.746180, 28.888109, 28.979446, 33.737079, 33.829258, 35.048137, 35.139869, 39.898342, 39.989845,
52.630050, 57.872253, 61.893707, 62.010196, 67.135734, 67.253265,
73.687667, 73.806480, 74.882408, 81.997367, 89.673179, 96.831947],dtype=np.float64)


RPhiErr = np.array([0.0005205, 0.0006515, 0.0004069, 0.0047973, 0.0055708, 0.0055551,
0.0054941, 0.0043147, 0.392515, 0.0041837, 0.429451, 0.0042747,
0.465744, 0.0038275, 0.0038912, 0.003795, 0.233091, 0.0038282,
0.265093, 0.0029776, 0.270927, 0.0030374, 0.31596, 0.0023857],dtype=np.float64)


matval = np.array([ 4.753e-6 , 4.06e-6 , 4.06e-6 , 0.00001872, 0.000028, 0.000028,
0.00001061, 0.00001061, 5.467e-6, 5.467e-6 , 8.882e-6 , 8.882e-6 ,
0.00001315, 9.93e-6, 4.358e-6 , 4.358e-6 , 4.691e-6, 4.691e-6,
4.88e-6 , 4.88e-6 , 9.76e-6 , 8.704e-6 , 8.388e-6, 8.37e-6],dtype=np.float64)

N = R.shape[0]

A = np.stack((np.ones_like(R),R,R**2),axis=-1)

             
#def Vinit(i,j):
    #res = 0
    #if i==j:
        #res += RPhiErr[i]**2
        
    #for l in range(min(i,j)):
        #res += (x[i,0] - x[l,0])*(x[j,0] - x[l,0])*matval[l]*k**2
        
    #return res
        
#V = Matrix(N,N,Vinit)

Vd = np.diag(RPhiErr**2)

Ri = R[:,np.newaxis,np.newaxis]
Rj = R[np.newaxis,:,np.newaxis]
Rl = R[np.newaxis,np.newaxis,:]
matvall = matval[np.newaxis,np.newaxis,:]


coeff = (Ri-Rl)*(Rj-Rl)*matvall
sumcoeff = np.zeros_like(Vd)
for i in range(N):
    for j in range(N):
        for l in range(min(i,j)):
            sumcoeff[i,j] += coeff[i,j,l]
                           

#k = 0.2

#ksqs = np.linspace(-1.,0.,10000)

def det(ksq):

    V = Vd + sumcoeff*ksq

    #print(V)

    Vinv = np.linalg.inv(V)

    varinv = np.transpose(A) @ Vinv @ A
    #var = np.linalg.inv(varinv)

    #det = 1./np.linalg.det(var)
    det = np.linalg.det(varinv)
    return det

    #print(det)
    

ksqs = np.linspace(-0.009,-0.0008,10000)
detvals = []
for ksq in ksqs:
    detval = det(ksq)
    detvals.append(detval)

detvals = np.array(detvals,dtype=np.float64)

plt.plot()
plt.plot(ksqs,detvals)
plt.ylabel("det(A^T V^-1 A)")
plt.xlabel("k^2")

#plt.show()

#bounds = (-0.0032,-0.0011)
#bounds = (-0.14137,-0.1)
bounds = (-0.031,-0.025)
#bounds = (-0.1,-0.005)
#bounds = (-0.00098,-0.00025)

d = optimize.root_scalar(det,bracket=bounds)
#d = optimize.bisect(det,-1.,0.)
#print(d)
d = d.root

print(1./np.sqrt(np.abs(d)))

plt.show()
                               
