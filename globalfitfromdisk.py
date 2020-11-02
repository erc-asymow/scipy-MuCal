import math
import ROOT
import numpy as np
import concurrent.futures

results = np.load('combinedgrads.npz')

gradfull = results["grad"]
hessfull = results["hess"]


#assert(0)

#print(grad[0])
#print(hess[0])


print("filling in lower triangular part of hessian")
for i in range(hessfull.shape[0]):
  print(i)
  hessfull[i,:i] = hessfull[:i,i]

print(hessfull[0,0])
print(hessfull[0,:10])
print(hessfull[:10,0])

print(hessfull[1,1])
print(hessfull[1,:10])
print(hessfull[:10,1])
  
  
#print(np.linalg.eigvalsh(hessfull))
#assert(0)


filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGradsParmInfo.root"
finfo = ROOT.TFile.Open(filenameinfo)

runtree = finfo.tree

parmset = set()
for iidx,parm in enumerate(runtree):
  parmtype = runtree.parmtype
  ieta = math.floor(runtree.eta/0.2)
  subdet = runtree.subdet
  layer = abs(runtree.layer)
  key = (parmtype, subdet, layer, ieta)
  parmset.add(key)
  
parmlist = list(parmset)
parmlist.sort()

parmmap = {}
for iparm,key in enumerate(parmlist):
  parmmap[key] = iparm
  
idxmap = []
for iidx,parm in enumerate(runtree):
  parmtype = runtree.parmtype
  ieta = math.floor(runtree.eta/0.2)
  subdet = runtree.subdet
  layer = abs(runtree.layer)
  key = (parmtype, subdet, layer, ieta)
  idxmap.append(parmmap[key])

#print(len(parmlist))
idxmap = np.array(idxmap)
print(idxmap)

nglobal = len(parmlist)
print(nglobal)

grad = np.zeros((nglobal,1), dtype=np.float64)
hess = np.zeros((nglobal,nglobal), dtype=np.float64)

print("reducing grad")
np.add.at(grad.ravel(), idxmap, gradfull.ravel())
print("reducing hess")
stepsize = 200

def fillhess(i):
  end = np.minimum(i+stepsize, hessfull.shape[0])
  idxs = nglobal*idxmap[i:end, np.newaxis] + idxmap[np.newaxis,:]
  np.add.at(hess.ravel(), idxs.ravel(), hessfull[i:end].ravel())

with concurrent.futures.ThreadPoolExecutor(32) as e:
  results = e.map(fillhess, range(0, hessfull.shape[0], stepsize))

for res in results:
  pass;

##idxs = nglobal*idxmap[:, np.newaxis] + idxmap[np.newaxis,:]
#np.add.at(hess.ravel(), idxs.ravel(), hessfull.ravel())

  
print("hessfull shape[0]", hessfull.shape[0])

#for i in range(0, hessfull.shape[0], stepsize):
#for i in range(hessfull.shape[0]):  
  #print(i)
  ##end = np.minimum(i+stepsize, hessfull.shape[0])
  #idxs = nglobal*idxmap[i:i+1, np.newaxis] + idxmap[np.newaxis,:]
  #np.add.at(hess.ravel(), idxs.ravel(), hessfull[i:i+1].ravel())

print(grad[0])
print(hess[0])
print(hess[0,0])
print(hess[0,:10])
print(hess[:10,0])

print(hess[1,1])
print(hess[1,:10])
print(hess[:10,1])

print(hess[-1,-1])

hessbak = hess
gradbak = grad

#assert(0)

print(np.linalg.eigvalsh(hess))


