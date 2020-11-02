import os
import multiprocessing

ncpu = multiprocessing.cpu_count()


forcecpu = True
if forcecpu:
    #os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ncpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import ROOT
import uproot4
import awkward1
import numpy as np
import scipy as scipy
from scipy import linalg
import matplotlib.pyplot as plt
import sys
import math
import concurrent.futures
import contextlib

#import jax
#import jax.numpy as jnp
#import tensorflow as tf


def ptparm(parms):
  p = np.abs(1./parms[0])
  theta = np.pi/2. - parms[1]
  pt = p*np.sin(theta)
  return pt
  
def deltaphi(phi1,phi2):
  if phi1 < -np.pi:
    phi1 += 2*np.pi
  elif phi1 >= np.pi:
    phi1 += -2*np.pi

  if phi2 < -np.pi:
    phi2 += 2*np.pi
  elif phi2 >= np.pi:
    phi2 += -2*np.pi
  
  dphi = phi2-phi1
  if dphi < -np.pi:
    dphi += 2*np.pi
  elif dphi >= np.pi:
    dphi += -2*np.pi
  return dphi


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

print(len(parmlist))
idxmap = np.array(idxmap)

nglobal = len(parmlist)


idxmap = idxmap.astype(np.int64)
#nglobal = np.int64(runtree.GetEntries())
print(nglobal)
grad = np.zeros((nglobal,1),dtype=np.float64)
hess = np.zeros((nglobal,nglobal), dtype=np.float64)

gradflat = grad.ravel()
hessflat = hess.ravel()

#TODO compute the splits accounting that only upper triangular indices will be updated
print(hessflat.shape)
hess_split = np.array_split(hessflat, ncpu)
nhess = len(hess_split)
print(nhess)
counts = []
for a in hess_split:
  counts.append(a.shape[0])
  
counts = np.array(counts)
binedges = np.cumsum(counts)
binedges = np.concatenate([[0],binedges])

print(counts)
print(binedges)

#assert(0)

filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root:tree"
#filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/testfile*.root:tree"
#files = []
#for i in range(1):
  #files.append(filename)

it = uproot4.iterate(filename, ["trackPt","globalidxv","gradv","hessv"], step_size=50000)


def handle(arrays, gradexec, hessexecs):
  print(arrays.shape)
  #idxs, lgrad, lhess = arrays
  
  trackpt = arrays["trackPt"]
  idxs = arrays["globalidxv"]
  gradv = arrays["gradv"]
  hessv = arrays["hessv"]
  
  print("selection")
  selidxs = trackpt>5.5
  idxs = idxs[selidxs]
  gradv = gradv[selidxs]
  hessv = hessv[selidxs]
  
  print("idxsflat")
  idxsflat = awkward1.flatten(idxs,highlevel=True)
  idxsflat = awkward1.to_numpy(idxsflat)
  #idxsflat = idxsflat.astype(np.int64)
  idxsflat = idxmap[idxsflat]


  #TODO make this faster (pack storage of upper triangular part in C++ job?)
  print("idxmangling")
  #restrict sum to upper triangular indices from source
  #upper triangular indices including diagonal
  nhits = awkward1.count(idxs, axis=1)
  #print(nhits)
  idxdualsource = awkward1.argcombinations(idxs, 2, replacement=True, axis=1)
  iidxsource, jidxsource = awkward1.unzip(idxdualsource)
  #TODO move multiplication before breakout
  print("idxcalc1")
  idxsource = nhits*iidxsource + jidxsource
  print("idxcalc1 done")
  iidx = idxs[iidxsource]
  jidx = idxs[jidxsource]
  #print(idxsource.shape)
  #print(idxsource[0].shape)
  ##print(hessv[idxsource])
  #idxsourceflat = 
  
  #print(
  #print(idxdualsource)
  #print(idxs[idxdualsource])
  ##idxdual = awkward1.combinations(idxs, 2, replacement=True, axis=1)
  #print(idxdual)
  #assert(0)
  
  
  iidxflat = awkward1.flatten(iidx, highlevel=True)
  jidxflat = awkward1.flatten(jidx, highlevel=True)
  iidxflat = awkward1.to_numpy(iidxflat)
  jidxflat = awkward1.to_numpy(jidxflat)
  
  #iidxflat = iidxflat.astype(np.int64)
  #jidxflat = jidxflat.astype(np.int64)
  iidxflat = idxmap[iidxflat]
  jidxflat = idxmap[jidxflat]
  
  #fill only upper triangular
  switch = iidxflat > jidxflat
  iidxflat, jidxflat = (np.where(switch, jidxflat, iidxflat), np.where(switch,iidxflat,jidxflat))
  
  #TODO move multiplication before breakout
  print("idxcalc2")
  hessidxsflat = nglobal*iidxflat + jidxflat
  print("idxcalc2 done")
  
  lgradflat = awkward1.flatten(gradv, highlevel=False)
  lhessflat = awkward1.flatten(hessv[idxsource], highlevel=False)
  
  print("sort")
  #sortidxs = np.argsort(hessidxsflat)
  #hessidxsflat = hessidxsflat[sortidxs]
  #lhessflat = lhessflat[sortidxs]
  
  hessbinidxs = np.digitize(hessidxsflat, binedges) - 1
  counts = np.bincount(hessbinidxs, minlength = nhess)
  #print(counts)
  #print(len(counts))
  #print(hessidxsflat.shape)
  splitidxs = np.cumsum(counts)[:-1]
  #print(splitidxs)
  
  sortidxs = np.argpartition(hessbinidxs, splitidxs)
  hessidxsflat = hessidxsflat[sortidxs]
  lhessflat = lhessflat[sortidxs]
  
  hessidxsflat_split = np.split(hessidxsflat, splitidxs)
  lhessflat_split = np.split(lhessflat, splitidxs)
  #print("last array", hessidxsflat_split[-1])
  
  #print(len(hessidxsflat_split))
  #assert(0)
  

  print("add")
  #np.add.at(gradflat, idxsflat, lgradflat)
  gradresult = gradexec.submit(np.add.at, gradflat, idxsflat, lgradflat)
  hessresults = []
  for i in range(nhess):
    hessresult = hessexecs[i].submit(np.add.at, hess_split[i], hessidxsflat_split[i]-binedges[i], lhessflat_split[i])
    hessresults.append(hessresult)
  #np.add.at(hessflat, hessidxsflat, lhessflat)
  #hessres = hessexecs[0].submit(np.add.at, hessflat, hessidxsflat, lhessflat)
  gradresult.result()
  for res in hessresults:
    res.result()
  #hessres.result()
  print("done")


#executor = concurrent.futures.ThreadPoolExecutor()
#executor

with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as e,\
    concurrent.futures.ThreadPoolExecutor(max_workers=1) as gradexec,\
    contextlib.ExitStack() as stack:
  
  hessexecs = [stack.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=1)) for i in range(ncpu)]
  
  def dohandle(arrays):
    return handle(arrays, gradexec, hessexecs)
  
  #map(dohandle, it)
  #for arrays in it:
    #dohandle(arrays)
  
  results = e.map(dohandle, it)
  for res in results:
    pass
  
#assert(0)


maxhits = 0

r = []
ralt = []

#for track in tree:
  #ljacref = np.array(track.jacrefv)
  ##ljacref = np.reshape(ljacref,(-1,5)).transpose()
  #ljacref = np.reshape(ljacref,(5,-1))
  #print(ljacref)
  
  #n= len(track.gradv)
  
  #lhess = np.array(track.hessv)
  #lhess = np.reshape(lhess,(n,n))
  
  #print(lhess)
  
  #assert(0)


##print(hess[:100,-100:])
##print(hess[-100:,:100])
print(np.diag(hess,k=1))
print(np.diag(hess,k=0))
print(np.diag(hess,k=-1))

hess + np.triu(hess, k=1).transpose()

#assert(0)

#print(hess[159,149])
#print(hess[149,159])
#print(hess.ravel()[137090])
#print(hess[918,917])
#print(hess.ravel()[844559])
#assert(0)

#print(np.mean(r))
#print(np.std(r))
#print(np.mean(ralt))
#print(np.std(ralt))

#assert(0)

#hess = hess[:nalign,:nalign]
#grad = grad[:nalign]

#hess = hess[nalign:nalign+nbfield,nalign:nalign+nbfield]
#grad = grad[nalign:nalign+nbfield]

#hess = hess[nalign+nbfield:,nalign+nbfield:]
#grad = grad[nalign+nbfield:]

#hess = np.block( [[hess[:nalign,:nalign], hess[:nalign,nalign+nbfield:]],
                  #[hess[nalign+nbfield:,:nalign], hess[nalign+nbfield:,nalign+nbfield:]]])
#grad = np.concatenate([grad[:nalign], grad[nalign+nbfield:]], axis=0)

e,v = np.linalg.eigh(hess)
#v0 = v[:,0]

print(e)
assert(0)

badidxs = []
for i in range(10):
  if e[i]<0.:
    idx = np.argmax(np.abs(v[:,i]))
    grad[idx] = 0.
    hess[idx] *= 0.
    hess[:,idx] *= 0.
    hess[idx,idx] = 1./1e-2**2
    badidxs.append(idx)

e,v = np.linalg.eigh(hess)
print(e)

xout = np.linalg.solve(hess,-grad)
cov = np.linalg.inv(hess)
errs = np.sqrt(np.diag(cov))

print(xout)

rcor = []
rcoralt = []
for track in tree:
  #print(track.trackPt)
  if track.trackPt < 5.5:
    continue
  
  ljacref = np.array(track.jacrefv)
  #ljacref = np.reshape(ljacref,(-1,5)).transpose()
  ljacref = np.reshape(ljacref,(5,-1))
  idxs = np.array(track.globalidxv)
  idxs = idxmap[idxs]
  
  parms = xout[idxs]
  dtk = ljacref@parms
  dtk = np.reshape(dtk,(5,))
  
  #print(ljacref)
  #print(parms)
  #print(dtk)
  
  tkcor = np.frombuffer(track.trackParms, dtype=np.float32) + dtk
  tkcoralt = np.frombuffer(track.refParms, dtype=np.float32) + dtk
  
  
  
  #r.append(track.trackPt/track.genPt)
  
  pt = ptparm(tkcor)
  ptalt = ptparm(tkcoralt)
  
  
  if track.genPt>0.:
    rcor.append(pt/track.genPt)
    rcoralt.append(ptalt/track.genPt)
  

r = np.array(r)
ralt = np.array(ralt)
rcor = np.array(rcor)
rcoralt = np.array(rcoralt)

#hr = np.histogram(r, bins=100, range=(0.9,1.1))

plt.plot()
plt.hist(r, bins=100, range=(0.9,1.1))
#plt.hist(ralt, bins=100, range=(0.9,1.1))
#plt.hist(rcor, bins=100, range=(0.9,1.1))
plt.hist(rcoralt, bins=100, range=(0.9,1.1))

plt.show()

#print("align:")
#print(v0[:nalign])
#print("bfield:")
#print(v0[nalign:nalign+nbfield])
#print("eloss")
#print(v0[nalign+nbfield:])

#print(np.argmax(np.abs(v0)))

#assert(0)

#print("before:")
#print(np.linalg.eigvalsh(hess))
#diagadd = np.zeros((nglobal,),np.float64)
##diagadd[:nalign] = 1./1e-2**2
#diagadd[nalign:nalign+nbfield] = 1/0.038**2
##diagadd[nalign:nalign+nbfield] = 1/0.00038**2
##diagadd[nalign+nbfield:] = 1./1e-4**2

#diagidxs = np.diag_indices_from(hess)
#np.add.at(hess, diagidxs, diagadd)
#print("after:")
#print(np.linalg.eigvalsh(hess))
#assert(0)
  
#np.fill_diagonal(hess,np.diag(hess)+diagadd)
  

e,v = np.linalg.eigh(hess)
print(e)


print(xout)
print(errs)

resmap = {}
#for parm in runtree:
  #iidx = parm.iidx
  #if iidx in badidxs:
    #continue
  #parmtype = parm.parmtype
  #subdet = parm.subdet
  #layer = parm.layer
  #ieta = math.floor(parm.eta/0.2)
  
  #resmap[(parmtype, subdet, abs(layer), ieta)] = (xout[iidx], errs[iidx])
  
for iparm,parm in enumerate(parmlist):
  resmap[parm] = (xout[iparm], errs[iparm])
  
  


#subdet = 1
#parmtype = 2
#layer = 1
for parmtype in range(4):
  for subdet in range(7):
    for layer in range(50):
      etas = []
      vals = []
      valerrs = []
      for ieta in range(-15,16):
        key = (parmtype, subdet, layer, ieta)
        if key in resmap:
          eta = 0.2*ieta
          etas.append(eta)
          val,err = resmap[key]
          vals.append(val)
          valerrs.append(err)
          print(key, resmap[key])
      if vals:
        plot = plt.figure()
        label = f"parmtype {parmtype}, subdet {subdet}, layer {layer}"
        plt.title(label)
        plt.xlabel("$Module Center \eta$")
        plt.errorbar(etas, vals, yerr=valerrs,fmt='none')
        
plt.show()
#assert(0)




#np.set_printoptions(threshold=sys.maxsize)



#print(v[:7])

#print(xout)
#ldlt = scipy.linalg.ldl(a, overwrite_a=True, check_finite=False)
#cho = scipy.linalg.cho_factor(hess, overwrite_a=True, check_finite=False)
#xout = scipy.linalg.cho_solve(cho, -grad, overwrite_b=True, check_finite=False)
#xout = scipy.linalg.lstsq(hess,-grad)

#print(grad)
#print(hess)


