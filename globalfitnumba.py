import os
import multiprocessing

ncpu = multiprocessing.cpu_count()


#forcecpu = True
#if forcecpu:
    ##os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ncpu}"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
from numba import jit, prange
import threading

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

#iidxmap = nglobal*idxmap
#jidxmap = idxmap

nglobal = np.int64(runtree.GetEntries())
#print(nglobal)
grad = np.zeros((nglobal,1),dtype=np.float64)
hess = np.zeros((nglobal,nglobal), dtype=np.float64)

gradflat = grad.ravel()
hessflat = hess.ravel()

gradlock = threading.Lock()
hesslock = threading.Lock()

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



@jit(nopython=True, nogil=True, parallel=False)
def add_in_place_nodup(arr, idxs, updates):
  for i in prange(idxs.shape[0]):
    arr[idxs[i]] += updates[i]


@jit(nopython=True, nogil=True, parallel=False)
def add_in_place_sorted(arr, idxs, updates):
  #print("add_in_place")
  #sortidxs = np.argsort(idxs, kind="mergesort")
  #idxs = idxs[sortidxs]
  #updates = updates[sortidxs]
  
  #idxs_split = np
  

  #print("done final sort")
  
  
  updates = updates.astype(np.float64)
  
  mask = np.empty(idxs.shape, dtype=np.bool_)
  mask[:1] = True
  mask[1:] = idxs[1:] != idxs[:-1]
  uniqueidxs = idxs[mask]
  binidxs = mask.astype(np.intp).cumsum()

  countidxs = np.concatenate( (np.nonzero(mask)[0], np.array((mask.size,),dtype=np.intp) ))
  counts = countidxs[1:] - countidxs[:-1]
  startidxs = counts.cumsum() - counts
  
  #print("uniqueidxs.shape", uniqueidxs.shape)
  
  #print("uniquesums")
  #uniquesums = np.empty(uniqueidxs.shape, dtype=np.float64)
  #for i in prange(uniqueidxs.shape[0]):
    #startidx = startidxs[i]
    #count = counts[i]
    #uniquesums[i] = np.sum(updates[startidx:startidx+count])    
  
  #print("final sum")
  for i in prange(uniqueidxs.shape[0]):
    startidx = startidxs[i]
    count = counts[i]
    uniquesum = np.sum(updates[startidx:startidx+count])
    #uniquesum = 0.
    #for j in prange(count):
      #uniquesum += updates[startidx + j]
    arr[uniqueidxs[i]] += uniquesum
    #arr[uniqueidxs[i]] += uniquesums[i]
  #print("done add in place")

@jit(nopython=True, nogil=True, parallel=False)
def add_in_place(arr, idxs, updates):
  Ntotal = idxs.shape[0]
  Nsections = ncpu
  #Nsections = 4
  Neach_section, extras = divmod(Ntotal, Nsections)
  #section_sizes = ([0] +
                    #extras * [Neach_section+1] +
                    #(Nsections-extras) * [Neach_section])
  section_sizes = ([0] +
                    [Neach_section + 1 for i in range(extras)] +
                    [Neach_section for i in range(Nsections-extras)])
  div_points = np.array(section_sizes, dtype=np.intp).cumsum()
  
  #subidxs = []
  #subupdates = []
  
  #subidxs = [idxs[div_points[i]:div_points[i+1]] for i in range(Nsections)]
  #subupdates = [updates[div_points[i]:div_points[i+1]] for i in range(Nsections)]
  
  #for i in range(Nsections):
    #subidx = idxs[div_points[i]:div_points[i+1]]
    #subupdate = updates[div_points[i]:div_points[i+1]]
    
    #subidxs.append(subidx)
    #subupdates.append(subupdates)
  
  #subsortidxs = [np.argsort(idxs[div_points[i]:div_points[i+1]]) for i in prange(Nsections)]
  
  #for i in prange(Nsections):
    #idxs[div_points[i]:div_points[i+1]] = idxs[div_points[i]:div_points[i+1]][subsortidxs[i]]
    #updates[div_points[i]:div_points[i+1]] = updates[div_points[i]:div_points[i+1]][subsortidxs[i]]
  
  #for i in prange(Nsections):
    #subidx = subidxs[i]
    #subsortidxs = np.argsort(subidx)

  
  #print("subsorts")
  #print(div_points)
  #print(idxs.shape)
  #print(idxs)
  #for i in prange(Nsections):
    #subidxs = idxs[div_points[i]:div_points[i+1]]
    #subupdates = updates[div_points[i]:div_points[i+1]]
    #subsortidxs = np.argsort(subidxs, kind="mergesort")

    #subidxs[...] = subidxs[subsortidxs]
    #subupdates[...] = subupdates[subsortidxs]

    
  #print("final sort")
  #print(idxs)
  sortidxs = np.argsort(idxs, kind="mergesort")
  idxs = idxs[sortidxs]
  updates = updates[sortidxs]
  #print("done final sort")
  
  add_in_place_sorted(arr, idxs, updates)


#def add_in_place(arr, idxs, updates):
  #print("sort")
  #sortidxs = np.argsort(idxs)
  #idxs = idxs[sortidxs]
  #updates = updates[sortidxs]
  
  #print("add_in_place_sorted")
  #add_in_place_sorted(arr, idxs, updates)
  #print("done add_in_place_sorted")
  

@jit(nopython=True, nogil=True, parallel=False)
def handlenp(idxsflat, iidxsflat, jidxsflat, lgradflat, lhessflat, gradflat, hessflat):
  ##idxsflat = idxmap[idxsflat]
  idxsflat = idxsflat.astype(np.int64)
  
  iidxsflat = idxmap[iidxsflat]
  #iidxsflat = iidxsflat.astype(np.int64)
  
  jidxsflat = idxmap[jidxsflat]
  #jidxsflat = jidxsflat.astype(np.int64)  
  
  #fill only upper triangular
  switch = iidxsflat > jidxsflat
  iidxsflat, jidxsflat = (np.where(switch, jidxsflat, iidxsflat), np.where(switch,iidxsflat,jidxsflat))

  hessidxsflat = nglobal*iidxsflat + jidxsflat
  
  #print("add")
  add_in_place(gradflat, idxsflat, lgradflat)
  add_in_place(hessflat, hessidxsflat, lhessflat)
  
  #add_in_place_nodup(gradflat, idxsflat, lgradflat)
  #add_in_place_nodup(hessflat, hessidxsflat, lhessflat)
  

#@jit(nopython=True, nogil=True, parallel=True)
def handle(arrays):
  print("handle:")
  print(arrays.shape)
  #idxs, lgrad, lhess = arrays
  
  #trackpt = arrays["trackPt"]
  gradv = arrays["gradv"]
  idxs = arrays["globalidxv"]
  
  hessv = arrays["hesspackedv"]
  iidxs = arrays["iidxhesspackedv"]
  jidxs = arrays["jidxhesspackedv"]
  
  #print(gradv.shape)
  #print(gradv)
  #print(type(gradv))
  
  #print(gradv.tolist())
  
  #gradvflat = np.concatenate(gradv)
  #print(gradvflat.shape)
  
  #print("selection")
  #selidxs = trackpt > 5.5
  #idxs = idxs[selidxs]
  #gradv = gradv[selidxs]
  #hessv = hessv[selidxs]
  #iidxs = iidxs[selidxs]
  #jidxs = jidxs[selidxs]
  
  print("flatten")
  idxsflat = awkward1.flatten(idxs)
  idxsflat = awkward1.to_numpy(idxsflat)


  iidxsflat = awkward1.flatten(iidxs)
  iidxsflat = awkward1.to_numpy(iidxsflat)


  jidxsflat = awkward1.flatten(jidxs)
  jidxsflat = awkward1.to_numpy(jidxsflat)

  
  lgradflat = awkward1.flatten(gradv)
  lhessflat = awkward1.flatten(hessv)
  
  lgradflat = awkward1.to_numpy(lgradflat)
  lhessflat = awkward1.to_numpy(lhessflat)
  
  
  ##idxsflat = idxmap[idxsflat]
  #idxsflat = idxsflat.astype(np.int64)
  
  ##iidxsflat = idxmap[iidxsflat]
  #iidxsflat = iidxsflat.astype(np.int64)
  
  ##jidxsflat = idxmap[jidxsflat]
  #jidxsflat = jidxsflat.astype(np.int64)  
  
  print("handlenp")
  handlenp(idxsflat, iidxsflat, jidxsflat, lgradflat, lhessflat, gradflat, hessflat)

  
  #for i in range(50):
    #print("handlenp")
    #handlenp(idxsflat, iidxsflat, jidxsflat, lgradflat, lhessflat, gradflat, hessflat)
  
  #idxsflat = idxmap[idxsflat]
  ##idxsflat = idxsflat.astype(np.int64)
  
  #iidxsflat = idxmap[iidxsflat]
  ##iidxsflat = iidxsflat.astype(np.int64)
  
  #jidxsflat = idxmap[jidxsflat]
  ##jidxsflat = jidxsflat.astype(np.int64)  
  
  ##fill only upper triangular
  #switch = iidxsflat > jidxsflat
  #iidxsflat, jidxsflat = (np.where(switch, jidxsflat, iidxsflat), np.where(switch,iidxsflat,jidxsflat))

  #hessidxsflat = nglobal*iidxsflat + jidxsflat

  

  
  ##print(gradflat.shape)
  ##print(idxsflat.shape)
  ##print(lgradflat.shape)
  ##print(hessflat.shape)
  ##print(hessidxsflat.shape)
  ##print(lhessflat.shape)
  
  ##print(gradflat.dtype)
  ##print(idxsflat.dtype)
  ##print(lgradflat.dtype)
  ##print(hessflat.dtype)
  ##print(hessidxsflat.dtype)
  ##print(lhessflat.dtype)  
  
  ##print("sort")
  ##sortidxs = np.argsort(idxsflat)
  ##idxsflat = idxsflat[sortidxs]
  ##lgradflat = lgradflat[sortidxs]
  
  ##sortidxs = np.argsort(hessidxsflat)
  ##hessidxsflat = hessidxsflat[sortidxs]
  ##lhessflat = lhessflat[sortidxs]
  ##print("done sort")

                        
  
  #print("add")
  #add_in_place(gradflat, idxsflat, lgradflat)
  #add_in_place(hessflat, hessidxsflat, lhessflat)
  #print("do add")
  #with gradlock:
    #add_in_place(gradflat, idxsflat, lgradflat)
  #with hesslock:
    #add_in_place(hessflat, hessidxsflat, lhessflat)
    
  #with gradlock:
    #add_in_place_sorted(gradflat, idxsflat, lgradflat)
  #with hesslock:
    #add_in_place_sorted(hessflat, hessidxsflat, lhessflat)
    
  print("done handle")


#filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root:tree"
filename = "/data/bendavid/muoncaldata/test2/testlarge.root:tree"
#filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/testfile*.root:tree"
#filename = "/data/bendavid/muoncaldata/test/test4.root:tree"
#files = []
#for i in range(1):
  #files.append(filename)
#step_size=100000



#it = uproot4.iterate(filename, ["trackPt","globalidxv","gradv","hesspackedv","iidxhesspackedv","jidxhesspackedv"], decompression_exector = concurrent.futures.ThreadPoolExecutor(max_workers=ncpu), interpretation_executor=concurrent.futures.ThreadPoolExecutor(max_workers=ncpu), filter_branch = lambda x: uproot4.asDtype(np.float64))

#it = uproot4.iterate(filename, ["trackPt","globalidxv","gradv","hesspackedv","iidxhesspackedv","jidxhesspackedv"], decompression_exector = concurrent.futures.ThreadPoolExecutor(max_workers=ncpu), interpretation_executor=concurrent.futures.ThreadPoolExecutor(max_workers=ncpu), step_size=1000000)


it = uproot4.iterate(filename, ["trackPt","trackParms","trackCov"], decompression_exector = concurrent.futures.ThreadPoolExecutor(max_workers=ncpu), interpretation_executor=concurrent.futures.ThreadPoolExecutor(max_workers=ncpu), step_size=1000000)

#it = uproot4.iterate(filename, ["trackPt","globalidxv","gradv","hesspackedv","iidxhesspackedv","jidxhesspackedv"],step_size=1000)

#ntot=1335552

#it = uproot4.iterate(filename,["trackPt","globalidxv","gradv","hesspackedv","iidxhesspackedv","jidxhesspackedv"], decompression_exector = concurrent.futures.ThreadPoolExecutor(max_workers=ncpu), interpretation_executor=concurrent.futures.ThreadPoolExecutor(max_workers=ncpu),step_size=20000)

#it2 = uproot4.iterate(filename,["trackPt","globalidxv","gradv","hesspackedv","iidxhesspackedv","jidxhesspackedv"], decompression_exector = concurrent.futures.ThreadPoolExecutor(max_workers=ncpu), interpretation_executor=concurrent.futures.ThreadPoolExecutor(max_workers=ncpu),entry_start=20000,entry_stop=40000,step_size=20000)


#its = 32*[it]
#its = []
#for i in range(32):
  #it = uproot4.iterate(filename,["trackPt","globalidxv","gradv","hesspackedv","iidxhesspackedv","jidxhesspackedv"], decompression_exector = concurrent.futures.ThreadPoolExecutor(max_workers=ncpu), interpretation_executor=concurrent.futures.ThreadPoolExecutor(max_workers=ncpu),entry_start=i*5000, entry_stop=(i+1)*5000, step_size=5000)
  #its.append(it)

#def handleiter(it):
  #for arrays in it:
    #handle(arrays)

#it = uproot4.iterate(filename, ["trackPt","globalidxv","gradv","hesspackedv","iidxhesspackedv","jidxhesspackedv"], cut="trackPt>5.5")

#print(it[5].shape)upro
#assert(0)

#map(handle, it)
print("start iterate")
for i,arrays in enumerate(it):
  print(i)
  print(arrays.shape)
  #gradv = arrays["gradv"]
  #print(type(gradv))
  #print(gradv.shape)
  ##print(type(array))
  ##handle(arrays)
assert(0)

with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as e:
  #results = e.map(handle,it)
  print("mapping on iterators")
  results = e.map(handleiter, its)
  for res in results:
    pass

#add_in_place.parallel_diagnostics(level=4)
#add_in_place_sorted.parallel_diagnostics(level=4)

assert(0)

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


