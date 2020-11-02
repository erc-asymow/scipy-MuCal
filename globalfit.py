import ROOT
import numpy as np
import scipy as scipy
from scipy import linalg
import matplotlib.pyplot as plt
import sys
import math

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

filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root"
#filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGradsdebug.root"
f = ROOT.TFile.Open(filename)
tree = f.tree
#parminfos = f.parminfos

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

#print(idxmap)
#assert(0)

#for parminfo in parminfos:
  #print(parminfo.eta)
  
#for parminfo in list(parminfos):
  #print(parminfo.eta)

#runtree = f.runtree

#nglobal = runtree.GetEntries()

#for track in tree:
  #nglobal = track.nglobalparms
  #nalign = track.nglobalparmsalignment
  #nbfield = track.nglobalparmsbfield
  #neloss = track.nglobalparmseloss
  #break

print(nglobal)
grad = np.zeros((nglobal,1),dtype=np.float64)
hess = np.zeros((nglobal,nglobal), dtype=np.float64)

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

i=0
for track in tree:
  if i%100==0:
    print(i)
    #break
  i += 1
  
  #n= len(track.gradv)
  #if n> maxhits:
    #maxhits = n
  
  #print(track.trackPt)
  if track.trackPt < 5.5:
    continue
  
  
  #atkparm = np.frombuffer(track.trackParms, dtype=np.float64)
  #print(atkparm)
  #print(atkparm.shape)
  #assert(0)
  
  
  #r.append(track.trackPt/track.genPt)
  
  pt = ptparm(track.trackParms)
  ptalt = ptparm(track.refParms)
  
  
  if track.genPt>0.:
    r.append(pt/track.genPt)
    ralt.append(ptalt/track.genPt)
  
  #r.append(deltaphi(track.trackPhi,track.genPhi))
  #ralt.append(deltaphi(track.refParms[2],track.genPhi))
  #ralt.append(track.refParms[2] - track.genPhi)
  
  lgrad = np.array(track.gradv)
  lhess = np.array(track.hessv)
  idxs = np.array(track.globalidxv)
  idxs = idxmap[idxs]
  

  
  #print(idxs)
  
  #print(lgrad.shape)
  #print(lhess.shape)
  #print(idxs.shape)
  
  nlocal = lgrad.shape[0]
  
  lgrad = np.reshape(lgrad,(nlocal,1))
  lhess = np.reshape(lhess,(nlocal,nlocal))
  lhess = lhess + np.triu(lhess, k=1).transpose()

  #fill in lower triangular part
  
  
  #print(np.diag(lhess,k=1))
  #print(np.diag(lhess,k=-1))
  
  #assert(0)
  
  np.add.at(grad,idxs,lgrad)
  #grad[idxs] += lgrad

  hessidxs = idxs[:,np.newaxis]*nglobal + idxs[np.newaxis,:]
  
  #print(idxs)
  #print(hessidxs[0])
  #print(hessidxs[1])
  #print(hessidxs.ravel())
  #assert(0)
  #hessidxs = idxs*nglobal + idxs[:,np.newaxis]
  #print(hessidxs.shape)
  #print(hessidxs)
  #assert(0)
  np.add.at(hess.ravel(),hessidxs.ravel(),lhess.ravel())
  #hess.ravel()[hessidxs.ravel()] += lhess.ravel()
  
  #print(hess[159,149])
  #print(hess[149,159])
  #print(hess.ravel()[137090])
  #assert(0)
  #hess.ravel()[hessidxs] += lhess.ravel()
  #assert(0)
  
  #np.add.at(hess,(idxs,idxs), lhess)
  #print(hess[idxs,idxs].shape)
  #break
  #grad[idxs] += lgrad
  #print(hess[idxs].shape)
  #hess[np.transpose((idxs,idxs))] += lhess
  #hess[idxs][:,:nlocal]
  #hess[idxs][:,idxs] += lhess
  #print(np.where(hess))
  #print(lgrad[0])
  #print(lhess[0,0])
  #print(grad[idxs[0]])
  #print(hess[idxs[0],idxs[0]])
  #print(np.sum(hess))
  #break
print("maxhits:")
print(maxhits)  


##print(hess[:100,-100:])
##print(hess[-100:,:100])
#print(np.diag(hess,k=1))
#print(np.diag(hess,k=-1))

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


