import math
import ROOT
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import threading
import scipy
import scipy.linalg
import sys
import scipy.sparse.linalg
import scipy.sparse
import h5py

np.set_printoptions(threshold=sys.maxsize)


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

#results = np.load('combinedgrads.npz')

#gradfull = results["grad"]
#hessfull = results["hess"]


##assert(0)

##print(grad[0])
##print(hess[0])


#print("filling in lower triangular part of hessian")
#for i in range(hessfull.shape[0]):
  #print(i)
  #hessfull[i,:i] = hessfull[:i,i]

#print(hessfull[0,0])
#print(hessfull[0,:10])
#print(hessfull[:10,0])

#print(hessfull[1,1])
#print(hessfull[1,:10])
#print(hessfull[:10,1])
  
  
#print(np.linalg.eigvalsh(hessfull))
#assert(0)

#filenamecor = "correctionResults_v171_genfull.root"
#filenamecor = "/data/home/bendavid/muoncal/scipy-MuCal/plotscale_v171_quality_biasm10_biasfield_constraintnofsr_iter0/correctionResults_v171_quality_biasm10_biasfield_constraintnofsr_iter0.root"
#filenamecor = "/data/home/bendavid/muoncal/scipy-MuCal/plotscale_v171_quality_biasm10_biasfield_constraintfsr28_iter0/correctionResults_v171_quality_biasm10_biasfield_constraintfsr28_iter0.root"
#filenamecor = "correctionResults_v186_jpsi_biasm10_biasfield_bz.root"
filenamecor = "correctionResults_v191_jpsi.root"




#filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGradsParmInfo.root"
#filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_10_2_23/work/trackTreeGradsParmInfo.root"
#filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_idealquality_constraint/210911_050035/0000/globalcor_0_1.root"
#filenameinfo = "infofullbz.root"
filenameinfo = "info_v202.root"
#filenameinfo = "info_v201.root"
#filenameinfo = "infofullbzdisky.root"
#filenameinfo = "infofullbzr.root"
finfo = ROOT.TFile.Open(filenameinfo)

runtree = finfo.runtree
#runtree = finfo.globalCorGen.runtree

nparmsfull = np.int64(runtree.GetEntries())


oldcors = np.zeros((nparmsfull,), dtype=np.float64)

if False:
    fcor = ROOT.TFile.Open(filenamecor)
    parmtree = fcor.Get("parmtree")

    #oldcors = []
    for iparm,entry in enumerate(parmtree):
        oldcors[iparm] = parmtree.x
        #oldcors.append(parmtree.x)

testsize = nparmsfull
#testsize = 20000
#testsize = 5


gradfull = np.zeros((testsize,1), dtype=np.float64)
hessfull = np.zeros((testsize, testsize), dtype=np.float64)

filenamegrads = "combinedgrads.hdf5"
#filenamegrads = "combinedgradsrec.root"
#filenamegrads = "/data/bendavid/muoncaldata/combinedgrads.root"
#fgrads = ROOT.TFile.Open(filenamegrads)
#gradtree = fgrads.tree

#row = np.zeros((nparmsfull,), dtype=np.float64)

#chunksize = 32

#fgrads = h5py.File(filenamegrads, rdcc_nbytes = nparmsfull*8*chunksize*4, rdcc_nslots = nparmsfull//chunksize*10)
fgrads = h5py.File("combinedgrads.hdf5")
gradd = fgrads["grad"]
hessd = fgrads["hess"]

gradfull[:,0] = 0.5*gradd[...]

print("loading grads")
#for i, entry in enumerate(gradtree):
for i in range(nparmsfull):
  #if i== testsize:
      #break
  #break
  #print(i)
  if i%5000 == 0:
      print(i)
  #gradfull[i,0] = entry.gradelem
  #hessfull[i] = entry.hessrow
  #gradfull[i,0] = 0.5*entry.gradelem
  #row[...] = entry.hessrow
  #hessfull[i] = 0.5*row[:testsize]
  hessfull[i] = 0.5*hessd[i]

#nfilled = 0
#for i in range(nparmsfull):
    #for j in range(nparmsfull):
        #if hessfull[i,j] != 0.:
            #nfilled += 1





#nonzero = np.count_nonzero(hessfull)
#print("nonzero", nonzero)
#assert(0)

print("filling in lower triangular part of hessian")
def filllower(i):
  hessfull[i,:i] = hessfull[:i,i]


#nfilled = np.count_nonzero(hessfull)

#print(nfilled)
#print(nparmsfull*nparmsfull)
#print(float(nfilled)/float(nparmsfull*nparmsfull))

#assert(0)
  
with concurrent.futures.ThreadPoolExecutor(64) as e:
  results = e.map(filllower, range(hessfull.shape[0]))
  
for result in results:
  pass


#for i in range(hessfull.shape[0]):
  #print(i)
  #hessfull[i,:i] = hessfull[:i,i]



parmset = set()
parmlistfull = []
for iidx,parm in enumerate(runtree):
    if iidx == testsize:
        break
    parmtype = runtree.parmtype
    ieta = math.floor(runtree.eta/0.1)
    #ieta = 0
    iphi = math.floor(runtree.phi/(math.pi/8.))
    #iphi = math.floor(runtree.phi/(math.pi/1024.))
    #iphi = 0
    #ieta = math.floor(runtree.eta/1.0)
    subdet = runtree.subdet
    layer = abs(runtree.layer)
    #if (parmtype<2) :
        #parmtype = -1
        #subdet = -1
        #layer = -1
        #ieta = 0
        #iphi = 0
    #elif (parmtype==3):
    ##else:
        #ieta = iidx
        #iphi = 0
      
  #if parmtype>1:
    #if (subdet==3 and layer==7) or (subdet==5 and layer==9):
      #subdet = -1
      #layer = -1
      #ieta = 0
      #parmtype = -1
    #key = (parmtype, subdet, layer, ieta)
    key = (parmtype, subdet, layer, (ieta,iphi))
    #key = (parmtype, subdet, layer, (ieta,iphi), runtree.bz)
    parmset.add(key)
    parmlistfull.append(key)
  
#parmlist = list(parmset)
#parmlist.sort()

parmlist = parmlistfull

parmmap = {}
for iparm,key in enumerate(parmlist):
  parmmap[key] = iparm
  
#idxmap = []
#for iidx, key in enumerate(parmlistfull):
  #idxmap.append(iidx)

#print(len(parmlist))
#idxmap = np.array(idxmap)
#print(idxmap)

#nglobal = len(parmlist)
#print(nglobal)

nglobal = gradfull.shape[0]
print(nglobal)
grad = gradfull
hess = hessfull
idxmap = np.arange(nglobal)


#bfact = 2.99792458e-3

for iparm,parm in enumerate(parmlist):
    parmtype, subdet, layer, ieta = parm
    if parmtype==-1:
        print(f"null parameter for index {iparm}")
        grad[iparm] = 0.
        hess[iparm, :] = 0.
        hess[:, iparm] = 0.
        hess[iparm,iparm] = 2.
        
    siga = 0.
    if parmtype == 0:
        siga = 5e-3
    elif parmtype == 1:
        if subdet < 2:
            siga = 5e-3
        else:
            siga = 5e-1
    elif parmtype == 2:
        #normally z displacement, temporarily spurious charge dependent x displacement
        siga = 5e-3
    elif parmtype == 5:
        siga = 5e-3
    elif parmtype == 6:
        siga = 0.038
        #siga = 0.76
        #siga = 0.2
    #elif parmtype == 7:
        ##siga = 1.0
        #siga = 0.76
        ##siga = 0.2
        ##siga = 1e-6
        ##grad[iparm] += -oldcors[iparm]/siga**2
    #elif parmtype == 8:
        #siga = 0.76
        ##siga = 0.5
    #elif parmtype == 9:
        #siga = math.log(2.)
    elif parmtype == 7:
        siga = math.log(2.)
        #siga = math.log(10.)
        #siga = 100.
        #siga = 1e-6
        #grad[iparm] += -math.log(1.1)/siga**2
        
        
    #if parmtype < 6:
    #if parmtype != 7:
    #if parmtype != 6:
    #if parmtype in [6,8]:
    #if parmtype not in [6,7] or subdet==5:
    #if parmtype not in [6,7]:
    #if parmtype == 8:
    #if parmtype in [6,7]:
    if False:
    #if parmtype == 1 and subdet > 1:
    #if parmtype == 1:
    #if parmtype in [1,6,7]:
    #if parmtype not in [7]:
    #if parmtype in [1, 6]:
    #if parmtype != 7:
    #if parmtype == 2 and subdet != 5:
    #if parmtype in [1, 5]:
        grad[iparm] = 0.
        hess[iparm,:] *= 0.
        hess[:,iparm] *= 0.
        
    grad[iparm] += oldcors[iparm]/siga**2
    hess[iparm, iparm] += 1./siga**2

#nalign = 0
#for parm in parmlist:
  #parmtype, subdet, layer, ieta = parm
  #if parmtype<3:
    #nalign+=1
    
#grad[nalign:] = 0.
#hess[nalign:] = 0.
#hess[:,nalign:] = 0.
#hess[nalign:,nalign:] = np.eye(nglobal-nalign)

#hessbak = hess
#gradbak = grad

#assert(0)

#print(np.linalg.eigvalsh(hess))


#v0 = v[:,0]

#print(e)
#assert(0)

#e,v = np.linalg.eigh(hess)
#e = np.linalg.eigvalsh(hess)
#print("hess eiganvalues")
#print(e)

#badidxs = []
#for i in range(nglobal):
  #if e[i]<0.:
    #idx = np.argmax(np.abs(v[:,i]))
    #grad[idx] = 0.
    #hess[idx] *= 0.
    #hess[:,idx] *= 0.
    #hess[idx,idx] = 2.
    #badidxs.append(idx)

#bconstraint = 0.01*3.8
#Jb = np.zeros((nglobal,1), dtype=np.float64)

#for iparm,parm in enumerate(parmlist):
  #parmtype, subdet, layer, ieta = parm
  #if parmtype==2:
    #Jb[iparm] = 1.

#hess += 2.*Jb*Jb.transpose()/bconstraint**2

#e,v = np.linalg.eigh(hess)
#print(e)


    
#print("making sparse")
#hesssparse = scipy.sparse.csc_matrix(hess)
#print("solving")
#xout = scipy.sparse.linalg.spsolve(hesssparse, -grad)

#xout[1:] = 0.05*3.8

#xout = scipy.sparse.linalg.spsolve(hess, -grad)
#xout = scipy.sparse.linalg.gmres(hess, -grad)
#xout = scipy.linalg.cho_solve(scipy.linalg.cho_factor(hess), -grad)
#xout = np.linalg.solve(hess,-grad)

#xout, residuals, rank, s = np.linalg.lstsq(hess, -grad, rcond=1e-8)
#xout, residuals, rank, s = np.linalg.lstsq(hess, -grad, rcond=5e-11)
#xout, residuals, rank, s = np.linalg.lstsq(hess, -grad, rcond=5e-6)
#xout, residuals, rank, s = np.linalg.lstsq(hess, -grad, rcond=None)

#rcond = 2./bconstraint**2/s[0]
#print("rcond", rcond)
#rcond = 1./e[-1]
#xout, residuals, rank, s = np.linalg.lstsq(hess, -grad, rcond=rcond)

##xout, residuals, rank, s = np.linalg.lstsq(hess, -grad, rcond=1e-14)
#print("rank", rank)
#print("singular values:")
#print(s)

#print("eigen decomposition")
#w, v = scipy.linalg.eigh(hess.transpose(), overwrite_a = True, subset_by_index = [0,4], driver = "evx")
#w, v = scipy.linalg.eigh(hess.transpose(), overwrite_a = True, subset_by_index = [0,4])

#print(w)


#assert(0)

print("decomposing")
#chol = scipy.linalg.cho_factor(hess, lower=True)

# transpose here is needed for the decomposition to occur in-place which speeds things up and saves a factor of 2 in memory
lower = True

chol = scipy.linalg.cho_factor(hess.transpose(), lower=lower, overwrite_a = True)
#chol = scipy.linalg.cho_factor(hess.transpose(), lower=lower, overwrite_a = False)


#chol = scipy.linalg.cholesky(hess.transpose(), lower=lower, overwrite_a = True)
#chol = scipy.linalg.cholesky(hess.transpose(), lower=lower, overwrite_a = False)
#chol =  scipy.linalg.lapack.dpotrf(hess.transpose(), lower=lower, overwrite_a = False)[0]
#chol = (chol, lower)

#print(chol[0])

print("solving")
xout = scipy.linalg.cho_solve(chol, -grad)

print("done solve")

#for iparm,parm in enumerate(parmlist):
    #parmtype, subdet, layer, ieta = parm
    #if parmtype == 2:
        #xout[iparm] = 0.


#print("computing inverse choleskey")
#cholinv = scipy.linalg.solve_triangular(chol[0], ident, lower=True)

#print("compute inverse")

#cholinv = scipy.linalg.lapack.dtrtri(chol[0], lower=lower, unitdiag = False, overwrite_c=True)[0]
#cholinv = scipy.linalg.lapack.dtrtri(chol[0], lower=chol[1], overwrite_c=True)[0]

#cov = scipy.linalg.lapack.dpotri(chol[0], lower = chol[1], overwrite_c = True)[0]
#cov = chol[0]

#print(cov)

#print(cholinv)

#cov = cholinv.transpose()@cholinv
#print(chol[0])
#print(cholinv)

#for i in range(testsize):
    #cov[i, i:] = cov[i:,i]

##ident = cholinv.transpose()*chol[0]
#ident = cov@hess
#print(ident)

#ident = np.identity(testsize, dtype = np.float64)
#cov = scipy.linalg.cho_solve(chol, ident, overwrite_b = True)
#cov = scipy.linalg.cho_solve(chol, ident.transpose(), overwrite_b = True)
#ident = None
#errs = np.sqrt(np.diag(cov))
#cov = None
errs = np.zeros_like(np.diag(hess))

print("done compute inverse")


#print("chol")
#print(chol)
#print("cholinv")
#print(cholinv)

doreplicas = False


print("compute toys")

ntoys = 100

if doreplicas:
    u = np.random.standard_normal((testsize,ntoys))

    print("xout.shape", xout.shape)

    #xtoys = xout + cholinv @ u
    xtoys = xout + scipy.linalg.solve_triangular(chol[0], u, lower=True)
    #xtoys = cholinv @ u
    #print("xtest")
    #print(cholinv.shape)
    print(xtoys.shape)
    #print(xtest)

print("done compute toys")



#for iparm,parm in enumerate(parmlist):
  #parmtype, subdet, layer, ieta = parm
  ##if parmtype < 2:
    ##hess[iparm, iparm] += 2.*1./1e-2**2
  #if parmtype == 2:
    #hess[iparm, iparm] += 2.*1./bconstraint**2
  ##if parmtype == 3:
    ##hess[iparm, iparm] += 2.*1./1e-4**2

#cov = np.linalg.inv(hess)
#errs = np.sqrt(2.*np.diag(cov))

#cov = hess
#errs = np.sqrt(2./np.diag(cov))




#write output file
#fout = ROOT.TFile.Open("correctionResults.root", "RECREATE")
fout = ROOT.TFile.Open("correctionResults.root", "RECREATE")
#fout = ROOT.TFile.Open("correctionResultsdebug.root", "RECREATE")


print("first loop")

idxmaptree = ROOT.TTree("idxmaptree","")
idx = np.empty((1), dtype=np.uint32)
idxmaptree.Branch("idx", idx, "idx/i")
for i in range(testsize):
    idx[0] = idxmap[i]
    idxmaptree.Fill()
    
idxmaptree.Write()

parmtree = ROOT.TTree("parmtree","")
x = np.empty((1), dtype=np.float32)
err = np.empty((1), dtype=np.float32)


xreplicas = np.empty((ntoys), dtype=np.float32)

print("second loop")

parmtree.Branch("x", x, "x/F")
parmtree.Branch("err", err, "err/F")
if doreplicas:
    parmtree.Branch("xreplicas", xreplicas , f"xreplicas[{ntoys}]/F")

for i in range(nglobal):
    #x[0] = xout[i]
    x[0] = xout[i] + oldcors[i]
    err[0] = errs[i]
    if doreplicas:
        xreplicas[...] = xtoys[i]
    parmtree.Fill()
    
parmtree.Write()
fout.Close()



#np.set_printoptions(threshold=sys.maxsize)



#print(v[:7])

#print(xout)
#ldlt = scipy.linalg.ldl(a, overwrite_a=True, check_finite=False)
#cho = scipy.linalg.cho_factor(hess, overwrite_a=True, check_finite=False)
#xout = scipy.linalg.cho_solve(cho, -grad, overwrite_b=True, check_finite=False)
#xout = scipy.linalg.lstsq(hess,-grad)

#print(grad)
#print(hess)





