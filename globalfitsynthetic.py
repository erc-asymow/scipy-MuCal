import ROOT
import numpy as np
import math

filenameinfo = "infofullbz.root"
finfo = ROOT.TFile.Open(filenameinfo)

runtree = finfo.runtree

nparmsfull = np.int64(runtree.GetEntries())
nglobal = nparmsfull
testsize = nparmsfull

idxmap = np.arange(nglobal)


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


#xreplicas = np.empty((ntoys), dtype=np.float32)

print("second loop")

parmtree.Branch("x", x, "x/F")
parmtree.Branch("err", err, "err/F")
#if doreplicas:
    #parmtree.Branch("xreplicas", xreplicas , f"xreplicas[{ntoys}]/F")

for i in range(nglobal):
    runtree.GetEntry(i)
    #x[0] = xout[i]
    parmtype = runtree.parmtype
    subdet = runtree.subdet
    
    
    #parmtype = parmlistfull[i][0]
    #subdet = parmlistfull[i][1]
    #bz = parmlistfull[i][-1]
    
    #if parmtype == 0:
        #x[0] = -runtree.dx
    #elif parmtype == 1:
        #x[0] = -runtree.dy
    #elif parmtype == 5:
        #x[0] = -runtree.dtheta
    #elif parmtype == 6:
    #if parmtype == 6:
        ##x[0] = runtree.bx
        #x[0] = 0.
    #elif parmtype == 7:
        ##x[0] = runtree.by
        #x[0] = 0.
    #elif parmtype == 8:
        #x[0] = (3.8/runtree.b0trivial)*(runtree.b0-runtree.b0trivial)
    #elif parmtype == 9:
        #x[0] = math.log(1.1)
        
    if parmtype == 7:
        x[0] = math.log(1.1)
    else:
        x[0] = 0.
    #x[0] = xout[i] + oldcors[i]
    err[0] = 0.
    #if doreplicas:
        #xreplicas[...] = xtoys[i]
    parmtree.Fill()
    
parmtree.Write()
fout.Close()
