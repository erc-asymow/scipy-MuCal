import ROOT
import numpy as np

filenamegradspacked = "combinedgradsdebugpacked.root"
filenamegradsfull = "combinedgradsdebugfull.root"

fpacked = ROOT.TFile.Open(filenamegradspacked)
ffull = ROOT.TFile.Open(filenamegradsfull)

gradtreepacked = fpacked.tree
gradtreefull = ffull.tree

filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v66_Gen_quality/210509_200135/0000/globalcor_0_1.root"
finfo = ROOT.TFile.Open(filenameinfo)
runtree = finfo.runtree
nparmsfull = np.int64(runtree.GetEntries())

##hesspacked = np.zeros((nparmsfull, nparmsfull), dtype=np.float64)
#hessfull = np.zeros((nparmsfull, nparmsfull), dtype=np.float64)

diffv = np.zeros((nparmsfull,))

for i,(packed,full) in enumerate(zip(gradtreepacked, gradtreefull)):
    if (i%1000 == 0):
        print(i)
    packedrow = np.asarray(packed.hessrow)
    fullrow = np.asarray(full.hessrow)
    
    #diff = packedrow - fullrow
    #diff = packedrow[i:] - fullrow[i:]
    diff = (packedrow[i:] - fullrow[i:])/np.where(np.equal(fullrow[i:],0.), 1., fullrow[i:])
    diffv[i] = np.max(np.abs(diff))
    
    #if i==49196:
    if i==58476:
        idxmax = np.argmax(np.abs(diff))
        print(diff[idxmax], packedrow[i+idxmax], fullrow[i+idxmax])
    
    #hesspacked[i] = packed.hessrow
    #hessfull[i] = full.hessrow
    #print(packedrow.shape)
    #print(packedrow)
    #print(fullrow)
    #diff = packedrow[i:] - fullrow[i:]
    #diff = packedrow - fullrow
    #print(diff)
    #print(np.max(np.abs(diff)))

print(np.argmax(diffv), np.max(diffv))
#print(np.max(np.abs(hesspacked - hessfull)))
