import numpy as np
import scipy.linalg
import ROOT
import sys
import concurrent.futures


np.set_printoptions(threshold=sys.maxsize)


filenamegrads = "combinedgrads.root"
#filenamegrads = "combinedgradsrec.root"
#filenamegrads = "/data/bendavid/muoncaldata/combinedgrads.root"
fgrads = ROOT.TFile.Open(filenamegrads)
gradtree = fgrads.tree

nparmsfull = np.int64(gradtree.GetEntries())
print(nparmsfull)

gradfull = np.zeros((nparmsfull,1), dtype=np.float64)
hessfull = np.zeros((nparmsfull, nparmsfull), dtype=np.float64)

print("loading grads")
for i, entry in enumerate(gradtree):
  #break
  if (i%1000 == 0):
    print(i)
  gradfull[i,0] = entry.gradelem
  hessfull[i] = entry.hessrow
  
print("filling in lower triangular part of hessian")
def filllower(i):
  hessfull[i,:i] = hessfull[:i,i]
  
with concurrent.futures.ThreadPoolExecutor(64) as e:
  results = e.map(filllower, range(hessfull.shape[0]))


#print("computing eigenvalues")
##eigvals = np.linalg.eigvalsh(hessfull)
#eigvalssmall = scipy.linalg.eigh(hessfull, eigvals_only=True, subset_by_index = [0, 10], driver="evx")

#print("computing eigenvalues large")
#eigvalslarge = scipy.linalg.eigh(hessfull, eigvals_only=True, subset_by_index = [nparmsfull-10, nparmsfull-1], driver="evx")

#nparmsfull = 100
#gradfull = gradfull[:nparmsfull]
#hessfull = hessfull[:nparmsfull,:nparmsfull]


#print(eigvalssmall)
#print(eigvalslarge)

print("doing lstsq solve")

xout, residuals, rank, s = np.linalg.lstsq(hessfull, -gradfull, rcond=None)

errs = np.zeros_like(xout)

print("rank:", rank)
print("s:")
print(s)

fout = ROOT.TFile.Open("correctionResults.root", "RECREATE")
#fout = ROOT.TFile.Open("correctionResults2016.root", "RECREATE")

idxmaptree = ROOT.TTree("idxmaptree","")
idx = np.empty((1), dtype=np.uint32)
idxmaptree.Branch("idx", idx, "idx/i")
for i in range(nparmsfull):
    idx[0] = i
    idxmaptree.Fill()
    
idxmaptree.Write()

parmtree = ROOT.TTree("parmtree","")
x = np.empty((1), dtype=np.float32)
err = np.empty((1), dtype=np.float32)

parmtree.Branch("x", x, "x/F")
parmtree.Branch("err", err, "err/F")

for i in range(nparmsfull):
    x[0] = xout[i]
    err[0] = errs[i]
    parmtree.Fill()
    
parmtree.Write()
fout.Close()
