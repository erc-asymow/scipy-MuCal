import ROOT
import numpy as np

#print(ROOT.gSystem.GetMakeSharedLib())

flags = ROOT.gSystem.GetMakeSharedLib().replace("-march=x86-64 -mtune=generic -O2", "-march=native -O3")
ROOT.gSystem.SetMakeSharedLib(flags)

#print(ROOT.gSystem.GetMakeSharedLib())


#assert(0)

ROOT.ROOT.EnableImplicitMT();

ROOT.gInterpreter.ProcessLine(".L fillhelpers.c+")

treename = "tree"
#filename = "/data/bendavid/muoncaldata/test3/testlz4large.root"
#filename = "/data/bendavid/muoncaldata/largetest/*.root"
filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root"

filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGradsParmInfo.root"
finfo = ROOT.TFile.Open(filenameinfo)
runtree = finfo.tree
nparms = int(runtree.GetEntries())
print(nparms)


gradhelper = ROOT.GradHelper(nparms)
hesshelper = ROOT.HessHelper(nparms)

d = ROOT.ROOT.RDataFrame(treename, filename)

d = d.Filter("trackPt>5.5")

dgrad = ROOT.BookHelper(ROOT.RDF.AsRNode(d), gradhelper, "gradv", "globalidxv")
dhess = ROOT.BookHelper(ROOT.RDF.AsRNode(d), hesshelper, "hesspackedv", "globalidxv")

gradval = dgrad.GetValue()
grad = np.frombuffer(gradval.data(), dtype=np.float64, count=nparms)

grad = np.reshape(grad, (nparms,1))

hessval = dhess.GetValue()
smallsize = 16*nparms
hess = np.empty((smallsize),dtype=np.float64)
for i in range(smallsize):
  hess[i] = hessval[i]
#hess = np.array(hessval[:16*nparms])
#hessvec = ROOT.ConvertHessResult(hessval)
#hess = np.frombuffer(hessvec.data(), dtype=np.float64, count=nparms*nparms)
#hess = np.frombuffer(hessvec.data(), dtype=np.float64, count=16*nparms)

#hess = np.reshape(hess, (nparms,nparms))
hess = np.reshape(hess, (16,nparms))

outfile = "combinedgrads.npz"
#np.savez_compressed(outfile, grad = grad, hess=hess)
np.savez(outfile, grad = grad, hess=hess)
