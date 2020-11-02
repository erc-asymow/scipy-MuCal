import ROOT

filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprecrefb///globalcor_0.root"

filenameres = "correctionResultsModule.root"


finfo = ROOT.TFile.Open(filenameinfo)
fres = ROOT.TFile.Open(filenameres)

infotree = finfo.runtree
parmtree = fres.parmtree

#for info in infotree:
  #print(info.layer)
  
#for parm in parmtree:
  #print(parm.x)

for info,parm in zip(infotree, parmtree):
  if info.parmtype==0 and info.subdet==1 and abs(info.layer)==1:
    print(parm.x)
  
