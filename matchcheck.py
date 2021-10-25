import ROOT
#ROOT.gROOT.SetBatch(False)
ROOT.ROOT.EnableImplicitMT()


#indata = "/data/shared/muoncal/MuonGunUL2016_v84_GenJpsiPhotosSingle_quality/210707_164413/0000/globalcor_*.root"
indata = "/data/shared/muoncal/MuonGunUL2016_v84_GenJpsiPhotosSingle_quality/210707_164413/0000/globalcor_0_2.root"

treename = "tree"
d = ROOT.ROOT.RDataFrame(treename,indata)

d = d.Filter("genPt > 3.5")

d = d.Define("nunmatched", "Sum(dxsimgen == -99.)")

h = d.Histo1D(("h","",101,-0.5, 100.5), "nunmatched")

h.Draw()
