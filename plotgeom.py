import ROOT
ROOT.gROOT.SetBatch(False)

finfo = ROOT.TFile.Open("/data/bendavid/muoncal/globalcor_design.root")
runtree = finfo.Get("runtree")

runtree.Print()

finfonom = ROOT.TFile.Open("/data/bendavid/muoncal/globalcor_nom.root")
runtreenom = finfonom.Get("runtree")

finfonomdigi = ROOT.TFile.Open("/data/bendavid/muoncal/globalcor_nomdigi.root")
runtreenomdigi = finfonomdigi.Get("runtree")

runtree.AddFriend(runtreenom, "nom")
runtree.AddFriend(runtreenomdigi, "nomdigi")
    
#runtree.Draw("x - nom.rho","subdet==0 || subdet==2 || subdet==3")
#runtree.Draw("z - nom.z","subdet==1 || subdet==4 || subdet==5")
runtree.Draw("nom.rho - nomdigi.rho","iidx==8")
#runtree.Draw("phi-nom.phi","abs(phi-nom.phi)<1e-4 && subdet==5 && z>0")
    
#runtree.Draw("x.nom:y.nom","subdet==0")
    
#runtree.Print()    
input("wait")
