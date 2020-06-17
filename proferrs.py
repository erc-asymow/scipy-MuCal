import ROOT

#f = ROOT.TFile.Open("/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muonGuntree.root")
f = ROOT.TFile.Open("/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/ZJToMuMu_mWPilot.root")
tree = f.Get("tree")

ptmin = 12.

prof = ROOT.TProfile("prof","",100,ptmin, 150.)
#prof = ROOT.TProfile("prof","",100,1./150.,1./ptmin)


#tree.Draw("tkptErr*tkptErr:1./gen_curv>>prof",f"1./gen_curv>{ptmin} && gen_eta>-2.4 && gen_eta<-2.3 && reco_charge>-2. && tkptErr*tkptErr<0.2","goff")

#tree.Draw("tkptErr*tkptErr:1./reco_curv>>prof",f"1./reco_curv>{ptmin} && gen_eta>-2.4 && gen_eta<-2.3 && reco_charge>-2. && tkptErr*tkptErr<0.2","goff")

#tree.Draw("tkptErr*tkptErr*reco_curv*reco_curv:reco_curv>>prof",f"1./reco_curv>{ptmin} && gen_eta>-2.4 && gen_eta<-2.3 && reco_charge>-2. && tkptErr*tkptErr<0.2","goff")

#tree.Draw("tkptErr*tkptErr*gen_curv*gen_curv:reco_curv>>prof",f"1./reco_curv>{ptmin} && gen_eta>-0.3 && gen_eta<-0.2 && reco_charge>-2. && tkptErr*tkptErr*gen_curv*gen_curv<3e-6","goff")

tree.Draw("cov1_03:pt1>>prof",f"pt1>{ptmin} && eta1>-2.4 && eta1<-2.3 && cov1_03<0. && cov1_03>-1e-5","goff")
#tree.Draw("cov1_00*pt1*pt1:pt1>>prof",f"pt1>{ptmin} && eta1>-2.4 && eta1<-2.3","goff")
#tree.Draw("cov1_03:pt1>>prof",f"pt1>{ptmin} && cov1_03>-0.8e-6  && eta1>-0.3 && eta1<-0.2","goff")

c = ROOT.TCanvas()
prof.Draw()

input("wait")
