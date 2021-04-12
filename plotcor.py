import ROOT
ROOT.gROOT.SetBatch(False)

#finfo = ROOT.TFile.Open("root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v27_Gen/201205_145326/0000/globalcor_0_1.root")
finfo = ROOT.TFile.Open("root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v32_Rec/210214_182643/0000/globalcor_0_1.root")

#fcor = ROOT.TFile.Open("correctionResults_align0.root")
#fcor = ROOT.TFile.Open("results_v27_aligdigi_01p67/correctionResults.root")
#fcor = ROOT.TFile.Open("results0123457/correctionResults.root")
#fcor = ROOT.TFile.Open("correctionResults.root")
fcor = ROOT.TFile.Open("results_V33_01p67quality/correctionResults.root")

#fgrads = ROOT.TFile.Open("results0123457/combinedgrads.root")
#fgrads = ROOT.TFile.Open("combinedgrads.root")

runtree = finfo.Get("runtree")
    
#runtree.Print()    
    
parmtree = fcor.Get("parmtree")
#gradtree = fgrads.Get("tree")

parmtree.AddFriend(runtree)
#parmtree.AddFriend(gradtree,"gradtree")

#c=ROOT.TCanvas()
#parmtree.Draw("x","parmtype==2 && subdet>3")
#parmtree.Draw("x","parmtype==2 && !(subdet==0 || subdet==2 || subdet==3)")
#parmtree.Draw("x","parmtype==2 && (subdet==1)")

#parmtree.Draw("-gradelem/hessrow[idx]","parmtype==2 && subdet>3 && abs(gradelem/hessrow[idx])>1.")
#parmtree.Draw("hessrow[43913]")
#parmtree.Draw("hessrow[43913]/sqrt(345330.46*hessrow[idx])")
#parmtree.Draw("gradelem:hessrow[idx]","parmtype==2 && subdet>3 && stereo==0")
#parmtree.Scan("idx:gradelem:hessrow[idx]","parmtype==2 && subdet>3 && abs(gradelem/hessrow[idx])>1.")
#parmtree.Draw("subdet","abs(x)>0.1 && parmtype==2")
#parmtree.Draw("x:runtree.z","parmtype==0")
#parmtree.Draw("x","parmtype==0 && subdet==5 && abs(x)<0.01")
#parmtree.Draw("x:runtree.z","parmtype==6 && subdet<5")
#parmtree.Draw("x:runtree.eta","parmtype==6 && subdet==1")
parmtree.Draw("x","runtree.z < 0. && parmtype==6 && subdet==5 && abs(x)<0.05")
#parmtree.Draw("x","parmtype==2")
#parmtree.Draw("eta","abs(x)>0.005 && parmtype==0")
#parmtree.Draw("subdet","abs(x)>1. && parmtype==2")

#input("wait")
