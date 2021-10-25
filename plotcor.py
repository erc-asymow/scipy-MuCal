import ROOT
ROOT.gROOT.SetBatch(False)

#finfo = ROOT.TFile.Open("root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v27_Gen/201205_145326/0000/globalcor_0_1.root")
#finfo = ROOT.TFile.Open("/data/shared/muoncal/MuonGunUL2016_v63_Rec_quality_bs/210507_003108/0000/globalcor_0_1.root")
#finfo = ROOT.TFile.Open("/data/shared/muoncal/MuonGunUL2016_v107_GenJpsiPhotosSingle_quality/210719_142518/0000/globalcor_0_1.root")
#filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_idealquality_constraint/210911_050035/0000/globalcor_0_1.root"
#filenameinfo = "testrecinfo.root"
#filenameinfo = "info.root"
#filenameinfo = "infofullbz.root"
##filenameinfo = "info_v201.root"
filenameinfo = "info_v202.root"
#filenameinfo = "infofullbzdisky.root"
#filenameinfo = "infofullbzr.root"

finfo = ROOT.TFile.Open(filenameinfo)

#finfo = ROOT.TFile.Open("/data/shared/muoncal/MuonGunUL2016_v124_Gen_quality/210729_121830/0000/globalcor_0_1.root")



#fcor = ROOT.TFile.Open("correctionResults_align0.root")
#fcor = ROOT.TFile.Open("results_v27_aligdigi_01p67/correctionResults.root")
#fcor = ROOT.TFile.Open("results0123457/correctionResults.root")
#fcor = ROOT.TFile.Open("correctionResults_iter0.root")

fcor = ROOT.TFile.Open("correctionResults.root")
#fcor = ROOT.TFile.Open("correctionResults_v206_data_nobsub.root")
#fcor = ROOT.TFile.Open("correctionResults_v186_jpsi_biasm10_biasfield.root")
#fcor = ROOT.TFile.Open("correctionResults_v186_jpsi_biasm10_biasfield_bz.root")


#fcor = ROOT.TFile.Open("/data/home/bendavid/muoncal/scipy-MuCal/plotscale_v171_quality_biasm10_biasfield_constraintnofsr_iter0/correctionResults_v171_quality_biasm10_biasfield_constraintnofsr_iter0.root")


#fcor = ROOT.TFile.Open("/data/home/bendavid/muoncal/scipy-MuCal/plotscale_v171_quality_biasm10_biasfield_constraintfsr28_iter0/correctionResults_v171_quality_biasm10_biasfield_constraintfsr28_iter0.root")
#fcor = ROOT.TFile.Open("correctionResultsdebug.root")
#fcor = ROOT.TFile.Open("correctionResults_v107_gunfullpt.root")

#fcor = ROOT.TFile.Open("plotscale_v124_corgunfullpt/correctionResults.root")
#fcor = ROOT.TFile.Open("correctionResults_reference.root")
#fcor = ROOT.TFile.Open("results_v66_gen/correctionResults.root")

fcor2 = ROOT.TFile.Open("correctionResults_v191_jpsi.root")
#fcor2 = ROOT.TFile.Open("/data/home/bendavid/muoncal/scipy-MuCal/plotscale_v171_quality_biasm10_biasfield_constraintfsr28_iter0/correctionResults_v171_quality_biasm10_biasfield_constraintfsr28_iter0.root")
#fcor = ROOT.TFile.Open("correctionResultsdebug.root")


fgrad = ROOT.TFile.Open("combinedgrads.root")
#fgrad = ROOT.TFile.Open("plotscale_v124_corgunfullpt/combinedgrads.root")

#fcor = ROOT.TFile.Open("correctionResultsgen.root")
#fcor = ROOT.TFile.Open("results_V33_01p67quality/correctionResults.root")

#fgrads = ROOT.TFile.Open("results0123457/combinedgrads.root")
#fgrads = ROOT.TFile.Open("combinedgrads.root")

runtree = finfo.Get("runtree")
gradtree = fgrad.Get("tree")
    
#runtree.Print()    
    
parmtree = fcor.Get("parmtree")
parmtree2 = fcor2.Get("parmtree")

#gradtree = fgrads.Get("tree")

parmtree.AddFriend(runtree)
parmtree.AddFriend(parmtree2, "parmtree2")
#parmtree.AddFriend(gradtree,"gradtree")
parmtree.AddFriend(gradtree,"gradtree")

#bfact = 2.99792458e-3


#c=ROOT.TCanvas()
#parmtree.Draw("x","parmtype==2 && subdet>3")
#parmtree.Draw("x","parmtype==2 && !(subdet==0 || subdet==2 || subdet==3)")
#parmtree.Draw("x","parmtype==2 && (subdet==1)")

#parmtree.Draw("x","parmtype==7")
#parmtree.Draw("x","parmtype==7 && abs(x)<0.05")
#parmtree.Draw("x","parmtype==6 && abs(x)<0.5")

#parmtree.Draw("-gradelem/hessrow[idx]","parmtype==2 && subdet>3 && abs(gradelem/hessrow[idx])>1.")
#parmtree.Draw("hessrow[43913]")
#parmtree.Draw("hessrow[43913]/sqrt(345330.46*hessrow[idx])")
#parmtree.Draw("gradelem:hessrow[idx]","parmtype==2 && subdet>3 && stereo==0")
#parmtree.Scan("idx:gradelem:hessrow[idx]","parmtype==2 && subdet>3 && abs(gradelem/hessrow[idx])>1.")
#parmtree.Draw("subdet","abs(x)>0.1 && parmtype==2")
#parmtree.Draw("x:runtree.z","parmtype==7")

#hprof = ROOT.TProfile("hprof","", 24, -2.4, 2.4)
#hprof = ROOT.TProfile("hprof","", 24, -300., 300.)
#hprof = ROOT.TProfile("hprof","", 24, 0., 120.)
#hprof = ROOT.TProfile("hprof","", 24, 0.5, 9.5)

#parmtree.Draw("parmtree2.x-x:z >> hprof", "parmtype==6", "goff")
#parmtree.Draw("parmtree2.x-x:layer >> hprof", "parmtype==6 && subdet==2", "goff")
#parmtree.Draw("parmtree2.x:layer >> hprof", "parmtype==6 && subdet==2", "goff")
#parmtree.Draw("x/xiavg","xiavg > 0.")
#parmtree.Draw("err","xiavg == 0. && err!=1.")
#parmtree.Draw("x:z >> hprof", "parmtype==6", "goff")
#parmtree.Draw("x/xi", "parmtype==7")
#parmtree.Draw("2./err", "parmtype==0 && subdet<2 && 2./err<1e-4")
#parmtree.Draw("x", "parmtype==6 && subdet==5")
#parmtree.Draw("x", "parmtype==6")
#parmtree.Draw("x:z", "parmtype==6")
#parmtree.Draw("runtree.bz - 3.8:z", "parmtype==6")
#parmtree.Draw("parmtree2.x - (runtree.bz-3.8):z", "parmtype==6")
#parmtree.Draw("3.8 + x - runtree.bz:runtree.eta", "parmtype==6")
#parmtree.Draw("x:runtree.z","parmtype==6")
#parmtree.Draw("3.8 - runtree.bz:runtree.eta", "parmtype==6")

#parmtree.Draw("3.8 - runtree.bz", "parmtype==6")
#parmtree.Draw("3.8 + x - runtree.bz", "parmtype==6")



#parmtree.Draw("x - parmtree2.x:runtree.z", "parmtype==6")
#parmtree.Draw("3.8 + x - runtree.bz:runtree.z", "parmtype==8")
#parmtree.Draw("x:runtree.z", "parmtype==6")
#parmtree.Draw("x:runtree.bx", "parmtype==6")
#parmtree.Draw("x:runtree.by", "parmtype==7")
#parmtree.Draw("x:3.8-runtree.bz", "parmtype==8")
##parmtree.Draw("sqrt(runtree.bx*runtree.bx + runtree.by*runtree.by):runtree.z", "parmtype==9")
#parmtree.Draw("runtree.bz:runtree.z", "parmtype==9")
#parmtree.Draw(f"3.8 + x/{bfact} - runtree.bz:runtree.z", "parmtype==8")
#parmtree.Draw(f"x/{bfact}:runtree.z", "parmtype==8")
#parmtree.Draw("sqrt(runtree.bx*runtree.bx + runtree.by*runtree.by):runtree.phi","parmtype==6")
#parmtree.Draw("runtree.by:runtree.phi","parmtype==6 && abs(runtree.z)>200.")
#parmtree.Draw("3.8 + x - runtree.bz:3.8 - runtree.bz", "parmtype==6")
#parmtree.Draw("3.8 - runtree.bz:runtree.z", "parmtype==6")
#parmtree.Draw("runtree.bz:runtree.z", "parmtype==6")
#parmtree.Draw("sqrt(runtree.bz*runtree.bz + runtree.bx*runtree.bx + runtree.by*runtree.by):runtree.z", "parmtype==6")

#parmtree.Draw("b0trivial + x*b0trivial/3.8 - b0:runtree.z", "parmtype==8")
#parmtree.Draw("(b0trivial + x*b0trivial/3.8 - b0)/b0:runtree.z", "parmtype==8")
#parmtree.Draw("(b0trivial + x*b0trivial/3.8 - b0)/b0", "parmtype==8")
#parmtree.Draw("x:runtree.layer","parmtype==7 && subdet==4")

#parmtree.Draw("x:runtree.phi","parmtype==6 && subdet==4")
parmtree.Draw("x:runtree.z","parmtype==6")

#prof = ROOT.TProfile("prof","",50, -300., 300., "s")
#parmtree.Draw("x:runtree.z>>prof","parmtype==6","goff")

#c = ROOT.TCanvas() 
#prof.Draw()

#parmtree.Draw("x","parmtype==7 && abs(runtree.z)>250.")
#parmtree.Draw("exp(x)","parmtype==9 && subdet==1")
#parmtree.Draw("x - parmtree2.x:runtree.z", "parmtype==6")


#parmtree.Draw("b0/b0trivial", "parmtype==6")
#parmtree.Draw("b0trivial - b0:z", "parmtype==8")
#parmtree.Draw("x:runtree.bz - 3.8", "parmtype==8")
#parmtree.Draw("x:runtree.bx", "parmtype==6")
#parmtree.Draw("x:runtree.by", "parmtype==7")
#parmtree.Draw("runtree.bz-3.8:runtree.z", "parmtype==7")

#parmtree.Draw("runtree.bz:runtree.z", "parmtype==8")
#parmtree.Draw("x-parmtree2.x", "parmtype==8")
#parmtree.Draw("x", "parmtype==8")

#parmtree.Draw("x/(runtree.bz - 3.8):z", "parmtype==6")
#parmtree.Draw("x/(runtree.bz - 3.8):runtree.bz - 3.8", "parmtype==6")
#parmtree.Draw("x/(runtree.bz - 3.8)", "parmtype==6 && abs(x/(runtree.bz - 3.8))<5.")
#parmtree.Draw("3.8 - runtree.bz:z", "parmtype==6")
#parmtree.Draw("(runtree.bz-3.8):z", "parmtype==6")
#parmtree.Draw("parmtree2.x - (runtree.bz-3.8):z", "parmtype==6")
#parmtree.Draw("x - (runtree.bz-3.8):z", "parmtype==6")
#parmtree.Draw("x - parmtree2.x:z", "parmtype==6")
#parmtree.Draw("x - parmtree2.x:z", "parmtype==7")
#parmtree.Draw("(runtree.bz-3.8):z", "parmtype==6")
#parmtree.Draw("x-parmtree2.x", "parmtype==6 && subdet==5")
#parmtree.Draw("x", "parmtype==6")
#parmtree.Draw("x", "parmtype==7 && subdet!=5 && abs(x)<1.0")
#parmtree.Draw("x", "parmtype==7 && subdet==5")

#hb = ROOT.TH2D("hb","", 100, -300., 300, 100, 0., 120.) 
#hb = ROOT.TProfile2D("hb","", 100, -300., 300, 100, 0., 120.) 

#runtree.Draw("bz:rho:z >> hb","(parmtype==6)","goff")

#hb.Draw("COLZ")


#hprof.Draw()

#parmtree.Draw("parmtree2.x-x", "parmtype==6")
#parmtree.Draw("x", "parmtype==6")

#parmtree.Draw("x","parmtype==6 && subdet<2")
#parmtree.Draw("x","parmtype==6 && subdet>2")
#parmtree.Draw("x:runtree.z","parmtype==6 && subdet>2")
#parmtree.Draw("runtree.rho:runtree.z","parmtype==6 && abs(x)>0.001")

#parmtree.Draw("x/runtree.xi","parmtype==7 && subdet==0 && layer==3")
#parmtree.Draw("x","parmtype==0 && subdet==3 && abs(x)<1e-3")
#parmtree.Draw("x-parmtree2.x","parmtype==0 && subdet==2")
#parmtree.Draw("x","parmtype==0 && subdet==2")
#parmtree.Draw("parmtree2.x","parmtype==0 && subdet==2")
#parmtree.Draw("gradtree.hessrow[gradtree.idx]","parmtype==0")

#parmtree.Draw("x","parmtype==6")
#parmtree.Draw("x","parmtype==6 && abs(x)<0.1")
#parmtree.Draw("x","parmtype==6 && subdet==0")
#parmtree.Draw("x","parmtype==0 && subdet==5 && abs(x)<0.01")
#parmtree.Draw("x:runtree.z","parmtype==6 && subdet<5")
#parmtree.Draw("x:runtree.eta","parmtype==6 && subdet==1")
#parmtree.Draw("x","runtree.z < 0. && parmtype==6 && subdet==5 && abs(x)<0.05")
#parmtree.Draw("x","parmtype==2")
#parmtree.Draw("eta","abs(x)>0.005 && parmtype==0")
#parmtree.Draw("subdet","abs(x)>1. && parmtype==2")

#input("wait")

