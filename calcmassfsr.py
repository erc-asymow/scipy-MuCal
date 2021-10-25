import ROOT

ROOT.gStyle.SetOptStat(111111)

ROOT.ROOT.EnableImplicitMT()


chain = ROOT.TChain("tree")


ROOT.gInterpreter.Declare("""
float deltaphi(float phi0, float phi1) {
    float dphi = phi1 - phi0;
    if (dphi > M_PI) {
        dphi -= 2.0*M_PI;
    }
    else if (dphi <= - M_PI) {
      dphi += 2.0*M_PI;
    }
    return dphi;

}
    
""")


ROOT.gInterpreter.Declare("""
    double alpharatio(float Muplus_pt, float Muplus_eta, float Muplus_phi, float Muminus_pt, float Muminus_eta, float Muminus_phi, float Muplusgen_pt, float Muplusgen_eta, float Muplusgen_phi, float Muminusgen_pt, float Muminusgen_eta, float Muminusgen_phi) {
        constexpr double mmu = 0.1056583745;
;
        
        ROOT::Math::PtEtaPhiMVector recplus(Muplus_pt, Muplus_eta, Muplus_phi, mmu);
        ROOT::Math::PtEtaPhiMVector recminus(Muminus_pt, Muminus_eta, Muminus_phi, mmu);
        
        ROOT::Math::PtEtaPhiMVector genplus(Muplusgen_pt, Muplusgen_eta, Muplusgen_phi, mmu);
        ROOT::Math::PtEtaPhiMVector genminus(Muminusgen_pt, Muminusgen_eta, Muminusgen_phi, mmu);
        
        const double cosalpharec = (recplus.x()*recminus.x() + recplus.y()*recminus.y() + recplus.z()*recminus.z())/recplus.P()/recminus.P();
        const double cosalphagen = (genplus.x()*genminus.x() + genplus.y()*genminus.y() + genplus.z()*genminus.z())/genplus.P()/genminus.P();
        
        return sqrt((1.-cosalpharec)/(1.-cosalphagen)) - 1.;
        //return sqrt(recplus.P()*recminus.P()/genplus.P()/genminus.P()*(1.-cosalpharec)/(1.-cosalphagen)) - 1.;
    }
""")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v109_RecJpsiPhotos_quality_constraint/210720_225832/0000/globalcor_0_1.root");
  
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v109_RecJpsiPhotos_quality_constraint/210720_225832/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v109_RecJpsiPhotos_quality_constraint/210720_225832/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v96_RecJpsiPhotosTail0_quality_noconstraint/210714_101530/0000/globalcor_*.root")

  
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v127_RecJpsiPhotos_quality_constraint/210731_153616/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v127_RecJpsiPhotos_quality_constraint/210731_153616/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v130_RecJpsiPhotos_quality_constraint/210802_204103/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v130_RecJpsiPhotos_quality_constraint/210802_204103/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v132_RecJpsiPhotos_quality_constraint/210803_024703/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v132_RecJpsiPhotos_quality_constraint/210803_024703/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v135_RecJpsiPhotos_quality_constraint/210805_130727/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v135_RecJpsiPhotos_quality_constraint/210805_130727/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v137_RecJpsiPhotos_idealquality/210806_103002/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v144_RecJpsiPhotos_quality_constraint/210807_192307/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v144_RecJpsiPhotos_quality_constraint/210807_192307/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v164_RecJpsiPhotos_idealquality_constraint/210903_020706/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v164_RecJpsiPhotos_idealquality_constraint/210903_020706/0001/globalcor_*.root");

chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecJpsiPhotos_quality_constraintfsr28/210930_202515/0000/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecJpsiPhotos_quality_constraintfsr28/210930_202515/0001/*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiG_quality_constraintfsr28/211007_111614/0000/*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiG_quality_constraintfsr28/211007_111614/0000/*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiFpost_quality_constraintfsr28/211007_111216/0000/*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v138a_RecJpsiPhotos_quality_constraint/210806_161218/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v138a_RecJpsiPhotos_quality_constraint/210806_161218/0001/globalcor_*.root")


d = ROOT.ROOT.RDataFrame(chain)

#d = d.Filter("Jpsigen_mass > 2.8 && Jpsigen_mass>0.")
d = d.Filter("Jpsigen_mass > 2.9 && Jpsigen_mass>0.")
#d = d.Filter("Jpsi_mass > 2.8 && Jpsigen_mass>0.")
#d = d.Filter("Jpsi_mass > 2.4 && Jpsi_mass < 3.8 && Jpsigen_mass>3.0968")
#d = d.Filter("Jpsi_mass > 2.8 && Jpsi_mass < 3.4 && Jpsigen_mass>3.0968")
#d = d.Filter("Jpsigen_mass>3.0968")
#d = d.Filter("Muplusgen_eta > 2.3 && Muminusgen_eta > 2.3 && Muplusgen_pt>1.1 && Muminusgen_pt>1.1")
#d = d.Filter("Muplusgen_eta > 2.3 && Muminusgen_eta > 2.3 && Muplusgen_pt>10. && Muminusgen_pt>10.")
#d = d.Filter("Muplusgen_eta > 2.3 && Muminusgen_eta > 2.3 && Muplusgen_pt>10. && Muminusgen_pt>10.")
#d = d.Filter("Muplusgen_eta > 2.3 && Muminusgen_eta > 2.3 && Muplusgen_pt>1.1 && Muminusgen_pt>1.1 && log(Muplusgen_pt/Muminusgen_pt)<2.")
#d = d.Filter("abs(Muplusgen_eta) < 0.8 && abs(Muminusgen_eta) < 0.8 && Muplusgen_pt>10. && Muminusgen_pt>10.")
#d = d.Filter("Muplusgen_eta > 2.3 && Muplusgen_pt>40. && Muminusgen_pt>1.1")
#d = d.Filter("Muplusgen_pt>2.0 && Muminusgen_pt>2.0")
#d = d.Filter("Jpsigen_eta>2.2")
#d = d.Filter("Muplusgen_pt>10.0 && Muminusgen_pt>2.0")
#d = d.Filter("Muplusgen_eta>2.3")
#d = d.Filter("abs(Muplusgen_eta)<0.8 && abs(Muminusgen_eta)<0.8")

#d = d.Filter("Muplusgen_pt>20.0 && Muplusgen_eta>2.2 && Muminusgen_pt>2.0")

#d = d.Filter("abs(Muplusgen_eta) < 0.8 && abs(Muminusgen_eta) < 0.8 && Muplusgen_pt>1.1 && Muminusgen_pt>1.1")

#d = d.Filter("Jpsigen_eta>2.2")

#d = d.Filter("chisqval/ndof < 1.1")

#d = d.Filter("(Muplus_pt*cosh(Muplus_eta)) < 100. && (Muminus_pt*cosh(Muminus_eta)) < 100.")
#d = d.Filter("Muplusgen_pt>1.1 && Muminusgen_pt>1.1 && (Muplusgen_pt*cosh(Muplusgen_eta)) < 100. && (Muminusgen_pt*cosh(Muminusgen_eta)) < 100.")


#d = d.Filter("Muplusgen_eta > 2.3 && Muminusgen_eta > 2.3 && Muplusgen_pt>1.1 && Muminusgen_pt>1.1")
#d = d.Filter("Muplusgen_eta < 0.8 && Muminusgen_eta < 0.8 && Muplusgen_pt>1.1 && Muminusgen_pt>1.1")

#d = d.Filter("Muplusgen_eta > 2.3 && Muplusgen_pt>1.1 &&")
#d = d.Filter("Muplus_eta > 2.3 && Muminus_eta > 2.3 && Muplus_pt>1.1 && Muminus_pt>1.1")
#d = d.Filter("Muplusgen_eta > 2.3 && Muminusgen_eta > 2.3 && Muplusgen_pt>1.1 && Muminusgen_pt>1.1")

#d = d.Filter("Muplusgen_eta > 2.3 && Muminusgen_eta > 2.3 && Muplusgen_pt>1.1 && Muminusgen_pt>1.1 && (Muplusgen_pt*cosh(Muplusgen_eta)) > 100. && (Muminusgen_pt*cosh(Muminusgen_eta)) > 100.")

#d = d.Filter("Muplus_pt>1.1 && Muminus_pt>1.1 && (Muplus_pt*cosh(Muplus_eta)) < 100. && (Muminus_pt*cosh(Muminus_eta)) < 100.")

#d = d.Filter("Mupluscons_pt>1.1 && Muminuscons_pt>1.1 && (Mupluscons_pt*cosh(Mupluscons_eta)) < 100. && (Muminuscons_pt*cosh(Muminuscons_eta)) < 100.")

#d = d.Filter("Muplus_nvalid > 8 && Muplus_nvalidpixel>1 && Muminus_nvalid > 8 && Muminus_nvalidpixel>1")
#d = d.Filter("Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel>0")
#d = d.Filter("Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel>0")

#d = d.Filter("Jpsigen_mass>3.0968")
#d = d.Filter("Jpsi_sigmamass > 0.")
#d = d.Filter("Jpsi_mass>2.8 && Jpsi_mass<3.4")
#d = d.Filter("(Muplus_nvalid == Muplus_nmatchedvalid && Muminus_nvalid == Muminus_nmatchedvalid)")
#d = d.Filter("(Muplus_nambiguousmatchedvalid + Muminus_nambiguousmatchedvalid) == 0 ")
#d = d.Filter("TMath::Prob(chisqval,ndof)>0.03")

#d = d.Filter("abs(Jpsi_mass-Jpsigen_mass)/Jpsi_sigmamass < 1.")

#d = d.Filter("std::isnan(Jpsi_sigmamass)")

#d = d.Filter("Jpsi_sigmamass<=0.")

#d = d.Filter("!(Muplus_nvalid == Muplus_nmatchedvalid && Muminus_nvalid == Muminus_nmatchedvalid)")
#d = d.Filter("(Muplus_nambiguousmatchedvalid + Muminus_nambiguousmatchedvalid) > 0 ")

#d = d.Define(

#d = d.Define("invmsq", "1./Jpsigen_mass/Jpsigen_mass")
#d = d.Define("invmsq", "1./Jpsi_mass/Jpsi_mass")

#d = d.Define("msmeared", "Jpsigen_mass + 0.06*gRandom->Gaus()")


#mean = d.Mean("msmeared")
#sigma = d.StdDev("msmeared")

mean = d.Mean("Jpsigen_mass")
sigma = d.StdDev("Jpsigen_mass")

dratio = 3.0969/3.09692

print(mean.GetValue())
print(sigma.GetValue())
#print(mean.GetValue()*3.09692**2-1.)
print(mean.GetValue()/3.09692-1.)

print(mean.GetValue()*dratio)

assert(0)

#mean = d.Mean("Jpsigen_mass")
#sigma = d.StdDev("Jpsigen_mass")

#mean = d.Mean("Jpsi_mass")
#sigma = d.StdDev("Jpsi_mass")

#d = d.Define("weight", "1./Jpsi_sigmamass/Jpsi_sigmamass")
#d = d.Define("weightedmass", "weight*Jpsi_mass")

#sumwmass = d.Sum("weightedmass")
#sumw = d.Sum("weight")

#wmean =

#d = d.Define("kr","Jpsi_mass/3.09692 - 1.")
#d = d.Define("kr","Jpsicons_mass/3.09692 - 1.")
#d = d.Define("kr","alpharatio(Muplus_pt, Muplus_eta, Muplus_phi, Muminus_pt, Muminus_eta, Muminus_phi, Muplusgen_pt, Muplusgen_eta, Muplusgen_phi, Muminusgen_pt, Muminusgen_eta, Muminusgen_phi)")

#d = d.Define("kr","alpharatio(Mupluscons_pt, Mupluscons_eta, Mupluscons_phi, Muminuscons_pt, Muminuscons_eta, Muminuscons_phi, Muplusgen_pt, Muplusgen_eta, Muplusgen_phi, Muminusgen_pt, Muminusgen_eta, Muminusgen_phi)")

d = d.Define("kr", "Muplusgen_pt*Muminusgen_pt/Mupluscons_pt/Muminuscons_pt - 1.")

#d = d.Define("kr","Muplusgen_pt/Muplus_pt - 1.")
#d = d.Define("krcons","Muplusgen_pt/Mupluscons_pt - 1.")

d = d.Define("krplus","Muplusgen_pt/Muplus_pt - 1.")
#d = d.Define("krconsplus","Muplusgen_pt/Mupluscons_pt - 1.")

#d = d.Define("krconsplus","Mupluscons_pt/Muplusgen_pt - 1.")

d = d.Define("krminus","Muminusgen_pt/Muminus_pt - 1.")
#d = d.Define("krconsminus","Muminusgen_pt/Muminuscons_pt - 1.")

d = d.Define("krconsplus","Muplusgen_pt/Mupluscons_pt - 1.")
d = d.Define("krconsminus","Muminusgen_pt/Muminuscons_pt - 1.")

#d = d.Define("krconsplus","Muplusgen_pt*cosh(Muplusgen_eta)/Mupluscons_pt/cosh(Mupluscons_eta) - 1.")
#d = d.Define("krconsminus","Muminusgen_pt*cosh(Muminusgen_eta)/Muminuscons_pt/cosh(Muminuscons_eta) - 1.")

#d = d.Define("kr","log(Muplusgen_pt/Muminusgen_pt)")

#d = d.Define("kr","(Jpsi_mass-Jpsigen_mass)/Jpsi_sigmamass")

#d = d.Define("kr","log(Muplusgen_pt/Muplus_pt)")
#d = d.Define("krcons","log(Muplusgen_pt/Mupluscons_pt)")

#d = d.Define("kr","Muminusgen_pt/Muminus_pt - 1.")
#d = d.Define("krcons","Muminusgen_pt/Muminuscons_pt - 1.")

#d = d.Define("kr","Muplus_pt/Muplusgen_pt - 1.")
#d = d.Define("krcons","Mupluscons_pt/Muplusgen_pt - 1.")

#d = d.Define("kr","Muplus_eta - Muplusgen_eta")
#d = d.Define("krcons","Mupluscons_eta - Muplusgen_eta")

#d = d.Define("kr","deltaphi(Muplusgen_phi, Muplus_phi)")
#d = d.Define("krcons","deltaphi(Muplusgen_phi, Mupluscons_phi)")

#d = d.Define("kr","Jpsi_x - Jpsigen_x")
#d = d.Define("krcons","Jpsicons_x - Jpsigen_x")

#d = d.Define("kr","Jpsi_y - Jpsigen_y")
#d = d.Define("krcons","Jpsicons_y - Jpsigen_y")


#d = d.Define("kr","Jpsi_z - Jpsigen_z")
#d = d.Define("krcons","Jpsicons_z - Jpsigen_z")

##d = d.Define("dx", "sqrt(pow(Jpsi_x - Jpsigen_x, 2) + pow(Jpsi_y - Jpsigen_y, 2) + pow(Jpsi_z - Jpsigen_z, 2))")
##d = d.Define("dxcons", "sqrt(pow(Jpsicons_x - Jpsigen_x, 2) + pow(Jpsicons_y - Jpsigen_y, 2) + pow(Jpsicons_z - Jpsigen_z, 2))")

#d = d.Define("dx", "sqrt(pow(Jpsi_x - Jpsigen_x, 2) + pow(Jpsi_y - Jpsigen_y, 2))")
#d = d.Define("dxcons", "sqrt(pow(Jpsicons_x - Jpsigen_x, 2) + pow(Jpsicons_y - Jpsigen_y, 2))")

#krlim = 0.01
#krlim = 1.
#krlim = 0.4
krlim = 0.1

#hmass = d.Histo1D(("hmass","", 200, 2.8/3.09692-1., 3.4/3.09692-1.), "kr")

#hmass = d.Histo1D(("hmass","", 200, -5.,5.), "kr")

#hmasscons = d.Histo1D(("hmasscons","", 200, -krlim, krlim), "krcons")



d = d.Define("chisqndof","chisqval/ndof")
#d = d.Define("chisqndof","TMath::Prob(chisqval,ndof)")

#hmass = d.Histo1D(("hmass","",100, 0.,5.),"chisqndof")
#hmass = d.Histo1D(("hmass","",400, -10.,10.),"kr")
#hmass = d.Histo1D(("hmass","",400, -0.4, 0.4),"krconsplus")
hkr = d.Histo1D(("hkr","",400, -0.1, 0.1),"kr")
hkrconsplus = d.Histo1D(("hkrconsplus","",400, -0.2, 0.2),"krconsplus")
hkrconsminus = d.Histo1D(("hkrconsminus","",400, -0.2, 0.2),"krconsminus")

#hmass = d.Histo1D(("hmass","",100, 0.,1e5),"gradmax")

#h2 = d.Histo2D(("h2","",100,-0.4,0.4,100,0.,5.), "krcons", "chisqndof")
#h2 = d.Histo2D(("h2","",100,-0.1,0.1,100,0.,2.), "krcons", "chisqndof")


hcor = d.Histo2D(("hcor","", 100, -krlim, krlim, 100, -krlim, krlim), "krplus", "krminus")
hcorcons = d.Histo2D(("hcorcons","", 100, -krlim, krlim, 100, -krlim, krlim), "krconsplus", "krconsminus")


#dxlim = 0.4

#hmass = d.Histo1D(("hmass","", 200, 0., dxlim), "dx")
#hmasscons = d.Histo1D(("hmasscons","", 200, 0., dxlim), "dxcons")


#hmass = d.Histo1D(("hmass","", 200, 2.4, 3.8), "Jpsi_mass")
#hmass = d.Histo2D(("hmass","", 100, 2.8, 3.4, 100, 0., 0.2), "Jpsi_mass", "Jpsi_sigmamass")
#hmass = d.Histo2D(("hmass","", 100, 2.8, 3.4, 100, 0., 0.05), "Jpsi_mass", "Jpsi_sigmamass")

hmass = d.Histo2D(("hmass","", 100, 2.8, 3.4, 100, -0.1, 0.1), "Jpsi_mass", "krconsplus")
#hmass2 = d.Histo2D(("hmass2","", 100, -3.,3., 100, -0.1, 0.1), "kr", "krconsplus")
hmass2 = d.Histo2D(("hmass2","", 100, -0.1,0.1, 100, -0.1, 0.1), "kr", "krconsplus")



print(mean.GetValue())
print(sigma.GetValue())
print(mean.GetValue()/3.09692-1.)
#print(mean.GetValue()/3.09692 - 1.)

#weightedavg = sumwmass.GetValue()/sumw.GetValue()
#print("weighted:")
#print(weightedavg)
#print(weightedavg/3.09692 - 1.)

c = ROOT.TCanvas()
hmass.Draw()
#hmasscons.SetLineColor(ROOT.kRed)
#hmasscons.Draw()
#hmass.Draw("SAME")

#c2 = ROOT.TCanvas()
#hcor.Draw("COLZ")

c3 = ROOT.TCanvas()

hmass2.Draw("COLZ")

c4 = ROOT.TCanvas()
hcorcons.Draw("COLZ")

c5 = ROOT.TCanvas()
hkrconsplus.Draw()

c5a = ROOT.TCanvas()
hkrconsminus.Draw()


c6 = ROOT.TCanvas()
hkr.Draw()

