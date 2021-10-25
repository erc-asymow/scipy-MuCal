import ROOT
import math

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
    
std::set<unsigned int> debugidxs = {45292, 45310, 45328, 45346, 45364, 45382, 45400, 45418, 45436, 45437, 45458, 45459, 45480, 45481, 45502, 45503, 45524, 45525, 45546, 45547, 45568, 45569, 45590, 45591};

ROOT::VecOps::RVec<bool> isdebugidx(const ROOT::VecOps::RVec<unsigned int> &idxs) {
    ROOT::VecOps::RVec<bool> match(idxs.size(), false);
    for (unsigned int i=0; i<idxs.size(); ++i) {
        match[i] = debugidxs.count(idxs[i])>0;
    }
    return match;
}

ROOT::VecOps::RVec<float> hessdiag(const ROOT::VecOps::RVec<float> &hess, unsigned int ndiag) {
    ROOT::VecOps::RVec<float> diag(ndiag, 0.);
    unsigned int idiag = 0;
    unsigned int k = 0;
    for (unsigned int i=0; i<ndiag; ++i) {
        for (unsigned int j=i; j<ndiag; ++j) {
            if (i==j) {
                diag[idiag] = hess[k];
                ++idiag;
            }
            ++k;
        }
    }
    return diag;
}

bool debugcheck(const ROOT::VecOps::RVec<bool> &mask) {

    std::cout << mask << std::endl;
    return true;
}

bool debugcheck(const ROOT::VecOps::RVec<unsigned int> &mask) {

    std::cout << mask << std::endl;
    return true;
}

ROOT::VecOps::RVec<bool> firstidxs(unsigned int nhits) {
    if (nhits==2) {
        return {true, false};
    }
    else if (nhits==4) {
        return {true, false, true, false};
    }
    return ROOT::VecOps::RVec<bool>(nhits, true);
}

ROOT::VecOps::RVec<bool> secondidxs(unsigned int nhits) {
    if (nhits==2) {
        return {false, true};
    }
    else if (nhits==4) {
        return {false, true, false, true};
    }
    return ROOT::VecOps::RVec<bool>(nhits, true);
    
}

bool hasdups(const ROOT::VecOps::RVec<unsigned int> &idxs, unsigned int nhitsplus, unsigned int nhitsminus) {
  const unsigned alignidx = 2*nhitsplus + 2*nhitsminus;
  std::vector<unsigned int> alignidxs(idxs.begin() + 2*nhitsplus + 2*nhitsminus, idxs.end());
  std::sort(alignidxs.begin(), alignidxs.end());
  auto dupit = std::adjacent_find(alignidxs.begin(), alignidxs.end());
  return dupit != alignidxs.end();
  
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

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v153a_RecJpsiPhotos_idealquality_constraint/210820_001516/0000/globalcor_0_2.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v158_RecJpsiPhotos_idealquality_constraint/210829_162156/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v158_RecJpsiPhotos_idealquality_constraint/210829_162156/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v164_RecJpsiPhotos_idealquality_constraint/210903_020706/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v164_RecJpsiPhotos_idealquality_constraint/210903_020706/0001/globalcor_*.root");

chain.Add("/data/shared/muoncal/MuonGunUL2016_v166_RecJpsiPhotos_idealquality_constraint/210907_053804/0000/globalcor_*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v166_RecJpsiPhotos_idealquality_constraint/210907_053804/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v138a_RecJpsiPhotos_quality_constraint/210806_161218/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v138a_RecJpsiPhotos_quality_constraint/210806_161218/0001/globalcor_*.root")


d = ROOT.ROOT.RDataFrame(chain)

d = d.Filter("Jpsigen_mass > 2.8 && Jpsigen_mass>0.")

#d = d.Filter("Jpsi_mass>2.5 && Jpsi_mass<3.7")
d = d.Filter("Jpsi_mass>2.8 && Jpsi_mass<3.4")
#d = d.Filter("abs(Jpsi_mass - 3.09692)/Jpsi_sigmamass < 1.0")
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
d = d.Filter("Muplusgen_pt>2.0 && Muminusgen_pt>2.0")
#d = d.Filter("Muplusgen_pt*cosh(Muplusgen_eta)>1.1 && Muminusgen_pt*cosh(Muminusgen_eta)>1.1")

#d = d.Filter("Muplus_pt>2.0 && Muminus_pt>2.0")
#d = d.Filter("Muplusgen_pt<10. && Muminusgen_pt<10.")

d = d.Filter("Muplusgen_eta>2.2 && Muminusgen_eta>2.2")

#d = d.Filter("Muplusgen_pt>15.")

#d = d.Filter("abs(Muplusgen_eta) < 0.8 && abs(Muminusgen_eta) < 0.8 && Muplusgen_pt>1.1 && Muminusgen_pt>1.1")
#d = d.Filter("Jpsi_sigmamass < 0.1")
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
d = d.Filter("Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel>0")
#d = d.Filter("Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel>0")

d = d.Filter("Jpsigen_mass>3.0968")

#d = d.Filter("!hasdups(globalidxv, Muplus_nhits, Muminus_nhits)")

#d = d.Filter("Muplus_nvalid == Muplus_nmatchedvalid && Muminus_nvalid == Muminus_nmatchedvalid && Muplus_nambiguousmatchedvalid == 0 && Muminus_nambiguousmatchedvalid == 0")
#d = d.Filter("Jpsi_sigmamass > 0.")
#d = d.Filter("Jpsi_mass>2.8 && Jpsi_mass<3.4")
#d = d.Filter("Jpsi_mass>2.9 && Jpsi_mass<3.29")
#d = d.Filter("abs(Jpsi_mass - 3.09692) < 0.3")
#d = d.Filter("(Muplus_nvalid == Muplus_nmatchedvalid && Muminus_nvalid == Muminus_nmatchedvalid)")
#d = d.Filter("(Muplus_nambiguousmatchedvalid + Muminus_nambiguousmatchedvalid) == 0 ")
#d = d.Filter("TMath::Prob(chisqval,ndof)>0.03")
#d = d.Filter("abs(Jpsigen_eta)>2.2")

#d = d.Filter("abs(Jpsi_mass-Jpsigen_mass)/Jpsi_sigmamass < 1.")

#d = d.Filter("std::isnan(Jpsi_sigmamass)")
#d = d.Define("globalidxdebug","isdebugidx(globalidxv)")
#d = d.Define("selidxs", "globalidxv[globalidxdebug]")
#d = d.Define("graddebug", "gradv[globalidxdebug]")
#d = d.Define("hessdiagdebug", "hessdiag(hesspackedv, globalidxv.size())[globalidxdebug]")
#d = d.Define("gradratio", "-graddebug/hessdiagdebug")

#d = d.Define("largeidxs", "abs(gradratio)>10.")

#d = d.Define("gradlarge", "graddebug[largeidxs]")
#d = d.Define("hesslarge", "hessdiagdebug[largeidxs]")


#d = d.Define("nhitsdebug", "graddebug.size()")

#d = d.Filter("nhitsdebug==2 || nhitsdebug==4")
#d = d.Filter("nhitsdebug==4")

#d = d.Define("didxsfirst", "firstidxs(nhitsdebug)")
#d = d.Define("didxssecond", "secondidxs(nhitsdebug)")

#d = d.Define("gradfirst","graddebug[didxsfirst]")
#d = d.Define("hessfirst","hessdiagdebug[didxsfirst]")

#d = d.Define("gradsecond","graddebug[didxssecond]")
#d = d.Define("hesssecond","hessdiagdebug[didxssecond]")

#d = d.Define("gradratiofirst", "-gradfirst/hessfirst")

#d = d.Define("krmass", "Jpsi_mass/3.09692 - 1.")
d = d.Define("krmass", "Jpsi_mass/3.09692 - 1.")

#d = d.Define("krmass", "3.09692*3.09692/Jpsi_mass/Jpsi_mass - 1.")

d = d.Define("kerr", "Jpsi_sigmamass/Jpsi_mass")


pmass = d.Profile1D(("pmass","",50, 0., 0.1), "kerr", "krmass")
herr = d.Histo1D(("herr", "", 50, 0., 0.1), "kerr")

c = ROOT.TCanvas()
pmass.Draw()

c2 = ROOT.TCanvas()
herr.Draw()

input("wait")
assert(0)

d = d.Define("weight", "3.09692*3.09692/Jpsi_sigmamass/Jpsi_sigmamass")

d = d.Define("weightedkr", "weight*krmass")


mean = d.Mean("krmass")

sumw = d.Sum("weight")
sumwkr = d.Sum("weightedkr")

meanval = mean.GetValue()
weightedmeanval = sumwkr.GetValue()/sumw.GetValue()

print("mean", meanval)
print("weightedmean", weightedmeanval)



input("wait")
assert(0)


#d = d.Define("massv", "ROOT::VecOps::RVec<float>(gradfirst.size(), Jpsi_mass)")

#d = d.Define("kr", "Muplusgen_pt/Mupluscons_pt-1.")
d = d.Define("kr", "Muplus_pt/Mupluscons_pt-1.")
#d = d.Define("kr", "Muplusgen_pt/Muplus_pt-1.")
#d = d.Define("krv", "ROOT::VecOps::RVec<float>(gradfirst.size(), kr)")


d = d.Define("krmass", "Jpsi_mass/3.09692 - 1.")
d = d.Define("krmassinv",  "3.09692/Jpsi_mass - 1.")
d = d.Define("krmassinvsq",  "3.09692*3.09692/Jpsi_mass/Jpsi_mass - 1.")
d = d.Define("krmasslog", "log(Jpsi_mass/3.09692)")
#d = d.Define("krmass", "3.09692/Jpsi_mass - 1.")
#d = d.Define("krmass", "3.09692*3.09692/Jpsi_mass/Jpsi_mass - 1.")
#d = d.Define("krmass", "Jpsi_mass*Jpsi_mass/3.09692/3.09692 - 1.")
#d = d.Define("krmass", "log(Jpsi_mass/3.09692)")

d = d.Define("pull", "(Jpsi_mass - 3.09692)/Jpsi_sigmamass")

d = d.Define("sigmasq", "Jpsi_sigmamass*Jpsi_sigmamass")

pmass = d.Profile1D(("pmass","",500, 0., 0.022), "sigmasq", "krmass")
pmassinvsq = d.Profile1D(("pmassinvsq","",500, 0., 0.022), "sigmasq", "krmassinvsq")
pmasspull = d.Profile1D(("pmasspull","",500, 0., 0.022), "sigmasq", "pull")

c = ROOT.TCanvas()
pmass.Draw()

c2 = ROOT.TCanvas()
pmassinvsq.Draw()

c3 = ROOT.TCanvas()
pmasspull.Draw()

input("wait")
assert(0)

#d = d.Define("krmom", "sqrt(Muplus_pt*cosh(Muplus_eta)*Muminus_pt*cosh(Muminus_eta)/Muplusgen_pt/Muminusgen_pt/cosh(Muplusgen_eta)/cosh(Muminusgen_eta)) - 1.")
#d = d.Define("krmom", "1./sqrt(Muplus_pt*cosh(Muplus_eta)*Muminus_pt*cosh(Muminus_eta)/Muplusgen_pt/Muminusgen_pt/cosh(Muplusgen_eta)/cosh(Muminusgen_eta)) - 1.")
d = d.Define("krmom", "1./(Muplus_pt*cosh(Muplus_eta)*Muminus_pt*cosh(Muminus_eta)/Muplusgen_pt/Muminusgen_pt/cosh(Muplusgen_eta)/cosh(Muminusgen_eta)) - 1.")


d = d.Define("krplus", "Muplus_pt/Mupluscons_pt-1.")
d = d.Define("krminus", "Muminus_pt/Muminuscons_pt-1.")

#d = d.Define("krplus", "Mupluskin_pt/Mupluskincons_pt-1.")
#d = d.Define("krminus", "Muminuskin_pt/Muminuskincons_pt-1.")

#d = d.Define("krplusg", "Muplusgen_pt/Muplus_pt-1.")
#d = d.Define("krminusg", "Muminusgen_pt/Muminus_pt-1.")

d = d.Define("krplusg", "Muplusgen_pt*cosh(Muplusgen_eta)/Muplus_pt/cosh(Muplus_eta)-1.")
d = d.Define("krminusg", "Muminusgen_pt*cosh(Muminusgen_eta)/Muminus_pt/cosh(Muminus_eta)-1.")

d = d.Define("krplusgcons", "Muplusgen_pt*cosh(Muplusgen_eta)/Mupluscons_pt/cosh(Mupluscons_eta)-1.")
d = d.Define("krminusgcons", "Muminusgen_pt*cosh(Muminusgen_eta)/Muminuscons_pt/cosh(Muminuscons_eta)-1.")

d = d.Define("kbalance", "Muplus_pt*cosh(Muplus_eta)/Muminus_pt/cosh(Muminus_eta) - 1.")
d = d.Define("kbalancecons", "Mupluscons_pt*cosh(Mupluscons_eta)/Muminuscons_pt/cosh(Muminuscons_eta) - 1.")
d = d.Define("kbalancekin", "Mupluskin_pt*cosh(Mupluskin_eta)/Muminuskin_pt/cosh(Muminuskin_eta) - 1.")

d = d.Define("logabsdeltachisq", "log10(abs(deltachisqval))")

d = d.Define("chisqndof", "chisqval/ndof")
#d = d.Define("chisqndof", "TMath::Prob(chisqval,ndof)")

d = d.Define("pull", "(Jpsi_mass - 3.09692)/Jpsi_sigmamass")

#d = d.Define("krplusg", "Muplus_pt/Muplusgen_pt-1.")
#d = d.Define("krminusg", "Muminus_pt/Muminusgen_pt-1.")

#d = d.Define("krplusg", "Muplus_pt*cosh(Muplus_eta)/Muplusgen_pt/cosh(Muplusgen_eta)-1.")
#d = d.Define("krminusg", "Muminus_pt*cosh(Muminus_eta)/Muminusgen_pt/cosh(Muminusgen_eta)-1.")


#d = d.Define("gradfirst","graddebug[0]")
#d = d.Define("hessfirst","hessdiagdebug[0]")

#d = d.Filter("nhitsdebug==1 || nhitsdebug==3")
#d = d.Filter("nhitsdebug==2 || nhitsdebug==4")


#d = d.Filter("debugcheck(globalidxdebug)")
#d = d.Filter("debugcheck(selidxs)")

#d = d.Filter("All(abs(gradratio)>10.)")

#gradsum = d.Sum("graddebug")
#hsum = d.Sum("hessdiagdebug")


#gradsumfirst = d.Sum("gradfirst")
#hsumfirst = d.Sum("hessfirst")

#gradsumsecond = d.Sum("gradsecond")
#hsumsecond = d.Sum("hesssecond")

#gradsum = d.Sum("gradfirst")
#hsum = d.Sum("hessfirst")

#gradsum = d.Sum("gradsecond")
##hsum = d.Sum("hesssecond")

#hgrad = d.Histo1D(("hgrad","",100,-50.,50.), "graddebug")
#hhess = d.Histo1D(("hhess","",100,0.,100.), "hessdiagdebug")

#hgradsecond = d.Histo1D(("hgradsecond","",100,-0.5,0.5), "gradsecond")
#hgrad = d.Histo1D(("hgrad","",100,-50.,50.), "gradfirst")
#hhess = d.Histo1D(("hhess","",100,0.,100.), "hessfirst")

#hratio =  d.Histo1D(("hratio","",100,-10.,10.), "gradratiofirst")

#hgradlarge = d.Histo1D(("hgradlarge","",100,-50.,50.), "gradlarge")
#hhesslarge = d.Histo1D(("hhesslarge","",100,0.,100.), "hesslarge")



#hgradmass = d.Histo2D(("hgradmass","",100,2.8,3.4, 100, -20.,20.), "massv","gradfirst")
#hgradmasssecond = d.Histo2D(("hgradmasssecond","",100,2.8,3.4, 100, -0.2,0.2), "massv","gradsecond")
#hgradmass = d.Histo2D(("hgradmass","",100,2.8,3.4, 100, -5.,5.), "massv","gradratiofirst")
#hgradmass = d.Histo2D(("hgradmass","",100,2.8,3.4, 100, 0.,100.), "massv","hessfirst")
#hgradkr = d.Histo2D(("hgradkr","",30,-0.1, 0.1, 30, -20.,20.), "krv","gradfirst")

#dnhits = d.Histo1D(("dnhits","", 10, -0.5, 9.5), "nhitsdebug")

#d = d.Define("kgenplus", "std::vector<float>(graddebug.size(), 1./Muplusgen_pt)")
#d = d.Define("kgenminus", "std::vector<float>(graddebug.size(), 1./Muminusgen_pt)")

d = d.Define("kgenplus", " 1./Muplusgen_pt")
d = d.Define("kgenminus", "1./Muminusgen_pt")

#pkplus = d.Profile1D(("pkplus", "", 30, 1./30., 1./2.0), "kgenplus", "graddebug")
#pkminus = d.Profile1D(("pkminus", "", 30, 1./30., 1./2.0), "kgenminus", "graddebug")

pconsplus = d.Profile1D(("pconsplus","",30, 1./30., 1./2.), "kgenplus", "krplus")
pconsminus = d.Profile1D(("pconsminus","",30, 1./30., 1./2.), "kgenminus", "krminus")

pmass = d.Profile1D(("pmass","",48, -2.4, 2.4), "Jpsigen_eta", "krmass")
pmassinv = d.Profile1D(("pmassinv","",48, -2.4, 2.4), "Jpsigen_eta", "krmassinv")
pmassinvsq = d.Profile1D(("pmassinvsq","",48, -2.4, 2.4), "Jpsigen_eta", "krmassinvsq")
pmasslog = d.Profile1D(("pmasslog","",48, -2.4, 2.4), "Jpsigen_eta", "krmasslog")

pmom = d.Profile1D(("pmom","",48, -2.4, 2.4), "Jpsigen_eta", "krmom")

dplus = d.Filter("Muplusgen_pt>20.")
dminus = d.Filter("Muminusgen_pt>20.")

#pconsplusg = dplus.Profile1D(("pconsplusg","",30, 1./30., 1./2.), "Muplusgen_eta", "krplusg")
#pconsminusg = dminus.Profile1D(("pconsminusg","",30, 1./30., 1./2.), "Muminusgen_eta", "krminusg")


pconsplusg = d.Profile1D(("pconsplusg","",48, -2.4, 2.4), "Muplusgen_eta", "krplusg")
pconsminusg = d.Profile1D(("pconsminusg","",48, -2.4, 2.4), "Muminusgen_eta", "krminusg")

hbalance = d.Histo1D(("hbalance", "", 100, -0.1, 0.1), "kbalance")
hbalancecons = d.Histo1D(("hbalancecons", "", 100, -0.1, 0.1), "kbalancecons")
hbalancekin = d.Histo1D(("hbalancekin", "", 100, -0.1, 0.1), "kbalancekin")

hdchisq = d.Histo1D(("hdchisq", "", 100, -5., 2.), "logabsdeltachisq")

hdchisqeta = d.Histo2D(("hdchisqeta", "", 48, -2.4,2.4, 100, -5., 2.), "Jpsigen_eta", "logabsdeltachisq")

#hmasschisq = d.Histo2D(("hmasschisq", "", 50, 3.0, 3.2, 50, 0., 2.0), "Jpsi_mass", "chisqndof")
#hmasschisq = d.Histo2D(("hmasschisq", "", 50, -0.1, 0.1, 50, 0., 2.0), "krplusgcons", "chisqndof")
hmasschisq = d.Histo2D(("hmasschisq", "", 50, -0.1, 0.1, 50, 0., 2.0), "krplus", "chisqndof")

hmass = d.Histo1D(("hmass", "", 100, 2.5, 3.7), "Jpsi_mass")
hmassinvsq = d.Histo1D(("hmassinvsq", "", 100, -0.3, 0.3), "krmassinvsq")
hsigma = d.Histo1D(("hsigma", "", 100, 0., 1.0), "Jpsi_sigmamass")
hpull = d.Histo1D(("hpull", "", 100, -5.0, 5.0), "pull")

meanmass = d.Mean("Jpsi_mass")
meanpull = d.Mean("pull")

print("meanmass", meanmass.GetValue())
print("meanpull", meanpull.GetValue())

#pkminus = d.Profile1D("pkminus", "", 100, 1./30., 1./2.0)

#res = -gradsum.GetValue()/hsum.GetValue()
#sigma = math.sqrt(2./hsum.GetValue())

#print(gradsum.GetValue())
#print(hsum.GetValue())
#print(f"res = {res} +- {sigma}")

#resfirst = -gradsumfirst.GetValue()/hsumfirst.GetValue()
#sigmafirst = math.sqrt(2./hsumfirst.GetValue())
#print(resfirst)
#print(sigmafirst)
#print(f"resfirst = {resfirst} +- {sigmafirst}")

#ressecond = -gradsumsecond.GetValue()/hsumsecond.GetValue()
#sigmasecond = math.sqrt(2./hsumsecond.GetValue())
#print(ressecond)
#print(sigmasecond)
#print(f"ressecond = {ressecond} +- {sigmasecond}")

#c = ROOT.TCanvas()
#hgrad.Draw()


#ca = ROOT.TCanvas()
#hgradsecond.Draw()



#c2 = ROOT.TCanvas()
#hhess.Draw()

#c3 = ROOT.TCanvas()
#hratio.Draw()

#c4 = ROOT.TCanvas()
#hgradlarge.Draw()

#c5 = ROOT.TCanvas()
#hhesslarge.Draw()


#c6 = ROOT.TCanvas()
#hgradmass.Draw("COLZ")

#c6a = ROOT.TCanvas()
#hgradmasssecond.Draw("COLZ")


#c7 = ROOT.TCanvas()
#hgradkr.Draw("COLZ")
#dnhits.Draw()

#c8 = ROOT.TCanvas()
#pkplus.Draw()

#c9 = ROOT.TCanvas()
#pkminus.Draw()

#c10 = ROOT.TCanvas()
#pconsplus.Draw()

#c11 = ROOT.TCanvas()
#pconsminus.Draw()

c12 = ROOT.TCanvas()
pmass.Draw()

c12bb = ROOT.TCanvas()
pmassinv.Draw()

c12b = ROOT.TCanvas()
pmassinvsq.Draw()

c12c = ROOT.TCanvas()
pmasslog.Draw()


#c12a = ROOT.TCanvas()
#pmom.Draw()

#c13 = ROOT.TCanvas()
#pconsplusg.Draw()

#c14 = ROOT.TCanvas()
#pconsminusg.Draw()


#c15 = ROOT.TCanvas()
#hbalance.Draw()

#c16 = ROOT.TCanvas()
#hbalancecons.Draw()


#c17 = ROOT.TCanvas()
#hbalancekin.Draw()

c18 = ROOT.TCanvas()
hdchisq.Draw()

c19 = ROOT.TCanvas()
hdchisqeta.Draw("COLZ")


c20 = ROOT.TCanvas()
hmasschisq.Draw("COLZ")

c21 = ROOT.TCanvas()
hmass.Draw()

c22 = ROOT.TCanvas()
hsigma.Draw()

c23 = ROOT.TCanvas()
hpull.Draw()

c24 = ROOT.TCanvas()
hmassinvsq.Draw()
