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

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v138a_RecJpsiPhotos_quality_constraint/210806_161218/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v138a_RecJpsiPhotos_quality_constraint/210806_161218/0001/globalcor_*.root")



#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_Rec_quality_nobs/210719_142714/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_Rec_quality_nobs/210719_142714/0001/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_RecJpsiPhotosSingle_quality/210719_142322/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_RecJpsiPhotosSingle_quality/210719_142322/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v158_Rec_idealquality/210829_164324/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v158_Rec_idealquality/210829_164324/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v158_RecJpsiPhotosSingle_idealquality/210829_164132/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v158_RecJpsiPhotosSingle_idealquality/210829_164132/0001/globalcor_*.root");


#chain.Add("/data/shared/muoncal/MuonGunUL2016_v166_Rec_idealquality/210907_054014/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v166_Rec_idealquality/210907_054014/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v166_RecJpsiPhotosSingle_idealquality/210907_053912/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v166_RecJpsiPhotosSingle_idealquality/210907_053912/0001/globalcor_*.root");
  
  

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_Rec_idealquality/210910_175214/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_Rec_idealquality/210910_175214/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotosSingle_idealquality/210910_175002/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotosSingle_idealquality/210910_175002/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v159_Rec_idealquality/210830_160945/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v159_Rec_idealquality/210830_160945/0001/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v159a_RecJpsiPhotosSingle_idealquality/210831_001801/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v159a_RecJpsiPhotosSingle_idealquality/210831_001801/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_Gen_quality/210728_174037/0000/globalcor_0_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_Gen_quality/210728_174037/0001/globalcor_0_*.root");

chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_Gen_idealquality/211002_182020/0000/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_Gen_idealquality/211002_182020/0001/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_GenJpsiPhotosSingle_idealquality/211002_181835/0000/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_GenJpsiPhotosSingle_idealquality/211002_181835/0001/*.root");


#chain.Add("/data/shared/muoncal/MuonGunUL2016_v158_RecJpsiPhotosSingle_idealquality/210829_164132/0000/globalcor_0_1.root")


d = ROOT.ROOT.RDataFrame(chain)

#d = d.Filter("TMath::Prob(chisqval,ndof) > 0.02")

#d = d.Filter("genEta>1.3 && genEta<1.5 && genPt>1.1 && genPt>3.5 && genPt < 5.0 && genCharge > 0.")
d = d.Filter("genEta>0. && genPt>1.1 && genPt<2.0 && genCharge > 0.")
d = d.Filter("hitidxv[nValidHits-1] >= 10188")


d = d.Define("dx","dysimgenconv[nValidHits-1]")

hdx = d.Histo1D(("hdx","",100,-5.,5.), "dx")

hdx.Draw()


input("wait")

d = d.Filter("nValidHits > 8 && nValidPixelHits>1")
d = d.Filter("edmval<1e-5")
#d = d.Filter("genCharge < 0.")
#d = d.Filter("genPt > 140. && genEta>2.3")
d = d.Filter("genPt > 100. && genPt<110.")

#d = d.Filter("abs(genEta)<0.8")
d = d.Filter("genEta>2.3")
#d = d.Filter("genPt > 140. && abs(genEta)<0.8")
#d = d.Filter("genPt > 2.0 && genPt < 5. && genEta>2.3")

#d = d.Filter("genPt > 100. && abs(genEta)<0.4")

#d = d.Define("deratio", "Sum(dE)/Sum(dEpred)")
#d = d.Define("deratio", "dE/dEpred")

#d = d.Filter("deratio > 5.")

startidx = 0

d = d.Define("recParms", "refParms")

d = d.Define("recPt", f"std::abs(1./recParms[{startidx}])*std::sin(M_PI_2 - recParms[{startidx} + 1])");
d = d.Define("recCharge", f"std::copysign(1.0,recParms[{startidx}])");

#d = d.Define("recPt", "trackPt")
#d = d.Define("recCharge", "trackCharge")


#d = d.Filter("recPt > 2.0")

a = 8.07036618e-04
c = 5.31792566e-07




d = d.Define("kr", "genPt*cosh(genEta)*genCharge*recParms[0] - 1.");
#d = d.Define("kerr", "sqrt(refCov[0])*genPt*cosh(genEta)")

#d = d.Define("kerr", f"sqrt({a}/genPt/genPt + {c})*(1./cosh(genEta))*genPt*cosh(genEta)")
d = d.Define("kerrraw", f"sqrt({a}/recPt/recPt + {c})")
d = d.Define("kerr", f"sqrt({a}*pow(1./recPt - 5.*kerrraw, 2) + {c})*(1./cosh(genEta))*genPt*cosh(genEta)")
#d = d.Define("kerr", f"sqrt({a}/recPt/recPt + {c})*(1./cosh(genEta))*genPt*cosh(genEta)")
#d = d.Define("kerr", f"sqrt({a}/recPt/recPt + {c})*cos(recParms[1])*genPt*cosh(genEta)")

d = d.Define("weight", "1./kerr/kerr")
d = d.Define("krweighted","weight*kr")

d = d.Filter("abs(kr)<0.5")

#d = d.Define("kr", "genPt*genCharge/recPt/recCharge - 1.");
#d = d.Define("krinv", "recPt*recCharge/genPt/genCharge - 1.");


mean = d.Mean("kr")

sumwkr = d.Sum("krweighted")
sumw = d.Sum("weight")

meanval = mean.GetValue()

wmeanval = sumwkr.GetValue()/sumw.GetValue()

print("meanval", meanval)
print("wmeanval", wmeanval)

assert(0)


input("wait")


d = d.Define("chisqndof", "chisqval/ndof")



#d = d.Define("dx", "dxrecgen[nValidHits - 1]")
#d = d.Define("dx", "dxsimgen[nValidHits - 1]")
#d = d.Define("dx", "dysimgen[nValidHits - 1]")


#d = d.Define("fbrem", "(trackPt*cosh(trackEta) - outPt*cosh(outEta))/trackPt/cosh(trackEta)")

krlim = 0.5
#krlim = 0.3
#krlim = 0.1
#hkr = d.Histo1D(("hkr","", 100, -krlim, krlim), "kr")
#hkrinv = d.Histo1D(("hkrinv","", 100, -krlim, krlim), "krinv")

hkrerr = d.Histo2D(("hkrerr", "", 100, -krlim, krlim, 50, 0., 0.025), "kr", "kerr")

hkr = d.Histo1D(("hkr", "", 100, -krlim, krlim), "kr")
hkerr = d.Histo1D(("hkerr", "", 50, 0., 0.5), "kerr")

hkz = d.Histo1D(("hkz", "", 100, -20., 20.), "genZ")

hkerrz = d.Histo2D(("hkerrz","", 50, -10., 10., 50, 0., 0.5), "genZ", "kerr")

#hkrinvfbrem = d.Histo2D(("hkrinvfbrem", "", 50, -krlim, krlim, 50, 0.,0.03), "krinv", "fbrem")
#hkrinvfbrem = d.Histo2D(("hkrinvfbrem", "", 50, -krlim, krlim, 50, 0.,0.03), "kr", "fbrem")
#hkrinvchisq = d.Histo2D(("hkrinvchisq", "", 50, -krlim, krlim, 50, 0.,1.5), "kr", "chisqndof")
#kkrin

#dplus = d.Filter("genCharge > 0.")
#dminus = d.Filter("genCharge < 0.")

#dxlim = 0.05
##hdxplus = dplus.Histo1D(("hdxplus","", 100, -dxlim, dxlim), "dx")
##hdxminus = dminus.Histo1D(("hdxminus","", 100, -dxlim, dxlim), "dx")

##hloss = d.Histo1D(("hloss", "", 1000, 0., 100.), "deratio")

##meanloss = d.Mean("deratio")

##print("meanloss",meanloss.GetValue())

#c = ROOT.TCanvas()
#hkr.Draw()
#hkrinv.SetLineColor(ROOT.kRed)
#hkrinv.Draw("SAME")


dstat = d.Filter("abs(kr) < 0.5")

mean = dstat.Mean("kr")
sumwkr = dstat.Sum("krweighted")
sumw = dstat.Sum("weight")

meanval = mean.GetValue()
weightedmeanval = sumwkr.GetValue()/sumw.GetValue()

print("mean", meanval)
print("weightedmean", weightedmeanval)


##c2 = ROOT.TCanvas()
##hdxplus.Draw()
##hdxminus.SetLineColor(ROOT.kRed)
##hdxminus.Draw("SAME")

#c3 = ROOT.TCanvas()
#hkrinvchisq.Draw("COLZ")
##hkrinvfbrem.Draw("COLZ")
##hloss.Draw()


#hkrinv.Draw()

c = ROOT.TCanvas()
hkrerr.Draw("COLZ")

c2 = ROOT.TCanvas()
hkr.Draw()

c3 = ROOT.TCanvas()
hkerr.Draw()

c4 = ROOT.TCanvas()
hkerrz.Draw("COLZ")

c5 = ROOT.TCanvas()
hkz.Draw()
