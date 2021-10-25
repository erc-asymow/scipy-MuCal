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

ROOT::VecOps::RVec<double> cleanratio(const ROOT::VecOps::RVec<float> &dE, const ROOT::VecOps::RVec<float> &dEpred) {
    unsigned int nvalid = 0;
    for (unsigned int i=0; i<dE.size(); ++i) {
        const bool valid = dE[i] != -99. && std::abs(dEpred[i]) > 0.;
        if (valid) {
            ++nvalid;
        }
    }
    
    ROOT::VecOps::RVec<double> res(nvalid);
    unsigned int ivalid = 0;
    for (unsigned int i=0; i<dE.size(); ++i) {
        const bool valid = dE[i] != -99. && std::abs(dEpred[i]) > 0.;
        if (valid) {
            res[ivalid] = dE[i]/dEpred[i];
            ++ivalid;
        }
    }
    
    return res;

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

chain.Add("/data/shared/muoncal/MuonGunUL2016_v177_Gen_idealquality/210913_173054/0000/globalcor_*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v177_Gen_idealquality/210913_173054/0001/globalcor_*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v177_GenJpsiPhotosSingle_idealquality/210913_172953/0000/globalcor_*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v177_GenJpsiPhotosSingle_idealquality/210913_172953/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v159_Rec_idealquality/210830_160945/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v159_Rec_idealquality/210830_160945/0001/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v159a_RecJpsiPhotosSingle_idealquality/210831_001801/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v159a_RecJpsiPhotosSingle_idealquality/210831_001801/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_Gen_quality/210728_174037/0000/globalcor_0_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_Gen_quality/210728_174037/0001/globalcor_0_*.root");


#chain.Add("/data/shared/muoncal/MuonGunUL2016_v158_RecJpsiPhotosSingle_idealquality/210829_164132/0000/globalcor_0_1.root")


d = ROOT.ROOT.RDataFrame(chain)

#d = d.Filter("TMath::Prob(chisqval,ndof) > 0.02")

d = d.Filter("nValidHits > 8 && nValidPixelHits>1")
d = d.Filter("edmval<1e-5")
#d = d.Filter("genCharge < 0.")
#d = d.Filter("genPt > 140. && genEta>2.3")
d = d.Filter("genPt > 1.1")
d = d.Filter("genEta > 2.3")

d = d.Filter("TMath::Prob(chisqval, ndof+5)>0.02")

#d = d.Define("eratio", "cleanratio(dE, dEpred)")
#d = d.Define("pvec", "std::vector<double>(eratio.size(), genPt*cosh(genEta))")
#d = d.Define("pvec", "std::vector<double>(eratio.size(), genPt)")

d = d.Filter("dE[nValidHits - 1] != -99. && abs(dEpred[nValidHits - 1]) > 0.")
d = d.Define("eratio", "dE[nValidHits-1]/dEpred[nValidHits-1]")
d = d.Define("pvec", "genPt")
#d = d.Define("pvec", "genPt*cosh(genEta)")

#ploss = d.Profile1D(("ploss","",50, 0., 1200.), "pvec", "eratio")
ploss = d.Profile1D(("ploss","",50, 0., 150.), "pvec", "eratio")
#ploss = d.Profile1D(("ploss","",50, 0., 150.), "genPt", "eratio")
#ploss = d.Histo1D(("ploss","", 100, 0.,2.), "eratio")


c = ROOT.TCanvas()
ploss.Draw()
