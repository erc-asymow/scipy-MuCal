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

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v138a_RecJpsiPhotos_quality_constraint/210806_161218/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v138a_RecJpsiPhotos_quality_constraint/210806_161218/0001/globalcor_*.root")

chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_Gen_quality/210728_174037/0000/globalcor_*.root")
chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_Gen_quality/210728_174037/0001/globalcor_*.root")


d = ROOT.ROOT.RDataFrame(chain)

#d = d.Filter("genPt>3.5 && genPt<4.5 && genEta>1.3 && genEta<1.5")
d = d.Filter("genPt>3.5")
d = d.Filter("Sum(dE)/Sum(dEpred) > 500.")
##d = d.Filter("genPt>100. && genPt<110. && genEta>1.3 && genEta<1.5")
#d = d.Filter("genPt>5. && genPt<10. && genEta>1.3 && genEta<1.5")

d = d.Define("dEratio", "Sum(dE)/Sum(dEpred)")
d = d.Define("dEsigmaratio", "Sum(sigmadE)/Sum(dEpred)")

#h = d.Histo1D(("h","", 10000, 0.,5000.), "dEratio")
#h = d.Histo1D(("h","", 100, -2.4,2.4), "genEta")
h = d.Histo1D(("h","", 100, 3.5,150.), "genPt")
hsigma = d.Histo1D(("h","", 100, 0.,10.), "dEsigmaratio")
#h = d.Histo2D(("h","", 100, 0.,2., 100, 3.5,100.), "dEratio", "genPt")

c = ROOT.TCanvas()
h.Draw("COLZ")
#h.Draw()

c2 = ROOT.TCanvas()
hsigma.Draw("COLZ")
#h.Draw()
