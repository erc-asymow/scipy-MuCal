
import ROOT
ROOT.ROOT.EnableImplicitMT()


chain = ROOT.TChain("tree")

chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_RecJpsiPhotos_idealquality_zeromaterial_noconstraint/210728_081911/0000/globalcor_*.root")

d = ROOT.ROOT.RDataFrame(chain)


d = d.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Jpsigen_mass > 3.0968 && Muplus_eta > 2.0 && Muminus_eta>2.0 && Jpsigen_pt>40.")


hmass = d.Histo1D(("hmass","", 800, 1.1, 5.1), "Jpsi_mass")


hmass.Draw("E")
                  
                  
