import ROOT
import math

ROOT.gStyle.SetOptStat(111111)

ROOT.ROOT.EnableImplicitMT()


chain = ROOT.TChain("tree")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecJpsiPhotos_quality_constraintfsr28/210930_202515/0000/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecJpsiPhotos_quality_constraintfsr28/210930_202515/0001/*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202a_RecDataJPsiH_quality_constraintfsr28/210930_203700/0000/*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecDataJPsiG_quality_constraintfsr28/210930_204012/0000/*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecDataJPsiFpost_quality_constraintfsr28/210930_204138/0000/*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiFpost_quality_constraintfsr28/211007_111216/0000/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiG_quality_constraintfsr28/211007_111614/0000/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiH_quality_constraintfsr28/211007_111815/0000/*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiH_quality_constraintfsr29/211012_233512/0000/*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiG_quality_constraintfsr29/211012_233641/0000/*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiFpost_quality_constraintfsr29/211012_233806/0000/*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecJpsiPhotos_quality_constraintfsr29/211012_233934/0000/*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecJpsiPhotos_quality_constraintfsr29/211012_233934/0001/*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintfsr29/211013_161309/0000/*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintfsr29/211013_161309/0001/*.root")

chain.Add("/data/shared/muoncal/MuonGunUL2016_v208_RecJpsiPhotos_quality_constraintfsr29_biasm10/211020_005505/0000/*.root")
chain.Add("/data/shared/muoncal/MuonGunUL2016_v208_RecJpsiPhotos_quality_constraintfsr29_biasm10/211020_005505/0001/*.root")

d = ROOT.ROOT.RDataFrame(chain)


cols = d.GetColumnNames()

largecols = ["globalidxv", "gradv", "hesspackedv", "Muplus_jacRef", "Muminus_jacRef", "Muplus_refParms", "Muminus_refParms"]

smallcols = []
for col in cols:
    if col not in largecols:
        smallcols.append(col)
        
d.Snapshot("tree","jpsisnapshot.root", smallcols)

#print(cols)
