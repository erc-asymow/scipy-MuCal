import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
ROOT.ROOT.EnableImplicitMT()

ROOT.gInterpreter.AddIncludePath("/usr/include/eigen3/")
ROOT.gSystem.CompileMacro("CVHCorrector.cc")

chain = ROOT.TChain("Events")
chain.Add("root://eoscms//store/cmst3/group/wmass/w-mass-13TeV/muonscaleNanoAOD/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/NanoV8MCPostVFP/211113_024025/0000/NanoV8MCPostVFP_weightFix_379.root")

d = ROOT.ROOT.RDataFrame(chain)

#split the nested vectors
d = d.Define("Muon_cvhGlobalIdxs", "splitNestedRVec(Muon_cvhGlobalIdxs_Vals, Muon_cvhGlobalIdxs_Counts)")

d = d.Define("Muon_cvhJacRef", "splitNestedRVec(Muon_cvhJacRef_Vals, Muon_cvhJacRef_Counts)")

d = d.Define("Muon_cvhbsJacRef", "splitNestedRVec(Muon_cvhbsJacRef_Vals, Muon_cvhbsJacRef_Counts)")

#most recent layer by layer corrections including background subtraction and corresponding bin selection

cormc = "root://eoscms//store/cmst3/group/wmass/bendavid/muoncalreduced/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintfsr29_cor/correctionResults.root"

cordata = "root://eoscms//store/cmst3/group/wmass/bendavid/muoncalreduced/MuonGunUL2016_v207_RecDataJPsiFpostGH_quality_constraintfsr29_cor/correctionResults.root"

# correction helpers
correctorsingle = ROOT.CVHCorrectorSingle(cormc)
corrector = ROOT.CVHCorrector(correctorsingle)


#correct muon collection

#without beamspot constraint
d = d.Define("Muon_p4WithChargeCor", corrector, ["Muon_cvhPt", "Muon_cvhEta", "Muon_cvhPhi", "Muon_mass", "Muon_cvhCharge", "Muon_cvhGlobalIdxs", "Muon_cvhJacRef"])

#with beamspot constraint
d = d.Define("Muon_p4WithChargeCorbs", corrector, ["Muon_cvhbsPt", "Muon_cvhbsEta", "Muon_cvhbsPhi", "Muon_mass", "Muon_cvhbsCharge", "Muon_cvhGlobalIdxs", "Muon_cvhbsJacRef"])




# Alternative: correct single muons
# at least one muon

d = d.Filter("nMuon > 0")

d = d.Filter("Muon_pt[0] > 15.")

# muon track refits succeeded

d = d.Filter("Muon_cvhPt[0] > 0. && Muon_cvhbsPt[0] > 0.")



#correct first muon

d = d.Define("Muon0_cvhPt", "Muon_cvhPt[0]")
d = d.Define("Muon0_cvhEta", "Muon_cvhEta[0]")
d = d.Define("Muon0_cvhPhi", "Muon_cvhPhi[0]")
d = d.Define("Muon0_mass", "Muon_mass[0]")
d = d.Define("Muon0_cvhCharge", "Muon_cvhCharge[0]")

d = d.Define("Muon0_cvhbsPt", "Muon_cvhbsPt[0]")
d = d.Define("Muon0_cvhbsEta", "Muon_cvhbsEta[0]")
d = d.Define("Muon0_cvhbsPhi", "Muon_cvhbsPhi[0]")
d = d.Define("Muon0_cvhbsCharge", "Muon_cvhbsCharge[0]")

d = d.Define("Muon0_cvhGlobalIdxs", "Muon_cvhGlobalIdxs[0]")
d = d.Define("Muon0_cvhJacRef", "Muon_cvhJacRef[0]")
d = d.Define("Muon0_cvhbsJacRef", "Muon_cvhbsJacRef[0]")

#without beamspot constraint
d = d.Define("Muon0_p4WithChargeCor", correctorsingle, ["Muon0_cvhPt", "Muon0_cvhEta", "Muon0_cvhPhi", "Muon0_mass", "Muon0_cvhCharge", "Muon0_cvhGlobalIdxs", "Muon0_cvhJacRef"])

#with beamspot constraint
d = d.Define("Muon0_p4WithChargeCorbs", correctorsingle, ["Muon0_cvhbsPt", "Muon0_cvhbsEta", "Muon0_cvhbsPhi", "Muon0_mass", "Muon0_cvhbsCharge", "Muon0_cvhGlobalIdxs", "Muon0_cvhbsJacRef"])


d = d.Define("corpt0", "Muon0_p4WithChargeCor.first.Pt()")

d = d.Define("corptratio", "corpt0/Muon0_cvhPt")

h = d.Histo1D(("h", "", 100, 0.9, 1.1), "corptratio")

c = ROOT.TCanvas()
h.Draw()
