import ROOT



#fmc = ROOT.TFile.Open("calmc.root")
#fdata = ROOT.TFile.Open("calibrationDATA.root")


fnamemc = "calmc.root"
fnamedata = "calibrationDATA.root"

filenames = [fnamemc, fnamedata]

gc = []

#for filename in filenames:
filename = fnamemc
f = ROOT.TFile.Open(filename)
scaleBinned = f.Get("scaleBinned")
scaleModel = f.Get("scaleModel")

scaleModel.SetLineColor(ROOT.kRed)

gc.append(scaleBinned)
gc.append(scaleModel)

c = ROOT.TCanvas()
gc.append(c)
scaleModel.Draw("HIST")
scaleBinned.Draw("ESAME")
    
