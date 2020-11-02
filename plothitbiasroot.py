import ROOT

#fname = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgundetnoms/globalcor_0.root"
#fname = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgundetmoreprecisenoms/globalcor_0.root"
#fname = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgundetmoremoreprecisenoms/globalcor_0.root"
fname = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/globalcor_0.root"

f = ROOT.TFile.Open(fname)
tree = f.tree

hlxexp = -5.116925e-02

hlxplus = ROOT.TH1D("hlxplus","",100, hlxexp - 1e-3, hlxexp + 1e-3)
hlxminus = ROOT.TH1D("hlxminus","",100, -hlxexp - 1e-3, -hlxexp + 1e-3)

hlxplusexp = ROOT.TH1D("hlxplusexp","",100, -1e-3, 1e-3)
hlxminusexp = ROOT.TH1D("hlxminusexp","",100, -1e-3, 1e-3)

hlxplus.GetXaxis().SetTitle("Local x (cm)")
hlxminus.GetXaxis().SetTitle("Local x (cm)")

hlxplusexp.GetXaxis().SetTitle("SimHit Local x - expected (cm)")
hlxminusexp.GetXaxis().SetTitle("SimHit Local x - expected (cm)")

hlxplusexp.SetLineColor(ROOT.kBlue)
hlxminusexp.SetLineColor(ROOT.kRed)

tree.Draw("simlocalxref >> hlxplus","genCharge>0","goff")
tree.Draw("simlocalxref >> hlxminus","genCharge<0","goff")

tree.Draw("simlocalxref + genCharge*5.116925e-02 >> hlxplusexp","genCharge>0","goff")
tree.Draw("simlocalxref + genCharge*5.116925e-02 >> hlxminusexp","genCharge<0","goff")

lineplus = ROOT.TLine(hlxexp, 0., hlxexp, 1000.)
lineplus.SetLineStyle(9)
lineplus.SetLineWidth(2)

lineminus = ROOT.TLine(-hlxexp, 0., -hlxexp, 1000.)
lineminus.SetLineStyle(9)
lineminus.SetLineWidth(2)

lineexp = ROOT.TLine(0., 0., 0., 1000.)
lineexp.SetLineStyle(9)
lineexp.SetLineWidth(2)


legplus = ROOT.TLegend(0.55, 0.5, 0.9, 0.7)
legplus.SetBorderSize(0)
legplus.SetFillStyle(0)
legplus.AddEntry(hlxplus, "SimHit", "L")
legplus.AddEntry(lineplus, "Expected", "L")


legminus = ROOT.TLegend(0.12, 0.5, 0.45, 0.7)
legminus.SetBorderSize(0)
legminus.SetFillStyle(0)
legminus.AddEntry(hlxminus, "SimHit", "L")
legminus.AddEntry(lineminus, "Expected", "L")

legexp = ROOT.TLegend(0.13, 0.67, 0.35, 0.9)
legexp.SetBorderSize(0)
legexp.SetFillStyle(0)
legexp.AddEntry(hlxplusexp, "mu+", "L")
legexp.AddEntry(hlxminusexp, "mu-", "L")
legexp.AddEntry(lineexp, "Expected", "L")


c = ROOT.TCanvas()
hlxplus.Draw("HIST")
lineplus.Draw()
legplus.Draw()
c.SaveAs("mupluslocalx.pdf")

c2 = ROOT.TCanvas()
hlxminus.Draw("HIST")
lineminus.Draw()
legminus.Draw()
c2.SaveAs("muminuslocalx.pdf")

c3 = ROOT.TCanvas()
hlxplusexp.SetStats(False)
hlxplusexp.Draw("HIST")
hlxminusexp.Draw("HISTSAME")
lineexp.Draw()
legexp.Draw()
c3.SaveAs("explocalx.pdf")


#print(hlx.Integral())

input("wait")
