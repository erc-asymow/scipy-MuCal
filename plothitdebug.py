import ROOT
ROOT.gROOT.SetBatch(False)
ROOT.ROOT.EnableImplicitMT()

#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgradsdebug/globalcor_*.root"
#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgradsmoredebug/globalcor_*.root"
#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/gennonidealgradsdebug/globalcor_*.root"
#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/gennonidealgradsdebugqbins//globalcor_*.root"
#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgradsdebugqbins//globalcor_*.root"
#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgradsdebugqbinsmorefull//globalcor_*.root"
###indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/gennonidealgradsdebugnotemplate//globalcor_*.root"
#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgradsdebugnotemplate//globalcor_*.root"

#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/resultshitqualitycheck/genidealnoquality/globalcor_*.root"
#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/resultshitqualitycheck/genidealquality/globalcor_*.root"
#indata = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/resultshitqualitycheck/genidealquality/globalcor_3.root"
#indata = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Rec_quality/210212_165132/0000/globalcor_*.root"
#indata = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Gen_quality/210212_164507/0000/globalcor_*.root"
#indata = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Gen_noquality/210212_154221/0000/globalcor_*.root"
#indata = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Gen_idealnoquality/210212_164813/0000/globalcor_*.root"
indata = "/data/shared/muoncal/resultsqualitycomparefulldet/MuonGunUL2016_v31_Gen/210213_015119/0000/globalcor_*.root"

treename = "tree"
d = ROOT.ROOT.RDataFrame(treename,indata)

#dlight = d
#dlight = dlight.Define("dxhit","dxrecgen[0]")
#dlight = dlight.Define("dxhiterr","dxerr[0]")
#g = dlight.Graph("dxhiterr","dxhit")


#d = d.Filter("genPt>5.5 && genEta>=-2.4 && genEta < -2.3")
d = d.Filter("genPt>5.5")
d = d.Filter("nValidHits > 8")


d = d.Define("ihit","1")

#d = d.Filter("hitidxv[0] == 9")
d = d.Filter("clusterCharge[ihit] != -99")

#d = d.Filter("clusterOnEdge[0]==0")

d = d.Define("dxhit","dxrecgen[ihit]")

d = d.Define("dxhiterr","dxerr[ihit]")
d = d.Define("dyhiterr","dyerr[ihit]")

d = d.Define("qdxhiterr","genCharge*dxhiterr")
d = d.Define("localxhit","localx[ihit]")
d = d.Define("localyhit","localy[ihit]")
d = d.Define("dxdzhit","localdxdz[ihit]")
d = d.Define("dydzhit","localdydz[ihit]")

d = d.Define("probxyHit","clusterProbXY[ihit]")

d = d.Define("qbinhit","clusterChargeBin[ihit]")
#d = d.Define("qbinhit","1.0")

d = d.Define("clusterSizeHit","clusterSize[ihit]")
d = d.Define("clusterSizeXHit","clusterSizeX[ihit]")
d = d.Define("clusterSizeYHit","clusterSizeY[ihit]")
d = d.Define("clusterChargeHit","clusterCharge[ihit]")
d = d.Define("onEdgeHit","clusterOnEdge[ihit]")
d = d.Define("thetaHit","asin(1./sqrt(dxdzhit*dxdzhit + dydzhit*dydzhit + 1.))")
d = d.Define("snstrip","clusterSN[clusterSN>-99. && clusterSN<30.]")


#d = d.Filter("thetaHit<0.5")

#d = d.Filter("qbinhit>0 && qbinhit<3")
#d = d.Filter("qbinhit<3")
#d = d.Filter("qbinhit < 4")
d = d.Filter("probxyHit > 0.000125 && probxyHit<1.5")
#d = d.Filter("clusterChargeHit < 50e3")

g = d.Graph("dxhiterr","dxhit")

#g = d.Graph("qbinhit", "dxhit")
g2 = d.Graph("dxhiterr","qbinhit")
g3 = d.Graph("dxhiterr","localxhit")
g4 = d.Graph("dxhiterr","dxdzhit")
g5 = d.Graph("dyhiterr","dydzhit")
g6 = d.Graph("dxdzhit", "dydzhit")
g7 = d.Graph("dxhiterr", "dyhiterr")
g9 = d.Graph("clusterChargeHit","qbinhit")
g10 = d.Graph("clusterChargeHit","onEdgeHit")

h4 = d.Histo1D(("h4","",30,0.,500e3),"clusterChargeHit")
h5 = d.Histo1D("thetaHit")
h6 = d.Histo1D("probxyHit")


h8 = d.Histo1D("snstrip")
g11 = d.Graph("genEta", "thetaHit")

dprof = d.Filter("abs(dxhit)<0.005")
dprof = d
t1 = dprof.Profile1D(("t1","",6,-0.5,5.5,"s"), "qbinhit","dxhit")
t2 = dprof.Profile1D(("t2","",6,-0.5,5.5), "qbinhit","dxhit")
t3 = dprof.Profile1D(("t3","",30,0,500e3),"clusterChargeHit","dxhit")

g8 = d.Graph("dxhiterr", "clusterChargeHit")

h2 = d.Histo1D("dxhiterr")
h3 = d.Histo1D("clusterChargeHit")
#p = d.Profile2D("dxhit","dxhiterr")

d = d.Filter("abs(dxhit)<0.01")
h1 = d.Histo1D("dxhit")
dhigh = d.Filter("dxhiterr >= 0.0012")
dlow = d.Filter("dxhiterr < 0.0012")

hhigh = dhigh.Histo1D("dxhit")
hlow = dlow.Histo1D("dxhit")

c = ROOT.TCanvas()
g.Draw("AP")

c2 = ROOT.TCanvas()
h1.Draw()

c3 = ROOT.TCanvas()
h2.Draw()

#c4 = ROOT.TCanvas()
#hhigh.Draw()

#c5 = ROOT.TCanvas()
#hlow.Draw()

c6 = ROOT.TCanvas()
g2.Draw("AP")

c7 = ROOT.TCanvas()
g3.Draw("AP")

c8 = ROOT.TCanvas()
g4.Draw("AP")

c9 = ROOT.TCanvas()
g5.Draw("AP")

c10 = ROOT.TCanvas()
g6.Draw("AP")

c11 = ROOT.TCanvas()
g7.Draw("AP")

c12 = ROOT.TCanvas()
g8.Draw("AP")

c13 = ROOT.TCanvas()
h3.Draw()

c14 = ROOT.TCanvas()
t1.Draw()

c15 = ROOT.TCanvas()
t2.Draw()
t2.GetXaxis().SetTitle("Template Charge Bin")
t2.GetYaxis().SetTitle("Bias (cm)")



c17 = ROOT.TCanvas()
g9.Draw("AP")

c16 = ROOT.TCanvas()
t3.GetXaxis().SetTitle("Cluster Charge (a.u.)")
t3.GetYaxis().SetTitle("Mean Bias (cm)")
t3.Draw()

c18 = ROOT.TCanvas()
h4.GetXaxis().SetTitle("Cluster Charge (a.u.)")
h4.GetYaxis().SetTitle("# of hits/bin")
h4.Draw()

c19 = ROOT.TCanvas()
g10.Draw("AP")

c20 = ROOT.TCanvas()
h5.Draw()

c21 = ROOT.TCanvas()
h6.Draw()
#p.Draw()

c22 = ROOT.TCanvas()
h8.Draw()

c23 = ROOT.TCanvas()
g11.Draw()

#input("wait")
