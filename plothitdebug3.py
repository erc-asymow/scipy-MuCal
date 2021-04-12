import math
import ROOT

from utils import lumitools

ROOT.gInterpreter.ProcessLine(".O3")

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(False)
ROOT.ROOT.EnableImplicitMT()

lumitools.init_lumitools()

jsonhelper = lumitools.make_jsonhelper("data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")

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
#indata = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/SingleMuon/MuonGunUL2016_v36_RecData_quality/210404_001131/0000/globalcor_*.root"
#indata = "/data/shared/muoncal/MuonGunUL2016_v36_RecData_quality/210404_001131/0000/globalcor_*.root"
#indata = "/data/shared/muoncal/MuonGunUL2016_v36plus_Rec_noquality/210404_025430/0000/globalcor_*.root"
#indata = "/data/shared/muoncal/MuonGunUL2016_v42_RecDataMuIsoH_noquality/210405_185116/0000/globalcor_*.root"
indata = "/data/shared/muoncal/MuonGunUL2016_v42_Rec_noquality/210405_185305/0000/globalcor_*.root"
#indata = "/data/shared/muoncal/MuonGunUL2016_v41_Rec_noquality/210405_115619/0000/globalcor_*.root"
#indata = "/data/shared/muoncal/resultsqualitycomparefulldet/MuonGunUL2016_v31_Gen/210213_015119/0000/globalcor_*.root"

treename = "tree"
d = ROOT.ROOT.RDataFrame(treename,indata)

d = d.Filter(jsonhelper, ["run", "lumi"], "jsonfilter")

#dlight = d
#dlight = dlight.Define("dxhit","dxrecgen[0]")
#dlight = dlight.Define("dxhiterr","dxerr[0]")
#g = dlight.Graph("dxhiterr","dxhit")


#d = d.Filter("genPt>5.5 && genEta>=-2.4 && genEta < -2.3")
#d = d.Filter("genPt>5.5 && genEta>=2.3 && genEta < 2.4")
#d = d.Filter("genPt>5.5 && abs(genEta)<1.5")
#d = d.Filter("genPt>5.5 && abs(genEta) < 2.4 && genEta<0.")
#d = d.Filter("genPt>5.5")
d = d.Define("refPt", "std::abs(1./refParms[0])*std::sin(M_PI_2 - refParms[1])")
d = d.Define("refEta", "-std::log(std::tan(0.5*(M_PI_2 - refParms[1])))")

#d = d.Filter("refPt > 5.5 && refEta > -2.4 && refEta < -2.3")
d = d.Filter("refPt > 26. && refEta > -2.4 && refEta < -2.3")
#d = d.Filter("trackPt > 5.5 && trackEta > -2.4 && trackEta < -2.3")
#d = d.Filter("genPt>5.5")
d = d.Filter("nValidHits > 8")
#d = d.Filter("abs(genEta)<0.5")


d = d.Define("ihit","0")

d = d.Filter("hitidxv[0] == 9")
#d = d.Filter("hitidxv[0] == 3") #central
d = d.Filter("clusterCharge[ihit] != -99")

#d = d.Filter("clusterOnEdge[0]==0")

d = d.Define("dxhit","dxrecgen[ihit] - deigx[ihit]")
#d = d.Define("dxhit","dxrecgen[ihit] - dlocalx[ihit]")
#d = d.Define("dxhit","dxrecgen[ihit]")
#d = d.Define("dxhit","dxrecsim[ihit]")
d = d.Define("dxhitum", "dxhit/1e-4")


#d = d.Define("dyhit","dyrecsim[ihit]")
d = d.Define("dyhit","dyrecgen[ihit]")
d = d.Define("dyhitum", "dyhit/1e-4")

d = d.Define("dxhiterr","dxerr[ihit]")
d = d.Define("dyhiterr","dyerr[ihit]")

d = d.Define("dxhiterrum","dxhiterr/1e-4")

#d = d.Filter("dxhiterrum > 20.")

d = d.Define("dxhitpull","dxhit/dxhiterr")

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
d = d.Define("corClusterChargeHit","clusterChargeHit*(1./sqrt(dxdzhit*dxdzhit + dydzhit*dydzhit + 1.))")
d = d.Define("snstrip","clusterSN[clusterSN>-99. && clusterSN<30.]")

#full: 1.743e8
#after clusterSizeXHit > 1 && !onEdgeHit: 1.344e8

#d = d.Filter("clusterSizeXHit > 1")
#d = d.Filter("clusterSizeXHit == 1")
#d = d.Filter("clusterSizeHit == 1")
#d = d.Filter("clusterSizeYHit > 1")
#d = d.Filter("!onEdgeHit")
#d = d.Filter("onEdgeHit")
dnoedge = d.Filter("!onEdgeHit")
dgood = dnoedge.Filter("clusterSizeXHit > 1")

dedge = d.Filter("onEdgeHit")
done = dnoedge.Filter("clusterSizeXHit == 1")

#d = d.Filter("abs(dxdzhit) < 0.15")
#d = d.Filter("abs(dxhiterrum-24.)>0.01 && abs(dxhiterrum-31.)>0.01")
#d = d.Filter("dxdzhit < 0.")
#d = d.Filter("dxdzhit < -0.15")
#d = d.Filter("dxdzhit > 0.15")
##d = d.Filter("dxhiterrum > 11.5")
#d = d.Filter("dxhiterrum < 40.")
#d = d.Filter("clusterChargeHit < 100e3")

#d = d.Filter("thetaHit<0.5")
#d = d.Filter("thetaHit>0.35")
#d = d.Filter("thetaHit>0.17")
#d = d.Filter("qbinhit>0 && qbinhit<3")
#d = d.Filter("qbinhit>0 && qbinhit<4")
#d = d.Filter("corClusterChargeHit > 10e3")
#d = d.Filter("corClusterChargeHit > 15e3")
#d = d.Filter("qbinhit < 3")
#d = d.Filter("qbinhit < 4")
#d = d.Filter("probxyHit > 0.000125 && probxyHit<1.5")
#d = d.Filter("clusterChargeHit < 50e3")

#dpxb9 = d.Filter("hitidxv[0] == 9")
dpxb9 = d

h = d.Histo2D(("h","",50,-3.0,3.0,100,0.,math.pi/2.),"genEta", "thetaHit")
h2 = d.Histo1D(("h2","",100,-2.5,2.5), "genEta")
h3 = d.Histo1D("clusterChargeHit")
h4 = d.Histo1D(("h4","",6, -0.5,5.5), "qbinhit")
h5 = d.Histo1D("corClusterChargeHit")
h6 = d.Histo1D(("h6","",100,-0.01,0.01),"dxhit")
h7 = d.Histo1D(("h7","",100,-5.,5.),"dxhitpull")
h8 = d.Histo1D(("h8","",500,0.,math.pi/2.), "thetaHit")

detaminus = d.Filter("genEta<0.")
detaplus = d.Filter("genEta>=0.")

h9 = detaminus.Histo1D(("h9","",500,0.,math.pi/2.), "thetaHit")
h10 = detaplus.Histo1D(("h10","",500,0.,math.pi/2.), "thetaHit")
#h11 = 


#h11 = d.Histo2D(("h11",

#g = d.Graph("dxhiterr","dxhit")

##g = d.Graph("qbinhit", "dxhit")
#g2 = d.Graph("dxhiterr","qbinhit")
#g3 = d.Graph("dxhiterr","localxhit")
#g4 = d.Graph("dxhiterr","dxdzhit")
#g5 = d.Graph("dyhiterr","dydzhit")
#g6 = d.Graph("dxdzhit", "dydzhit")
#g7 = d.Graph("dxhiterr", "dyhiterr")
#g9 = d.Graph("clusterChargeHit","qbinhit")
#g10 = d.Graph("clusterChargeHit","onEdgeHit")

#h4 = d.Histo1D(("h4","",30,0.,500e3),"clusterChargeHit")
#h5 = d.Histo1D("thetaHit")
#h6 = d.Histo1D("probxyHit")


#h8 = d.Histo1D("snstrip")
#g11 = d.Graph("genEta", "thetaHit")

#dprof = d.Filter("abs(dxhit)<0.005")
#dprof = d
dprof = dpxb9
#t1 = dprof.Profile1D(("t1","",6,-0.5,5.5,"s"), "qbinhit","dxhit")
t2 = dprof.Profile1D(("t2","",6,-0.5,5.5), "qbinhit","dxhitum")
t3 = dprof.Profile1D(("t3","",30,0,500e3),"clusterChargeHit","dxhitum")
#t3 = dprof.Profile1D(("t3","",30,0,100e3),"corClusterChargeHit","dxhitum")
#t3 = dprof.Profile1D(("t3","",30,0,500e3),"clusterChargeHit","dxhit")
t4 = dprof.Profile1D(("t4","",30,0.,math.pi/2.),"thetaHit","dxhitum")
t5 = dprof.Profile1D(("t5","",30,0,50.),"dxhiterrum","dxhitum")
t5a = dnoedge.Profile1D(("t5a","",30,0,50.),"dxhiterrum","dxhitum")
t5b = dgood.Profile1D(("t5b","",30,0,50.),"dxhiterrum","dxhitum")

t6 = dprof.Profile1D(("t6","",21,-0.5,20.5),"clusterSizeXHit","dxhitum")
t7 = dprof.Profile1D(("t7","",21,-0.5,20.5),"clusterSizeYHit","dyhitum")

h11 = dprof.Histo2D(("h11","",30,0.,400e3, 30,0,50.),"clusterChargeHit","dxhiterrum")
h12 = dprof.Histo2D(("h12","",6,-0.5,5.5, 30,0,50.),"qbinhit","dxhiterrum")
#g = dprof.Graph("dxdzhit","dxhiterrum")
g = dprof.Graph("clusterSizeYHit","dxhiterrum")
#h11 = dprof.Histo2D(("h11","",30,0.,100e3, 30,0,50.),"corClusterChargeHit","dxhiterrum")
h13 = dprof.Histo1D("dxhiterrum")

h14 = dprof.Histo2D(("h14","",51,-0.5,50.5,30,0.,50.),"clusterSizeHit","dxhiterrum")

h15 = dprof.Histo1D("dxdzhit");
h16 = dprof.Histo1D("dydzhit");
h17 = dprof.Histo1D(("h17","",21,-0.5,20.5),"clusterSizeXHit")
h18 = dprof.Histo1D(("h18","",21,-0.5,20.5),"clusterSizeYHit")

h19 = d.Histo1D(("h19","",100,-100.,100.),"dxhitum")
h19a = dedge.Histo1D(("h19a","",100,-100.,100.),"dxhitum")
h19b = done.Histo1D(("h19b","",100,-100.,100.),"dxhitum")
h19c = dgood.Histo1D(("h19c","",100,-100.,100.),"dxhitum")


d = d.Define("weight","1./dxhiterr/dxhiterr")
d = d.Define("wdx","weight*dxhit");

num = d.Sum("wdx")
den = d.Sum("weight")

#print("cordx", num.GetValue()/den.GetValue())


c = ROOT.TCanvas()
h.GetXaxis().SetTitle("gen #eta")
h.GetYaxis().SetTitle("First Hit Incidence Angle (radians)")
h.Draw("COLZ")
c.SaveAs("angleeta.pdf")

c2 = ROOT.TCanvas()
h2.Draw("COLZ")

c3 = ROOT.TCanvas()
h3.Draw()

c4 = ROOT.TCanvas()
h4.Scale(1./h4.Integral())
h4.Draw()

c5 = ROOT.TCanvas()
h5.Draw()


c6 = ROOT.TCanvas()
h6.Draw()


c7 = ROOT.TCanvas()
h7.Draw()

c8 = ROOT.TCanvas()
h8.Draw()

c9 = ROOT.TCanvas()
h9.SetLineColor(ROOT.kRed)
h10.SetLineColor(ROOT.kBlue)
h9.GetXaxis().SetTitle("First Hit Incidence angle (radians)")
h9.GetYaxis().SetTitle("# of hits/bin")
h9.Draw()
h10.Draw("HISTSAME")

leg = ROOT.TLegend(0.6,0.6,0.8,0.8)
leg.SetFillStyle(0)
leg.SetBorderSize(0)
leg.AddEntry(h9.GetValue(), "gen #eta < 0", "L")
leg.AddEntry(h10.GetValue(), "gen #eta > 0", "L")
leg.Draw()

c9.SaveAs("angleeta1d.pdf")

#g8 = d.Graph("dxhiterr", "clusterChargeHit")

#h2 = d.Histo1D("dxhiterr")
#h3 = d.Histo1D("clusterChargeHit")
##p = d.Profile2D("dxhit","dxhiterr")

#d = d.Filter("abs(dxhit)<0.01")
#h1 = d.Histo1D("dxhit")
#dhigh = d.Filter("dxhiterr >= 0.0012")
#dlow = d.Filter("dxhiterr < 0.0012")

#hhigh = dhigh.Histo1D("dxhit")
#hlow = dlow.Histo1D("dxhit")

#c = ROOT.TCanvas()
#g.Draw("AP")

#c2 = ROOT.TCanvas()
#h1.Draw()

#c3 = ROOT.TCanvas()
#h2.Draw()

##c4 = ROOT.TCanvas()
##hhigh.Draw()

##c5 = ROOT.TCanvas()
##hlow.Draw()

#c6 = ROOT.TCanvas()
#g2.Draw("AP")

#c7 = ROOT.TCanvas()
#g3.Draw("AP")

#c8 = ROOT.TCanvas()
#g4.Draw("AP")

#c9 = ROOT.TCanvas()
#g5.Draw("AP")

#c10 = ROOT.TCanvas()
#g6.Draw("AP")

#c11 = ROOT.TCanvas()
#g7.Draw("AP")

#c12 = ROOT.TCanvas()
#g8.Draw("AP")

#c13 = ROOT.TCanvas()
#h3.Draw()

#c14 = ROOT.TCanvas()
#t1.Draw()

c15 = ROOT.TCanvas()
t2.Draw()
t2.GetXaxis().SetTitle("Template Charge Bin")
t2.GetYaxis().SetTitle("Bias (cm)")
c15.SaveAs("biaschargebin.pdf")


#c17 = ROOT.TCanvas()
#g9.Draw("AP")

c16 = ROOT.TCanvas()
t3.GetXaxis().SetTitle("Cluster Charge (a.u.)")
t3.GetYaxis().SetTitle("Mean Bias (#mum)")
t3.Draw()
t3.GetYaxis().SetRangeUser(-20.,20.)
c16.SaveAs("biascharge.pdf")

c16a = ROOT.TCanvas()
t4.Draw()

ct5 = ROOT.TCanvas()
t5.GetXaxis().SetTitle("Hit Uncertainty (#mum)")
t5.GetYaxis().SetTitle("Mean Bias (#mum)")
t5.Draw()
#t5a.SetLineColor(ROOT.kRed)
#t5a.SetMarkerColor(ROOT.kRed)
#t5b.SetLineColor(ROOT.kMagenta)
#t5b.SetMarkerColor(ROOT.kMagenta)
#t5a.Draw("SAME")
#t5b.Draw("SAME")
ct5.SaveAs("biashiterr.pdf")


ct5a = ROOT.TCanvas()
t5a.GetXaxis().SetTitle("Hit Uncertainty (#mum)")
t5a.GetYaxis().SetTitle("Mean Bias (#mum)")
t5a.Draw()
ct5a.SaveAs("biashiterrnoedge.pdf")

ct5b = ROOT.TCanvas()
t5b.GetXaxis().SetTitle("Hit Uncertainty (#mum)")
t5b.GetYaxis().SetTitle("Mean Bias (#mum)")
t5b.Draw()
ct5b.SaveAs("biashiterrgood.pdf")


c16c = ROOT.TCanvas()
h11.GetXaxis().SetTitle("Cluster Charge (a.u.)")
h11.GetYaxis().SetTitle("Hit Uncertainty (#mum)")
h11.Draw("COLZ")
c16c.SaveAs("hiterrcharge.pdf")

c17 = ROOT.TCanvas()
h12.Draw("COLZ")

c8 = ROOT.TCanvas()
g.Draw("AP")

c9 = ROOT.TCanvas()
h13.Draw()


c10 = ROOT.TCanvas()
h14.Draw("COLZ")

c11 = ROOT.TCanvas()
h15.Draw()

c12 = ROOT.TCanvas()
h16.Draw()

c13 = ROOT.TCanvas()
h17.Draw()

c14 = ROOT.TCanvas()
h18.Draw()

c15 = ROOT.TCanvas()
t6.Draw()

c16 = ROOT.TCanvas()
t7.Draw()

c17 = ROOT.TCanvas()
h19a.SetLineColor(ROOT.kRed)
h19b.SetLineColor(8)
h19c.SetLineColor(ROOT.kMagenta)

h19.Scale(1./h19.Integral())
h19a.Scale(1./h19a.Integral())
h19b.Scale(1./h19b.Integral())
h19c.Scale(1./h19c.Integral())

h19c.GetXaxis().SetTitle("local x Reco-Gen (#mum)")
h19c.GetYaxis().SetTitle("Fraction of hits per bin")

h19c.Draw("HIST")
h19a.Draw("HISTSAME")
h19b.Draw("HISTSAME")
h19.Draw("HISTSAME")

legdx = ROOT.TLegend(0.6, 0.6, 0.86, 0.86)
legdx.SetFillStyle(0)
legdx.SetBorderSize(0)
legdx.AddEntry(h19.GetValue(), "All Hits", "L")
legdx.AddEntry(h19a.GetValue(), "Module Edge", "L")
legdx.AddEntry(h19b.GetValue(), "One Pixel Width", "L")
legdx.AddEntry(h19c.GetValue(), "Other", "L")
legdx.Draw()

c17.SaveAs("dxhit.pdf")

#c18 = ROOT.TCanvas()
#h4.GetXaxis().SetTitle("Cluster Charge (a.u.)")
#h4.GetYaxis().SetTitle("# of hits/bin")
#h4.Draw()

#c19 = ROOT.TCanvas()
#g10.Draw("AP")

#c20 = ROOT.TCanvas()
#h5.Draw()

#c21 = ROOT.TCanvas()
#h6.Draw()
##p.Draw()

#c22 = ROOT.TCanvas()
#h8.Draw()

#c23 = ROOT.TCanvas()
#g11.Draw()

#input("wait")
