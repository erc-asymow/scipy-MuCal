import ROOT
import numpy as np

#ROOT.gStyle.SetOptStat(111100)


f = ROOT.TFile.Open("/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/ZJToMuMu_mWPilot.root")

tree = f.Get("tree")

hdxyplus = ROOT.TH1D("hdxyplus","",100,-0.01,0.01)
hdxyminus = ROOT.TH1D("hdxyminus","",100,-0.01,0.01)


hdzplus = ROOT.TH1D("hdzplus","",100,-0.1,0.1)
hdzminus = ROOT.TH1D("hdzminus","",100,-0.1,0.1)

tree.Draw("dxy1>>hdxyplus","pt1>100. && eta1>-2.4 && eta1<-2.2","goff")
tree.Draw("dxy2>>hdxyminus","pt2>100. && eta2>-2.4 && eta2<-2.2","goff")

tree.Draw("dz1>>hdzplus","pt1>100. && eta1>-2.4 && eta1<-2.2","goff")
tree.Draw("dz2>>hdzminus","pt2>100. && eta2>-2.4 && eta2<-2.2","goff")

hdxyminus.SetLineColor(ROOT.kRed)

c=ROOT.TCanvas()
hdxyplus.Draw("HIST")
hdxyminus.Draw("HISTSAME")

hdzminus.SetLineColor(ROOT.kRed)

c2=ROOT.TCanvas()
hdzplus.Draw("HIST")
hdzminus.Draw("HISTSAME")


#resxlast = ROOT.TH1D("resx","",100,-200.,200.)

#for track in tree:
    #nhits = len(track.simx)
    #resxlast.Fill(track.localx[nhits-1] - track.simx[nhits-1])
    ##print("nhits",nhits)
    ##for ihit in range(nhits):
        ##print("simx", track.simx[ihit])


##gc = []

#c = ROOT.TCanvas
#resxlast.Draw("E")


input("wait")
