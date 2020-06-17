import ROOT
import numpy as np

ROOT.ROOT.EnableImplicitMT()

#ROOT.gStyle.SetOptStat(111100)


f = ROOT.TFile.Open("/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muonGuntree.root")

tree = f.Get("tree")

#hdphiplus = ROOT.TH1D("hdphiplus","",100,-0.01,0.01)
#hdphiminus = ROOT.TH1D("hdphiminus","",100,-0.01,0.01)


#hdthetaplus = ROOT.TH1D("hdthetaplus","",100,-0.1,0.1)
#hdthetaminus = ROOT.TH1D("hdthetaminus","",100,-0.1,0.1)


#tree.Draw("2.*atan(exp(-reco_eta)) - 2.*atan(exp(-gen_eta)) >>hdthetaplus","1./gen_curv>100. && gen_eta>-2.4 && gen_eta<-2.2 && gen_charge>0","goff")

#tree.Draw("2.*atan(exp(-reco_eta)) - 2.*atan(exp(-gen_eta)) >>hdthetaminus","1./gen_curv>100. && gen_eta>-2.4 && gen_eta<-2.2 && gen_charge<0","goff")

#tree.Draw("ROOT::Math::VectorUtil::Phi_mpi_pi(reco_phi-gen_phi)>>hdphiplus","1./gen_curv>100. && gen_eta>-2.4 && gen_eta<-2.2 && gen_charge>0","goff")
#tree.Draw("ROOT::Math::VectorUtil::Phi_mpi_pi(reco_phi-gen_phi)>>hdphiminus","1./gen_curv>100. && gen_eta>-2.4 && gen_eta<-2.2 && gen_charge<0","goff")


#treename = "tree"
#fname = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muonGuntree.root"

#d = ROOT.ROOT.RDataFrame(treename, fname)

#d = d.Filter("1./gen_curv>100. && gen_eta>-2.3 && gen_eta<-2.1 && reco_charge>-2.")
##d = d.Filter("1./gen_curv>120. && gen_eta>-2.4 && gen_eta>0.1 && gen_eta<0.2")
##d = d.Filter("1./gen_curv>120. && gen_eta>-2.4 && gen_eta>0.1 && gen_eta<0.2")
##d = d.Filter("1./gen_curv>5.5 && 1./gen_curv<10. && gen_eta>-2.4 && gen_eta<-2.2 && reco_charge>-2.")
#d = d.Define("dphi","ROOT::Math::VectorUtil::Phi_mpi_pi(reco_phi-gen_phi)")
#d = d.Define("dtheta","2.*atan(exp(-reco_eta)) - 2.*atan(exp(-gen_eta))")

#dplus = d.Filter("gen_charge>0.")
#dminus = d.Filter("gen_charge<0.")


treename = "tree"
fname = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/trackTreeP.root"

d = ROOT.ROOT.RDataFrame(treename, fname)

#d = d.Filter("1./gen_curv>120. && gen_eta>-2.4 && gen_eta<-2.2 && reco_charge>-2.")
#d = d.Filter("1./gen_curv>120. && gen_eta>-2.4 && gen_eta>0.1 && gen_eta<0.2")
#d = d.Filter("1./gen_curv>120. && gen_eta>-2.4 && gen_eta>0.1 && gen_eta<0.2")
d = d.Filter("genPt>100.")
d = d.Define("dphi","ROOT::Math::VectorUtil::Phi_mpi_pi(trackPhi-genPhi)")
d = d.Define("dtheta","2.*atan(exp(-trackEta)) - 2.*atan(exp(-genEta))")

dplus = d.Filter("genCharge>0.")
dminus = d.Filter("genCharge<0.")


hdphi = d.Histo1D(("hdphi","",100,-0.001,0.001), "dphi")
hdphiplus = dplus.Histo1D(("hdphiplus","",100,-0.001,0.001), "dphi")
hdphiminus = dminus.Histo1D(("hdphiminus","",100,-0.001,0.001), "dphi")

hdthetaplus = dplus.Histo1D(("hdthetaplus","",100,-5e-4,5e-4), "dtheta")
hdthetaminus = dminus.Histo1D(("hdthetaminus","",100,-5e-4,5e-4), "dtheta")

#hdthetaplus = dplus.Histo1D( "dtheta")
#hdthetaminus = dminus.Histo1D( "dtheta")


#tree.Draw("dphi2>>hdphiminus","pt2>100. && eta2>-2.4 && eta2<-2.2","goff")

#tree.Draw("dtheta1>>hdthetaplus","pt1>100. && eta1>-2.4 && eta1<-2.2","goff")
#tree.Draw("dtheta2>>hdthetaminus","pt2>100. && eta2>-2.4 && eta2<-2.2","goff")


hdphi.GetXaxis().SetTitle("#phi_{reco} - #phi_{gen}")
c3 = ROOT.TCanvas()
hdphi.Draw()

hdphiminus.SetLineColor(ROOT.kRed)
hdphiplus.GetXaxis().SetTitle("#phi_{reco} - #phi_{gen}")



c=ROOT.TCanvas()
hdphiplus.Draw("HIST")
hdphiminus.Draw("HISTSAME")
c.Update()

hdthetaminus.SetLineColor(ROOT.kRed)

c2=ROOT.TCanvas()
hdthetaplus.Draw("HIST")
hdthetaminus.Draw("HISTSAME")
c.Update()


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
