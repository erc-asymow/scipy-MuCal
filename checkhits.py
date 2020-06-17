import ROOT
import numpy as np

ROOT.gStyle.SetOptStat(111100)


f1 = ROOT.TFile.Open("/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/trackTreeNoSmearN.root")
f2 = ROOT.TFile.Open("/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/trackTreeNoSmearP.root")

fs = [f1,f2]
#fs = [f1]

hists = []

for iq,f in enumerate(fs):
    tree = f.Get("tree")

    resxfirst = ROOT.TH1D(f"resxfirst_{iq}","",100,-0.005,0.005)
    resyfirst = ROOT.TH1D(f"resyfirst_{iq}","",100,-0.005,0.005)
    reszfirst = ROOT.TH1D(f"reszfirst_{iq}","",100,-0.005,0.005)

    resxsecond = ROOT.TH1D(f"resxsecond_{iq}","",100,-0.005,0.005)
    resysecond = ROOT.TH1D(f"resysecond_{iq}","",100,-0.005,0.005)
    reszsecond = ROOT.TH1D(f"reszsecond_{iq}","",100,-0.005,0.005)


    resxlast = ROOT.TH1D(f"resxlast_{iq}","",100,-0.1,0.1)
    resylast = ROOT.TH1D(f"resylast_{iq}","",100,-10.,10.)
    reszlast = ROOT.TH1D(f"reszlast_{iq}","",100,-0.05,0.05)
    
    simytec = ROOT.TH1D(f"simytec_{iq}","",100,-0.0002,0.0002)

    for track in tree:
        #if track.genPt<140.:
            #continue
        nhits = len(track.simx)
        
        #print("detector layer stereo 0", track.detector[0],track.layer[0],track.stereo[0])
        ##print("layer n1", track.layer[nhits-1])
        
        #print("detector layer stereo n1", track.detector[nhits-1],track.layer[nhits-1],track.stereo[nhits-1])
        #print("detector layer stereo n2", track.detector[nhits-2],track.layer[nhits-2],track.stereo[nhits-2])

        #print("stereo 0", track.stereo[0])
        
        for ihit in range(nhits):
            det = track.detector[ihit]
            layer = track.layer[ihit]
            stereo = track.stereo[ihit]
            
            if det==0 and layer==1 and stereo==0:
                #print("found pxb1")
                resxfirst.Fill(track.localx[ihit] - track.simx[ihit])
                resyfirst.Fill(track.localy[ihit] - track.simy[ihit])
                reszfirst.Fill(track.localz[ihit] - track.simz[ihit])
                
            elif det==1 and layer==-1 and stereo==0:
                #print("found pxd1")
                resxsecond.Fill(track.localx[ihit] - track.simx[ihit])
                resysecond.Fill(track.localy[ihit] - track.simy[ihit])
                reszsecond.Fill(track.localz[ihit] - track.simz[ihit])
                
                simytec.Fill(track.phi[ihit] - track.globalsim_phi[ihit])

            elif det==5 and layer==-9 and stereo==0:
                #print("found tec last")
                resxlast.Fill(track.localx[ihit] - track.simx[ihit])
                resylast.Fill(track.localy[ihit] - track.simy[ihit])
                reszlast.Fill(track.localz[ihit] - track.simz[ihit])
                
                
    
    hists.append(simytec)
    #print("nhits",nhits)
    #for ihit in range(nhits):
        #print("simx", track.simx[ihit])

hminus = hists[0]

#hist = hists[0]
hplus = hists[1]

hminus.SetLineColor(ROOT.kRed)

c = ROOT.TCanvas()
#hist.Draw("HIST")
hplus.Draw("HIST")
hminus.Draw("HISTSAME")

#gc = []

#c = ROOT.TCanvas()
#resxfirst.Draw("E")

#c2 = ROOT.TCanvas()
#resyfirst.Draw("E")

#c3 = ROOT.TCanvas()
#reszfirst.Draw("E")

#c4 = ROOT.TCanvas()
#resxsecond.Draw("E")

#c5 = ROOT.TCanvas()
#resysecond.Draw("E")

#c6 = ROOT.TCanvas()
#reszsecond.Draw("E")


#c7 = ROOT.TCanvas()
#resxlast.Draw("E")

#c8 = ROOT.TCanvas()
#resylast.Draw("E")

#c9 = ROOT.TCanvas()
#reszlast.Draw("E")



input("wait")
