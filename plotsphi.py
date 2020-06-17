import ROOT
import numpy as np
from fittingFunctionsBinned import computeTrackLength
from root_numpy import array2hist, fill_hist, hist2array

nEtaBins = 12
etas = np.linspace(-2.4,2.4,nEtaBins+1)
etas = 0.5*(etas[1:] + etas[:-1])
l = computeTrackLength(etas)

nPhiBins = 8
nBins = nEtaBins*nPhiBins

ROOT.gStyle.SetOptStat(0)

gc = []


#fnom = ROOT.TFile.Open("calibrationMC48binsrescompare.root")
fnom = ROOT.TFile.Open("calibrationMCeta12phi1.root")
fphi = ROOT.TFile.Open("calibrationMCeta12phi8.root")

#files = [ffine, fnom]
#parms = ["A","e","M","a","c"]
parms = ["A","e","M", "W","a","c","g","d"]

for parm in parms:
    hists = []
    
    hphi = fphi.Get(parm)
    
    
    hetain = fnom.Get(parm)
    heta = ROOT.TH1D(f"{parm}_eta", parm, nEtaBins, -0.5, nBins-0.5)
    for ibin in range(nEtaBins):
        heta.SetBinContent(ibin+1, hetain.GetBinContent(ibin+1))
        heta.SetBinError(ibin+1, hetain.GetBinError(ibin+1))
        
    hists.append(hphi)
    hists.append(heta)
    
    gc.append(hphi)
    gc.append(heta)
    
    #for f in files:
        #h = f.Get(parm)
        ##print(h.GetYaxis().GetRangeUser())
        
        #gc.append(h)
        #hists.append(h)
    
    hdiff = ROOT.TH1D(f"{parm}_diff", parm, nBins,-0.5,nBins-0.5)
    for ibin in range(nBins):
        ieta = ibin//nPhiBins
        hdiff.SetBinContent(ibin+1, hphi.GetBinContent(ibin+1) - heta.GetBinContent(ieta+1))
        hdiff.SetBinError(ibin+1, hphi.GetBinError(ibin+1))
        
    hdiffnull = ROOT.TH1D(f"{parm}_diffnull", parm, nEtaBins,-0.5,nBins-0.5)
    for ieta in range(nEtaBins):
        hdiffnull.SetBinContent(ieta+1, 0.)
        hdiffnull.SetBinError(ieta+1, heta.GetBinError(ieta+1))

        
    hdiff.SetLineColor(ROOT.kRed)
    gc.append(hdiff)
    
    c = ROOT.TCanvas()
    gc.append(c)
    
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
    gc.append(pad1)
    pad1.SetBottomMargin(0);
    #pad1.SetGridx();
    pad1.Draw();
    pad1.cd()
    
    #hists[0].SetMaximum(max(hists[0].GetMaximum(),hists[1].GetMaximum()))
    #hists[0].SetMinimum(min(hists[0].GetMinimum(),hists[1].GetMinimum()))
    
    ymax = (max(hists[0].GetMaximum(),hists[1].GetMaximum()))
    ymin = (min(hists[0].GetMinimum(),hists[1].GetMinimum()))
    
    #hists[0].SetMinimum(ymin)
    #hists[0].SetMaximum(ymax)
    hists[0].GetYaxis().SetRangeUser(ymin,ymax)
        
        
    hists[1].SetLineWidth(2)
    #hists[1].SetMarkerStyle(8)
    
    hists[0].SetLineColor(ROOT.kRed)
    hists[0].Draw("E")
    
    hists[1].Draw("ESame")
    

    
    if parms=="A":
        hists[0].GetYaxis().SetTitle("Magnetic Field Correction A")

    
    hists[0].GetYaxis().SetTitleOffset(1.2)
    
    if parm in ["A","W"]:
        leg = ROOT.TLegend(0.4,0.1, 0.6,0.3)
    else:
        leg = ROOT.TLegend(0.4,0.7, 0.6,0.87)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(hists[0],"#eta-#phi binned", "LP")
    leg.AddEntry(hists[1],"#eta binned", "LP")
    leg.Draw()
    gc.append(leg)
    #c.Modified()
    
    if parms=="e":
        hists[0].GetYaxis().SetRangeUser(-0.02,0.02)
        
    if parms=="M":
        hists[0].GetYaxis().SetRangeUser(-0.1e-3,0.1e-3)    
        
    c.cd()
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.3);
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.2)
    #pad2.SetGridx()
    pad2.Draw()
    pad2.cd()
    #pad2.SetLogx()        

    hdiff.GetXaxis().SetTitleFont(43)
    hdiff.GetXaxis().SetTitleSize(20)
    hdiff.GetXaxis().SetTitleOffset(1.2)
    
    hdiff.GetYaxis().SetTitleFont(43)
    hdiff.GetYaxis().SetTitleSize(20)
    hdiff.GetYaxis().SetTitleOffset(1.)
    
    hdiff.GetXaxis().SetTitle("#eta-#phi bin")
    hdiff.GetYaxis().SetTitle("difference")
    
    hdiff.Draw("E")
    hdiffnull.Draw("ESAME")
    
        
    c.Update()
    

        
    #c = ROOT.TCanvas()
    #hdiff.Draw("E")
    #c.Update()
    
    gc.append(c)
    
    
    
    c.SaveAs(f"{parm}_phicomp.pdf")
    
input("wait")
    
