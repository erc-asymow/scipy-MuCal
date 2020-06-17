import ROOT
import numpy as np
from fittingFunctionsBinned import computeTrackLength
from root_numpy import array2hist, fill_hist, hist2array

nEtaBins = 48
etas = np.linspace(-2.4,2.4,nEtaBins+1)
etas = 0.5*(etas[1:] + etas[:-1])
l = computeTrackLength(etas)

ROOT.gStyle.SetOptStat(0)

gc = []


#fnom = ROOT.TFile.Open("calibrationMC48binsrescompare.root")
fnom = ROOT.TFile.Open("calibrationMCeta48phi1.root")
ffine = ROOT.TFile.Open("calibrationMCeta480phi1.root")

files = [ffine, fnom]
#parms = ["A","e","M","a","c"]
parms = ["A","e","M", "W","a","c","g","d"]

hA = fnom.Get("A")
hW = fnom.Get("W")
ha = fnom.Get("a")
hc = fnom.Get("c")

hAscaled = hA.Clone("hAscaled")
hascaled = ha.Clone("hascaled")
hcscaled = hc.Clone("hcscaled")
hWscaled = hW.Clone("hWscaled")

gc.append(hascaled)
gc.append(hcscaled)
gc.append(hWscaled)

for ibin in range(nEtaBins):
    il = l[ibin]
    
    Ascale = 1./il**2
    hAscaled.SetBinContent(ibin+1, hAscaled.GetBinContent(ibin+1)*Ascale)
    hAscaled.SetBinError(ibin+1, hAscaled.GetBinError(ibin+1)*Ascale)    
    
    hascaled.SetBinContent(ibin+1, -hascaled.GetBinContent(ibin+1)*il**2)
    hascaled.SetBinError(ibin+1, hascaled.GetBinError(ibin+1)*il**2)
    
    cscale = 1.
    hcscaled.SetBinContent(ibin+1, hcscaled.GetBinContent(ibin+1)*cscale)
    hcscaled.SetBinError(ibin+1, hcscaled.GetBinError(ibin+1)*cscale)
    
    wscale = il**4
    hWscaled.SetBinContent(ibin+1, hWscaled.GetBinContent(ibin+1)*wscale)
    hWscaled.SetBinError(ibin+1, hWscaled.GetBinError(ibin+1)*wscale)    


c = ROOT.TCanvas()
gc.append(c)
hascaled.SetLineColor(ROOT.kRed)
hAscaled.Draw("E")
hAscaled.GetYaxis().SetTitle("A*(L/L0)^2")
hAscaled.GetYaxis().SetTitleOffset(0.8)
#hascaled.Draw("ESAME")
c.Update()
c.SaveAs("Ascaled.pdf")

c = ROOT.TCanvas()
gc.append(c)
hcscaled.SetLineColor(ROOT.kRed)
hWscaled.GetYaxis().SetTitle("W/(L/L0)^4")
hWscaled.GetYaxis().SetTitleOffset(0.8)
hWscaled.Draw("E")
#hcscaled.Draw("ESAME")
c.Update()
c.SaveAs("Wscaled.pdf")


#input("wait")
#assert(0)

for parm in parms:
    h = fnom.Get(parm)
    gc.append(h)
    
    c = ROOT.TCanvas()
    gc.append(c)
    h.Draw("E")
    
    c.Update()
    
    c.SaveAs(f"{parm}.pdf")
    
#input("wait")
#assert(0)

for parm in parms:
    hists = []
    for f in files:
        h = f.Get(parm)
        #print(h.GetYaxis().GetRangeUser())
        
        gc.append(h)
        hists.append(h)
    
    c = ROOT.TCanvas()
    gc.append(c)
    
    #hists[0].SetMaximum(max(hists[0].GetMaximum(),hists[1].GetMaximum()))
    #hists[0].SetMinimum(min(hists[0].GetMinimum(),hists[1].GetMinimum()))
    
    ymax = (max(hists[0].GetMaximum(),hists[1].GetMaximum()))
    ymin = (min(hists[0].GetMinimum(),hists[1].GetMinimum()))
    
    #hists[0].SetMinimum(ymin)
    #hists[0].SetMaximum(ymax)
    hists[0].GetYaxis().SetRangeUser(ymin,ymax)
        
        
    hists[1].SetLineWidth(2)
    hists[1].SetMarkerStyle(8)
    
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
    leg.AddEntry(hists[0],"Finely Binned", "LP")
    leg.AddEntry(hists[1],"Nominal", "LP")
    leg.Draw()
    gc.append(leg)
    #c.Modified()
    
    if parms=="e":
        hists[0].GetYaxis().SetRangeUser(-0.02,0.02)
        
    if parms=="M":
        hists[0].GetYaxis().SetRangeUser(-0.1e-3,0.1e-3)    
    c.Update()
    
    
    
    c.SaveAs(f"{parm}_bincomp.pdf")
    

    
    
    
input("wait")
