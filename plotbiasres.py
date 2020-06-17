import ROOT
import numpy as np
from fittingFunctionsBinned import computeTrackLength
from root_numpy import array2hist, fill_hist, hist2array

nEtaBins = 24
etas = np.linspace(-2.4,2.4,nEtaBins+1)
etas = 0.5*(etas[1:] + etas[:-1])
l = computeTrackLength(etas)

ROOT.gStyle.SetOptStat(0)

gc = []


fnom = ROOT.TFile.Open("calibrationMC.root")
#fnom = ROOT.TFile.Open("calibrationMCeta48phi1.root")
ferrs = ROOT.TFile.Open("calibrationMCErrsgensqrtalpha1rel.root")

#fnom = ROOT.TFile.Open("calibrationMCAeMWcor.root")
#ferrs = ROOT.TFile.Open("calibrationMCErrsabcd.root")

files = [fnom, ferrs]
#parms = ["A","e","M","a","c"]

parms = ["A","e","M", "W","a","c","g","d"]
errparms = ["a","c","g","d"]

#parms = ["A","e","M", "W","a","c","b","d"]
#errparms = ["a","c","b","d"]

hA = fnom.Get("A")
hW = fnom.Get("W")
ha = fnom.Get("a")
hc = fnom.Get("c")
hY = fnom.Get("Y")
hZ = fnom.Get("Z")

haerrs = ferrs.Get("a")
hcerrs = ferrs.Get("c")

hAscaled = hA.Clone("hAscaled")
hascaled = ha.Clone("hascaled")
hcscaled = hc.Clone("hcscaled")
hWscaled = hW.Clone("hWscaled")
hYscaled = hY.Clone("hYscaled")
hZscaled = hZ.Clone("hZscaled")


gc.append(hAscaled)
gc.append(hascaled)
gc.append(hcscaled)
gc.append(hWscaled)

L0=108.-4.4

for ibin in range(nEtaBins):
    il = l[ibin]
    labs = L0/il
    eta = etas[ibin]
    theta = 2.*np.arctan(np.exp(-eta))
    sec4theta = (1./np.cos(theta))**4
    ncsc4theta = (1./np.sin(theta))**4
    cos2theta = np.cos(theta)**2
    
    Ascale = 1./il**4
    hAscaled.SetBinContent(ibin+1, hAscaled.GetBinContent(ibin+1)*Ascale)
    hAscaled.SetBinError(ibin+1, hAscaled.GetBinError(ibin+1)*Ascale)    
    
    anom = ha.GetBinContent(ibin+1)
    aerrs = haerrs.GetBinContent(ibin+1)
    #a = np.sqrt(anom - aerrs)
    #a = np.sqrt(a)
    aerr = ha.GetBinError(ibin+1)
    
    #hascaled.SetBinContent(ibin+1, a)
    
    cnom = hc.GetBinContent(ibin+1)
    cerrs = hcerrs.GetBinContent(ibin+1)
    #c = hc.GetBinContent(ibin+1) - hcerrs.GetBinContent(ibin+1)
    #c = np.sqrt(cnom) - np.sqrt(cerrs)
    #c = cnom - cerrs
    #c = 10.*(cerrs - cnom)
    #c = np.sqrt(cerrs - cnom)
    #c = np.sqrt(cerrs) - np.sqrt(cnom)
    #c = -np.sqrt(cnom)
    c = 0.3**2*3.8**2*labs**2*cnom/16.*0.1
    #c = L0**2*cnom
    
    #c = np.sqrt(np.abs(cerrs - cnom))
    cerr = hc.GetBinError(ibin+1)
    hcscaled.SetBinContent(ibin+1, c)
    
    anom  = ha.GetBinContent(ibin+1)
    aerrs = haerrs.GetBinContent(ibin+1)
    #aout = -5*(anom-aerrs)
    #aout = -2.*ncsc4theta*anom
    #aout = -2.*sec4theta*anom
    #aout = -0.3**2*3.8**2*L0*sec4theta/8.*anom
    #aout = -0.3**2*3.8**2*labs**2/8.*anom
    #aout = -0.3**2*3.8**2*labs**2/8.*(anom-aerrs)*sec4theta
    #aout = 0.3**2*3.8**2*labs**2*anom/il**2/256.*ncsc4theta
    aout = 0.3**2*3.8**2*labs**2*anom*il**2/256.
    
    #aout = -0.3**2*3.8**2*L0*ncsc4theta/8.*anom
    #aout = -2.*anom
    #aout = -np.sqrt(anom)
    #aout = -2*anom
    hascaled.SetBinContent(ibin+1,aout)
    hascaled.SetBinError(ibin+1, 2.*hascaled.GetBinError(ibin+1))
    
    wscale = 1./il**4
    hWscaled.SetBinContent(ibin+1, wscale*hWscaled.GetBinContent(ibin+1))
    hWscaled.SetBinError(ibin+1, wscale*hWscaled.GetBinError(ibin+1))
    
    yscale = 1./il**4
    hYscaled.SetBinContent(ibin+1, yscale*hYscaled.GetBinContent(ibin+1))
    hYscaled.SetBinError(ibin+1, yscale*hYscaled.GetBinError(ibin+1))
    
    #zscale = 1./il**2
    zscale = 1.
    hZscaled.SetBinContent(ibin+1, zscale*hZscaled.GetBinContent(ibin+1))
    hZscaled.SetBinError(ibin+1, zscale*hZscaled.GetBinError(ibin+1))
        
    #cscale = 1.
    #hcscaled.SetBinContent(ibin+1, hcscaled.GetBinContent(ibin+1)*cscale)
    #hcscaled.SetBinError(ibin+1, hcscaled.GetBinError(ibin+1)*cscale)

hascaled.SetLineColor(ROOT.kRed)

#c = ROOT.TCanvas()
#gc.append(c)
#hcscaled.SetLineColor(ROOT.kRed)
#hAscaled.Draw("E")
#hAscaled.GetYaxis().SetTitle("A/L^2")
#hascaled.Draw("ESAME")
##ha.Draw("ESAME")
#c.Update()
#c.SaveAs("Ascaled.png")

#input("wait")
#assert(0)

c = ROOT.TCanvas()
gc.append(c)
#hWscaled.SetLineColor(ROOT.kRed)
hZscaled.Draw("E")
hascaled.SetLineColor(ROOT.kRed)
hascaled.Draw("ESAME")
#hWscaled.Draw("E")
#hascaled.Draw("ESAME")
c.Update()

input("wait")
assert(0)

for parm in errparms:
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
    #hists[1].SetMarkerStyle(8)
    
    hists[0].SetLineColor(ROOT.kRed)
    hists[0].Draw("E")
    
    hists[1].Draw("ESame")
    

    
    if parms=="A":
        hists[0].GetYaxis().SetTitle("Magnetic Field Correction A")

    
    hists[0].GetYaxis().SetTitleOffset(1.2)
    
    leg = ROOT.TLegend(0.32,0.7, 0.68,0.87)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(hists[0],"Scale/Res Fit", "LP")
    leg.AddEntry(hists[1],"Track Uncertainty Fit", "LP")
    leg.Draw()
    gc.append(leg)
    #c.Modified()
    
    if parms=="e":
        hists[0].GetYaxis().SetRangeUser(-0.02,0.02)
        
    if parms=="M":
        hists[0].GetYaxis().SetRangeUser(-0.1e-3,0.1e-3)    
    c.Update()
    
    
    
    c.SaveAs(f"{parm}_errcomp.pdf")
    


input("wait")
#assert(0)

#for parm in parms:
    #h = fnom.Get(parm)
    #gc.append(h)
    
    #c = ROOT.TCanvas()
    #gc.append(c)
    #h.Draw("E")
    
    #c.Update()
    
    #c.SaveAs(f"{parm}.png")
    
#input("wait")
assert(0)

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
    
    
    
    c.SaveAs(f"{parm}_bincomp.png")
    

    
    
    
input("wait")
