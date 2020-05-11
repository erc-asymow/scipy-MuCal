import ROOT

ROOT.gStyle.SetOptStat(0)

fdata = ROOT.TFile.Open("calibrationMC.root")
fmc = ROOT.TFile.Open("calibrationMC.root")

files = [fmc, fdata]
#parms = ["A","e","M","a","c"]
parms = ["A","e","M", "W","a","c","b","d"]

gc = []

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
        
        

    
    hists[1].SetLineColor(ROOT.kRed)
    hists[0].Draw("E")
    
    hists[1].Draw("ESame")
    

    
    if parms=="A":
        hists[0].GetYaxis().SetTitle("Magnetic Field Correction A")

    
    hists[0].GetYaxis().SetTitleOffset(1.2)
    
    leg = ROOT.TLegend(0.4,0.7, 0.6,0.87)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(hists[0],"MC", "LP")
    leg.AddEntry(hists[1],"Data", "LP")
    #leg.Draw()
    gc.append(leg)
    #c.Modified()
    
    if parms=="e":
        hists[0].GetYaxis().SetRangeUser(-0.02,0.02)
        
    if parms=="M":
        hists[0].GetYaxis().SetRangeUser(-0.1e-3,0.1e-3)    
    c.Update()
    
    
    
    c.SaveAs(f"{parm}.pdf")
    
for comp in ["scale","sigma"]:
    hist0 = fdata.Get(f"{comp}Model")
    hist1 = fdata.Get(f"{comp}Binned")
    
    gc.append(hist0)
    gc.append(hist1)
    
    hist1.SetLineColor(ROOT.kRed)
    
    #hist1.GetXaxis().SetRangeUser(3000,3100)
    hist1.GetYaxis().SetTitle(comp)
    
    c = ROOT.TCanvas()
    hist1.Draw("E")
    hist0.Draw("ESAME")
    
    gc.append(c)
    
    leg = ROOT.TLegend(0.4,0.1, 0.6,0.3)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(hists[0],"Model", "LP")
    leg.AddEntry(hists[1],"Binned", "LP")
    leg.Draw()
    gc.append(leg)
    c.Update()
    
    c.SaveAs(f"{comp}.pdf")
    
    
input("wait")
