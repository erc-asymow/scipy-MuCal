import ROOT

ROOT.gInterpreter.ProcessLine(".O3")
ROOT.ROOT.EnableImplicitMT()


import numpy as np
import h5py
from utils import lumitools


status = ROOT.gInterpreter.Declare('#include "aggregategrads.cpp"')

assert(status)


#lumitools.init_lumitools()
jsonhelper = lumitools.make_jsonhelper("data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")

#nparms = 10
#m = ROOT.SymMatrixAtomic(10)

#row = np.zeros((nparms,), dtype=np.float64)

#m.fill_row(0,row.data)

#assert(0)


filenameinfo = "info_v202.root"
finfo = ROOT.TFile.Open(filenameinfo)
runtree = finfo.Get("runtree")

nparms = int(runtree.GetEntries())



print(nparms)

etas = np.linspace(-2.4, 2.4, 49)
#logpts = np.linspace(-2.5,2.5,21)
#cosphis = np.array([round(-1. + 2.*i/20,2) for i in range(21)], dtype=np.float64)
#logpts = np.linspace(-3.5,3.5,71)
#cosphis = [0.8, 0.859, 0.893, 0.916, 0.932, 0.944, 0.954, 0.961, 0.967, 0.972, 0.976, 0.98, 0.983, 0.986, 0.989, 0.991, 0.993, 0.995, 0.997, 0.998,1.]
#masses = np.linspace(2.9, 3.3, 1001)
logpts = np.linspace(-2.5,2.5,41)
cosphis = np.array([round(-1. + 2.*i/20,2) for i in range(21)],dtype=np.float64)
masses = np.linspace(2.9, 3.3, 101)


print(etas)
print(logpts)
print(cosphis)
print(masses)

neta = etas.shape[0] - 1
nlogpt = logpts.shape[0] - 1
ncosphi = cosphis.shape[0] - 1
nmass = masses.shape[0] - 1

fullidxs = -1*np.ones((neta+2,neta+2,nlogpt+2,ncosphi+2),dtype=np.int32)

#fullidxs = -1*np.ones((50,50,22,22),dtype=np.int32)

filenamefits = "fitsJDATA.hdf5"
#ffits = h5py.file(filen
with h5py.File(filenamefits) as ffits:
    print(ffits.keys())
    #assert(0)

    sigpdf = ffits["sigpdf"][...]
    bkgpdf = ffits["bkgpdf"][...]

    wsig = sigpdf/(sigpdf+bkgpdf)



    print("wsig.shape", wsig.shape)

    #assert(0)


    good_idx = ffits["good_idx"][...].astype(np.int32) + 1
    good_idx = (good_idx[0], good_idx[1], good_idx[2], good_idx[3])

    linidxs = np.arange(good_idx[0].shape[0])
    print(linidxs)

    #assert(0)

    #print(good_idx)

    fullidxs[good_idx] = linidxs



    #valid = np.where(fullidxs >= 0)

    #print(valid)

    #hdata = full[good_idx]

    #print(good_idx.shape)
    #print(fullidxs)
    #print(hdata)
    #print(full)


    #print(ffits["good_idx"])




@ROOT.Numba.Declare(["float", "float", "float", "float", "float", "float", "float", "unsigned int"], "double")
def massweight(ptplus, etaplus, phiplus, ptminus, etaminus, phiminus, mass, run):

    #if run == 1:
        #return 1.

    logpt = np.log(ptplus/ptminus)
    cosphi = np.cos(phiplus - phiminus)

    idx0 = np.digitize([etaplus], etas)
    idx1 = np.digitize([etaminus], etas)
    idx2 = np.digitize([logpt], logpts)
    idx3 = np.digitize([cosphi], cosphis)

    idxt = (idx0[0], idx1[0], idx2[0], idx3[0])

    #idxt = (np.digitize(etaplus, etas), np.digitize(etaminus, etas), np.digitize(logpt, logpts), np.digitize(cosphi, cosphis))

    idx = fullidxs[idxt]

    if idx == -1:
        return 0.

    massidx = np.digitize([mass], masses) - 1
    massidx = massidx[0]

    if massidx < 0 or massidx >= wsig.shape[-1]:
        return 0.


    if run == 1:
        return 1.

    return wsig[idx, massidx]


#assert(0)


chainjpsi = ROOT.TChain("tree")

#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiG_quality_constraintfsr28/211007_111614/0000/globalcor_0_1.root")

#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiG_quality_constraintfsr28/211007_111614/0000/*.root")
#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiG_quality_constraintfsr28/211007_111614/0000/*.root")
#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecDataJPsiFpost_quality_constraintfsr28/211007_111216/0000/*.root")

#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecJpsiPhotos_quality_constraintfsr29/211011_190957/0000/*.root")
#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v206_RecJpsiPhotos_quality_constraintfsr29/211011_190957/0001/*.root")

#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecJpsiPhotos_quality_constraintnofsr/211012_234314/0000/*.root")
#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecJpsiPhotos_quality_constraintnofsr/211012_234314/0001/*.root")

#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintnofsr/211013_160905/0000/*.root")
#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintnofsr/211013_160905/0001/*.root")

#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintfsr29/211013_161309/0000/*.root")
#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintfsr29/211013_161309/0001/*.root")



#chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiH_quality_constraintfsr29/211012_233512/0000/globalcor_0_1.root")

chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiH_quality_constraintfsr29/211012_233512/0000/*.root")
chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiG_quality_constraintfsr29/211012_233641/0000/*.root")
chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiFpost_quality_constraintfsr29/211012_233806/0000/*.root")

dj = ROOT.ROOT.RDataFrame(chainjpsi)

dj = dj.Filter(jsonhelper, ["run", "lumi"], "jsonfilter")


#dj = dj.Filter("Mupluscons_pt > 1.1 && Muminuscons_pt > 1.1 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");

#dj = dj.Filter("Mupluscons_pt > 1.1 && Muminuscons_pt > 1.1 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.9 && Jpsi_mass<3.3");

#dj = dj.Filter("Mupluscons_pt > 1.1 && Muminuscons_pt > 1.1 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.9 && Jpsi_mass<3.3 && Muplus_muonLoose && Muminus_muonLoose");

#dj = dj.Filter("Mupluscons_pt > 1.1 && (Mupluscons_pt > 4.0 || abs(Mupluscons_eta) > 1.2) && Muminuscons_pt > 1.1 && (Muminuscons_pt > 4.0 || abs(Muminuscons_eta) > 1.2) && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.9 && Jpsi_mass<3.3 && Muplus_muonLoose && Muminus_muonLoose");


dj = dj.Filter("Muplus_pt > 1. && Muminus_pt > 1. && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.9 && Jpsi_mass<3.3");

#dj = dj.Filter("Muplus_pt > 1.5 && Muminus_pt > 1.5 && Muplus_pt < 23. && Muminus_pt < 23. && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.9 && Jpsi_mass<3.3");


#dj = dj.Filter("Muplus_muonLoose && Muminus_muonLoose")

#dj = dj.Filter("Jpsigen_mass > 3.0968")


dj = dj.Filter("valid(gradv) && valid(hesspackedv) && edmval < 1e-5");


#dj = dj.Define("massweightval", "Numba::massweight(Muplus_pt, Muplus_eta, Muplus_phi, Muminus_pt, Muminus_eta, Muminus_phi, Jpsi_mass, run)")


dj = dj.Define("massweightval", "1.0")

dj = dj.Filter("massweightval > 0.")


debugweights = False

if debugweights:
    hmass = dj.Histo1D(("hmass", "", 100, 2.9, 3.3), "Jpsi_mass")
    hmassweighted = dj.Histo1D(("hmassweighted", "", 100, 2.9, 3.3), "Jpsi_mass", "massweightval")


    c = ROOT.TCanvas()
    hmass.SetLineColor(ROOT.kRed)
    hmass.Draw("HIST")
    hmassweighted.Draw("HISTSAME")


    input("weight")

gradhelperj = ROOT.GradHelper(nparms)
hesshelperj = ROOT.HessHelper(nparms)

grad = dj.Book(gradhelperj, ["gradv", "globalidxv", "massweightval"])
hess = dj.Book(hesshelperj, ["hesspackedv", "globalidxv", "massweightval"])


#gradval = grad.GetResult()
#hessval = hess.GetResult()

#print(gradval[0])


#chunksize = 32

#fout = h5py.File("combinedgrads.hdf5", "w", rdcc_nbytes = nparms*8*chunksize*4, rdcc_nslots = nparms//chunksize*10)

fout = h5py.File("combinedgrads.hdf5", "w")

gradout = fout.create_dataset("grad", (nparms,), dtype=np.float64, compression="lzf")
hessout = fout.create_dataset("hess", (nparms, nparms), dtype=np.float64, compression="lzf", chunks=(1, nparms))


gradout[...] = grad

#assert(0)

hessrow = np.zeros((nparms,), dtype=np.float64)

for i in range(nparms):
  if i%1000 == 0:
      print(i)
  #if i>61200:
    #print(i)
  hess.fill_row(i, hessrow)
  hessout[i] = hessrow
  #hess.fill_row(i, hessout[i])
  #if i < 10:
    #print(hessout[i])

#hess.fill_row(0, hessrow.data)

#print(hessrow)
