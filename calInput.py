import ROOT
import pickle
import numpy as np
from header import CastToRNode
from root_numpy import array2hist, fill_hist
import argparse
import itertools
from scipy.stats import binned_statistic_dd

ROOT.gROOT.ProcessLine(".L src/module.cpp+")
ROOT.gROOT.ProcessLine(".L src/applyCalibration.cpp+")


parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-smearedMC', '--smearedMC', default=False, action='store_true', help='Use smeared gen mass in MC, omit for using reco mass')
parser.add_argument('-isData', '--isData', default=False, action='store_true', help='Use if data, omit if MC')
parser.add_argument('-runClosure', '--runClosure', default=False, action='store_true', help='Use to apply full calibration. If omit, rescale data for B map and leave MC as it is')
parser.add_argument('-dataDir', '--dataDir', default='/scratchssd/emanca/wproperties-analysis/muonCalibration/Minimisation', type=str, help='set the directory for input data')

args = parser.parse_args()
isJ = args.isJ
smearedMC = args.smearedMC
isData = args.isData
runClosure = args.runClosure
dataDir = args.dataDir

ROOT.ROOT.EnableImplicitMT()
RDF = ROOT.ROOT.RDataFrame

restrictToBarrel = True

if isJ:
    cut = 'pt1>4.3 && pt2>4.3 && pt1<25. && pt2<25.'# && mass>2.9 && mass<3.3'
    
else:
    cut = 'pt1>20.0 && pt2>20.0 && mass>80. && mass<100.'

if restrictToBarrel:
    cut+= '&& fabs(eta1)<0.8 && fabs(eta2)<0.8'

else:
    cut+= '&& fabs(eta1)<2.4 && fabs(eta2)<2.4' 

if not isData:

    cut+= '&& mcpt1>0. && mcpt2>0.'

if isJ:
    inputFileMC ='%s/muonTree.root' % dataDir
    inputFileD ='%s/muonTreeData.root' % dataDir
else:
    inputFileMC ='%s/muonTreeMCZ.root' % dataDir
    inputFileD ='%s/muonTreeDataZ.root' % dataDir

if isData: inputFile = inputFileD
else: inputFile = inputFileMC

def makeData(inputFile, genMass=False, smearedMass=False):

    d = RDF('tree',inputFile)

    if smearedMass:
        NSlots = d.GetNSlots()
        ROOT.gInterpreter.ProcessLine('''
                        std::vector<TRandom3> myRndGens({NSlots});
                        int seed = 1; // not 0 because seed 0 has a special meaning
                        for (auto &&gen : myRndGens) gen.SetSeed(seed++);
                        '''.format(NSlots = NSlots))

    print(cut)

    d = d.Filter(cut)\
        .Define('v1', 'ROOT::Math::PtEtaPhiMVector(pt1,eta1,phi1,0.105)')\
        .Define('v2', 'ROOT::Math::PtEtaPhiMVector(pt2,eta2,phi2,0.105)')
    
    if smearedMass:
        d = d.Define('v1sm', 'ROOT::Math::PtEtaPhiMVector(mcpt1+myRndGens[rdfslot_].Gaus(0., cErr1*mcpt1),eta1,phi1,0.105)')\
            .Define('v2sm', 'ROOT::Math::PtEtaPhiMVector(mcpt2+myRndGens[rdfslot_].Gaus(0., cErr2*mcpt2),eta2,phi2,0.105)')\
            .Define('smearedgenMass', '(v1sm+v2sm).M()')

    f = ROOT.TFile.Open('%s/bFieldMap.root' % dataDir)
    bFieldMap = f.Get('bfieldMap')

    if runClosure: print('taking corrections from', '{}/scale_{}_80X_13TeV.root'.format(dataDir, 'DATA' if isData else 'MC'))

    f2 = ROOT.TFile.Open('{}/scale_{}_80X_13TeV.root'.format(dataDir, 'DATA' if isData else 'MC'))
    A = f2.Get('magnetic')
    e = f2.Get('e')
    M = f2.Get('B')

    module = ROOT.applyCalibration(bFieldMap, A, e, M, isData, runClosure)

    d = module.run(CastToRNode(d))
    
    mass = 'corrMass'

    if smearedMass:
        mass = 'smearedgenMass'
    
    cols=[mass,'eta1', 'pt1', 'phi1', 'eta2', 'pt2', 'phi2']
    
    if genMass:
        cols.append('genMass')
    
    data = d.AsNumpy(columns=cols)

    return data


def makeGenDataset(data, etas, pts, masses):

    datasetGen = np.array([data['eta1'],data['eta2'],data['pt1'],data['pt2'],data['genMass']])
    histoGen, edges = np.histogramdd(datasetGen.T, bins = [etas,etas,pts,pts,masses])

    return histoGen


def makepkg(data, etas, pts, masses, good_idx, smearedMass=False):

    mass = 'corrMass'

    if smearedMass:
        mass = 'smearedgenMass'

    dataset = np.array([data['eta1'],data['eta2'],data['pt1'],data['pt2'],data[mass]])
    
    histo, edges = np.histogramdd(dataset.T, bins = [etas,etas,pts,pts,masses])
    
    #compute mean pt in each bin (integrating over mass)
    massesfull = np.array([masses[0],masses[-1]])
    histoden,_ = np.histogramdd(dataset.T, bins = [etas,etas,pts,pts,massesfull])
    
    histopt1,_ = np.histogramdd(dataset.T, bins = [etas,etas,pts,pts,massesfull], weights=1./dataset[2])
    ret1 = histopt1/histoden
    
    histopt2,_ = np.histogramdd(dataset.T, bins = [etas,etas,pts,pts,massesfull], weights=1./dataset[3])
    ret2 = histopt2/histoden
    
    #remove spurious mass dimension
    ret1 = np.squeeze(ret1, axis=-1)
    ret2 = np.squeeze(ret2, axis=-1)

    pkg = {}
    pkg['dataset'] = histo[good_idx]
    pkg['edges'] = edges
    pkg['binCenters1'] = ret1[good_idx]
    pkg['binCenters2'] = ret2[good_idx]
    pkg['good_idx'] = good_idx
    
    return pkg


mcrecomass = "mass"
if smearedMC:
    mcrecomass = "smearedgenMass"

dataD = makeData(inputFileD)
dataMC = makeData(inputFileMC, genMass=True, smearedMass=smearedMC)

etas = np.arange(-0.8, 1.2, 0.4)
if isJ:
    masses = np.arange(2.9,3.304,0.004)
else:
    masses = np.arange(75.,115.04,0.4)
nPtBins = 5
ptquantiles = np.linspace(0.,1.,nPtBins+1)
print(ptquantiles)
pts = np.quantile(np.concatenate((dataMC['pt1'],dataMC['pt2']),axis=0),ptquantiles)
print(pts)

histoGen = makeGenDataset(dataMC,etas,pts,masses)

good_idx = np.nonzero(np.sum(histoGen,axis=-1)>4000.)

histoGen = histoGen[good_idx]

pkgD = makepkg(dataMC, etas, pts, masses, good_idx)
pkgMC = makepkg(dataMC, etas, pts, masses, good_idx, smearedMass=smearedMC)

if not isJ:
    pklfileGen = 'calInputZMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(pts)-1)
    filehandler = open('calInputZMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(pts)-1), 'wb')
else:
    pklfileGen = 'calInputJMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(pts)-1)
    filehandler = open('calInputJMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(pts)-1), 'wb')
    
with open(pklfileGen, 'wb') as filehandler:
    pickle.dump(histoGen, filehandler)

pklfileBase = 'calInput{}'.format('J' if isJ else 'Z')
pklfileData = pklfileBase + 'DATA'

if smearedMC:
    pklfileMC = pklfileBase + 'MCsmear'
else:
    pklfileMC = pklfileBase + 'MC'

pklfileData+='_{}etaBins_{}ptBins'.format(len(etas)-1, len(pts)-1)
pklfileData+='.pkl'

pklfileMC+='_{}etaBins_{}ptBins'.format(len(etas)-1, len(pts)-1)
pklfileMC+='.pkl'

with open(pklfileMC, 'wb') as filehandler:
    pickle.dump(pkgMC, filehandler)
    
with open(pklfileData, 'wb') as filehandler:
    pickle.dump(pkgD, filehandler)
