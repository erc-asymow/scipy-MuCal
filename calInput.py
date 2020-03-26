import ROOT
import pickle
import numpy as np
from root_numpy import array2hist, fill_hist
import argparse
import itertools

parser = argparse.ArgumentParser("")
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help="Use to run on JPsi, omit to run on Z")
parser.add_argument('-isData', '--isData', default=False, action='store_true', help="Use if data, omit if MC")
parser.add_argument('-tag', '--tag', type=str, default="TEST",      help="folder to save output")

args = parser.parse_args()
isJ = args.isJ
isData = args.isData
tag = args.tag


ROOT.ROOT.EnableImplicitMT()
RDF = ROOT.ROOT.RDataFrame

restrictToBarrel = True

if isJ:
    cut = 'pt1>3. && pt2>3. && mass>2.9 && mass<3.3'
    
else:
    cut = 'pt1>20.0 && pt2>20.0 && mass>80. && mass<100.'

if restrictToBarrel:
    cut+= '&& fabs(eta1)<0.8 && fabs(eta2)<0.8'

else:
    cut+= '&& fabs(eta1)<2.4 && fabs(eta2)<2.4' 

if not isData:

    cut+= '&& mcpt1>0. && mcpt2>0.'

if isJ:
    inputFileMC ='/scratchssd/emanca/wproperties-analysis/muonCalibration/muonTree.root'
    inputFileD ='/scratchssd/emanca/wproperties-analysis/muonCalibration/muonTreeData.root'
else:
    inputFileMC ='/scratchssd/emanca/wproperties-analysis/muonCalibration/muonTreeMCZ.root'
    inputFileD ='/scratchssd/emanca/wproperties-analysis/muonCalibration/muonTreeDataZ.root'

if isData: inputFile = inputFileD
else: inputFile = inputFileMC

d = RDF('tree',inputFile)

NSlots = d.GetNSlots()
ROOT.gInterpreter.ProcessLine('''
                std::vector<TRandom3> myRndGens({NSlots});
                int seed = 1; // not 0 because seed 0 has a special meaning
                for (auto &&gen : myRndGens) gen.SetSeed(seed++);
                '''.format(NSlots = NSlots))

print cut
d = d.Filter(cut)\
	 .Define('v1', 'ROOT::Math::PtEtaPhiMVector(pt1,eta1,phi1,0.105)')\
     .Define('v2', 'ROOT::Math::PtEtaPhiMVector(pt2,eta2,phi2,0.105)')\
	 .Define('rapidity', 'float((v1+v2).Rapidity())').Filter('fabs(rapidity)<0.8')\
     .Define('v1sm', 'ROOT::Math::PtEtaPhiMVector(mcpt1+myRndGens[rdfslot_].Gaus(0., cErr1*pt1),eta1,phi1,0.105)')\
     .Define('v2sm', 'ROOT::Math::PtEtaPhiMVector(mcpt2+myRndGens[rdfslot_].Gaus(0., cErr2*pt2),eta2,phi2,0.105)')\
     .Define('smearedgenMass', '(v1sm+v2sm).M()')

etas = np.arange(-0.8, 1.2, 0.4)
pts = np.array((3.,7.,10.,15.,20.))
mass = np.arange(2.9,3.304,0.004)
#etas = np.array((-0.8,0.8))
#pts = np.array((3.,20.))

#phis = np.arange(-np.pi, np.pi+2.*np.pi/6.,2.*np.pi/6.)
phis = np.array((-np.pi,np.pi))

data = d.AsNumpy(columns=["mass","eta1", "pt1", "phi1", "eta2", "pt2", "phi2"])

"""
eta1
eta2
genMass
phi2
phi1
pt2
pt1
"""

dataset = np.array([val for val in data.values()])
histo, edges = np.histogramdd(dataset.T, bins = [etas,etas,mass,phis,phis,pts,pts])



if not isJ:
    filehandler = open('calInputZMC.pkl', 'w')
else:
    filehandler = open('calInputJMC.pkl', 'w')
pickle.dump(histo, filehandler)


if not isData:

    dataGen = d.AsNumpy(columns=["genMass","eta1", "pt1", "phi1", "eta2", "pt2", "phi2"])

    """
    eta1
    eta2
    mass
    phi2
    phi1
    pt2
    pt1
    """
    datasetGen = np.array([val for val in dataGen.values()])
    histoGen, edges = np.histogramdd(datasetGen.T, bins = [etas,etas,mass,phis,phis,pts,pts])

    if not isJ:
        filehandler = open('calInputZMCgen.pkl', 'w')
    else:
        filehandler = open('calInputJMCgen.pkl', 'w')
    pickle.dump(histoGen, filehandler)




