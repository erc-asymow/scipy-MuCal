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

def datasetSplitter(dataRaw, etas, phis):

    dataset = {}
    data = {}

    for key,value in dataRaw.iteritems():
        data[key]=np.array(value)

    allBins=[]
    for i in range(1,len(etas)):
        for j in range(1,len(phis)):

                allBins.append(i+(len(etas)-1)*(j-1)-1)

    #figure out in which bin of eta and phi you are
    etaBin1 = np.digitize(np.array([data["eta1"]]), etas)
    etaBin2 = np.digitize(np.array([data["eta2"]]), etas)
    phiBin1 = np.digitize(np.array([data["phi1"]]), phis)
    phiBin2 = np.digitize(np.array([data["phi2"]]), phis)

    print allBins
    
    for bin1,bin2 in itertools.product(allBins, repeat=2):

        print bin1,bin2

        tmp = {}
        
        for key, value in data.iteritems():

            tmp[key] = value[ np.where((etaBin1+(len(etas)-1)*(phiBin1-1)-1==bin1) & (etaBin2+(len(etas)-1)*(phiBin2-1)-1==bin2))[1]]

        tmp["s1"] = 2*np.exp(-tmp["eta1"])/(1+np.exp(-2*tmp["eta1"]))
        tmp["s2"] = 2*np.exp(-tmp["eta2"])/(1+np.exp(-2*tmp["eta2"]))
        tmp["c1"] = 1./tmp["pt1"]
        tmp["c2"] = 1./tmp["pt2"]

        dataset[(bin1,bin2)] = tmp

    return dataset


ROOT.ROOT.EnableImplicitMT()
RDF = ROOT.ROOT.RDataFrame

restrictToBarrel = True

if isJ:
    cut = 'pt1>5 && pt2>5 && mass>3.0 && mass<3.2'
    
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


data = d.AsNumpy(columns=["smearedgenMass", "massErr","rapidity","eta1", "pt1", "phi1", "eta2", "pt2", "phi2"])

etas = np.arange(-0.8, 1.2, 0.4)
#phis = np.arange(-np.pi, np.pi+2.*np.pi/6.,2.*np.pi/6.)
phis = np.array((-np.pi,np.pi))

dataset = datasetSplitter(data,etas,phis)

filehandler = open('calInputZMCsm.pkl', 'w')
pickle.dump(dataset, filehandler)


if not isData:

    dataGen = d.AsNumpy(columns=["genMass", "massErr","rapidity","eta1", "pt1", "phi1", "eta2", "pt2", "phi2"])

    datasetGen = datasetSplitter(dataGen,etas,phis)

    filehandler = open('calInputZMCgen.pkl', 'w')
    pickle.dump(datasetGen, filehandler)




