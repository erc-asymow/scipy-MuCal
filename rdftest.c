#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TMatrix.h"
#include "TSystem.h"
#include <ROOT/TTreeProcessorMT.hxx>
// #include </usr/include/eigen3/Eigen/Dense>

float testfuncfixed2(float arg0, float arg1) {
  return arg0 + arg1;
}

float testfuncfixed11(float arg0, float arg1, float arg2, float arg3, float arg4, float arg5, float arg6, float arg7, float arg8, float arg9, float arg10) {
  return arg0 + arg1 + arg2 + arg3 + arg4 + arg5 + arg6 + arg7 + arg8 + arg9 + arg10;
}

void rdftest() {
  
//   filename = "root://eoscms.cern.ch///store/group/phys_smp/emanca/data/*.root";
//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v3/200817_213237/0000/*.root";
  
//   const std::string filename = "/eos/cms/store/cmst3/group/wmass/bendavid/muoncal/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v3/200817_213237/0000/*.root";
//   const std::string filename = "/data/bendavid/muoncaldatalarge/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v4/200820_002243/0000/*.root";
//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v23_Gen/201201_232356/0000/globalcor_*.root";
  const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v30_Rec210206_025546/0000/globalcor_0_1.root";
//   const std::string filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealpxbgenxygenstart/globalcor_*.root";
  
//   gSystem->Setenv("XRD_REQUESTTIMEOUT","30");
  
//   Eigen::initParallel();
  ROOT::EnableImplicitMT(32);
  ROOT::TTreeProcessorMT::SetMaxTasksPerFilePerWorker(1);

  std::string treename = "tree";
//   std::string outfilename = "correctedTracks.root";
  
//   TFile::SetOpenTimeout(1);
  
//   std::string path = "root://eoscms.cern.ch//store/group/phys_smp/emanca/data/";
//   TChain *chain = new TChain("tree");
// //   chain->SetOption("TIMEOUT=1");
//   std::ifstream istrm("filelist.txt", std::ios::in);
//   std::string line;
//   while (std::getline(istrm, line)) {
//     std::string fname = path + line;
//     std::cout << fname << std::endl;
// //     TFile *ftest = TFile::Open(fname.c_str(), "TIMEOUT=5");
//     TFile *ftest = TFile::Open(fname.c_str());
//     ftest->Close();
//     chain->AddFile(fname.c_str());
//   }
  
//   chain->Draw("jacrefv");
  
//   return;
  
//   ROOT::RDataFrame d(*chain);

  
  ROOT::RDataFrame d(treename, filename);
  
//   auto d2 = d.Define("testdef", "testfuncfixed2(genEta, genEta)");
  auto d2 = d.Define("testdef", "testfuncfixed11(genEta, genEta, genEta, genEta, genEta, genEta, genEta, genEta, genEta, genEta, genEta)");
//   auto d2 = d.Define("testdef", testfuncfixed11, { "genEta", "genEta", "genEta", "genEta", "genEta", "genEta", "genEta", "genEta", "genEta", "genEta", "genEta" });
  
  auto sum = d2.Sum("testdef");
  std::cout << sum.GetValue() << std::endl;
  
//   auto dcut = d.Filter("genPt>0. && genEta<-2.3");
//   auto dcut = d.Filter("genPt>0.");
  
//   auto d2 = d.Define("corParms", correctParms, { "refParms", "jacrefv", "globalidxv" });
  
//   auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "run", "lumi", "event", "corParms" } );
  
//   std::cout << "starting snapshot:" << std::endl;
//     auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "corParms", "gradmax", "hessmax" } );
// 
//     
//   std::cout << "done snapshot:" << std::endl;

}
