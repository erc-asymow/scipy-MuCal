#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TMatrix.h"
// #include </usr/include/eigen3/Eigen/Dense>

void thintrees(const char *filename="/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root") {
  
//   filename = "root://eoscms.cern.ch///store/group/phys_smp/emanca/data/*.root";
  
//   Eigen::initParallel();
  ROOT::EnableImplicitMT();
  
 
  std::string treename = "tree";
  std::string outfilename = "TrackTree.root";
  
  
//   std::string path = "root://eoscms.cern.ch//store/group/phys_smp/emanca/data/";
//   TChain *chain = new TChain("tree");
//   std::ifstream istrm("filelist.txt", std::ios::in);
//   std::string line;
//   while (std::getline(istrm, line)) {
//     std::string fname = path + line;
//     std::cout << fname << std::endl;
//     TFile *ftest = TFile::Open(fname.c_str());
//     ftest->Close();
//     chain->AddFile(fname.c_str());
//   }
  
//   chain->Draw("jacrefv");
  
//   return;
  
//   ROOT::RDataFrame d(*chain);

  const std::string path = "root://eoscms.cern.ch//store/group/phys_smp/emanca/data/*.root";
  
//   ROOT::RDataFrame d(treename, filename);

  ROOT::RDataFrame d(treename, path);
  
//   auto d2 = d.Define("corParms", correctParms, { "refParms", "jacrefv", "globalidxv" });
  
//   auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "run", "lumi", "event", "corParms" } );
  
  std::cout << "starting snapshot:" << std::endl;
//     auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "corParms" } );
  
  auto d3 = d.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "nJacRef", "nParms", "jacrefv" } );

    
  std::cout << "done snapshot:" << std::endl;

}
