#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include "TTree.h"
#include </usr/include/eigen3/Eigen/Dense>

void applycorrections(const char *filename="/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root") {
  
//   filename = "root://eoscms.cern.ch///store/group/phys_smp/emanca/data/*.root";
  
  Eigen::initParallel();
  ROOT::EnableImplicitMT();
  
  TFile *fcor = TFile::Open("correctionResults.root");
  
  TTree *idxmaptree = static_cast<TTree*>(fcor->Get("idxmaptree"));
  
  unsigned int idx;
  idxmaptree->SetBranchAddress("idx", &idx);
  
  std::vector<unsigned int> idxmap;
  idxmap.reserve(idxmaptree->GetEntries());
  
  for (long long ientry=0; ientry < idxmaptree->GetEntries(); ++ientry) {
    idxmaptree->GetEntry(ientry);
    idxmap.push_back(idx);
  }
  
  TTree *parmtree = static_cast<TTree*>(fcor->Get("parmtree"));
  
  float x;
  parmtree->SetBranchAddress("x", &x);
  
  std::vector<float> xout;
  xout.reserve(parmtree->GetEntries());
  
  for (long long ientry=0; ientry < parmtree->GetEntries(); ++ientry) {
    parmtree->GetEntry(ientry);
    xout.push_back(x);
  }
  
  auto correctParms = [&xout, &idxmap](ROOT::VecOps::RVec<float> const& refParms, ROOT::VecOps::RVec<float> const& jacRef, ROOT::VecOps::RVec<unsigned int> const& globalidxv) {
    using Vector5f = Eigen::Matrix<float, 5, 1>;
    
    const unsigned int nparms = globalidxv.size();
    Eigen::VectorXf xtrk(nparms);
    for (unsigned int i = 0; i < nparms; ++i) {
      xtrk[i] = xout[idxmap[globalidxv[i]]];
    }
    
    const Eigen::Map<const Vector5f> refParmsEig(refParms.data());
    const Eigen::Map<const Eigen::Matrix<float, 5, Eigen::Dynamic, Eigen::RowMajor>> jacRefEig(jacRef.data(), 5, nparms);

    std::array<float, 5> corParms;
    Eigen::Map<Vector5f> corParmsEig(corParms.data());
    
    corParmsEig = refParmsEig + jacRefEig*xtrk;
    
    return corParms;
    
  };
  
  std::string treename = "tree";
  std::string outfilename = "correctedTracks.root";
  
  ROOT::RDataFrame d(treename, filename);
  
  auto d2 = d.Define("corParms", correctParms, { "refParms", "jacrefv", "globalidxv" });
  
//   auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "run", "lumi", "event", "corParms" } );
  
    auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "corParms" } );

}
