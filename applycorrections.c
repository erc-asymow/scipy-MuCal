#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TMatrix.h"
#include "TSystem.h"
#include <ROOT/TTreeProcessorMT.hxx>
#include </usr/include/eigen3/Eigen/Dense>

void applycorrections() {
  
//   filename = "root://eoscms.cern.ch///store/group/phys_smp/emanca/data/*.root";
//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v3/200817_213237/0000/*.root";
  
//   const std::string filename = "/eos/cms/store/cmst3/group/wmass/bendavid/muoncal/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v3/200817_213237/0000/*.root";
//   const std::string filename = "/data/bendavid/muoncaldatalarge/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v4/200820_002243/0000/*.root";
//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v23_Gen/201201_232356/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v30_Rec210206_025546/0000/globalcor_*.root";
//   const std::string filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recnopxb/globalcor_*.root";
//   const std::string filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recnopxbgt1/globalcor_*.root";
//   const std::string filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealpxbgenxygenstart/globalcor_*.root";
  
//     const std::string filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recideal/globalcor_*.root";
//     const std::string filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recnonidealdebugreallynotemplate/globalcor_*.root";
//     const std::string filename = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Rec_quality/210212_165132/0000/globalcor_*.root";
//     const std::string filename = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Rec_noquality/210212_165222/0000/globalcor_*.root";
//     const std::string filename = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v4_Rec_quality/210212_220352/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/resultsqualitycomparefulldet/MuonGunUL2016_v31_Rec/210213_015544/0000/globalcor_*.root";
//   const std::string filename = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v32_Rec/210214_182643/0000/globalcor_*.root";
  
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v33_Rec_quality/210228_002026/0000/globalcor_*.root";
//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v37_Rec_quality/210311_001553/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v38_Rec_quality/210403_232303/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v39_Rec_quality/210403_232723/0000/globalcor_*.root";
  
  const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v44_Rec_quality/210411_162903/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v33_Rec_noquality/210228_002110/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGUNUL2016Fwd_v33_Rec_idealquality/210228_002451/0000/globalcor_*.root";
  
  
//   gSystem->Setenv("XRD_REQUESTTIMEOUT","30");
  
  Eigen::initParallel();
  ROOT::EnableImplicitMT(32);
//   ROOT::TTreeProcessorMT::SetMaxTasksPerFilePerWorker(1);

  
//   TFile *fcor = TFile::Open("correctionResults.root");
  TFile *fcor = TFile::Open("results_V37_01p567quality/correctionResults.root");
//   TFile *fcor = TFile::Open("correctionResultsV23.root");
//   TFile *fcor = TFile::Open("correctionResultsIdeal.root");
//   TFile *fcor = TFile::Open("correctionResultsV30.root");
//   TFile *fcor = TFile::Open("correctionResults2016.root");
//   TFile *fcor = TFile::Open("correctionResultsModule.root");
//   TFile *fcor = TFile::Open("correctionResultsFullGen.root");
  
//   TFile *fcor = TFile::Open("correctionResultsNoElossNoBfield.root");
  
  
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
  
//   using Vector5f = Eigen::Matrix<float, 5, 1>;

  
  auto correctParms = [&xout, &idxmap](ROOT::VecOps::RVec<float> const& refParms, ROOT::VecOps::RVec<float> const& jacRef, ROOT::VecOps::RVec<unsigned int> const& globalidxv) {
    using Vector5f = Eigen::Matrix<float, 5, 1>;
    
    const unsigned int nparms = globalidxv.size();
    Eigen::VectorXf xtrk(nparms);
    for (unsigned int i = 0; i < nparms; ++i) {
      xtrk[i] = xout[idxmap[globalidxv[i]]];
    }
    

    
    const Eigen::Map<const Vector5f> refParmsEig(refParms.data());
    const Eigen::Map<const Eigen::Matrix<float, 5, Eigen::Dynamic, Eigen::RowMajor>> jacRefEig(jacRef.data(), 5, nparms);

//     std::cout << "qopref = " << refParms[0] << std::endl;
//     for (unsigned int i = 0; i < nparms; ++i) {
//       std::cout << "i = " << i << " xtrk[i] = " << xtrk[i] << " jacref(0,i) = " << jacRefEig(0,i) << " diff = " << jacRefEig(0,i)*xtrk[i] << std::endl;
//     }
    
    std::array<float, 5> corParms;
    Eigen::Map<Vector5f> corParmsEig(corParms.data());
    
//     corParmsEig = refParmsEig + jacRefEig*xtrk;
    corParmsEig = (refParmsEig.cast<double>() + jacRefEig.cast<double>()*xtrk.cast<double>()).cast<float>();
    
//     std::cout << "total diff = " << corParmsEig[0] - refParmsEig[0] << std::endl;
    
    return corParms;
    
  };
  
//   auto correctParms = [&xout, &idxmap](ROOT::VecOps::RVec<float> const& refparmsv, ROOT::VecOps::RVec<float> const& jacrefv, ROOT::VecOps::RVec<unsigned int> const& globalidxv) -> std::array<float, 5> {
//     
//     const unsigned int nparms = globalidxv.size();
// //     Eigen::VectorXf xtrk(nparms);
//     TMatrixT<float> xtrk(nparms, 1);
//     for (unsigned int i = 0; i < nparms; ++i) {
// //       xtrk[i] = xout[idxmap[globalidxv[i]]];
//       xtrk(i,0) = xout[idxmap[globalidxv[i]]];
//     }
//     
//     
//     TMatrixT<float> refParms(5,1);
//     for (unsigned int i=0; i<5; ++i) {
//       refParms(i,0) = refparmsv[i];
// //       refParms(i,0) = refparmsv.at(i,0.);
//     }
//     
// //     Eigen::Matrix<float, 5, Eigen::Dynamic> jacRefEig(5, nparms);
//     TMatrixT<float> jacRef(5, nparms);
//     for (unsigned int i=0; i<5; ++i) {
//       for (unsigned int j=0; j<nparms; ++j) {
//         jacRef(i,j) = jacrefv[5*i + j];
// //         jacRef(i,j) = jacrefv.at(5*i + j, 0.);
// //         jacRefEig(i,j) = 0.;
//       }
//     }
//     
//     const TMatrixT<float> corParms = refParms + jacRef*xtrk;
//     
//     std::array<float, 5> corParmsArr;
//     for (unsigned int i=0; i<5; ++i) {
//       corParmsArr[i] = corParms(i,0);
//     }
//     
//     return corParmsArr;
//     
// //     assert(jacRef.size() == 5*nparms);
// //     assert(refParms.size() == 5);
// // //     
// // //     const Eigen::Map<const Vector5f> refParmsEig(refParms.data());
// // //     const Eigen::Map<const Eigen::Matrix<float, 5, Eigen::Dynamic, Eigen::RowMajor>> jacRefEig(jacRef.data(), 5, nparms);
// // 
// // //     const Vector5f refParmsEig = Eigen::Map<const Vector5f>(*refParms.data());
// // //     const Eigen::Matrix<float, 5, Eigen::Dynamic> jacRefEig = Eigen::Map<const Eigen::Matrix<float, 5, Eigen::Dynamic, Eigen::RowMajor>>(*jacRef.data(), 5, nparms);
// // 
// //     Vector5f refParmsEig;
// //     for (unsigned int i=0; i<5; ++i) {
// //       refParmsEig[i] = float(refParms[i]);
// // //       refParmsEig[i] = 0.;
// //     }
// //     
// //     Eigen::Matrix<float, 5, Eigen::Dynamic> jacRefEig(5, nparms);
// //     for (unsigned int i=0; i<5; ++i) {
// //       for (unsigned int j=0; j<nparms; ++j) {
// //         jacRefEig(i,j) = float(jacRef[5*i + j]);
// // //         jacRefEig(i,j) = 0.;
// //       }
// //     }
// //     
// //     
// // //     ROOT::VecOps::RVec<float> corParms(5);
// //     std::array<float, 5> corParms;
// //     Eigen::Map<Vector5f>(corParms.data()) = refParmsEig + jacRefEig*xtrk;
// //     
// // //     Eigen::Map<Vector5f> corParmsEig(corParms.data());
// //     
// // //     corParmsEig = refParmsEig + jacRefEig*xtrk;
// //     
// // //     const Vector5f corParmsEig = refParmsEig + jacRefEig*xtrk;
// //     
// // //     std::array<float, 5> corParms;
// // //     for (unsigned int i=0; i<5; ++i) {
// // //       corParms[i] = corParmsEig[i];
// // //     }
// // //     Eigen::Map<Vector5f>(corParms.data()) = corParmsEig;
// //     
// // //     std::cout << "refParmsEig" << std::endl;
// // //     std::cout << refParmsEig << std::endl;
// // //     std::cout << "corParms" << std::endl;
// // //     std::cout << Eigen::Map<Vector5f>(corParms.data()) << std::endl;
// //     
// //     return corParms;
//     
//   };
  
//   std::string treename = "globalCor/tree";
  std::string treename = "tree";
  std::string outfilename = "correctedTracks.root";
  
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
//   auto dcut = d.Filter("genPt>0. && genEta<-2.3");
//   auto dcut = d.Filter("genPt>0.");
  
  auto d2 = d.Define("corParms", correctParms, { "refParms", "jacrefv", "globalidxv" });
  
//   auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "run", "lumi", "event", "corParms" } );
  
  std::cout << "starting snapshot:" << std::endl;
    auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "corParms", "hitidxv", "nValidHits" } );

    
  std::cout << "done snapshot:" << std::endl;

}
