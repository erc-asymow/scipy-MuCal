#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TMatrix.h"
#include "TSystem.h"
#include <ROOT/TTreeProcessorMT.hxx>
#include </usr/include/eigen3/Eigen/Dense>

#include "Math/GenVector/PxPyPzM4D.h"

#include "Math/GenVector/LorentzVector.h"
#include "Math/Vector4Dfwd.h"

// float calcmass(ROOT::VecOps::RVec<float> const& parms0, ROOT::VecOps::RVec<float> const& parms1) {
float calcmass(const std::array<float, 3> &parms0, const std::array<float, 3> &parms1) {

  constexpr double mmu = 0.1056583745;

  
  const double p0 = std::abs(1./parms0[0]);
  const double theta0 = M_PI_2 - parms0[1];
  const double phi0 = parms0[2];
  
  const double p1 = std::abs(1./parms1[0]);
  const double theta1 = M_PI_2 - parms1[1];
  const double phi1 = parms1[2];
  
  const ROOT::Math::PxPyPzMVector mom0(p0*std::sin(theta0)*std::cos(phi0), p0*std::sin(theta0)*std::sin(phi0), p0*std::cos(theta0), mmu);
  
  const ROOT::Math::PxPyPzMVector mom1(p1*std::sin(theta1)*std::cos(phi1), p1*std::sin(theta1)*std::sin(phi1), p1*std::cos(theta1), mmu);
  
  return (mom0 + mom1).mass();
  
}

ROOT::Math::PxPyPzMVector calcmom(const std::array<float, 3> &parms) {
  constexpr double mmu = 0.1056583745;

  const double p = std::abs(1./parms[0]);
  const double theta = M_PI_2 - parms[1];
  const double phi = parms[2];

  return ROOT::Math::PxPyPzMVector(p*std::sin(theta)*std::cos(phi), p*std::sin(theta)*std::sin(phi), p*std::cos(theta), mmu);
}

void applycorrectionsjpsi() {
  
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
  
//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v44_Rec_quality/210411_162903/0000/globalcor_*.root";

  
  //   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v45_Rec_quality_bs_v2/210411_194557/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v45_Rec_quality_nobs_v2/210413_084740/0000/globalcor_*.root";
  
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v61_Rec_quality_nobs/210505_074303/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v107_RecJpsiPhotosSingle_quality/210719_142322/0000/globalcor_0_1.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v80_Rec_quality_nobs/210703_155540/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v68_Rec_quality_nobs/210621_095523/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v69_Rec_quality_nobs/210621_181808/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v70_Rec_quality_nobs/210622_184454/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v66_RecJpsiPhotosSingle_quality/210509_200944/0000/globalcor_*.root";
  
  
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v33_Rec_noquality/210228_002110/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGUNUL2016Fwd_v33_Rec_idealquality/210228_002451/0000/globalcor_*.root";
  
  TChain chain("tree");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v66_Rec_quality_nobs/210509_195947/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v84_RecJpsiPhotosSingle_quality/210707_164015/0000/globalcor_*.root");

//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v87_Rec_quality_nobs/210709_013206/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v87_RecJpsiPhotosSingle_quality/210709_013609/0000/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v88_Rec_quality_nobs/210709_112232/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v88_RecJpsiPhotosSingle_quality/210709_112548/0000/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v70_Rec_quality_nobs/210622_184454/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v90_RecJpsiPhotosSingle_quality/210711_155348/0000/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v92_Rec_quality_nobs/210712_163945/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v92_RecJpsiPhotosSingle_quality/210712_163622/0000/globalcor_*.root");
  

//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v93_Rec_quality_nobs/210712_211420/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v93_RecJpsiPhotosSingle_quality/210712_211742/0000/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v94_Rec_quality_nobs/210713_193329/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v94_RecJpsiPhotosSingle_quality/210713_193009/0000/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v97_Rec_quality_nobs/210714_153858/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v97_RecJpsiPhotosSingle_quality/210714_153322/0000/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v98_Rec_quality_nobs/210715_164101/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v98_RecJpsiPhotosSingle_quality/210715_163814/0000/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_Rec_quality_nobs/210719_142714/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_Rec_quality_nobs/210719_142714/0001/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_RecJpsiPhotosSingle_quality/210719_142322/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_RecJpsiPhotosSingle_quality/210719_142322/0001/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v125_RecJpsiPhotos_quality_constraint_nograds/210730_184742/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v125_RecJpsiPhotos_quality_constraint_nograds/210730_184742/0001/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v125a_RecJpsiPhotos_quality_noconstraint/210730_185206/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v125a_RecJpsiPhotos_quality_noconstraint/210730_185206/0001/globalcor_*.root");

//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecJpsiPhotos_quality_constraintfsr28/210930_202515/0000/*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecJpsiPhotos_quality_constraintfsr28/210930_202515/0001/*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintfsr29/211013_161309/0000/*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintfsr29/211013_161309/0001/*.root");

//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v208_RecZMuMu_quality_nobs/211014_121741/0000/*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v208_RecZMuMu_quality_nobs/211014_121741/0001/*.root");


  chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiH_quality_constraintfsr29/211012_233512/0000/*.root");
  chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiG_quality_constraintfsr29/211012_233641/0000/*.root");
  chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataJPsiFpost_quality_constraintfsr29/211012_233806/0000/*.root");

//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataZMuMuH_quality/211012_180657/0000/*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataZMuMuG_quality/211012_180844/0000/*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v207_RecDataZMuMuFpost_quality/211012_181018/0000/*.root");

//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v202a_RecDataJPsiH_quality_constraintfsr28/210930_203700/0000/*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecDataJPsiG_quality_constraintfsr28/210930_204012/0000/*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecDataJPsiFpost_quality_constraintfsr28/210930_204138/0000/*.root");

//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v124_Rec_quality_nobs/210729_122020/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v124_Rec_quality_nobs/210729_122020/0001/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v124_RecJpsiPhotosSingle_quality/210729_122728/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v124_RecJpsiPhotosSingle_quality/210729_122728/0001/globalcor_*.root");
  
  
//   gSystem->Setenv("XRD_REQUESTTIMEOUT","30");
  
//   Eigen::initParallel();
  ROOT::EnableImplicitMT();
//   ROOT::TTreeProcessorMT::SetMaxTasksPerFilePerWorker(1);

  
  TFile *fcor = TFile::Open("correctionResults.root");
//   TFile *fcor = TFile::Open("plotscale_v107_cor/correctionResults.root");
  
  
//   TFile *fcor = TFile::Open("results_V37_01p567quality/correctionResults.root");
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
    using Vector3f = Eigen::Matrix<float, 3, 1>;
    
    const unsigned int nparms = globalidxv.size();
    Eigen::VectorXf xtrk(nparms);
    for (unsigned int i = 0; i < nparms; ++i) {
      xtrk[i] = xout[idxmap[globalidxv[i]]];
//       const unsigned int iidx = globalidxv[i];
//       const float scale = iidx >=47916 ? std::sqrt(2.) : 1.;
//       xtrk[i] = scale*xout[idxmap[iidx]];
    }
    

    
    const Eigen::Map<const Vector3f> refParmsEig(refParms.data());
    const Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor>> jacRefEig(jacRef.data(), 3, nparms);

//     std::cout << "qopref = " << refParms[0] << std::endl;
//     for (unsigned int i = 0; i < nparms; ++i) {
//       std::cout << "i = " << i << " xtrk[i] = " << xtrk[i] << " jacref(0,i) = " << jacRefEig(0,i) << " diff = " << jacRefEig(0,i)*xtrk[i] << std::endl;
//     }
    
    std::array<float, 3> corParms;
    Eigen::Map<Vector3f> corParmsEig(corParms.data());
    
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
  std::string outfilename = "correctedTracksjpsi.root";
  
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

  
//   ROOT::RDataFrame d(treename, filename);
  ROOT::RDataFrame d(chain);
//   auto dcut = d.Filter("genPt>0. && genEta<-2.3");
//   auto dcut = d.Filter("genPt>0.");
  
//   auto da = d.Filter("genEta > 1.3 && genEta < 1.5");
  
  auto d2 = d.Define("Muplus_corParms", correctParms, { "Muplus_refParms", "Muplus_jacRef", "globalidxv" });
  auto d3 = d2.Define("Muminus_corParms", correctParms, { "Muminus_refParms", "Muminus_jacRef", "globalidxv" });

//   auto d4 = d3.Define("Jpsicor_mass", calcmass, { "Muplus_corParms", "Muminus_corParms" });
  
  auto d4 = d3.Define("Mupluscor_mom", calcmom, { "Muplus_corParms" } );
  auto d5 = d4.Define("Muminuscor_mom", calcmom, { "Muminus_corParms" } );
  auto d6 = d5.Define("Jpsicor_mom", "Mupluscor_mom + Muminuscor_mom");

  auto d7 = d6.Define("Mupluscor_pt", "Mupluscor_mom.pt()");
  auto d8 = d7.Define("Mupluscor_eta", "Mupluscor_mom.eta()");
  auto d9 = d8.Define("Mupluscor_phi", "Mupluscor_mom.phi()");

  auto d10 = d9.Define("Muminuscor_pt", "Muminuscor_mom.pt()");
  auto d11 = d10.Define("Muminuscor_eta", "Muminuscor_mom.eta()");
  auto d12 = d11.Define("Muminuscor_phi", "Muminuscor_mom.phi()");

  auto d13 = d12.Define("Jpsicor_pt", "Jpsicor_mom.pt()");
  auto d14 = d13.Define("Jpsicor_eta", "Jpsicor_mom.eta()");
  auto d15 = d14.Define("Jpsicor_phi", "Jpsicor_mom.phi()");
  auto d16 = d15.Define("Jpsicor_mass", "Jpsicor_mom.mass()");


  auto const &allcols = d16.GetColumnNames();
  
  std::vector<std::string> outcols;
  for (const std::string &col : allcols) {
    if (col == "hesspackedv") {
      continue;
    }
    else if (col == "gradv") {
      continue;
    }
    else if (col == "globalidxv") {
      continue;
    }
    else if (col == "Muplus_jacRef") {
      continue;
    }
    else if (col == "Muminus_jacRef") {
      continue;
    }
    else if (col == "Muplus_refParms") {
      continue;
    }
    else if (col == "Muminus_refParms") {
      continue;
    }
    else if (col == "Mupluscor_mom") {
      continue;
    }
    else if (col == "Muminuscor_mom") {
      continue;
    }
    else if (col == "Jpsicor_mom") {
      continue;
    }

    outcols.push_back(col);
  }
  
//   auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "run", "lumi", "event", "corParms" } );
  
  std::cout << "starting snapshot:" << std::endl;
//     auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "corParms", "hitidxv", "nValidHits", "nValidPixelHits" } );
//     auto d3 = d2.Snapshot(treename, outfilename,  { "trackPt", "trackPtErr", "trackEta", "trackPhi", "trackCharge", "trackParms", "trackCov", "refParms", "refCov", "genParms", "genPt", "genEta", "genPhi", "genCharge", "corParms", "nValidHits", "nValidPixelHits" } );
//     
//   d4.Snapshot(treename, outfilename,  outcols);
  d16.Snapshot(treename, outfilename,  outcols);

    
  std::cout << "done snapshot:" << std::endl;

}
