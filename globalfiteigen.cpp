#include </usr/include/eigen3/Eigen/Dense>
#include </usr/include/eigen3/Eigen/Sparse>
#include</usr/include/eigen3/Eigen/IterativeLinearSolvers>
#include "TFile.h"
#include "TTree.h"
#include <vector>
#include <iostream>
#include <ROOT/RDataFrame.hxx>


using namespace Eigen;

class GradHelper : public ROOT::Detail::RDF::RActionImpl<GradHelper> {

public:
  using Result_t = std::pair<VectorXd, MatrixXd>;
  
  GradHelper(unsigned int nparms) : nparms_(nparms), grad_(new Result_t()) {}   
  GradHelper(GradHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
  GradHelper(const GradHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}
  
  std::shared_ptr<Result_t> GetResultPtr() const { return grad_; }

  void Exec(unsigned int slot, unsigned int idx, double gradelem,  ROOT::VecOps::RVec<double> const& hessrow) {
//   void Exec(unsigned int slot, ULong64_t idx, double gradelem,  ROOT::VecOps::RVec<double> const& hessrow) {
    grad_->first[idx] = gradelem;
    grad_->second.row(idx) = Map<const VectorXd>(hessrow.data(), hessrow.size());
//     grad_->second.row(idx) = Map<const VectorXd>(hessrow.data(), nparms_);
  }
  void InitTask(TTreeReader *, unsigned int) {}

  void Initialize() {
    grad_->first.resize(nparms_);
    grad_->second.resize(nparms_, nparms_);   
  }

    
  void Finalize() {}

   std::string GetActionName(){
      return "GradHelper";
   }
   
   
private:
  unsigned int nparms_;
  std::shared_ptr<Result_t> grad_;
   
};

int main() {
  
  Eigen::initParallel();
  ROOT::EnableImplicitMT();

  std::cout << "eigen nthreads" << std::endl;
  std::cout << Eigen::nbThreads() << std::endl;
  
//   TFile *f = TFile::Open("combinedgrads.root");
//   TTree *tree = (TTree*)f->Get("tree");
  
//   const char* filenameinfo = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v23_ActualGen/201201_232503/0000/globalcor_0_1.root";
  
//   const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v30_Gen210206_025446/0000/globalcor_0_1.root";
//   const char* filenameinfo = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v4_Gen_quality/210212_220639/0000/globalcor_0_1.root";
  
//   const char* filenameinfo = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v32_Rec/210214_182643/0000/globalcor_0_1.root";
//   const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v33_Gen_idealquality/210307_155804/0000/globalcor_0_1.root";
//     const char* filenameinfo = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v37_Gen_quality/210311_001434/0000/globalcor_0_1.root";
//     const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v107_GenJpsiPhotosSingle_quality/210719_142518/0000/globalcor_0_1.root";
    
    const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v148_RecJpsiPhotos_quality_constraint/210811_193709/0000/globalcor_0_1.root";
    
//     const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v124_Gen_quality/210729_121830/0000/globalcor_0_1.root";
//     const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v61_Gen_quality/210505_074057/0000/globalcor_0_1.root";
//   const char* filenameinfo = "/data/shared/muoncal/MuonGUNUL2016Fwd_v33_Gen_idealquality/210228_002318/0000/globalcor_0_1.root";


// //     const std::string filenameresults = "correctionResults_iter0.root";
//     const std::string filenameresults = "correctionResults_v107_gunfullpt.root";
    const std::string filenameresults = "";


//   const char* filenameinfo = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorGen_v29/200901_214453/0000/globalcorgen_100.root";
//   const char* filenameinfo = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorRec_v28/200829_122617/0000/globalcor_1.root";

  TFile *finfo = TFile::Open(filenameinfo);
//   TTree *runtree = static_cast<TTree*>(finfo->Get("tree"));
//   TTree *runtree = static_cast<TTree*>(finfo->Get("globalCorGen/runtree"));
  TTree *runtree = static_cast<TTree*>(finfo->Get("runtree"));

  const unsigned int nparms = runtree->GetEntries();
  
  
  std::vector<int> parmtypes(nparms, -1);
  std::vector<int> subdets(nparms, -1);
  std::vector<int> layers(nparms, -1);
  std::vector<float> xis(nparms, -1.);
  
  unsigned int iidx;
  int parmtype;
  int subdet;
  int layer;
  float xi;
  runtree->SetBranchAddress("iidx", &iidx);
  runtree->SetBranchAddress("parmtype", &parmtype);
  runtree->SetBranchAddress("subdet", &subdet);
  runtree->SetBranchAddress("layer", &layer);
  runtree->SetBranchAddress("xi", &xi);
  for (unsigned int i=0; i<runtree->GetEntries(); ++i) {
    runtree->GetEntry(i);
    parmtypes[iidx] = parmtype;
    subdets[iidx] = subdet;
    layers[iidx] = layer;
    xis[iidx] = xi;
  }
  
  finfo->Close();
  
  std::vector<float> oldres(nparms, 0.);
  if (!filenameresults.empty()) {
    TFile *fold = TFile::Open(filenameresults.c_str());
    TTree *treeold = static_cast<TTree*>(fold->Get("parmtree"));
    
    float val;
    treeold->SetBranchAddress("x", &val);
    
    for (unsigned int iparm = 0; iparm < nparms; ++iparm) {
      treeold->GetEntry(iparm);
      
      float outval = val;
//       if (parmtypes[iparm] == 7) {
//         outval = std::max(val, -0.95f*xis[iparm]);
//       }
      
      if (parmtypes[iparm] < 6) {
        oldres[iparm] = -outval;
      }
        
//       oldres[iparm] = outval;

      std::cout << "iparm = " << iparm << " oldres = " << oldres[iparm] << std::endl;
      
    }
    
    fold->Close();
  }
  
  
//   assert(0);
  
  GradHelper gradhelper(nparms);
  
  const std::string filename = "combinedgrads.root";
//   const std::string filename = "combinedgrads_v61_gen.root";
//   const std::string filename = "combinedgrads_v66_genfixed.root";
  
//   const std::string filename = "combinedgradsgen.root";
//   const std::string filename = "combinedgrads_jpsi.root";
  
//   const std::string filename = "results_V33_01p567quality/combinedgrads.root";
  
//   const std::string filename = "combinedgradsrec.root";
  ROOT::RDataFrame dgrad("tree", filename);
  
//   auto dgradtmp = dgrad.Filter("idx<
  
  auto grads = dgrad.Book<unsigned int, double, ROOT::VecOps::RVec<double>>(std::move(gradhelper), { "idx", "gradelem", "hessrow" });
//   auto grads = dgrad.Book<ULong64_t, double, ROOT::VecOps::RVec<double>>(std::move(gradhelper), { "rdfentry_", "gradelem", "hessrow" });
  
  
  VectorXd& grad = grads->first;
  MatrixXd& hess = grads->second;
  
  //fill lower diagonal values of hessian
//   hess.triangularView<StrictlyLower>() = hess.triangularView<StrictlyUpper>().transpose();
  
  
  //fill lower diagonal values of hessian
  std::cout << "filling lower triangular values" << std::endl;
//     hess.triangularView<StrictlyLower>() = hess.triangularView<StrictlyUpper>().transpose();

  
#pragma omp parallel for
  for (unsigned int i = 0; i<nparms; ++i) {
    hess.block<1, Dynamic>(i, 0, 1, i) = hess.block<Dynamic, 1>(0, i, i, 1).transpose().eval();
  }
  
  std::cout << "adding priors" << std::endl;
  for (unsigned int i=0; i<parmtypes.size(); ++i) {
    const int parmtype = parmtypes[i];
    const int subdet = subdets[i];
    const int layer = layers[i];
    const float xi = xis[i];
    
    
    std::cout << "i = " << i << " parmtype = " << parmtype << " before: grad = " << grad(i) << " hess = " << hess(i,i) << std::endl;

    
//     if (subdet == 0 && layer == 1) {
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.;
//     }
    
//     if (parmtype == 1 && (subdet == 2 || subdet == 3)) {
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.;
//     }
//     if (parmtype == 2 && subdet < 4) {
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.;      
//     }
    
//     if (parmtype == 1 && subdet>1) {
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.;
//     }
//     if (parmtype == 2) {
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.;      
//     }
    
//     if (parmtype==0) {
//       // rotation (in-plane, ie around z axis)
//       // 0.01 radians
//       hess(i,i) += 2.*1./pow(1e-2, 2);
//     }
//     else if (parmtype>0 && parmtype<3) {
//       // translation (in-plane)
//       hess(i,i) += 2.*1./pow(1e-1, 2);
//     }
//     else if (parmtype==3) {
//       // b-field
// //       hess.row(i) *= 0.;
// //       hess.col(i) *= 0.;
// //       grad[i] = 0.;
//       hess(i,i) += 2.*1./pow(0.2, 2);
//     }
//     else if (parmtype==4) {
//       // material
// //       hess.row(i) *= 0.;
// //       hess.col(i) *= 0.;
// //       grad[i] = 0.;
//       hess(i,i) += 2.*1./pow(1e-4, 2);
//     }
    
    if (parmtype == 0) {
      
//       if (subdet > 3) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.;
//       }
      
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.; 
      
//       const double siga = 1e-1;
      
      const double siga = 5e-3;
//       const double siga = 5e-4;
//       const double siga = 1e-7;
      
//       hess(i,i) += 2.*1./pow(1e-1, 2);
      grad(i) += 2.*oldres[i]/siga/siga;
      hess(i,i) += 2./siga/siga;
    }
    else if (parmtype == 1) {
      // translation
//       if (parmtype == 1 && subdet>1 && subdet<4) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.; 
//       }
//       if (parmtype == 1 && subdet>1) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.; 
//       }
      
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.; 
      
//       const double siga = 1e-1;
//       const double siga = 5e-3;
      const double siga = subdet < 2 ? 5e-3 : 5e-1;
      
//       const double siga = 5e-4;
      
//       const double siga = 1e-7;
      
      grad(i) += 2.*oldres[i]/siga/siga;
      hess(i,i) += 2./siga/siga;
//       hess(i,i) += 2.*1./pow(1e-1, 2);
//       hess(i,i) += 2.*1./pow(1e-2, 2);
    }
    else if (parmtype == 2) {
      //linearization is a bad approximation?
      
//       if (subdet>1) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.;        
//       }
//       if (subdet > 0) {
//       if (true) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.;
//       }
      
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.; 

      const double siga = 1e-1;
      grad(i) += 2.*oldres[i]/siga/siga;
      hess(i,i) += 2./siga/siga;
      
//       hess(i,i) += 2.*1./pow(1e-1, 2);
    }
    else if (parmtype < 5) {
      // rotations out of plane
      // linearization is a bad approximation?
      // 0.01 radians
//       if (subdet>1) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.;        
//       }
//       if (subdet > 0) {
//       if (true) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.;
//       }
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.; 
      
      const double siga = 1e-2;
      grad(i) += 2.*oldres[i]/siga/siga;
      hess(i,i) += 2./siga/siga;
      
//       hess(i,i) += 2.*1./pow(1e-2, 2);
    }    
    else if (parmtype < 6) {
      // rotation
      // 0.01 radians
//       if (subdet > 1) {
//       if (subdet > 3) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.;
//       }
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.; 
      
//       const double siga = 1e-2;
      const double siga = 5e-3;
//       const double siga = 5e-5;
//       const double siga = 1e-9;
      
      
      
      grad(i) += 2.*oldres[i]/siga/siga;
      hess(i,i) += 2./siga/siga;
      

//       hess(i,i) += 2.*1./pow(1e-2, 2);
    }
    else if (parmtype==6) {
      // b-field
//       if (subdet < 2) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.; 
//       }
      
//       hess.row(i) *= 0.;
//       hess.col(i) *= 0.;
//       grad[i] = 0.;
      
//       if (subdet==5) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.;
//       }
      
//       const double sigb = 0.2;
      //       const double sigb = 0.0038;

      
      
      const double sigb = 0.038;
      grad(i) += 2.*oldres[i]/sigb/sigb;
      hess(i,i) += 2.*1./sigb/sigb;
      
//       const double sigb = 1e-6;
//       grad(i) += -2.*0.038/sigb/sigb;
//       hess(i,i) += 2.*1./sigb/sigb;
      
      
    }
    else if (parmtype==7) {
      // material
//       if (subdet < 2 ) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.; 
//       }

//       if (subdet==5) {
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.;
//       }
      
//         hess.row(i) *= 0.;
//         hess.col(i) *= 0.;
//         grad[i] = 0.;
//       
//       const double sigxi = 1e-5;
//       const double sigxi = 1e-4;
//       const double sigxi = 1e-2;
//       const double sigxi = subdet < 2 ? 1e-3 : 0.5*xi;
//       const double sigxi = 0.2*xi;
//       const double sigxi = 2.0*xi;
//       const double sigxi = 0.2*xi;
      
      
      //"nominal" config
//       const double sigxi = 2.0*xi;
      const double sigxi = 1.0;
//       const double sigxi = 1000.0;
//       const double sigxi = 0.5*xi;
//       const double sigxi = 0.1*xi;
//       const double sigxi = 1e-2;

      grad(i) += 2.*oldres[i]/sigxi/sigxi;
      hess(i,i) += 2.*1./sigxi/sigxi;
      
      
//       grad(i) += -2.*0.1/1e-4/1e-4;
//       hess(i,i) += 2.*1./1e-4/1e-4;
      
//       hess(i,i) += 2.*1./pow(1e-4, 2);
    }
    std::cout << "i = " << i << " parmtype = " << parmtype << " after:  grad = " << grad(i) << " hess = " << hess(i,i) << std::endl;

    
    
    
  }
  
//   assert(0);
  
//   std::cout << "computing svd decomposition" << std::endl;
// 
//   BDCSVD<MatrixXd> svd(hess);
//   std::cout << "singular values:" << std::endl;
//   std::cout << svd.singularValues().transpose() << std::endl;
  
//   std::cout << "computing eigenvalues" << std::endl;
//   SelfAdjointEigenSolver<MatrixXd> es(hess, EigenvaluesOnly);
  
//   std::cout << "eigenvalues:" << std::endl;
//   std::cout << es.eigenvalues().transpose() << std::endl;
  
//   assert(false);
  
//   std::cout << "convert to sparse" << std::endl;
//   SparseMatrix<double> sparsehess = hess.sparseView();
//   
//   BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> solver;
//   
//   std::cout << "compute" << std::endl;
//   solver.compute(sparsehess);
//   
//   std::cout << "solving" << std::endl;
//   VectorXd xout = -solver.solve(grad);
//   
//   return 0;
//   VectorXd xout = grad;
  
//   std::cout << hess.diagonal() << std::endl;
  
  std::cout << "decomposing" << std::endl;
// 
//   
// //   LDLT<Ref<MatrixXd>, Upper> hessd(hess);
//   
  
  
  PartialPivLU<Ref<MatrixXd>> hessd(hess);
   
  std::cout << "solving" << std::endl;

  VectorXd xout = -hessd.solve(grad);
  
  std::cout << "computing errors" << std::endl;
  
  VectorXd errs = 2./hess.diagonal().array().sqrt();
  
  
//   constexpr unsigned int testsize = 10000;
//   
//   MatrixXd smallhess = hess.topLeftCorner<testsize,testsize>();
//   const VectorXd smallgrad = grad.head<testsize>();
//   
//   PartialPivLU<Ref<MatrixXd>> hessd(smallhess);
// //   LDLT<Ref<MatrixXd>, Upper> hessd(hess);
// //   LLT<Ref<MatrixXd>, Upper> hessd(hess);
//  
//   
//   
//   std::cout << "solving" << std::endl;
// 
//   VectorXd xout = -hessd.solve(smallgrad);
//   
//   std::cout << "computing errors" << std::endl;
// 
//   MatrixXd cov = hessd.inverse();
//   
// //   MatrixXd cov = MatrixXd::Zero(hess.rows(), hess.cols());
// //   
// //   #pragma omp parallel for
// //   for (unsigned int i = 0; i < hess.rows(); ++i) {
// //     VectorXd onehot = VectorXd::Zero(hess.rows());
// //     onehot[i] = 1.;
// //     cov.col(i) = hessd.solve(onehot);
// //   }
// //   
//   std::cout << "done computing errors" << std::endl;

  
//   const MatrixXd cov = hessd.solve(MatrixXd::Identity(testsize,testsize));
  
//   VectorXd errs = hessd.solve(MatrixXd::Identity(testsize,testsize)).diagonal().array().sqrt();
  
//   VectorXd errs = cov.diagonal().array().sqrt();
  
  
  
//   VectorXd errs = std::sqrt(2.)*hessd.inverse().diagonal().array().sqrt();

  
//   
//   std::cout << xout << std::endl;
//   
//   return 0;
  
//   TFile *fout = TFile::Open("correctionResults.root","RECREATE");
  TFile *fout = TFile::Open("correctionResultsdebug.root","RECREATE");
  
  
//   TFile *fout = TFile::Open("correctionResultsgen.root","RECREATE");
  TTree *idxmaptree = new TTree("idxmaptree", "");
  
  unsigned int idx;
  idxmaptree->Branch("iidx",&iidx);
  idxmaptree->Branch("idx",&idx);
  for (unsigned int i=0; i<nparms; ++i) {
    iidx = i;
    idx = i;
    idxmaptree->Fill();
  }
  
  TTree *parmtree = new TTree("parmtree", "");
  
  float x;
  float err;
  
  parmtree->Branch("idx", &idx);
  parmtree->Branch("x", &x);
  parmtree->Branch("err", &err);
  
  for (unsigned int i=0; i<nparms; ++i) {
    idx = i;
    x = xout[i];
//     x = xout[i] + oldres[i];
    
    
//     if (parmtypes[i] == 7) {
//       x = std::max(x, -0.95f*xis[i]);
//     }
    err = errs[i];
    parmtree->Fill();
  }
  
  fout->Write();
  fout->Close();
  
  
  
  
}
