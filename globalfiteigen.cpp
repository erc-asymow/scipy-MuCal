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
    grad_->first[idx] = gradelem;
    grad_->second.row(idx) = Map<const VectorXd>(hessrow.data(), hessrow.size());
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
  
  const char* filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantint4genhelixprecrefb//globalcor_0.root";

//   const char* filenameinfo = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorGen_v29/200901_214453/0000/globalcorgen_100.root";
//   const char* filenameinfo = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorRec_v28/200829_122617/0000/globalcor_1.root";

  TFile *finfo = TFile::Open(filenameinfo);
//   TTree *runtree = static_cast<TTree*>(finfo->Get("tree"));
//   TTree *runtree = static_cast<TTree*>(finfo->Get("globalCorGen/runtree"));
  TTree *runtree = static_cast<TTree*>(finfo->Get("runtree"));

  const unsigned int nparms = runtree->GetEntries();
  
  
  std::vector<int> parmtypes(nparms, -1);
  
  unsigned int iidx;
  int parmtype;
  runtree->SetBranchAddress("iidx", &iidx);
  runtree->SetBranchAddress("parmtype", &parmtype);
  for (unsigned int i=0; i<runtree->GetEntries(); ++i) {
    runtree->GetEntry(i);
    parmtypes[iidx] = parmtype;
  }
  
  finfo->Close();
  
  
  
  GradHelper gradhelper(nparms);
  
  const std::string filename = "combinedgrads.root";
//   const std::string filename = "combinedgradsrec.root";
  ROOT::RDataFrame dgrad("tree", filename);
  
  auto grads = dgrad.Book<unsigned int, double, ROOT::VecOps::RVec<double>>(std::move(gradhelper), { "idx", "gradelem", "hessrow" });
  
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
    if (parmtype<2) {
      hess(i,i) += 2.*1./pow(1e-1, 2);
    }
    else if (parmtype==2) {
      hess(i,i) += 2.*1./pow(0.2, 2);
    }
    else if (parmtype==3) {
      hess(i,i) += 2.*1./pow(1e-4, 2);
    }
  }
  
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
//   
  std::cout << "solving" << std::endl;
//   
  VectorXd xout = -hessd.solve(grad);
  
  std::cout << "computing errors" << std::endl;
  
  VectorXd errs = 2./hess.diagonal().array().sqrt();
  
//   VectorXd errs = std::sqrt(2.)*hessd.inverse().diagonal().array().sqrt();

  
//   
//   std::cout << xout << std::endl;
//   
//   return 0;
  
  TFile *fout = TFile::Open("correctionResults.root","RECREATE");
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
    err = errs[i];
    parmtree->Fill();
  }
  
  fout->Write();
  fout->Close();
  
  
  
  
}
