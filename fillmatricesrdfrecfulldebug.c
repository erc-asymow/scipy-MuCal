#include <atomic>
#include <chrono>
#include <algorithm>
#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include <ROOT/TTreeProcessorMT.hxx>
#include <stdlib.h>  

class GradHelper : public ROOT::Detail::RDF::RActionImpl<GradHelper> {

public:
  using Result_t = std::vector<double>;
  
  GradHelper(unsigned int nparms) : nparms_(nparms), grad_(std::make_shared<Result_t>()) {}  
//   GradHelper(unsigned int nparms, std::shared_ptr<Result_t> grad) : nparms_(nparms), grad_(grad) {}
  GradHelper(GradHelper && other) = default;
//   GradHelper(GradHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
//   GradHelper(const GradHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}
  
  std::shared_ptr<Result_t> GetResultPtr() const { return grad_; }

  void Exec(unsigned int slot, ROOT::VecOps::RVec<float> const& vec, ROOT::VecOps::RVec<unsigned int> const& idxs) {
    std::vector<double>& grad = gradtmp_[slot];
    
    for (unsigned int i=0; i<vec.size(); ++i) {
      const unsigned int& idx = idxs[i];
      grad[idx] += vec[i];
    }
    
  }
  void InitTask(TTreeReader *, unsigned int) {}

  void Initialize() {
//     const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetImplicitMTPoolSize() : 1;
    const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetThreadPoolSize() : 1;
    gradtmp_.clear();
    gradtmp_.resize(nslots, std::vector<double>(nparms_, 0.));
    
    if (grad_->empty()) {
      grad_->clear();
      grad_->resize(nparms_, 0.);    
    }
  }

    
  void Finalize() {
    for (auto const& grad: gradtmp_) {
      for (unsigned int i=0; i<grad_->size(); ++i) {
        (*grad_)[i] += grad[i];
      }
    }
  }

   std::string GetActionName(){
      return "GradHelper";
   }
   
   
private:
  unsigned int nparms_;
  std::vector<std::vector<double> > gradtmp_;
  std::shared_ptr<Result_t> grad_;
   
};

class HessHelper : public ROOT::Detail::RDF::RActionImpl<HessHelper> {

public:
  using Result_t = std::vector<std::atomic<double> >;
//   using Result_t = std::vector<double>;
//   using Data_t = std::vector<std::atomic<double> >;
  
  HessHelper(unsigned int nparms) : nparms_(nparms), grad_(std::make_shared<Result_t>()) {}   
//   HessHelper(unsigned int nparms, std::shared_ptr<Result_t> grad) : nparms_(nparms), grad_(grad) {}
  HessHelper(HessHelper && other) = default;
//   HessHelper(HessHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
//   HessHelper(const HessHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}
  
  std::shared_ptr<Result_t> GetResultPtr() const { return grad_; }
//   std::shared_ptr<Result_t> GetResultPtr() const { 
//     return std::shared_ptr<Result_t>(reinterpret_cast<Result_t*>(gradatom_.get()));
//   }

  void Exec(unsigned int slot, ROOT::VecOps::RVec<float> const& vec, ROOT::VecOps::RVec<unsigned int> const& idxs) {
    
    unsigned int k = 0;
    for (unsigned int i=0; i<idxs.size(); ++i) {
      const unsigned int& iidx = idxs[i];
      const unsigned long long ioffset = (unsigned long long)(iidx)*(unsigned long long)(nparms_);
      for (unsigned int j=0; j<idxs.size(); ++j) {
        const unsigned int& jidx = idxs[j];
        const unsigned long long idx = ioffset + (unsigned long long)(jidx);
        
//         std::cout << "vec.size() = " << vec.size() << " idxs.size = " << idxs.size() << std::endl;
//         std::cout << "k = " << k << " iidx = " << iidx << " jidx = " << jidx << " ioffset = " << ioffset << " idx = " << idx << std::endl;
        
        std::atomic<double>& ref = (*grad_)[idx];
//         std::cout << "loaded atomic ref" << std::endl;
        const double diff = vec[k];
//         std::cout << "got diff" << std::endl;
        double old = ref.load();
//         std::cout << "got old" << std::endl;
        double desired = old + diff;
//         std::cout << "got desired" << std::endl;
        while (!ref.compare_exchange_weak(old, desired))
        {
            desired = old + diff;
        }
//         std::cout << "done cmpx loop" << std::endl;
//         if ((idxs[i]==15 && idxs[j]==61060) || idx==976960) {
//           std::cout << "i: " << i << " j: " << j << " idxs[i]: " << idxs[i] << " idxs[j]: " << idxs[j] << " offset: " << offset << " idx: " << idx << " diff: " << diff << std::endl;          
//         }
        ++k;
      }
    }

    
  }
  void InitTask(TTreeReader *, unsigned int) {}

  void Initialize() {
//     const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetImplicitMTPoolSize() : 1;
    const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetThreadPoolSize() : 1;
    
//     grad_ = std::shared_ptr<std::vector<double> >(new std::vector<double>());
    
    const unsigned long long nparmsull = nparms_;
    const unsigned long long nhess = nparmsull*nparmsull;
    
    if (grad_->empty()) {
      std::cout<< "allocating huge atomic double vector of size " << nhess << std::endl;

      std::vector<std::atomic<double> > tmp(nhess);
      grad_->swap(tmp);
      std::cout<< "initializing values" << std::endl;
      for (unsigned long long i=0; i<grad_->size(); ++i) {
        (*grad_)[i] = 0.;
      }
      std::cout<< "done huge vector" << std::endl;
    }
    timestamp_ = std::chrono::steady_clock::now();
  }

    
  void Finalize() {
//     std::vector<double>& tmp = *reinterpret_cast<std::vector<double>*>(&gradatom_);
//     grad_->swap(tmp);
    
//     double *data = reinterpret_cast<double*>(gradatom_.get());
//     double *data = reinterpret_cast<double*>(gradatom_);
//     std::vector<double> tmp(std::move(data), std::move(data+nparms_*nparms_));
//     grad_->swap(tmp);
//     grad_.reset(new std::vector<double>(std::move(data), std::move(data+nparms_*nparms_)));
    
    std::cout << "val0 = " << (*grad_)[0] << std::endl;
    std::cout << "val1 = " << (*grad_)[1] << std::endl;
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    auto timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end-timestamp_);
    
    std::cout << "Elapsed time = " << timediff.count() << std::endl;
    
  }

   std::string GetActionName(){
      return "HessHelper";
   }
      
private:
  unsigned long long nparms_;
  std::shared_ptr<Result_t> grad_;
//   std::unique_ptr<std::atomic<double>[]> gradatom_;
//   std::atomic<double>* gradatom_;
//   std::vector<std::vector<unsigned long long> > tmpidxs_;
//   std::vector<unsigned long long> offsets_;
  std::chrono::steady_clock::time_point timestamp_;
};


float maxelement(ROOT::VecOps::RVec<float> const& vec) {
  float maxval = 0.;
  for (unsigned int i=0; i<vec.size(); ++i) {
    float absval = std::abs(vec[i]);
    if (absval>maxval) {
      maxval = absval;
    }
  }
//   if (maxval > 1e5) {
//     std::cout << "maxval = " << maxval << std::endl;
//   }
  return maxval;
}

float maxelementhess(ROOT::VecOps::RVec<float> const& vec) {
  float maxval = 0.;
  for (unsigned int i=0; i<vec.size(); ++i) {
    float absval = std::abs(vec[i]);
    if (absval>maxval) {
      maxval = absval;
    }
  }
  if (maxval > 1e8) {
    std::cout << "maxvalhess = " << maxval << std::endl;
  }
  return maxval;
}

bool valid(ROOT::VecOps::RVec<float> const& vec) {
//   for (unsigned long long i=0; i<vec.size(); ++i) {
//     if (std::isnan(vec[i]) || std::isinf(vec[i])) {
//       return false;
//     }
//   }
  for (auto val : vec) {
    if (std::isnan(val) || std::isinf(val)) {
      return false;
    }
  }
  return true;
}

void fillmatricesrdfrecfulldebug() {
    
//   std::atomic<double> t;
//   std::cout<< "is lock free: " << t.is_lock_free() << std::endl;
  
  
//   std::cout << ROOT::GetImplicitMTPoolSize() << std::endl;

  setenv("XRD_PARALLELEVTLOOP", "16", true);
  
//   ROOT::EnableImplicitMT();
//   ROOT::TTreeProcessorMT::SetMaxTasksPerFilePerWorker(1);
  
//   std::cout << ROOT::GetImplicitMTPoolSize() << std::endl;
 
//   const char* filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGradsParmInfo.root";
//   TFile *finfo = TFile::Open(filenameinfo);
//   TTree *runtree = static_cast<TTree*>(finfo->Get("tree"));
//   unsigned int nparms = runtree->GetEntries();
//   const unsigned int nparms = 61068;
  
//   const char* filenameinfo = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v23_ActualGen/201201_232503/0000/globalcor_0_1.root";
//   const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v23_ActualGen/201201_232503/0000/globalcor_0_1.root";
//   const char* filenameinfo = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgrads/globalcor_0.root";
//   const char* filenameinfo = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v4_Gen_quality/210212_220639/0000/globalcor_0_1.root";
  
//   const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v33_Gen_quality/210228_001934/0000/globalcor_0_1.root";
//   const char* filenameinfo = "/data/shared/muoncal/MuonGUNUL2016Fwd_v33_Gen_idealquality/210228_002318/0000/globalcor_0_1.root";
  
//   const char* filenameinfo = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v37_Gen_quality/210311_001434/0000/globalcor_0_1.root";
//   std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v37_Gen_quality/210311_001434/0000/globalcor_*.root";
  
//   const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v38_Gen_quality/210403_232146/0000/globalcor_0_1.root";
//   std::string filename = "/data/shared/muoncal/MuonGunUL2016_v38_Gen_quality/210403_232146/0000/globalcor_*.root";
  
//   const char* filenameinfo = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v45_Rec_quality_bs_v2/210411_194557/0000/globalcor_0_10.root";
//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v45_Rec_quality_bs_v2/210411_194557/0000/globalcor_*.root";
//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v45_Rec_quality_bs_v2/210411_194557/0000/globalcor_0_10.root";
//   const std::string filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v45_Rec_quality_bs_v2/210411_194557/0000/globalcor_0_10.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v45_Rec_quality_bs_v2/210411_194557/0000/globalcor_*.root";
  
//   const std::string filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v63_Rec_quality_bs/210507_003108/0000/globalcor_0_1.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v63_Rec_quality_bs/210507_003108/0000/globalcor_*.root";
  
  const std::string filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v66_Gen_quality/210509_200135/0000/globalcor_0_1.root";
  const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v66_Gen_quality/210509_200135/0000/globalcor_*.root";
  
//   const std::string filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v45_Rec_quality_nobs_v2/210413_084740/0000/globalcor_0_1.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v45_Rec_quality_nobs_v2/210413_084740/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v45_Rec_quality_nobs_v2/210413_084740/0000/globalcor_0_1.root";
  
//   const std::string filenamejpsi = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/JPsiToMuMuGun_Pt5To30-pythia8-photos/MuonGunUL2016_v45_RecJpsiPhotos_quality_constraint_v2/210411_195122/0000/globalcor_*.root ";
//   const std::string filenamejpsi = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/JPsiToMuMuGun_Pt5To30-pythia8-photos/MuonGunUL2016_v45_RecJpsiPhotos_quality_constraint_v2/210411_195122/0000/globalcor_0_10.root ";
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v48_RecJpsiPhotos_quality_constraint/210414_072114/0000/globalcor_*.root";
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v49_RecJpsiPhotos_quality_constraint/210418_112257/0000/globalcor_*.root";
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v50_RecJpsiPhotos_quality_constraint/210421_161515/0000/globalcor_*.root";
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v51_RecJpsiPhotos_quality_constraint/210424_105652/0000/globalcor_*.root";
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v52_RecJpsiPhotos_quality_constraint/210424_221459/0000/globalcor_*.root";
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v52_GenJpsiPhotos_quality/210426_215703/0000/globalcor_*.root";
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v53_GenJpsiPhotos_quality/210428_111554/0000/globalcor_*.root";
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v64_RecJpsiPhotos_quality_constraint/210508_125028/0000/globalcor_*.root";
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v66_RecJpsiPhotos_quality_constraint/210509_200735/0000/globalcor_*.root";
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v66_GenJpsiPhotos_quality/210510_204631/0000/globalcor_*.root";
    
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v66_GenJpsiPhotos_quality/210510_204631/0000/globalcor_*.root";  
    const std::string filenamejpsi = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/globalcor_0.root";  
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v60_RecJpsiPhotos_quality_constraint/210430_223931/0000/globalcor_*.root";
  
//   const char* filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v30_Gen210206_025446/0000/globalcor_0_1.root";
  
//   const char* filenameinfo = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorRec_v28/200829_122617/0000/globalcor_1.root";

  TFile *finfo = TFile::Open(filenameinfo.c_str());
//   TTree *runtree = static_cast<TTree*>(finfo->Get("tree"));
//   TTree *runtree = static_cast<TTree*>(finfo->Get("globalCorGen/runtree"));
//   TTree *runtree = static_cast<TTree*>(finfo->Get("globalCor/runtree"));
    TTree *runtree = static_cast<TTree*>(finfo->Get("runtree"));
  unsigned int nparms = runtree->GetEntries();
//   const unsigned int nparms = 61068;
    
  
  std::cout << "nparms: " << nparms << std::endl;
  
  std::string treename = "tree";
//   std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v23_ActualGen/201201_232503/0000/globalcor_*.root";
//   std::string filename = "/data/shared/muoncal/MuonGunUL2016_v23_ActualGen/201201_232503/0000/globalcor_*.root";
//   std::string filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgrads/globalcor_*.root";
//   std::string filename = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v4_Gen_quality/210212_220639/0000/globalcor_*.root";
//   std::string filename = "/data/shared/muoncal/MuonGunUL2016_v33_Gen_quality/210228_001934/0000/globalcor_*.root";
//   std::string filename = "/data/shared/muoncal/MuonGUNUL2016Fwd_v33_Gen_idealquality/210228_002318/0000/globalcor_*.root";
//   std::string filename = "/data/shared/muoncal/MuonGunUL2016_v33_Gen_idealquality/210307_155804/0000/globalcor_*.root";
  
//   std::string filename = "/data/shared/muoncal/MuonGunUL2016_v30_Gen210206_025446/0000/globalcor_*.root";
  
//   std::string filename = "/data/bendavid/muoncaldata/test3/testlz4large.root";
//   std::string filename = "/data/bendavid/muoncaldata/test3/testzliblarge.root";
//   std::string filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root";
//   std::string filename = "/data/bendavid/muoncaldata/largetest/*.root";

  
  
  GradHelper gradhelper(nparms);
  HessHelper hesshelper(nparms);
  
  ROOT::RDataFrame d(treename, filename);
  
//   auto d2 = d.Filter("genPt > 5.5 && abs(genEta)<2.4");
//   auto d2 = d.Filter("genPt > 5.5");
//   auto d2 = d.Filter("genPt > 5.5 && nValidHits > 9 && nValidPixelHits > 0 && trackCharge*trackPt/genPt/genCharge > 0.3");
  
  auto d2a = d.Define("refPt", "std::abs(1./refParms[0])*std::sin(M_PI_2 - refParms[1])");
  auto d2b = d2a.Define("refEta", "-std::log(std::tan(0.5*(M_PI_2 - refParms[1])))");
  auto d2 = d2b.Filter("refPt > 3.5 && nValidHits > 9 && nValidPixelHits > 0 && std::abs(refEta)<2.2");
//   auto d2 = d.Filter("genPt > 5.5 && nValidHitsFinal > 9 && nValidPixelHitsFinal > 0");
//   auto d2 = d.Filter("genPt > 5.5 && nValidHits > 9 && nValidPixelHits > 0");
//   auto d2 = d.Filter("genPt > 5.5 && nValidHits > 9 && nValidPixelHits > 0 && genEta>-2.4 && genEta<-2.3");
  
//   auto d2 = d.Filter("genPt > 5.5 && genEta>-1.7 && genEta>-1.7 && genEta<-1.4");
//   auto d2 = d.Filter("trackPt > 5.5");
//   auto d3 = d.Define("gradmax", "maxelement(gradv)");
//   auto d3 = d2.Filter("maxelement(gradv) < 1e5 && maxelementhess(hesspackedv)<1e8");
//   auto d3 = d2.Filter("maxelement(gradv) < 1e5");
  auto d3 = d2.Filter("maxelement(gradv) < 1e5");
//   auto d3 = d2.Filter("maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv)");
//   auto d3 = d2.Filter("true");
//   auto d3 = d2.Filter("maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv) && maxelementhess(hesspackedv)<1e8");
  
  
//   auto grad0 = d3.Book<ROOT::VecOps::RVec<float>,  ROOT::VecOps::RVec<unsigned int> >(std::move(gradhelper), {"gradv", "globalidxv"});
//   auto hess0 = d3.Book<ROOT::VecOps::RVec<float>,  ROOT::VecOps::RVec<unsigned int> >(std::move(hesshelper), {"hesspackedv", "globalidxv"});
// //   
// //   auto gradcounts = d3.Histo1D({"gradcounts", "", int(nparms), -0.5, double(nparms)-0.5}, "globalidxv");
// // 
//   std::cout << (*grad0)[0] << std::endl;
//   std::cout << (*hess0)[0] << std::endl;

  std::cout << "starting second rdf" << std::endl;

  
  ROOT::RDataFrame dj(treename, filenamejpsi);
//   auto dj2a = dj.Filter("std::abs(Jpsi_mass - 3.09692f) < 0.1");
//   auto dj2b = dj2a.Filter("valid(gradv) && valid(hesspackedv)");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5 && std::abs(Jpsi_mass-3.09692)<90e-6 && Jpsi_pt < 28. && maxelement(gradv) < 1e5");
  
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5");
//   auto dj2 = dj.Filter("Muplusgen_pt > 5.5 && Muminusgen_pt > 5.5");
//   auto dj2 = dj.Filter("Muplus_pt > 3.5 && Muminus_pt > 3.5 && Muplus_nvalid > 9 && Muplus_nvalidpixel>0 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 0 && std::abs(Muplus_eta)<2.2 && std::abs(Muminus_eta)<2.2");
  auto dj2 = dj.Filter("Muplus_pt > 3.5 && Muminus_pt > 3.5 && Muplus_nvalid > 9 && Muplus_nvalidpixel>0 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 0");
//   auto dj2 = dj.Filter("Muplus_pt > 3.5 && Muminus_pt > 3.5 && Muplus_nvalid > 9 && Muplus_nvalidpixel>0 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 0");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5 && TMath::Prob(chisqval, ndof)>1e-3");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5 && std::abs(Jpsigen_mass - 3.09692) < 180e-6");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5 && Jpsi_pt < 28.");
//   auto dj3 = dj2.Filter("true");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && TMath::Prob(chisqval, ndof) > 1e-3 && Jpsi_pt<28.");
  auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5");
//   auto dj3 = dj2.Filter("maxelement(gradv) < 5e5 && valid(gradv) && valid(hesspackedv)");
//   auto dj3 = dj2.Filter("maxelement(gradv) < 1e5");
//   auto dj3 = dj2.Filter("maxelement(gradv) < 13e3");
//   auto dj3 = dj2.Filter("TMath::Prob(chisqval, ndof) > 1e-2 && maxelement(gradv) < 1e5");
    
  GradHelper gradhelperj(nparms);
  HessHelper hesshelperj(nparms);
  
//   gradhelperj.GetResultPtr()->swap(*grad0);
//   hesshelperj.GetResultPtr()->swap(*hess0);
  
  auto grad = dj3.Book<ROOT::VecOps::RVec<float>,  ROOT::VecOps::RVec<unsigned int> >(std::move(gradhelperj), {"gradv", "globalidxv"});
  auto hess = dj3.Book<ROOT::VecOps::RVec<float>,  ROOT::VecOps::RVec<unsigned int> >(std::move(hesshelperj), {"hessv", "globalidxv"});
  
//   auto &grad = grad0;
//   auto &hess = hess0;
  
  std::cout << (*grad)[0] << std::endl;
  std::cout << (*hess)[0] << std::endl;
  
  std::cout << "done second rdf" << std::endl;
  
//   TFile *fgrads = new TFile("combinedgrads.root", "RECREATE");
  TFile *fgrads = new TFile("combinedgradsdebugfull.root", "RECREATE");
  TTree *tree = new TTree("tree", "");
  
  unsigned int idx;
  double gradelem;
  std::vector<double> hessrow(nparms);
  
  std::ostringstream leaflist;
  leaflist<< "hessrow[" << nparms << "]/D";
  
  tree->Branch("idx", &idx);
  tree->Branch("gradelem", &gradelem);
  tree->Branch("hessrow", hessrow.data(), leaflist.str().c_str(), nparms);
  
  std::vector<unsigned long long> offsets;
  offsets.reserve(nparms);
  unsigned long long k = 0;
  for (unsigned int i = 0; i < nparms; ++i) {
    offsets.push_back(k - i);
    k += nparms - i;
  }
  
  std::cout << "offsets[15]: " << offsets[15] << " offsets[16]: " << offsets[16] << std::endl;
  
  for (unsigned int i = 0; i < nparms; ++i) {
    idx = i;
    gradelem = (*grad)[i];
//     hessrow.clear();
//     hessrow.resize(nparms, 0.);
    const unsigned long long offset = (unsigned long long)(i)*(unsigned long long)(nparms);
//     for (unsigned int j = 0; j < i; ++j) {
//       hessrow[j] = 0.;
//     }
    for (unsigned int j = 0; j < nparms; ++j) {
      const unsigned long long idx = offset + (unsigned long long)(j);
      hessrow[j] = (*hess)[idx];
//       if ((i==15 && j==61060) || idx==976960) {
//         std::cout << "i: " << i << " j: " << j << " offset: " << offset << " idx: " << idx << " val: " << (*hess)[idx] << std::endl;
//       }
    }
    tree->Fill();
  }
  
  tree->Write();
//   gradcounts->Write();
  fgrads->Close();
    
}

