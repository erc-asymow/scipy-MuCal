#include <atomic>
#include <chrono>
#include <algorithm>
#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include "TSystem.h"

class GradHelper : public ROOT::Detail::RDF::RActionImpl<GradHelper> {

public:
  using Result_t = std::vector<double>;
  
  GradHelper(unsigned int nparms) : nparms_(nparms), grad_(new Result_t()) {}   
  GradHelper(GradHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
  GradHelper(const GradHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}
  
  std::shared_ptr<Result_t> GetResultPtr() const { return grad_; }

  void Exec(unsigned int slot, unsigned int idx, double val) {
    std::vector<double>& grad = gradtmp_[slot];
    grad[idx] += val;
    
  }
  void InitTask(TTreeReader *, unsigned int) {}

  void Initialize() {
    const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetImplicitMTPoolSize() : 1;
    gradtmp_.clear();
    gradtmp_.resize(nslots, std::vector<double>(nparms_, 0.));
    
    grad_->clear();
    grad_->resize(nparms_, 0.);    
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
  
  HessHelper(unsigned int nparms) : nparms_(nparms), grad_(new Result_t()) {}   
  HessHelper(HessHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
  HessHelper(const HessHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}
  
  std::shared_ptr<Result_t> GetResultPtr() const { return grad_; }
//   std::shared_ptr<Result_t> GetResultPtr() const { 
//     return std::shared_ptr<Result_t>(reinterpret_cast<Result_t*>(gradatom_.get()));
//   }

  void Exec(unsigned int slot, unsigned long long iidx, unsigned int jidx, double val) {
    
    const unsigned long long& ioffset = offsets_[iidx];
    const unsigned long long& joffset = offsets_[jidx];
    const unsigned long long idx = jidx >= iidx ? ioffset + jidx : joffset + iidx;
    
    std::atomic<double>& ref = (*grad_)[idx];
    double old = ref.load();
    double desired = old + val;
    while (!ref.compare_exchange_weak(old, desired))
    {
        desired = old + val;
    }

    
  }
  void InitTask(TTreeReader *, unsigned int) {}

  void Initialize() {
    const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetImplicitMTPoolSize() : 1;
    
//     grad_ = std::shared_ptr<std::vector<double> >(new std::vector<double>());
    
    offsets_.clear();
    offsets_.reserve(nparms_);
    
    unsigned long long k = 0;
    for (unsigned int i = 0; i < nparms_; ++i) {
      offsets_.push_back(k - i);
      k += nparms_ - i;
    }
    
    std::cout << "offsets_[15]: " << offsets_[15] << " offsets_[16]: " << offsets_[16] << std::endl;
    
    unsigned long long nsym = nparms_*(nparms_+1)/2;
    
    std::cout<< "allocating huge atomic double vector of size " << nsym << std::endl;

    std::vector<std::atomic<double> > tmp(nsym);
    grad_->swap(tmp);
    std::cout<< "initializing values" << std::endl;
    for (unsigned long long i=0; i<grad_->size(); ++i) {
      (*grad_)[i] = 0.;
    }
    std::cout<< "done huge vector" << std::endl;
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
  std::vector<unsigned long long> offsets_;
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
  if (maxval > 1e5) {
    std::cout << "maxval = " << maxval << std::endl;
  }
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
  for (unsigned long long i=0; i<vec.size(); ++i) {
    if (std::isnan(vec[i]) || std::isinf(vec[i])) {
      return false;
    }
  }
  return true;
}

void fillmatricessparse() {
    
//   std::atomic<double> t;
//   std::cout<< "is lock free: " << t.is_lock_free() << std::endl;
  gSystem->Setenv("XRD_REQUESTTIMEOUT","10");
  
//   std::cout << ROOT::GetImplicitMTPoolSize() << std::endl;

  
  ROOT::EnableImplicitMT();
  
//   std::cout << ROOT::GetImplicitMTPoolSize() << std::endl;
 
//   const char* filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGradsParmInfo.root";
//   const char* filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_10_2_23/work/trackTreeGradsParmInfo.root";
//   const char* filenameinfo = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorGen_v12/200823_134027/0000/globalcorgen_1.root";
//   const char* filenameinfo = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorGenFwd_v12/200824_105616/0000/globalcorgen_103.root";
  const char* filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantint4genhelixprecrefb//globalcor_0.root";
//   const char* filenameinfo = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorRec_v28/200829_122617/0000/globalcor_1.root";

  TFile *finfo = TFile::Open(filenameinfo);
//   TTree *runtree = static_cast<TTree*>(finfo->Get("tree"));
//   TTree *runtree = static_cast<TTree*>(finfo->Get("globalCorGen/runtree"));
//   TTree *runtree = static_cast<TTree*>(finfo->Get("globalCor/runtree"));
    TTree *runtree = static_cast<TTree*>(finfo->Get("runtree"));
  unsigned int nparms = runtree->GetEntries();
//   const unsigned int nparms = 61068;
  
  std::cout << "nparms: " << nparms << std::endl;
  
//   std::string treename = "tree";
//   std::string filename = "/data/bendavid/muoncaldata/test3/testlz4large.root";
//   std::string filename = "/data/bendavid/muoncaldata/test3/testzliblarge.root";
//   std::string filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root";
//   std::string filename = "/data/bendavid/muoncaldata/largetest/*.root";

//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v3/200817_213237/0000/*.root";
//   const std::string filename = "/data/bendavid/muoncaldatalarge/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v4/200820_002243/0000/*.root";
  
//   const std::string filename = "/data/bendavid/cmsswdevslc6/CMSSW_10_2_23/work/trackTreeGrads.root";
//   const std::string filename = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCor_v6/200821_082048/0000/*.root";
//   const std::string filename = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorGen_v12/200823_134027/0000/globalcorgen_*.root";
//   const std::string filename = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorGen_GBLchisq/200908_222942/0000/globalcorgen_*.root";
  
    const std::string filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantint4genhelixprecrefb//globalcor_*.root";

  
//   const std::string filename = "/data/bendavid/muoncaldatalarge/MuonGunGlobalCorRec_v28/200829_122617/0000/globalcor_*.root";
  
  GradHelper gradhelper(nparms);
  HessHelper hesshelper(nparms);
  
//   ROOT::RDataFrame dgrad("gradtree", "fixedGrads.root");
//   ROOT::RDataFrame dgrad("globalCorGen/gradtree", filename);
//   ROOT::RDataFrame dgrad("globalCor/gradtree", filename);
  ROOT::RDataFrame dgrad("gradtree", filename);
  auto grad = dgrad.Book<unsigned int, double>(std::move(gradhelper), { "idx", "gradval" });
  
//   ROOT::RDataFrame dhess("globalCorGen/hesstree", filename);
//   ROOT::RDataFrame dhess("globalCor/hesstree", filename);
  ROOT::RDataFrame dhess("hesstree", filename);

  auto hess = dhess.Book<unsigned int, unsigned int, double>(std::move(hesshelper), {"iidx", "jidx", "hessval" });
  
//   auto gradcounts = d3.Histo1D({"gradcounts", "", int(nparms), -0.5, double(nparms)-0.5}, "globalidxv");

  std::cout << (*grad)[0] << std::endl;
  std::cout << (*hess)[0] << std::endl;

  TFile *fgrads = new TFile("combinedgrads.root", "RECREATE");
//   TFile *fgrads = new TFile("combinedgradsrec.root", "RECREATE");
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
    const unsigned long long& offset = offsets[i];
    for (unsigned int j = 0; j < i; ++j) {
      hessrow[j] = 0.;
    }
    for (unsigned int j = i; j < nparms; ++j) {
      const unsigned long long idx = offset + j;
      hessrow[j] = (*hess)[idx];
      if ((i==15 && j==61060) || idx==976960) {
        std::cout << "i: " << i << " j: " << j << " offset: " << offset << " idx: " << idx << " val: " << (*hess)[idx] << std::endl;
      }
    }
    tree->Fill();
  }
  
  tree->Write();
//   gradcounts->Write();
  fgrads->Close();
    
}

