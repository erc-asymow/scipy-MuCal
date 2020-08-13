#include <atomic>
#include <chrono>
#include <ROOT/RDataFrame.hxx>
#include "TFile.h"

class GradHelper : public ROOT::Detail::RDF::RActionImpl<GradHelper> {

public:
  using Result_t = std::vector<double>;
  
  GradHelper(unsigned int nparms) : nparms_(nparms), grad_(new Result_t()) {}   
  GradHelper(GradHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
  GradHelper(const GradHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}
  
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

  void Exec(unsigned int slot, ROOT::VecOps::RVec<float> const& vec, ROOT::VecOps::RVec<unsigned int> const& idxs) {
    
    unsigned int k = 0;
    for (unsigned int i=0; i<idxs.size(); ++i) {
      const unsigned int& iidx = idxs[i];
      const unsigned long long& ioffset = offsets_[iidx];
      for (unsigned int j=i; j<idxs.size(); ++j) {
        const unsigned int& jidx = idxs[j];
        const unsigned long long& joffset = offsets_[jidx];
        const unsigned long long idx = jidx >= iidx ? ioffset + jidx : joffset + iidx;

        std::atomic<double>& ref = (*grad_)[idx];
        const double& diff = vec[k];
        double old = ref.load();
        double desired = old + diff;
        while (!ref.compare_exchange_weak(old, desired))
        {
            desired = old + diff;
        }
//         if ((idxs[i]==15 && idxs[j]==61060) || idx==976960) {
//           std::cout << "i: " << i << " j: " << j << " idxs[i]: " << idxs[i] << " idxs[j]: " << idxs[j] << " offset: " << offset << " idx: " << idx << " diff: " << diff << std::endl;          
//         }
        ++k;
      }
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

void fillmatricesrdf(const char *filename) {
    
//   std::atomic<double> t;
//   std::cout<< "is lock free: " << t.is_lock_free() << std::endl;
  
  
//   std::cout << ROOT::GetImplicitMTPoolSize() << std::endl;

  
  ROOT::EnableImplicitMT();
  
//   std::cout << ROOT::GetImplicitMTPoolSize() << std::endl;
 
//   const char* filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGradsParmInfo.root";
//   TFile *finfo = TFile::Open(filenameinfo);
//   TTree *runtree = static_cast<TTree*>(finfo->Get("tree"));
//   unsigned int nparms = runtree->GetEntries();
  const unsigned int nparms = 61068;
  
  std::cout << "nparms: " << nparms << std::endl;
  
  std::string treename = "tree";
//   std::string filename = "/data/bendavid/muoncaldata/test3/testlz4large.root";
//   std::string filename = "/data/bendavid/muoncaldata/test3/testzliblarge.root";
//   std::string filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root";
//   std::string filename = "/data/bendavid/muoncaldata/largetest/*.root";

  
  
  GradHelper gradhelper(nparms);
  HessHelper hesshelper(nparms);
  
  ROOT::RDataFrame d(treename, filename);
  
  auto d2 = d.Filter("trackPt > 5.5");
//   auto d3 = d.Define("gradmax", "maxelement(gradv)");
//   auto d3 = d2.Filter("maxelement(gradv) < 1e5 && maxelementhess(hesspackedv)<1e8");
  auto d3 = d2.Filter("maxelement(gradv) < 1e5");
  
  
  auto grad = d3.Book<ROOT::VecOps::RVec<float>,  ROOT::VecOps::RVec<unsigned int> >(std::move(gradhelper), {"gradv", "globalidxv"});
  auto hess = d3.Book<ROOT::VecOps::RVec<float>,  ROOT::VecOps::RVec<unsigned int> >(std::move(hesshelper), {"hesspackedv", "globalidxv"});
  
  auto gradcounts = d3.Histo1D({"gradcounts", "", int(nparms), -0.5, double(nparms)-0.5}, "globalidxv");

  std::cout << (*grad)[0] << std::endl;
  std::cout << (*hess)[0] << std::endl;

  TFile *fgrads = new TFile("combinedgrads.root", "RECREATE");
  TTree *tree = new TTree("tree", "");
  
  double gradelem;
  std::vector<double> hessrow(nparms);
  
  std::ostringstream leaflist;
  leaflist<< "hessrow[" << nparms << "]/D";
  
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
  gradcounts->Write();
  fgrads->Close();
    
}

