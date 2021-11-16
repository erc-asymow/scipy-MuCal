#include <atomic>
#include <chrono>
#include <algorithm>
#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include <ROOT/TTreeProcessorMT.hxx>
#include <stdlib.h>  

// using grad_t = double;
using grad_t = float;


class GradHelper : public ROOT::Detail::RDF::RActionImpl<GradHelper> {
  
public:
  using Result_t = std::vector<double>;
  
  GradHelper(unsigned int nparms) : nparms_(nparms), grad_(std::make_shared<Result_t>()) {}  
//   GradHelper(unsigned int nparms, std::shared_ptr<Result_t> grad) : nparms_(nparms), grad_(grad) {}
  GradHelper(GradHelper && other) = default;
//   GradHelper(GradHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
//   GradHelper(const GradHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}
  
  std::shared_ptr<Result_t> GetResultPtr() const { return grad_; }

  void Exec(unsigned int slot, ROOT::VecOps::RVec<grad_t> const& vec, ROOT::VecOps::RVec<unsigned int> const& idxs) {
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

  void Exec(unsigned int slot, ROOT::VecOps::RVec<grad_t> const& vec, ROOT::VecOps::RVec<unsigned int> const& idxs) {
    
    unsigned int k = 0;
    for (unsigned int i=0; i<idxs.size(); ++i) {
      const unsigned int& iidx = idxs[i];
      const unsigned long long& ioffset = offsets_[iidx];
      for (unsigned int j=i; j<idxs.size(); ++j) {
        const unsigned int& jidx = idxs[j];
        const unsigned long long& joffset = offsets_[jidx];
        const unsigned long long idx = jidx >= iidx ? ioffset + jidx : joffset + iidx;

        const double diff = (iidx==jidx && i!=j) ? 2.*vec[k] : vec[k];
        
        std::atomic<double>& ref = (*grad_)[idx];
//         const double& diff = vec[k];
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
//     const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetImplicitMTPoolSize() : 1;
    const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetThreadPoolSize() : 1;
    
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
    
    if (grad_->empty()) {
      std::cout<< "allocating huge atomic double vector of size " << nsym << std::endl;

      std::vector<std::atomic<double> > tmp(nsym);
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
  std::vector<unsigned long long> offsets_;
  std::chrono::steady_clock::time_point timestamp_;
};


grad_t maxelement(ROOT::VecOps::RVec<grad_t> const& vec) {
  grad_t maxval = 0.;
  for (unsigned int i=0; i<vec.size(); ++i) {
    grad_t absval = std::abs(vec[i]);
    if (absval>maxval) {
      maxval = absval;
    }
  }
//   if (maxval > 1e5) {
//     std::cout << "maxval = " << maxval << std::endl;
//   }
  return maxval;
}

grad_t maxelementhess(ROOT::VecOps::RVec<grad_t> const& vec) {
  grad_t maxval = 0.;
  for (unsigned int i=0; i<vec.size(); ++i) {
    grad_t absval = std::abs(vec[i]);
    if (absval>maxval) {
      maxval = absval;
    }
  }
  if (maxval > 1e8) {
    std::cout << "maxvalhess = " << maxval << std::endl;
  }
  return maxval;
}

bool valid(ROOT::VecOps::RVec<grad_t> const& vec) {
//   for (unsigned long long i=0; i<vec.size(); ++i) {
//     if (std::isnan(vec[i]) || std::isinf(vec[i])) {
//       return false;
//     }
//   }
  for (auto val : vec) {
    if (std::isnan(val) || std::isinf(val)) {
//       printf("invalid\n");
      return false;
    }
  }
  return true;
}

bool hasdups(const ROOT::VecOps::RVec<unsigned int> &idxs, unsigned int nhitsplus, unsigned int nhitsminus) {
  const unsigned alignidx = 2*nhitsplus + 2*nhitsminus;
  std::vector<unsigned int> alignidxs(idxs.begin() + 2*nhitsplus + 2*nhitsminus, idxs.end());
  std::sort(alignidxs.begin(), alignidxs.end());
  auto dupit = std::adjacent_find(alignidxs.begin(), alignidxs.end());
  return dupit != alignidxs.end();
  
}


bool isconsistent(const ROOT::VecOps::RVec<unsigned int> &idxs, unsigned int nhitsplus, unsigned int nhitsminus, unsigned int nvalidplus, unsigned int nvalidminus, unsigned int nvalidpixelplus, unsigned int nvalidpixelminus) {
  const unsigned nparms = 2*nhitsplus + 2*nhitsminus + 2*nvalidplus + 2*nvalidminus + nvalidpixelplus + nvalidpixelminus;
  
  return nparms == idxs.size();
  
//   std::cout << "alignidx = " << alignidx << " total = " << idxs.size() << " nhitsplus = " << nhitsplus << " nhitsminus = " << nhitsminus << " nvalidplus = " << nvalidplus << " nvalidminus = " << nvalidminus << std::endl;
//   assert(alignidx < idxs.size());
  
  
//   std::vector<unsigned int> alignidxs(idxs.begin() + 2*nhitsplus + 2*nhitsminus, idxs.end());
//   std::sort(alignidxs.begin(), alignidxs.end());
//   auto dupit = std::adjacent_find(alignidxs.begin(), alignidxs.end());
//   return dupit != alignidxs.end();
  
}

void fillmatricesrdfrec() {
    
//   std::atomic<double> t;
//   std::cout<< "is lock free: " << t.is_lock_free() << std::endl;
  
  
//   std::cout << ROOT::GetImplicitMTPoolSize() << std::endl;

  setenv("XRD_PARALLELEVTLOOP", "16", true);
  
  ROOT::EnableImplicitMT();
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
  
//   const std::string filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v107_GenJpsiPhotosSingle_quality/210719_142518/0000/globalcor_0_1.root";
  
//   const std::string filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_idealquality_constraint/210911_050035/0000/globalcor_0_1.root";
  
//   const std::string filenameinfo = "info.root";
//   const std::string filenameinfo = "infofullbz.root";
  const std::string filenameinfo = "info_v202.root";
//   const std::string filenameinfo = "info_v201.root";
//   const std::string filenameinfo = "infofullbzdisky.root";
//     const std::string filenameinfo = "infofullbzr.root";
  
  
//   const std::string filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v124_Gen_quality/210729_121830/0000/globalcor_0_1.root";
  
//   const std::string filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v66_Gen_quality/210509_200135/0000/globalcor_0_1.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v66_Gen_quality/210509_200135/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v66_Rec_quality_bs/210511_004711/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v66_Rec_quality_nobs/210509_195947/0000/globalcor_*.root";
//   const std::string filename = "/data/shared/muoncal/MuonGunUL2016_v67_Rec_quality_nobs/210613_175312/0000/globalcor_*.root";
//   const std::string filename = filenameinfo;
  TChain chain("tree");
//   chain.Add(filenameinfo.c_str());
  
  chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_Rec_idealquality/210910_175214/0000/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v120_Rec_quality_bs/210725_153917/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v120_Rec_quality_bs/210725_153917/0001/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v148_Rec_quality_bs/210812_042225/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v148_Rec_quality_bs/210812_042225/0001/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v124_Gen_quality/210729_121830/0000/globalcor_*.root");

  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v115_Rec_quality_nobs/210721_193853/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v115_Rec_quality_nobs/210721_193853/0001/globalcor_*.root");
  
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

  
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v67_RecJpsiPhotos_quality_constraint/210511_011120/0000/globalcor_*.root";  
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v67_RecJpsiPhotosTail_quality_constraint/210615_124648/0000/globalcor_*.root";
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v67_RecJpsiPhotosTail0_quality_constraint/210616_212646/0000/globalcor_*.root";
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v80_RecJpsiPhotosTail0_quality_constraint/210703_155721/0000/globalcor_*.root";
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v74_RecJpsiPhotosTail0_quality_constraint/210628_223927/0000/globalcor_*.root";
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v71_RecJpsiPhotosTail0_quality_constraint/210624_204606/0000/globalcor_*.root";
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v68_RecJpsiPhotosTail0_quality_constraint/210621_095255/0000/globalcor_*.root";
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v69_RecJpsiPhotosTail0_quality_constraint/210621_182428/0000/globalcor_*.root";
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v70_RecJpsiPhotosTail0_quality_constraint/210622_184224/0000/globalcor_*.root";
    
  TChain chainjpsi("tree");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v109_RecJpsiPhotos_quality_constraint/210720_225832/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v109_RecJpsiPhotos_quality_constraint/210720_225832/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v126_GenJpsiPhotos_quality_constraint/210730_185921/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v126_GenJpsiPhotos_quality_constraint/210730_185921/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v127_RecJpsiPhotos_quality_constraint/210731_153616/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v127_RecJpsiPhotos_quality_constraint/210731_153616/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v130_RecJpsiPhotos_quality_constraint/210802_204103/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v130_RecJpsiPhotos_quality_constraint/210802_204103/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v146_RecJpsiPhotos_quality_constraint/210809_232349/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v146_RecJpsiPhotos_quality_constraint/210809_232349/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v147_RecJpsiPhotos_quality_constraint/210810_144929/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v147_RecJpsiPhotos_quality_constraint/210810_144929/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v148_RecJpsiPhotos_quality_constraint/210811_193709/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v148_RecJpsiPhotos_quality_constraint/210811_193709/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v149_RecJpsiPhotos_idealquality_constraint_biasm10/210816_163109/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v149_RecJpsiPhotos_idealquality_constraint/210816_164138/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v149_RecJpsiPhotos_idealquality_zeromaterial_constraint/210817_033501/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v150_RecJpsiPhotos_idealquality_constraint/210817_175059/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v153a_RecJpsiPhotos_idealquality_constraint/210820_001516/0000/globalcor_*.root");
  
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v154_RecJpsiPhotos_idealquality_constraint/210820_122952/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v156_RecJpsiPhotos_idealquality_constraint/210829_002423/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v156_RecJpsiPhotos_idealquality_constraint/210829_002423/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v157_RecJpsiPhotos_idealquality_constraint/210829_035722/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v157_RecJpsiPhotos_idealquality_constraint/210829_035722/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v158_RecJpsiPhotos_idealquality_constraint/210829_162156/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v158_RecJpsiPhotos_idealquality_constraint/210829_162156/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v160_RecJpsiPhotos_idealquality_constraint/210831_182013/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v160_RecJpsiPhotos_idealquality_constraint/210831_182013/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v161_RecJpsiPhotos_idealquality_constraint/210901_104346/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v161_RecJpsiPhotos_idealquality_constraint/210901_104346/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v164_RecJpsiPhotos_idealquality_constraint/210903_020706/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v164_RecJpsiPhotos_idealquality_constraint/210903_020706/0001/globalcor_*.root");


//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v165a_RecJpsiPhotos_idealquality_constraint/210904_183901/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v165a_RecJpsiPhotos_idealquality_constraint/210904_183901/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v166_RecJpsiPhotos_idealquality_constraint/210907_053804/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v166_RecJpsiPhotos_idealquality_constraint/210907_053804/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v167_RecJpsiPhotos_idealquality_constraint/210908_011211/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v167_RecJpsiPhotos_idealquality_constraint/210908_011211/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v168_RecJpsiPhotos_idealquality_constraint/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v169_RecJpsiPhotos_idealquality_constraint/210908_130028/0001/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v169_RecJpsiPhotos_idealquality_constraint/210908_130028/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v170_RecJpsiPhotos_idealquality_constraint/210908_211115/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v170_RecJpsiPhotos_idealquality_constraint/210908_211115/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v172_RecJpsiPhotos_idealquality_constraint/210910_174808/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v172_RecJpsiPhotos_idealquality_constraint/210910_174808/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v173_RecJpsiPhotos_idealquality_constraint/210911_014941/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v173_RecJpsiPhotos_idealquality_constraint/210911_014941/0001/globalcor_*.root");

  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_idealquality_constraint/210911_050035/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_idealquality_constraint/210911_050035/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_quality_constraint_biasm10_biasfield/210914_224608/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_quality_constraint_biasm10_biasfield/210914_224608/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171a_RecJpsiPhotos_quality_constraintfsr28_biasm10_biasfield/210914_234555/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171a_RecJpsiPhotos_quality_constraintfsr28_biasm10_biasfield/210914_234555/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter0/210915_052414/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter0/210915_052414/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171a_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter0/210915_090037/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171a_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter0/210915_090037/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171a_RecJpsiPhotos_quality_constraintfsr28_biasm10_biasfield_iter0/210915_064129/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v171a_RecJpsiPhotos_quality_constraintfsr28_biasm10_biasfield_iter0/210915_064129/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v174_RecJpsiPhotos_idealquality_constraint/210911_074615/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v175_RecJpsiPhotos_idealquality_constraint/210912_215354/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v175_RecJpsiPhotos_idealquality_constraint/210912_215354/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v176_RecJpsiPhotos_idealquality_constraint/210913_141934/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v176_RecJpsiPhotos_idealquality_constraint/210913_141934/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v181_RecJpsiPhotos_quality_constraint_biasm10_biasfield/210916_041724/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v181_RecJpsiPhotos_quality_constraint_biasm10_biasfield/210916_041724/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v184_RecJpsiPhotos_quality_constraint_biasm10_biasfield/210917_172452/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v184_RecJpsiPhotos_quality_constraint_biasm10_biasfield/210917_172452/0001/globalcor_*.root");
  
// chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v186_RecJpsiPhotos_quality_constraint_biasm10_biasfield/210918_204424/0000/globalcor_*.root");
// chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v186_RecJpsiPhotos_quality_constraint_biasm10_biasfield/210918_204424/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v187_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter1_bz/210920_102146/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v187_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter1_bz/210920_102146/0001/globalcor_*.root");

//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v188_RecJpsiPhotos_quality_constraint_biasm10/210921_055627/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v188_RecJpsiPhotos_quality_constraint_biasm10/210921_055627/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v193_RecJpsiPhotos_quality/210924_135219/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v193_RecJpsiPhotos_quality/210924_135219/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v194_RecJpsiPhotos_quality/210924_172109/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v194_RecJpsiPhotos_quality/210924_172109/0001/globalcor_*.root");
  
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v195_RecJpsiPhotos_quality_constraint/210925_184120/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v195_RecJpsiPhotos_quality_constraint/210925_184120/0001/globalcor_*.root");
  
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v196_RecJpsiPhotos_quality_constraint/210925_190747/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v196_RecJpsiPhotos_quality_constraint/210925_190747/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v197a_RecJpsiPhotos_quality_constraint/210925_193954/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v197a_RecJpsiPhotos_quality_constraint/210925_193954/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v198_RecJpsiPhotos_quality_constraint/210927_154710/0000/globalcor_*.root");

//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v199_RecJpsiPhotos_quality_constraint/210929_185302/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v199_RecJpsiPhotos_quality_constraint/210929_185302/0001/globalcor_*.root");

//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v200_RecJpsiPhotos_quality_constraint/210929_191259/0000/*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v200_RecJpsiPhotos_quality_constraint/210929_191259/0001/*.root");

//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecJpsiPhotos_quality_constraintfsr28/210930_202515/0000/*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecJpsiPhotos_quality_constraintfsr28/210930_202515/0001/*.root");
  
  chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v202a_RecDataJPsiH_quality_constraintfsr28/210930_203700/0000/*.root");
  chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecDataJPsiG_quality_constraintfsr28/210930_204012/0000/*.root");
  chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v202_RecDataJPsiFpost_quality_constraintfsr28/210930_204138/0000/*.root");


//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v201_RecJpsiPhotos_quality_constraint/210930_024849/0000/*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v201_RecJpsiPhotos_quality_constraint/210930_024849/0001/*.root");

//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v188_RecJpsiPhotos_quality_constraint_biasm10_iter1/210921_153750/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v188_RecJpsiPhotos_quality_constraint_biasm10_iter1/210921_153750/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v191_RecJpsiPhotos_quality/210922_084651/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v191_RecJpsiPhotos_quality/210922_084651/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v192_RecJpsiPhotos_quality_iter1/210922_142844/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v192_RecJpsiPhotos_quality_iter1/210922_142844/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v186_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter1/210919_064829/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v186_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter1/210919_064829/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v181_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter1/210916_100334/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v181_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter1/210916_100334/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v187_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter1/210919_142948/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v187_RecJpsiPhotos_quality_constraint_biasm10_biasfield_iter1/210919_142948/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v152_RecJpsiPhotos_idealquality_constraint/210818_180345/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v144_RecJpsiPhotos_quality_constraint/210807_192307/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v144_RecJpsiPhotos_quality_constraint/210807_192307/0001/globalcor_*.root");
  
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v140_RecJpsiPhotos_quality_constraint/210806_175826/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v140_RecJpsiPhotos_quality_constraint/210806_175826/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v143_RecJpsiPhotos_quality_constraint/210807_034655/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v143_RecJpsiPhotos_quality_constraint/210807_034655/0001/globalcor_*.root");

    
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v136_RecJpsiPhotos_quality_constraint/210805_192633/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v136_RecJpsiPhotos_quality_constraint/210805_192633/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v133_RecJpsiPhotos_quality_constraint/210803_172304/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v133_RecJpsiPhotos_quality_constraint/210803_172304/0001/globalcor_*.root");
    
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v132_RecJpsiPhotos_quality_constraint/210803_024703/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v132_RecJpsiPhotos_quality_constraint/210803_024703/0001/globalcor_*.root");
    
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v129_RecJpsiPhotos_quality_constraint/210801_152925/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v129_RecJpsiPhotos_quality_constraint/210801_152925/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v128_RecJpsiPhotos_quality_constraint/210801_090738/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v128_RecJpsiPhotos_quality_constraint/210801_090738/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v124_RecJpsiPhotos_quality_constraint/210729_122545/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v124_RecJpsiPhotos_quality_constraint/210729_122545/0001/globalcor_*.root");
  
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v117_RecJpsiPhotos_quality_constraint/210723_144009/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v117_RecJpsiPhotos_quality_constraint/210723_144009/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v118_RecJpsiPhotos_quality_constraint_bs/210723_181056/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v118_RecJpsiPhotos_quality_constraint_bs/210723_181056/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v116_RecJpsiPhotos_quality_constraint/210722_170211/0000/globalcor_*.root");
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v116_RecJpsiPhotos_quality_constraint/210722_170211/0001/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v113_RecJpsiPhotos_quality_constraint/210721_170757/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v114_RecJpsiPhotos_quality_constraint/210721_193447/0000/globalcor_*.root");
  
//   chainjpsi.Add("/data/shared/muoncal/MuonGunUL2016_v112_RecJpsiPhotos_quality_constraint/210721_131953/0000/globalcor_*.root");
  
//     const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v67_GenJpsiPhotos_quality/210511_123004/0000/globalcor_*.root";  

    
//   const std::string filenamejpsi = "/data/shared/muoncal/MuonGunUL2016_v67_GenJpsiPhotos_quality/210511_123004/0000/globalcor_0_*.root";  
    
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

  
  if (false) {
    GradHelper gradhelper(nparms);
    HessHelper hesshelper(nparms);
    
  //   ROOT::RDataFrame d(treename, filename);
    ROOT::RDataFrame d(chain);
    
  //   auto d2 = d.Filter("genPt > 5.5 && abs(genEta)<2.4");
  //   auto d2 = d.Filter("genPt > 5.5");
  //   auto d2 = d.Filter("genPt > 5.5 && nValidHits > 9 && nValidPixelHits > 0 && trackCharge*trackPt/genPt/genCharge > 0.3");
    
    auto d2a = d.Define("refPt", "std::abs(1./refParms[0])*std::sin(M_PI_2 - refParms[1])");
    auto d2b = d2a.Define("refEta", "-std::log(std::tan(0.5*(M_PI_2 - refParms[1])))");
    auto d2 = d2b.Filter("refPt > 3.5 && nValidHits > 9 && nValidPixelHits > 0");
  //   auto d2 = d.Filter("genPt > 5.5 && nValidHitsFinal > 9 && nValidPixelHitsFinal > 0");
  //   auto d2 = d.Filter("genPt > 5.5 && nValidHits > 9 && nValidPixelHits > 0");
  //   auto d2 = d.Filter("genPt > 5.5 && nValidHits > 9 && nValidPixelHits > 0 && genEta>-2.4 && genEta<-2.3");
    
  //   auto d2 = d.Filter("genPt > 5.5 && genEta>-1.7 && genEta>-1.7 && genEta<-1.4");
  //   auto d2 = d.Filter("trackPt > 5.5");
  //   auto d3 = d.Define("gradmax", "maxelement(gradv)");
  //   auto d3 = d2.Filter("maxelement(gradv) < 1e5 && maxelementhess(hesspackedv)<1e8");
  //   auto d3 = d2.Filter("maxelement(gradv) < 1e5");
  //   auto d3 = d2.Filter("maxelement(gradv) < 1e5 && edmval < 1e-3");
    
  //   auto d3 = d2.Filter("maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv)");
    auto d3 = d2.Filter("true");
  //   auto d3 = d2.Filter("maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv) && maxelementhess(hesspackedv)<1e8");
    
    
  //   auto grad0 = d3.Book<ROOT::VecOps::RVec<grad_t>,  ROOT::VecOps::RVec<unsigned int> >(std::move(gradhelper), {"gradv", "globalidxv"});
  //   auto hess0 = d3.Book<ROOT::VecOps::RVec<grad_t>,  ROOT::VecOps::RVec<unsigned int> >(std::move(hesshelper), {"hesspackedv", "globalidxv"});
  // //   
  // //   auto gradcounts = d3.Histo1D({"gradcounts", "", int(nparms), -0.5, double(nparms)-0.5}, "globalidxv");
  // // 
  //   std::cout << (*grad0)[0] << std::endl;
  //   std::cout << (*hess0)[0] << std::endl;

  }
  std::cout << "starting second rdf" << std::endl;

  
//   ROOT::RDataFrame dj(treename, filenamejpsi);
  ROOT::RDataFrame dj(chainjpsi);
//   auto dj2a = dj.Filter("std::abs(Jpsi_mass - 3.09692f) < 0.1");
//   auto dj2b = dj2a.Filter("valid(gradv) && valid(hesspackedv)");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5 && std::abs(Jpsi_mass-3.09692)<90e-6 && Jpsi_pt < 28. && maxelement(gradv) < 1e5");
  
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5");
//   auto dj2 = dj.Filter("Muplusgen_pt > 5.5 && Muminusgen_pt > 5.5");
//   auto dj2 = dj.Filter("Muplus_pt > 3.5 && Muminus_pt > 3.5 && Muplus_nvalid > 9 && Muplus_nvalidpixel>0 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 0 && std::abs(Muplus_eta)<2.2 && std::abs(Muminus_eta)<2.2");
//   auto dj2 = dj.Filter("Muplusgen_pt > 5.5 && Muminusgen_pt > 5.5 && Muplus_nvalid > 9 && Muplus_nvalidpixel>0 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 0 && abs(Muplus_eta)<2.4 & abs(Muminus_eta)<2.4");
  
//   auto dj2 = dj.Filter("Muplus_pt > 3.5 && Muminus_pt > 3.5 && Muplus_nvalid > 9 && Muplus_nvalidpixel>0 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 0");
  
//   auto dj2 = dj.Filter("Muplus_pt > 3.5 && Muminus_pt > 3.5 && Muplus_nvalid > 9 && Muplus_nvalidpixel>0 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 0");
  
//   auto dj2 = dj.Filter("Muplus_pt > 0.2 && Muminus_pt > 0.2 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0");
//     auto dj2 = dj.Filter("Muplus_pt > 3.5 && Muminus_pt > 3.5 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0");

//     auto dj2 = dj.Filter("Muplus_pt > 0.9 && Muminus_pt > 0.9 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0");
    
//     auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && !hasdups(globalidxv, Muplus_nhits, Muminus_nhits, Muplus_nvalid, Muminus_nvalid)");
  
//     auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && (Jpsi_pt*std::cosh(Jpsi_eta)) < 20. && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && isconsistent(globalidxv, Muplus_nhits, Muminus_nhits, Muplus_nvalid, Muminus_nvalid, Muplus_nvalidpixel, Muminus_nvalidpixel)");
  
// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0");


// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0");
// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");

// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");
  
  
// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");

// auto dj2 = dj.Filter("Mupluscons_pt > 1.1 && Muminuscons_pt > 1.1 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");
  
// auto dj2 = dj.Filter("Mupluscons_pt > 4.0 && Muminuscons_pt > 4.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");

// auto dj2 = dj.Filter("Mupluscons_pt > 4.0 && Muminuscons_pt > 4.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");
  
auto dj2 = dj.Filter("Mupluscons_pt > 1.1 && Muminuscons_pt > 1.1 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");

// auto dj2 = dj.Filter("Muplusgen_pt > 1.1 && Muminusgen_pt > 1.1 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");

// auto dj2 = dj.Filter("Mupluscons_pt > 2.0 && Muminuscons_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");

// auto dj2 = dj.Filter("Muplus_pt > 2.0 && Muminus_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");

// auto dj2 = dj.Filter("Muplusgen_pt > 2.0 && Muminusgen_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");
  
// auto dj2 = dj.Filter("Muplusgen_pt > 2.0 && Muminusgen_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && Muplusgen_pt>10. && Muminusgen_pt>10.");
  
// auto dj2 = dj.Filter("Muplusgen_pt > 2.0 && Muminusgen_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0");

// auto dj2 = dj.Filter("Muplusgen_pt > 2.0 && Muminusgen_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");


// auto dj2 = dj.Filter("Mupluscons_pt > 2.0 && Muminuscons_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");
// 

// auto dj2 = dj.Filter("Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");
    
// auto dj2 = dj.Filter("Muplusgen_pt > 2.0 && Muminusgen_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && abs(deltachisqval)<1e-2"); 

// auto dj2 = dj.Filter("Muplusgen_pt > 2.0 && Muminusgen_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && Muplusgen_pt<10. && Muminusgen_pt<10."); 

// auto dj2 = dj.Filter("Mupluscons_pt > 3.5 && Muminuscons_pt > 3.5 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");
  
// auto dj2 = dj.Filter("Mupluscons_pt > 3.5 && Muminuscons_pt > 3.5 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");
  
// auto dj2 = dj.Filter("Mupluscons_pt > 3.5 && Muminuscons_pt > 3.5 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && !hasdups(globalidxv, Muplus_nhits, Muminus_nhits)");
  
// auto dj2 = dj.Filter("Muplusgen_pt > 3.5 && Muminusgen_pt > 3.5 && Muplus_nvalid > 9 && Muplus_nvalidpixel>1 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 1 && Jpsi_mass>2.8 && Jpsi_mass<3.4");


// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && (Muplus_pt*cosh(Muplus_eta)) < 100. && (Muminus_pt*cosh(Muminus_eta)) < 100.");


// auto dj2 = dj.Filter("Mupluscons_pt > 1.1 && Muminuscons_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && (Mupluscons_pt*cosh(Mupluscons_eta)) < 100. && (Muminuscons_pt*cosh(Muminuscons_eta)) < 100.");

// auto dj2 = dj.Filter("Muplusgen_pt > 1.1 && Muminusgen_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && (Muplusgen_pt*cosh(Muplusgen_eta)) < 100. && (Muminusgen_pt*cosh(Muminusgen_eta)) < 100.");
  

// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4");

// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && !hasdups(globalidxv, Muplus_nhits, Muminus_nhits)");
  
// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && Muplus_nvalid == Muplus_nmatchedvalid && Muminus_nvalid == Muminus_nmatchedvalid && Muplus_nambiguousmatchedvalid == 0 && Muminus_nambiguousmatchedvalid == 0");

// auto dj2 = dj.Filter("Mupluscons_pt > 2.0 && Muminuscons_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && Muplus_nvalid == Muplus_nmatchedvalid && Muminus_nvalid == Muminus_nmatchedvalid && Muplus_nambiguousmatchedvalid == 0 && Muminus_nambiguousmatchedvalid == 0");
  
  




// auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 9 && Muplus_nvalidpixel>1 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 1 && Jpsi_mass>2.8 && Jpsi_mass<3.4");
    
//     auto dj2 = dj.Filter("Muplusgen_pt > 1.5 && Muminusgen_pt > 1.5 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0");
    
  
//   auto dj2 = dj.Filter("Muplus_pt > 3.5 && Muminus_pt > 3.5 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0");
  
  
//   auto dj2 = dj.Filter("Muplus_pt > 3.5 && Muminus_pt > 3.5 && Muplus_nvalid > 9 && Muplus_nvalidpixel>0 && Muminus_nvalid > 9 && Muminus_nvalidpixel > 0");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5 && TMath::Prob(chisqval, ndof)>1e-3");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5 && std::abs(Jpsigen_mass - 3.09692) < 180e-6");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5 && Jpsi_pt < 28.");
//   auto dj3 = dj2.Filter("true");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && TMath::Prob(chisqval, ndof) > 1e-3 && edmval < 1e-3");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5 && edmval < 1e-3");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5 && edmval < 1e-3 && valid(gradv) && valid(hesspackedv)");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5 && edmval < 1e-3 && valid(gradv) && valid(hesspackedv)");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv) && std::abs(Jpsi_mass - 3.09692) < 1e-4");

//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv)");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv)");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && TMath::Prob(chisqval, ndof)>0.02");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv)");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && TMath::Prob(chisqval, ndof)>0.02");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && TMath::Prob(chisqval_iter0, ndof)>0.02");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && TMath::Prob(chisqval, ndof)>1e-3");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && chisqval < 1e9");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && TMath::Prob(chisqval, ndof)>0.02 && edmval < 1e-5");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && edmval < 1e-5 && abs(deltachisqval)<1e-2");


//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && edmval < 1e-5");
  auto dj3 = dj2.Filter("valid(gradv) && valid(hesspackedv) && edmval < 1e-5");

//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && edmval < 1e-5 && TMath::Prob(chisqval,ndof)>0.02");
  
  
//   auto dj3 = dj2.Filter("valid(gradv) && valid(hesspackedv) && edmval < 1e-5");
  
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && valid(gradv) && valid(hesspackedv) && Jpsi_sigmamass/3.09692 < 0.03");
  
//   auto dj3 = dj2.Filter("maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv)");
   
  
  
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv) && TMath::Prob(chisqval,(Muplus_nvalid + Muplus_nvalidpixel + Muminus_nvalid + Muminus_nvalidpixel - 9 + 1))>1e-3");
  
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && edmval < 1e-3 && valid(gradv) && valid(hesspackedv)");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5 && valid(gradv) && valid(hesspackedv)");
//   auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968 && maxelement(gradv) < 1e5");
//   auto dj3 = dj2.Filter("maxelement(gradv) < 5e5 && valid(gradv) && valid(hesspackedv)");
//   auto dj3 = dj2.Filter("maxelement(gradv) < 1e5");
//   auto dj3 = dj2.Filter("maxelement(gradv) < 13e3");
//   auto dj3 = dj2.Filter("TMath::Prob(chisqval, ndof) > 1e-2 && maxelement(gradv) < 1e5");
    
  GradHelper gradhelperj(nparms);
  HessHelper hesshelperj(nparms);
  
//   gradhelperj.GetResultPtr()->swap(*grad0);
//   hesshelperj.GetResultPtr()->swap(*hess0);
  
  auto grad = dj3.Book<ROOT::VecOps::RVec<grad_t>,  ROOT::VecOps::RVec<unsigned int> >(std::move(gradhelperj), {"gradv", "globalidxv"});
  auto hess = dj3.Book<ROOT::VecOps::RVec<grad_t>,  ROOT::VecOps::RVec<unsigned int> >(std::move(hesshelperj), {"hesspackedv", "globalidxv"});
  
//   auto &grad = grad0;
//   auto &hess = hess0;
  
  std::cout << (*grad)[0] << std::endl;
  std::cout << (*hess)[0] << std::endl;
  
  std::cout << "done second rdf" << std::endl;
  
  TFile *fgrads = new TFile("combinedgrads.root", "RECREATE");
//   TFile *fgrads = new TFile("combinedgradsdebugpacked.root", "RECREATE");
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

