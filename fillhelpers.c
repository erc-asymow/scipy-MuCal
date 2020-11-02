#include <atomic>
#include <chrono>
#include <ROOT/RDataFrame.hxx>

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

    
    std::vector<unsigned long long>& tmpidxs = tmpidxs_[slot];
    tmpidxs.resize(vec.size());
    
    unsigned int k=0;
    for (unsigned int i=0; i<idxs.size(); ++i) {
      const unsigned long long idx = idxs[i];
      const unsigned long long iidx =  nparms_*idx;
      for (unsigned int j=i; j<idxs.size(); ++j) {
        const unsigned long long jidx = idxs[j];
        tmpidxs[k] = iidx + jidx;
        ++k;
      }
    }

//     ntotals_[slot] += vec.size();
    
    for (unsigned int i=0; i<vec.size(); ++i) {
      const unsigned long long idx = tmpidxs[i];
      
      std::atomic<double>& ref = (*grad_)[idx];
      const double& diff = vec[i];
      double old = ref.load();
      double desired = old + diff;
      while (!ref.compare_exchange_weak(old, desired))
      {
           desired = old + diff;
//            nmisses_[slot]++;
      }
      
    }
    
  }
  void InitTask(TTreeReader *, unsigned int) {}

  void Initialize() {
    const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetImplicitMTPoolSize() : 1;
    
//     grad_ = std::shared_ptr<std::vector<double> >(new std::vector<double>());
    
    tmpidxs_.clear();
    tmpidxs_.resize(nslots);
    
    
    
    std::cout<< "allocating huge atomic double vector of size " << nparms_*nparms_ << std::endl;
    unsigned long long size = nparms_*nparms_;

    
//     std::vector<std::atomic<double> > tmp(size);
    std::vector<std::atomic<double> > tmp(size);
//     std::cout << "swapping" << std::endl;
    grad_->swap(tmp);
//     gradatom_ = std::shared_ptr<std::vector<std::atomic<double> > >(new std::vector<std::atomic<double> >);
//     gradatom_->swap(tmp);
//     gradatom_ = std::unique_ptr<std::atomic<double>*>(new std::atomic<double>[nparms_*nparms_]);
//     gradatom_ = new std::atomic<double>[size];
    std::cout<< "initializing values" << std::endl;
//     memset(grad_->data(), 0, grad_->size()*sizeof(double));
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
  std::vector<std::vector<unsigned long long> > tmpidxs_;
  std::chrono::steady_clock::time_point timestamp_;
};

auto BookHelper(ROOT::RDF::RNode& df, GradHelper& helper, const std::string& vals, const std::string& idxs) {
  return df.Book<ROOT::VecOps::RVec<float>,  ROOT::VecOps::RVec<unsigned int> >(std::move(helper), {vals, idxs} );
}

auto BookHelper(ROOT::RDF::RNode& df, HessHelper& helper, const std::string& vals, const std::string& idxs) {
  return df.Book<ROOT::VecOps::RVec<float>,  ROOT::VecOps::RVec<unsigned int> >(std::move(helper), {vals, idxs} );
}

// std::vector<double>* ConvertHessResult(ROOT::RDF::RResultPtr<std::vector<std::atomic<double> > >& res) {
//   return reinterpret_cast<std::vector<double>* >(res.GetPtr());
// // ROOT::VecOps::RVec<double> ConvertHessResult(std::vector<std::atomic<double> >& res) {
// // double* ConvertHessResult(std::vector<std::atomic<double> >& res) {
// //   return reinterpret_cast<double*>(res.data());
// //   ROOT::VecOps::RVec<double> out(reinterpret_cast<double*>(res.data()), res.size());
// //   return out;
// //   std::vector<double> out;
// //   out.swap(*reinterpret_cast<std::vector<double>* >(&res));
// //   return out;
// // //   return reinterpret_cast<double*>(res.data());
// // //   return reinterpret_cast<std::vector<double>* >(&res);
// }

std::vector<double>* ConvertHessResult(std::vector<std::atomic<double> >* res) {
  return reinterpret_cast<std::vector<double>* >(res);
// ROOT::VecOps::RVec<double> ConvertHessResult(std::vector<std::atomic<double> >& res) {
// double* ConvertHessResult(std::vector<std::atomic<double> >& res) {
//   return reinterpret_cast<double*>(res.data());
//   ROOT::VecOps::RVec<double> out(reinterpret_cast<double*>(res.data()), res.size());
//   return out;
//   std::vector<double> out;
//   out.swap(*reinterpret_cast<std::vector<double>* >(&res));
//   return out;
// //   return reinterpret_cast<double*>(res.data());
// //   return reinterpret_cast<std::vector<double>* >(&res);
}
