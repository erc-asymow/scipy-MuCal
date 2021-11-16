#include <atomic>
#include <chrono>
#include <algorithm>
#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include <ROOT/TTreeProcessorMT.hxx>
#include <stdlib.h>

// using grad_t = double;
// using grad_t = float;
//
class SymMatrixAtomic {
public:
  SymMatrixAtomic() = default;

  SymMatrixAtomic(unsigned int nparms) {


    const unsigned long long nparmsl = nparms;
    const unsigned long long nsym = nparmsl*(nparmsl+1)/2;

    offsets_.reserve(nparms + 1);

    unsigned long long k = 0;
    for (unsigned long long i = 0; i < nparmsl; ++i) {
//       offsets_.push_back(k - i);
      offsets_.push_back(k);
      k += nparmsl - i;
    }
    assert(k == nsym);
//     std::cout << "k = " << k << " nsym = " << nsym << std::endl;
//     offsets_.push_back(k - nparmsl);
    offsets_.push_back(nsym);

    std::cout<< "allocating huge atomic double vector of size " << nsym << std::endl;

    std::vector<std::atomic<double> > tmp(nsym);
    data_.swap(tmp);
    std::cout<< "initializing values" << std::endl;
    for (unsigned long long i=0; i<data_.size(); ++i) {
      data_[i] = 0.;
    }
    std::cout<< "done huge vector" << std::endl;
  }

  double fetch_add(unsigned int iidx, unsigned int jidx, double val) {
    const unsigned long long ioffset = offsets_[iidx];
    const unsigned long long joffset = offsets_[jidx];

    const unsigned long long idx = jidx >= iidx ? ioffset + jidx - iidx : joffset + iidx - jidx;

//     const double diff = (iidx==jidx && i!=j) ? 2.*val : val;
    const double diff = val;

    std::atomic<double>& ref = data_[idx];
    double old = ref.load();
    double desired = old + diff;
    while (!ref.compare_exchange_weak(old, desired))
    {
        desired = old + diff;
    }

    return desired;
  }

  void fill_row(unsigned int i, double *rowData) {
//     std::fill(rowData.begin(), rowData.begin() + i, 0.);
    if (i > 61200) {
      std::cout << "i = " << i << " offsets i = " << offsets_[i] << " offsets i+1 = " << offsets_[i+1] << " diffoffset = " << offsets_[i+1] - offsets_[i] << std::endl;
    }
    std::fill(rowData, rowData + i, 0.);
    std::copy(data_.begin() + offsets_[i], data_.begin() + offsets_[i+1], rowData + i);
  }



private:
  std::vector<unsigned long long> offsets_;
  std::vector<std::atomic<double> > data_;
};

class GradHelper : public ROOT::Detail::RDF::RActionImpl<GradHelper> {

public:
  using grad_t = float;

  using Result_t = std::vector<double>;

  GradHelper(unsigned int nparms) : nparms_(nparms), grad_(std::make_shared<Result_t>()) {}
//   GradHelper(unsigned int nparms, std::shared_ptr<Result_t> grad) : nparms_(nparms), grad_(grad) {}
//   GradHelper(GradHelper && other) = default;
//   GradHelper(GradHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
//   GradHelper(const GradHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}

  std::shared_ptr<Result_t> GetResultPtr() const { return grad_; }

  void Exec(unsigned int slot, ROOT::VecOps::RVec<grad_t> const& vec, ROOT::VecOps::RVec<unsigned int> const& idxs) {
    Exec(slot, vec, idxs, 1.);
  }

  void Exec(unsigned int slot, ROOT::VecOps::RVec<grad_t> const& vec, ROOT::VecOps::RVec<unsigned int> const& idxs, double w) {
    std::vector<double>& grad = gradtmp_[slot];

    for (unsigned int i=0; i<vec.size(); ++i) {
      const unsigned int& idx = idxs[i];
      grad[idx] += w*vec[i];
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
  using grad_t = float;

  using Result_t = SymMatrixAtomic;
//   using Result_t = std::vector<std::atomic<double> >;
//   using Result_t = std::vector<double>;
//   using Data_t = std::vector<std::atomic<double> >;

  HessHelper(unsigned int nparms) : grad_(std::make_shared<Result_t>(nparms)) {}

//   HessHelper(unsigned int nparms) : nparms_(nparms), grad_(std::make_shared<Result_t>()) {}
//   HessHelper(unsigned int nparms, std::shared_ptr<Result_t> grad) : nparms_(nparms), grad_(grad) {}
//   HessHelper(HessHelper && other) = default;
//   HessHelper(HessHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
//   HessHelper(const HessHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}

  std::shared_ptr<Result_t> GetResultPtr() const { return grad_; }
//   std::shared_ptr<Result_t> GetResultPtr() const {
//     return std::shared_ptr<Result_t>(reinterpret_cast<Result_t*>(gradatom_.get()));
//   }

  void Exec(unsigned int slot, ROOT::VecOps::RVec<grad_t> const& vec, ROOT::VecOps::RVec<unsigned int> const& idxs) {
    Exec(slot, vec, idxs, 1.);
  }

  void Exec(unsigned int slot, ROOT::VecOps::RVec<grad_t> const& vec, ROOT::VecOps::RVec<unsigned int> const& idxs, double w) {

    unsigned int k = 0;
    for (unsigned int i=0; i<idxs.size(); ++i) {
      const unsigned int iidx = idxs[i];
      for (unsigned int j=i; j<idxs.size(); ++j) {
        const unsigned int jidx = idxs[j];
//         const double val = vec[k];

        const double val = (iidx==jidx && i!=j) ? 2.*w*vec[k] : w*vec[k];


        grad_->fetch_add(iidx, jidx, val);

        ++k;
      }
    }


  }
  void InitTask(TTreeReader *, unsigned int) {}

  void Initialize() {
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

//     std::cout << "val0 = " << (*grad_)[0] << std::endl;
//     std::cout << "val1 = " << (*grad_)[1] << std::endl;

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    auto timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end-timestamp_);

    std::cout << "Elapsed time = " << timediff.count() << std::endl;

  }

   std::string GetActionName(){
      return "HessHelper";
   }

private:
//   unsigned long long nparms_;
  std::shared_ptr<Result_t> grad_;
//   std::unique_ptr<std::atomic<double>[]> gradatom_;
//   std::atomic<double>* gradatom_;
//   std::vector<std::vector<unsigned long long> > tmpidxs_;
//   std::vector<unsigned long long> offsets_;
  std::chrono::steady_clock::time_point timestamp_;
};

bool valid(ROOT::VecOps::RVec<float> const& vec) {
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
