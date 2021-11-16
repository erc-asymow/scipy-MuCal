#include <ROOT/RVec.hxx>
#include "Math/GenVector/PtEtaPhiM4D.h"
#include "TFile.h"
#include "TTree.h"
#include <Eigen/Dense>
#include <memory>

using ROOT::VecOps::RVec;

template<typename T>
ROOT::VecOps::RVec<ROOT::VecOps::RVec<T>> splitNestedRVec(const ROOT::VecOps::RVec<T> &vec, const ROOT::VecOps::RVec<int> &counts) {
  using ROOT::VecOps::RVec;

  RVec<RVec<T>> res;
  res.reserve(counts.size());

  int total = 0;
  for (unsigned int i = 0; i < counts.size(); ++i) {
    const int count = counts[i];
    const int end = total + count;
    res.emplace_back(vec.begin() + total, vec.begin() + end);
    total += count;
  }

  return res;
}

template ROOT::VecOps::RVec<ROOT::VecOps::RVec<int>> splitNestedRVec(const ROOT::VecOps::RVec<int>&, const ROOT::VecOps::RVec<int>&);
template ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>> splitNestedRVec(const ROOT::VecOps::RVec<float>&, const ROOT::VecOps::RVec<int>&);
template ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>> splitNestedRVec(const ROOT::VecOps::RVec<double>&, const ROOT::VecOps::RVec<int>&);

class CVHCorrectorSingle {
public:

  using V = ROOT::Math::PtEtaPhiM4D<double>;

  CVHCorrectorSingle(std::string_view filename) {
    TFile *fcor = TFile::Open("correctionResults.root");

    TTree *idxmaptree = static_cast<TTree*>(fcor->Get("idxmaptree"));

    unsigned int idx;
    idxmaptree->SetBranchAddress("idx", &idx);

    idxmap_->reserve(idxmaptree->GetEntries());

    for (long long ientry=0; ientry < idxmaptree->GetEntries(); ++ientry) {
      idxmaptree->GetEntry(ientry);
      idxmap_->push_back(idx);
    }

    TTree *parmtree = static_cast<TTree*>(fcor->Get("parmtree"));

    float x;
    parmtree->SetBranchAddress("x", &x);

    x_->reserve(parmtree->GetEntries());

    for (long long ientry=0; ientry < parmtree->GetEntries(); ++ientry) {
      parmtree->GetEntry(ientry);
      x_->push_back(x);
    }
  }


  std::pair<V, int> operator() (float pt, float eta, float phi, float mass, int charge, const RVec<int> &idxs, const RVec<float> &jac) {

    if (pt < 0.) {
      return std::make_pair<V, int>(V(), -99);
    }

    const double theta = 2.*std::atan(std::exp(-double(eta)));
    const double lam = M_PI_2 - theta;
    const double p = double(pt)/std::sin(theta);
    const double qop = double(charge)/p;

    const Eigen::Matrix<double, 3, 1> curvmom(qop, lam, phi);

    const auto nparms = idxs.size();

    const Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor>> jacMap(jac.data(), 3, nparms);

    Eigen::VectorXd xtrk(nparms);
    for (unsigned int i = 0; i < nparms; ++i) {
      xtrk[i] = (*x_)[(*idxmap_)[idxs[i]]];
    }

    const Eigen::Matrix<double, 3, 1> curvmomcor = curvmom + jacMap.cast<double>()*xtrk;

    const double qopcor = curvmomcor[0];
    const double lamcor = curvmomcor[1];
    const double phicor = curvmomcor[2];

    const double pcor = 1./std::abs(qopcor);
    const int qcor = std::copysign(1., qopcor);

    const double ptcor = pcor*std::cos(lamcor);

    const double thetacor = M_PI_2 - lamcor;
    const double etacor = -std::log(std::tan(0.5*thetacor));

    return std::make_pair<V, int>(V(ptcor, etacor, phicor, mass), int(qcor));

  }

protected:

  std::shared_ptr<std::vector<unsigned int>> idxmap_ = std::make_shared<std::vector<unsigned int>>();
  std::shared_ptr<std::vector<double>> x_ = std::make_shared<std::vector<double>>();
};


class CVHCorrector : public CVHCorrectorSingle {
public:


  CVHCorrector(std::string_view filename) : CVHCorrectorSingle(filename) {}
  CVHCorrector(const CVHCorrectorSingle &corsingle) : CVHCorrectorSingle(corsingle) {}

  RVec<std::pair<V, int>> operator () (const RVec<float> &ptv, const RVec<float> &etav, const RVec<float> &phiv, const RVec<float> &massv, const RVec<int> &chargev, const RVec<RVec<int>> &idxsv, const RVec<RVec<float>> &jacv) {
    RVec<std::pair<V, int>> res;
    res.reserve(ptv.size());

    for (unsigned int i = 0; i < ptv.size(); ++i) {
      res.emplace_back(CVHCorrectorSingle::operator()(ptv[i], etav[i], phiv[i], massv[i], chargev[i], idxsv[i], jacv[i]));
    }

    return res;
  }
};
