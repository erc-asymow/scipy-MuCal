#include <atomic>
#include <chrono>
#include <algorithm>
#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include <ROOT/TTreeProcessorMT.hxx>
#include <stdlib.h>  
#include </usr/include/eigen3/Eigen/Dense>


// using grad_t = double;
using grad_t = float;


class GradHelper : public ROOT::Detail::RDF::RActionImpl<GradHelper> {
  
  
public:

  
  using Result_t = std::pair<Eigen::VectorXd, Eigen::MatrixXd>;
  
  GradHelper() : nparms_(48*3), res_(std::make_shared<Result_t>()) {}  
//   GradHelper(unsigned int nparms, std::shared_ptr<Result_t> grad) : nparms_(nparms), grad_(grad) {}
  GradHelper(GradHelper && other) = default;
//   GradHelper(GradHelper && other) : nparms_(other.nparms_), grad_(other.grad_) {}
//   GradHelper(const GradHelper &other) : nparms_(other.nparms_), grad_(other.grad_) {}
  
  std::shared_ptr<Result_t> GetResultPtr() const { return res_; }

  void Exec(unsigned int slot, float mass, float masserr, float eta0, float eta1, float pt0, float pt1, float phi0, float phi1) {
//     std::vector<double>& grad = gradtmp_[slot];
    
//     std::cout << "Exec" << std::endl;
    
    Eigen::VectorXd &grad = gradtmp_[slot];
    Eigen::MatrixXd &hess = hesstmp_[slot];
    
    const unsigned int ieta0 = std::clamp(std::floor((eta0 + 2.4)/0.1), 0., 47.);
    const unsigned int ieta1 = std::clamp(std::floor((eta1 + 2.4)/0.1), 0., 47.);
    
//     const unsigned int ieta0 = 0;
//     const unsigned int ieta1 = 0;
    
    const std::array<unsigned int, 6> idxs = {{ 3*ieta0, 3*ieta0 + 1, 3*ieta0 + 2, 3*ieta1, 3*ieta1 + 1, 3*ieta1 + 2 }};
    
    const double q0 = 1.;
    const double q1 = -1.;
    
    constexpr double mu = 0.1056583745;
    constexpr double m0 = 3.09692;
    
    const ROOT::Math::PtEtaPhiMVector mom0(pt0, eta0, phi0, mu);
    const ROOT::Math::PtEtaPhiMVector mom1(pt1, eta1, phi1, mu);
    
    const double massval = (mom0 + mom1).mass();
    
    const double sigm = 2.*std::pow(massval, -3)*masserr;
    
//     const double sigm = 1.;
//     const double sigm = masserr;
    
    const double p0 = pt0*std::cosh(eta0);
    const double p1 = pt1*std::cosh(eta1);
    
    const double theta0 = 2.*std::atan(exp(-eta0));
    const double theta1 = 2.*std::atan(exp(-eta1));
    
    const double dx0 = std::sin(theta0)*std::cos(phi0);
    const double dy0 = std::sin(theta0)*std::sin(phi0);
    const double dz0 = std::cos(theta0);
    
    const double dx1 = std::sin(theta1)*std::cos(phi1);
    const double dy1 = std::sin(theta1)*std::sin(phi1);
    const double dz1 = std::cos(theta1);
    
    const double cosalpha = dx0*dx1 + dy0*dy1 + dz0*dz1;
          
    const double xf0 = std::pow(mu, 2);
    const double xf1 = cosalpha*p0*p1;
    const double xf2 = 2.0*xf1;
    const double xf3 = -xf2;
    const double xf4 = std::pow(p0, 2);
    const double xf5 = xf0 + 1.0*xf4;
    const double xf6 = std::sqrt(xf5);
    const double xf7 = std::pow(p1, 2);
    const double xf8 = 1.0*xf7;
    const double xf9 = xf0 + xf8;
    const double xf10 = std::sqrt(xf9);
    const double xf11 = xf10*xf6;
    const double xf12 = 1.0/(2*xf0 + 2*xf11 + xf3) - 1/std::pow(m0, 2);
    const double xf13 = xf4/xf6;
    const double xf14 = xf10*xf13;
    const double xf15 = 2.0*xf14;
    const double xf16 = xf15 + xf3;
    const double xf17 = xf12*xf16;
    const double xf18 = 1.0*xf1;
    const double xf19 = -xf18;
    const double xf20 = xf0 + xf11 + xf19;
    const double xf21 = std::pow(sigm, -2);
    const double xf22 = 0.5*xf21;
    const double xf23 = xf22/std::pow(xf20, 2);
    const double xf24 = std::pow(xf20, -4);
    const double xf25 = xf22*xf24;
    const double xf26 = 4.0*xf1;
    const double xf27 = 6.0*xf14;
    const double xf28 = 2.0*std::pow(p0, 4)*xf10/std::pow(xf5, 3.0/2.0);
    const double xf29 = xf12*xf23;
    const double xf30 = -xf26;
    const double xf31 = 4.0*xf14;
    const double xf32 = 0.25/std::pow(xf20, 3);
    const double xf33 = xf32*(xf30 + xf31);
    const double xf34 = xf17*xf21;
    const double xf35 = 1.0/pt0;
    const double xf36 = xf26*xf35;
    const double xf37 = -xf31*xf35 + xf36;
    const double xf38 = xf32*xf34;
    const double xf39 = xf2*xf35;
    const double xf40 = -xf15*xf35 + xf39;
    const double xf41 = xf21*xf40;
    const double xf42 = 0.125*xf24;
    const double xf43 = xf16*xf42;
    const double xf44 = xf29*(xf27*xf35 - xf28*xf35 - xf36) + xf41*xf43;
    const double xf45 = q0*xf26;
    const double xf46 = pt0*xf45;
    const double xf47 = pt0*q0;
    const double xf48 = xf31*xf47 - xf46;
    const double xf49 = xf2*xf47;
    const double xf50 = xf15*xf47 - xf49;
    const double xf51 = xf21*xf43;
    const double xf52 = q0*xf27;
    const double xf53 = q0*xf28;
    const double xf54 = xf29*(-pt0*xf52 + pt0*xf53 + xf46) + xf50*xf51;
    const double xf55 = 1.0/xf10;
    const double xf56 = xf55*xf6;
    const double xf57 = xf56*xf7;
    const double xf58 = 4.0*xf57;
    const double xf59 = xf30 + xf58;
    const double xf60 = 2.0*xf57;
    const double xf61 = xf3 + xf60;
    const double xf62 = 2.0*xf13*xf55*xf7;
    const double xf63 = xf29*(xf2 - xf62) + xf51*xf61;
    const double xf64 = 1.0/pt1;
    const double xf65 = xf26*xf64;
    const double xf66 = -xf58*xf64 + xf65;
    const double xf67 = xf2*xf64;
    const double xf68 = -xf60*xf64 + xf67;
    const double xf69 = xf62*xf64;
    const double xf70 = xf29*(-xf67 + xf69) + xf51*xf68;
    const double xf71 = q1*xf26;
    const double xf72 = pt1*xf71;
    const double xf73 = pt1*q1;
    const double xf74 = xf58*xf73 - xf72;
    const double xf75 = xf2*xf73;
    const double xf76 = xf60*xf73 - xf75;
    const double xf77 = xf62*xf73;
    const double xf78 = xf29*(xf75 - xf77) + xf51*xf76;
    const double xf79 = xf12*xf41;
    const double xf80 = std::pow(pt0, 2);
    const double xf81 = 1.0/xf80;
    const double xf82 = xf32*xf79;
    const double xf83 = xf41*xf42;
    const double xf84 = xf29*(-xf45 + xf52 - xf53) + xf50*xf83;
    const double xf85 = xf35*xf62;
    const double xf86 = xf29*(-xf39 + xf85) + xf61*xf83;
    const double xf87 = xf29*(-xf35*xf69 + xf39*xf64) + xf68*xf83;
    const double xf88 = xf29*(-xf39*xf73 + xf73*xf85) + xf76*xf83;
    const double xf89 = xf12*xf21;
    const double xf90 = xf50*xf89;
    const double xf91 = xf32*xf90;
    const double xf92 = std::pow(q0, 2)*xf80;
    const double xf93 = xf21*xf42;
    const double xf94 = xf50*xf93;
    const double xf95 = xf29*(-xf47*xf62 + xf49) + xf61*xf94;
    const double xf96 = xf29*(-xf47*xf67 + xf47*xf69) + xf68*xf94;
    const double xf97 = xf29*(-xf47*xf77 + xf49*xf73) + xf76*xf94;
    const double xf98 = xf61*xf89;
    const double xf99 = xf32*xf98;
    const double xf100 = xf56*xf8;
    const double xf101 = 6.0*xf57;
    const double xf102 = 2.0*std::pow(p1, 4)*xf6/std::pow(xf9, 3.0/2.0);
    const double xf103 = xf61*xf93;
    const double xf104 = xf103*xf68 + xf29*(xf101*xf64 - xf102*xf64 - xf65);
    const double xf105 = q1*xf101;
    const double xf106 = q1*xf102;
    const double xf107 = xf103*xf76 + xf29*(-pt1*xf105 + pt1*xf106 + xf72);
    const double xf108 = xf68*xf89;
    const double xf109 = xf108*xf32;
    const double xf110 = std::pow(pt1, 2);
    const double xf111 = 1.0/xf110;
    const double xf112 = xf29*(xf105 - xf106 - xf71) + xf68*xf76*xf93;
    const double xf113 = xf76*xf89;
    const double xf114 = xf113*xf32;
    const double xf115 = std::pow(q1, 2)*xf110;
    const double dchisqdA0 = xf17*xf23;
    const double d2chisqdA0dA0 = xf25*std::pow(xf14 + xf19, 2) + xf29*(xf26 - xf27 + xf28) + xf33*xf34;
    const double d2chisqdA0de0 = xf37*xf38 + xf44;
    const double d2chisqdA0dM0 = xf38*xf48 + xf54;
    const double d2chisqdA0dA1 = xf38*xf59 + xf63;
    const double d2chisqdA0de1 = xf38*xf66 + xf70;
    const double d2chisqdA0dM1 = xf38*xf74 + xf78;
    const double dchisqde0 = xf29*xf40;
    const double d2chisqde0dA0 = xf33*xf79 + xf44;
    const double d2chisqde0de0 = xf25*std::pow(-xf14*xf35 + xf18*xf35, 2) + xf29*(xf26*xf81 - xf27*xf81 + xf28*xf81) + xf37*xf82;
    const double d2chisqde0dM0 = xf48*xf82 + xf84;
    const double d2chisqde0dA1 = xf59*xf82 + xf86;
    const double d2chisqde0de1 = xf66*xf82 + xf87;
    const double d2chisqde0dM1 = xf74*xf82 + xf88;
    const double dchisqdM0 = xf29*xf50;
    const double d2chisqdM0dA0 = xf33*xf90 + xf54;
    const double d2chisqdM0de0 = xf37*xf91 + xf84;
    const double d2chisqdM0dM0 = xf25*std::pow(xf14*xf47 - xf18*xf47, 2) + xf29*(xf26*xf92 - xf27*xf92 + xf28*xf92) + xf48*xf91;
    const double d2chisqdM0dA1 = xf59*xf91 + xf95;
    const double d2chisqdM0de1 = xf66*xf91 + xf96;
    const double d2chisqdM0dM1 = xf74*xf91 + xf97;
    const double dchisqdA1 = xf29*xf61;
    const double d2chisqdA1dA0 = xf33*xf98 + xf63;
    const double d2chisqdA1de0 = xf37*xf99 + xf86;
    const double d2chisqdA1dM0 = xf48*xf99 + xf95;
    const double d2chisqdA1dA1 = xf25*std::pow(xf100 + xf19, 2) + xf29*(-xf101 + xf102 + xf26) + xf59*xf99;
    const double d2chisqdA1de1 = xf104 + xf66*xf99;
    const double d2chisqdA1dM1 = xf107 + xf74*xf99;
    const double dchisqde1 = xf29*xf68;
    const double d2chisqde1dA0 = xf108*xf33 + xf70;
    const double d2chisqde1de0 = xf109*xf37 + xf87;
    const double d2chisqde1dM0 = xf109*xf48 + xf96;
    const double d2chisqde1dA1 = xf104 + xf109*xf59;
    const double d2chisqde1de1 = xf109*xf66 + xf25*std::pow(-xf100*xf64 + xf18*xf64, 2) + xf29*(-xf101*xf111 + xf102*xf111 + xf111*xf26);
    const double d2chisqde1dM1 = xf109*xf74 + xf112;
    const double dchisqdM1 = xf29*xf76;
    const double d2chisqdM1dA0 = xf113*xf33 + xf78;
    const double d2chisqdM1de0 = xf114*xf37 + xf88;
    const double d2chisqdM1dM0 = xf114*xf48 + xf97;
    const double d2chisqdM1dA1 = xf107 + xf114*xf59;
    const double d2chisqdM1de1 = xf112 + xf114*xf66;
    const double d2chisqdM1dM1 = xf114*xf74 + xf25*std::pow(xf100*xf73 - xf18*xf73, 2) + xf29*(-xf101*xf115 + xf102*xf115 + xf115*xf26);
    Eigen::Matrix<double, 1, 6> g;
    g(0,0) = dchisqdA0;
    g(0,1) = dchisqde0;
    g(0,2) = dchisqdM0;
    g(0,3) = dchisqdA1;
    g(0,4) = dchisqde1;
    g(0,5) = dchisqdM1;
    Eigen::Matrix<double, 6, 6> h;
    h(0,0) = d2chisqdA0dA0;
    h(0,1) = d2chisqdA0de0;
    h(0,2) = d2chisqdA0dM0;
    h(0,3) = d2chisqdA0dA1;
    h(0,4) = d2chisqdA0de1;
    h(0,5) = d2chisqdA0dM1;
    h(1,0) = d2chisqde0dA0;
    h(1,1) = d2chisqde0de0;
    h(1,2) = d2chisqde0dM0;
    h(1,3) = d2chisqde0dA1;
    h(1,4) = d2chisqde0de1;
    h(1,5) = d2chisqde0dM1;
    h(2,0) = d2chisqdM0dA0;
    h(2,1) = d2chisqdM0de0;
    h(2,2) = d2chisqdM0dM0;
    h(2,3) = d2chisqdM0dA1;
    h(2,4) = d2chisqdM0de1;
    h(2,5) = d2chisqdM0dM1;
    h(3,0) = d2chisqdA1dA0;
    h(3,1) = d2chisqdA1de0;
    h(3,2) = d2chisqdA1dM0;
    h(3,3) = d2chisqdA1dA1;
    h(3,4) = d2chisqdA1de1;
    h(3,5) = d2chisqdA1dM1;
    h(4,0) = d2chisqde1dA0;
    h(4,1) = d2chisqde1de0;
    h(4,2) = d2chisqde1dM0;
    h(4,3) = d2chisqde1dA1;
    h(4,4) = d2chisqde1de1;
    h(4,5) = d2chisqde1dM1;
    h(5,0) = d2chisqdM1dA0;
    h(5,1) = d2chisqdM1de0;
    h(5,2) = d2chisqdM1dM0;
    h(5,3) = d2chisqdM1dA1;
    h(5,4) = d2chisqdM1de1;
    h(5,5) = d2chisqdM1dM1;




    
    
    for (unsigned int i = 0; i < idxs.size(); ++i) {
      const unsigned int iidx = idxs[i];
      grad[iidx] += g[i];
//       if (std::isnan(g[i])) {
//         std::cout << "grad nan" << std::endl;
//         std::cout << g << std::endl;
//       }
      for (unsigned int j = 0; j < idxs.size(); ++j) {
        const unsigned int jidx = idxs[j];
        hess(iidx, jidx) += h(i, j);
      }
    }
    
//     grad += g;
//     hess += h;


    
//     Eigen::Matrix<double, 6, 1> g;
//     Eigen::Matrix<double, 6, 6> h;
    
//     const double msq = mass*mass;
//     constexpr double msq0 = 3.09692*3.09692;
//     
// //     const double wsig = 1.;
//     const double wsig = 1./masserr/masserr;
//     
//     const double dmsq = 2.*(msq - msq0)*wsig;
//     
//     g[0] = msq;
//     g[1] = -msq/pt0;
//     g[2] = q0*pt0*msq;
//     g[3] = msq;
//     g[4] = -msq/pt1;
//     g[5] = q1*pt1*msq;
//     
//     h(0,0) = 0.;
//     h(0,1) = 0.;
//     h(0,2) = 0.;
//     h(0,3) = msq;
//     h(0,4) = -msq/pt1;
//     h(0,5) = q1*pt1*msq;
//     h(1,0) = 0.;
//     h(1,1) = 0.;
//     h(1,2) = 0.;
//     h(1,3) = -msq/pt0;
//     h(1,4) = msq/pt0/pt1;
//     h(1,5) = -msq*q1*pt1/pt0;
//     h(2,0) = 0.;
//     h(2,1) = 0.;
//     h(2,2) = 0.;
//     h(2,3) = q0*pt0*msq;
//     h(2,4) = -q0*pt0/pt1*msq;
//     h(2,5) = q0*q1*pt0*pt1*msq;
//     h(3,0) = msq;
//     h(3,1) = -msq/pt0;
//     h(3,2) = q0*pt0*msq;
//     h(3,3) = 0.;
//     h(3,4) = 0.;
//     h(3,5) = 0.;
//     h(4,0) = -msq/pt1;
//     h(4,1) = msq/pt0/pt1;
//     h(4,2) = -q0*pt0/pt1*msq;
//     h(4,3) = 0.;
//     h(4,4) = 0.;
//     h(4,5) = 0.;
//     h(5,0) = q1*pt1*msq;
//     h(5,1) = -q1*pt1/pt0*msq;
//     h(5,2) = q0*q1*pt0*pt1*msq;
//     h(5,3) = 0.;
//     h(5,4) = 0.;
//     h(5,5) = 0.;
//     
// //     std::cout << g << std::endl;
// //     std::cout << h << std::endl;
//     
//     for (unsigned int i = 0; i < idxs.size(); ++i) {
//       const unsigned int iidx = idxs[i];
//       grad[iidx] += dmsq*g[i];
//       for (unsigned int j = 0; j < idxs.size(); ++j) {
//         const unsigned int jidx = idxs[j];
//         hess(iidx, jidx) += dmsq*h(i, j) + 2.*wsig*g[i]*g[j];
//       }
//     }
    
  }
  void InitTask(TTreeReader *, unsigned int) {}

  void Initialize() {
//     const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetImplicitMTPoolSize() : 1;
    
    std::cout << "Initialize" << std::endl;
    
    const unsigned int nslots = ROOT::IsImplicitMTEnabled() ? ROOT::GetThreadPoolSize() : 1;
    
    
    gradtmp_.clear();
    gradtmp_.resize(nslots, Eigen::VectorXd::Zero(nparms_));
    
    hesstmp_.clear();
    hesstmp_.resize(nslots, Eigen::MatrixXd::Zero(nparms_, nparms_));
    
  }

    
  void Finalize() {
    
    std::cout << "Finalize" << std::endl;
    
    res_->first = Eigen::VectorXd::Zero(nparms_);
    for (auto const& grad: gradtmp_) {
      res_->first += grad;
    }
    
    res_->second = Eigen::MatrixXd::Zero(nparms_, nparms_);
    for (auto const& hess: hesstmp_) {
      res_->second += hess;
    }
  }

   std::string GetActionName(){
      return "GradHelper";
   }
   
   
private:
  unsigned int nparms_;
  std::vector<Eigen::VectorXd> gradtmp_;
  std::vector<Eigen::MatrixXd> hesstmp_;
  std::shared_ptr<Result_t> res_;
   
};


void fillmatricesrdfsimpleinvmsq() {
    
//   std::atomic<double> t;
//   std::cout<< "is lock free: " << t.is_lock_free() << std::endl;
  
  
//   std::cout << ROOT::GetImplicitMTPoolSize() << std::endl;

  setenv("XRD_PARALLELEVTLOOP", "16", true);
  
  ROOT::EnableImplicitMT();

  
  TChain chain("tree");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v134_RecJpsiPhotos_quality_constraint/210805_125523/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v134_RecJpsiPhotos_quality_constraint/210805_125523/0001/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v135_RecJpsiPhotos_quality_constraint/210805_130727/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v135_RecJpsiPhotos_quality_constraint/210805_130727/0001/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v137_RecJpsiPhotos_idealquality/210806_103002/0000/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v146_RecJpsiPhotos_quality_constraint/210809_232349/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v146_RecJpsiPhotos_quality_constraint/210809_232349/0001/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v166_RecJpsiPhotos_idealquality_constraint/210907_053804/0000/globalcor_*.root");
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v166_RecJpsiPhotos_idealquality_constraint/210907_053804/0001/globalcor_*.root");


  chain.Add("/data/shared/muoncal/MuonGunUL2016_v167_RecJpsiPhotos_idealquality_constraint/210908_011211/0000/globalcor_*.root");
  chain.Add("/data/shared/muoncal/MuonGunUL2016_v167_RecJpsiPhotos_idealquality_constraint/210908_011211/0001/globalcor_*.root");
  
//   chain.Add("/data/shared/muoncal/MuonGunUL2016_v135_RecJpsiPhotos_quality_constraint/210805_130727/0000/globalcor_0_17.root");


  ROOT::RDataFrame dj(chain);
  
//   auto dj2 = dj.Filter("Muplus_pt > 1.1 && Muminus_pt > 1.1 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.8 && Jpsi_mass<3.4 && Jpsi_sigmamass > 0.");
    auto dj2 = dj.Filter("Muplusgen_pt > 2.0 && Muminusgen_pt > 2.0 && Muplus_nvalid > 3 && Muplus_nvalidpixel>0 && Muminus_nvalid > 3 && Muminus_nvalidpixel > 0  && Jpsi_sigmamass > 0.  && Jpsi_mass>2.8 && Jpsi_mass<3.4 && TMath::Prob(chisqval, ndof)>0.02");
//   auto dj2 = dj.Filter("Muplus_pt > 5.5 && Muminus_pt > 5.5 && Muplus_nvalid > 2 && Muplus_nvalidpixel>0 && Muminus_nvalid > 2 && Muminus_nvalidpixel > 0 && Jpsi_mass>2.6 && Jpsi_mass<3.6 && Jpsi_sigmamass>0.");
//&& abs(Jpsi_mass-Jpsigen_mass)/Jpsi_sigmamass < 1.
  auto dj3 = dj2.Filter("Jpsigen_mass > 3.0968");
  
  GradHelper gradhelperj;
  
  auto res = dj3.Book<float, float, float, float, float, float, float, float >(std::move(gradhelperj), {"Jpsi_mass", "Jpsi_sigmamass", "Muplus_eta", "Muminus_eta", "Muplus_pt", "Muminus_pt", "Muplus_phi","Muminus_phi"});
  
  const Eigen::VectorXd &grad = res->first;
  const Eigen::MatrixXd &hess = res->second;
  
//   std::cout << grad << std::endl;  
//   std::cout << hess << std::endl;
  
  Eigen::LDLT<Eigen::MatrixXd> Cinvd(hess);
  
  const Eigen::VectorXd corparms = Cinvd.solve(-grad);
  
  const Eigen::VectorXd errs = Cinvd.solve(Eigen::MatrixXd::Identity(grad.rows(), grad.rows())).diagonal().array().sqrt();
  
  constexpr unsigned int nbins = 48;
//   constexpr unsigned int nbins = 1;
  
  for (unsigned int ieta = 0; ieta < nbins; ++ieta) {
//     std::cout << corparms[3*ieta] << " " << corparms[3*ieta+1] << " " << corparms[3*ieta+2] << std::endl;
    std::cout << corparms[3*ieta] << " +- " << errs[3*ieta] << " " << corparms[3*ieta+1] << " +- " << errs[3*ieta+1] << " " << corparms[3*ieta+2] << " +- " << errs[3*ieta+2] << std::endl;
  }




    
}

