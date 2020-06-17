import ROOT
import numpy as np

ROOT.ROOT.EnableImplicitMT()

ROOT.gInterpreter.AddIncludePath("/usr/include/eigen3/")

#ROOT.gInterpreter.Declare("""
##include <Eigen/Dense>
    
#typedef Eigen::Matrix<double,5,5> Matrix5d;
#typedef Eigen::Matrix<double,5,1> Vector5d;

#typedef ROOT::Math::SMatrix<double,5> SMatrix55;
#typedef ROOT::Math::SVector<double,5> SVector5;

#auto eig(double c00, double c01, double c02, double c03, double c04, double c11, double c12, double c13, double c14, double c22, double c23, double c24, double c33, double c34, double c44) {
    
    #Matrix5d M;
    #M << c00,c01,c02,c03,c04,  c01,c11,c12,c13,c14,  c02,c12,c22,c23,c24,  c03,c13,c23,c33,c34,  c04,c14,c24,c34,c44;
    
    #Eigen::SelfAdjointEigenSolver<Matrix5d> s;
    #s.compute(M);
    
    #double ea[5];
    #Eigen::Map<Vector5d>( ea, s.eigenvalues().rows(), s.eigenvalues().cols() ) = s.eigenvalues();
    #SVector5 e(ea,5);
    
    #double va[5*5];
    #Eigen::Map<Matrix5d>( va, s.eigenvectors().cols(), s.eigenvectors().rows() ) = s.eigenvectors().transpose();
    #SMatrix55 v(va,5*5);
    
    #return std::make_tuple(e,v);
#}
#""")

#ROOT.gInterpreter.Declare("""
##include <Eigen/Dense>
    
#typedef Eigen::Matrix<double,5,5> CovIn;
#typedef Eigen::Matrix<double,3,5> Jac;
#typedef Eigen::Matrix<double,3,3> CovOut;
#typedef Eigen::Matrix<double,3,1> EigVals;
#typedef ROOT::Math::SMatrix<double,3> SMatrix33;
#typedef ROOT::Math::SVector<double,3> SVector3;
    
#auto covrphi(double c00, double c01, double c02, double c03, double c04, double c11, double c12, double c13, double c14, double c22, double c23, double c24, double c33, double c34, double c44, double k, double eta) {
    
    #double theta = 2.*atan(exp(-eta));
    
    #CovIn covin;
    #covin << c00,c01,c02,c03,c04,
         #c01,c11,c12,c13,c14,
         #c02,c12,c22,c23,c24,
         #c03,c13,c23,c33,c34,
         #c04,c14,c24,c34,c44;
         
    #Jac jac;
    #jac << 1./sin(theta), k/tan(theta), 0., 0., 0.,
           #0., 0., 1., 0., 0.,
           #0., 0., 0., 1., 0.;
           
    #CovOut covout = jac*covin*jac.transpose();
    #Eigen::SelfAdjointEigenSolver<CovOut> s(covout);
    
    
    #double ca[3*3];
    #Eigen::Map<CovOut>(ca, covout.rows(), covout.cols() ) = covout;
    #SMatrix33 c(ca,3*3);
    
    #double ea[3];
    #Eigen::Map<EigVals>( ea, s.eigenvalues().rows(), s.eigenvalues().cols() ) = s.eigenvalues();
    #SVector3 e(ea,3);
    
    #double va[3*3];
    #Eigen::Map<CovOut>( va, s.eigenvectors().cols(), s.eigenvectors().rows() ) = s.eigenvectors().transpose();
    #SMatrix33 v(va,3*3);
    
    #return std::make_tuple(c,e,v);    
#}
#""")


ROOT.gInterpreter.Declare("""
#include <Eigen/Dense>
    
typedef Eigen::Matrix<double,5,5> CovIn;
typedef Eigen::Matrix<double,3,5> Jac;
typedef Eigen::Matrix<double,3,3> CovOut;
typedef Eigen::Matrix<double,3,1> EigVals;
typedef ROOT::Math::SMatrix<double,3> SMatrix33;
typedef ROOT::Math::SVector<double,3> SVector3;
    
auto covrphi(double c00, double c01, double c02, double c03, double c04, double c11, double c12, double c13, double c14, double c22, double c23, double c24, double c33, double c34, double c44, double k, double eta) {
    
    double theta = 2.*atan(exp(-eta));
    
    CovIn covin;
    covin << c00,c01,c02,c03,c04,
         c01,c11,c12,c13,c14,
         c02,c12,c22,c23,c24,
         c03,c13,c23,c33,c34,
         c04,c14,c24,c34,c44;
         
    Jac jac;
    jac << 1./sin(theta), k/tan(theta), 0., 0., 0.,
           0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0.;
           
    CovOut covout = jac*covin*jac.transpose();
    
    double ca[3*3];
    Eigen::Map<CovOut>(ca, 3, 3) = covout;
    SMatrix33 c(ca,3*3);
    
    double va[3*3];
    Eigen::Map<CovOut>(va, 3, 3) = covout.inverse();
    SMatrix33 v(va,3*3);
    
    return std::make_pair(c,v);    
}
""")


treename = "tree"
fname = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muGunCov.root"

d = ROOT.ROOT.RDataFrame(treename, fname)
#d = d.Filter("1./reco_curv>148. && 1./reco_curv<150. && reco_eta>-2.4 && reco_eta<-2.3 && reco_charge>-2.")
#d = d.Define("eig", "eig(cov00,cov01,cov02,cov03,cov04,cov11,cov12,cov13,cov14,cov22,cov23,cov24,cov33,cov34,cov44)")
d = d.Define("covrphi", "covrphi(cov00,cov01,cov02,cov03,cov04,cov11,cov12,cov13,cov14,cov22,cov23,cov24,cov33,cov34,cov44,reco_curv,reco_eta)")
#d.Snapshot("tree","/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muGunCov_eig.root")
d.Snapshot("tree","/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muGunCov_covrphiinv.root")

treename = "tree"
fname = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/JpsiToMuMu_JpsiPt8_pythia8.root"

d = ROOT.ROOT.RDataFrame(treename, fname)
#d = d.Filter("mcpt1 > 6. && mcpt1<7. && mceta1>-2.4 && mceta1<-2.3")
for idx in [1,2]:
    #d = d.Define(f"eig{idx}", f"eig(cov{idx}_00,cov{idx}_01,cov{idx}_02,cov{idx}_03,cov{idx}_04,cov{idx}_11,cov{idx}_12,cov{idx
    d = d.Define(f"covrphi{idx}", f"covrphi(cov{idx}_00,cov{idx}_01,cov{idx}_02,cov{idx}_03,cov{idx}_04,cov{idx}_11,cov{idx}_12,cov{idx}_13,cov{idx}_14,cov{idx}_22,cov{idx}_23,cov{idx}_24,cov{idx}_33,cov{idx}_34,cov{idx}_44, 1./pt{idx},eta{idx})")
#d.Snapshot("tree","/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/JpsiToMuMu_JpsiPt8_pythia8_eig.root")
d.Snapshot("tree","/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/JpsiToMuMu_JpsiPt8_pythia8_covrphiinv.root")




#input("wait")
