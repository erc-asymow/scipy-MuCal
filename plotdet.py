import ROOT
import numpy as np

ROOT.ROOT.EnableImplicitMT()

#ROOT.gStyle.SetOptStat(111100)


#f = ROOT.TFile.Open("/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muGunCov.root")

#tree = f.Get("tree")

#hdphiplus = ROOT.TH1D("hdphiplus","",100,-0.01,0.01)
#hdphiminus = ROOT.TH1D("hdphiminus","",100,-0.01,0.01)


#hdthetaplus = ROOT.TH1D("hdthetaplus","",100,-0.1,0.1)
#hdthetaminus = ROOT.TH1D("hdthetaminus","",100,-0.1,0.1)


#tree.Draw("2.*atan(exp(-reco_eta)) - 2.*atan(exp(-gen_eta)) >>hdthetaplus","1./gen_curv>100. && gen_eta>-2.4 && gen_eta<-2.2 && gen_charge>0","goff")

#tree.Draw("2.*atan(exp(-reco_eta)) - 2.*atan(exp(-gen_eta)) >>hdthetaminus","1./gen_curv>100. && gen_eta>-2.4 && gen_eta<-2.2 && gen_charge<0","goff")

#tree.Draw("ROOT::Math::VectorUtil::Phi_mpi_pi(reco_phi-gen_phi)>>hdphiplus","1./gen_curv>100. && gen_eta>-2.4 && gen_eta<-2.2 && gen_charge>0","goff")
#tree.Draw("ROOT::Math::VectorUtil::Phi_mpi_pi(reco_phi-gen_phi)>>hdphiminus","1./gen_curv>100. && gen_eta>-2.4 && gen_eta<-2.2 && gen_charge<0","goff")


#treename = "tree"
#fname = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muonGuntree.root"

#d = ROOT.ROOT.RDataFrame(treename, fname)

#d = d.Filter("1./gen_curv>100. && gen_eta>-2.3 && gen_eta<-2.1 && reco_charge>-2.")
##d = d.Filter("1./gen_curv>120. && gen_eta>-2.4 && gen_eta>0.1 && gen_eta<0.2")
##d = d.Filter("1./gen_curv>120. && gen_eta>-2.4 && gen_eta>0.1 && gen_eta<0.2")
##d = d.Filter("1./gen_curv>5.5 && 1./gen_curv<10. && gen_eta>-2.4 && gen_eta<-2.2 && reco_charge>-2.")
#d = d.Define("dphi","ROOT::Math::VectorUtil::Phi_mpi_pi(reco_phi-gen_phi)")
#d = d.Define("dtheta","2.*atan(exp(-reco_eta)) - 2.*atan(exp(-gen_eta))")

#dplus = d.Filter("gen_charge>0.")
#dminus = d.Filter("gen_charge<0.")

#print(15*["double"])

#print(ROOT.Numba)

#@ROOT.Numba.Declare(15*["double"], "double")
#def symdet(c00,c01,c02,c03,c04,c11,c12,c13,c14,c22,c23,c24,c33,c34,c44):
    #M = np.array([[c00,c01,c02,c03,c04],
                 #[c01,c11,c12,c13,c14],
                 #[c02,c12,c22,c23,c24],
                 #[c03,c13,c23,c33,c34],
                 #[c04,c14,c24,c34,c44]], dtype=np.float64)
    
    #return np.linalg.det(M)


#ROOT.gInterpreter.Declare("""
#double symdet(double c00, double c01, double c02, double c03, double c04, double c11, double c12, double c13, double c14, double c22, double c23, double c24, double c33, double c34, double c44) {
    
    #double v[25] = {c00,c01,c02,c03,c04,  c01,c11,c12,c13,c14,  c02,c12,c22,c23,c24,  c03,c13,c23,c33,c34,  c04,c14,c24,c34,c44};
    
    #ROOT::Math::SMatrix<double,5,5> M(v,25);
    #double det;
    #M.Det(det);
    #//return log(det);
    #return det;
#}
#""")

#ROOT.gInterpreter.AddIncludePath("/usr/include/eigen3/")

#ROOT.gInterpreter.Declare("""
##include <Eigen/Dense>
    
#double symdeteigen(double c00, double c01, double c02, double c03, double c04, double c11, double c12, double c13, double c14, double c22, double c23, double c24, double c33, double c34, double c44) {
    
    #typedef Eigen::Matrix<double,5,5> Matrix5d;
    
    #Matrix5d M;
    #M << c00,c01,c02,c03,c04,  c01,c11,c12,c13,c14,  c02,c12,c22,c23,c24,  c03,c13,c23,c33,c34,  c04,c14,c24,c34,c44;
    
    #Eigen::SelfAdjointEigenSolver<Matrix5d> s;
    
    #//return M.determinant();
    #s.compute(M);
    #return s.eigenvalues()[0];
#}
#""")



ROOT.gInterpreter.Declare("""

typedef ROOT::Math::SMatrix<double,5> SMatrix55;
typedef ROOT::Math::SVector<double,5> SVector5;    

double eigval(const std::tuple<SVector5, SMatrix55> &eig) {
    
    const auto &e = std::get<0>(eig);
    const auto &v = std::get<1>(eig);
    
    double maxval = 0.;
    unsigned int imax = 0;
    for (unsigned int i=0; i<5; ++i) {
        double val = v(0,i)*v(0,i) + v(2,i)*v(2,i);
        if (val > maxval && v(0,i)*v(2,i)>0.) {
            maxval = val;
            imax = i;
        }
    }
    //return log(e(imax));
    return std::abs(v(2,imax));
    
}
""")


treename = "tree"
#fname = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muGunCov_eig.root"
fname = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muGunCov_covrphi.root"

d = ROOT.ROOT.RDataFrame(treename, fname)

#d = d.Filter("1./gen_curv>148. && 1./gen_curv<150.&& reco_charge>-2.")
d = d.Filter("1./gen_curv>5.5 && 1./gen_curv<7. && reco_charge>-2.")
#d = d.Filter("1./gen_curv>148. && 1./gen_curv<150. && gen_eta>-2.4 && gen_eta<-2.3 && reco_charge>-2.")
dhits = d.Filter("tkHits>15")
#d = d.Filter("covrphi.first(0,1)>4e-7")

#d = d.Filter("abs(reco_curv/gen_curv-1.)<0.005")
#d = d.Filter("abs(reco_phi-gen_phi)<1e-5")

#d = d.Define("covrphiarr","*covrphi.Array()")
#d = d.Define("covrphiarr","std::array<double, 3> d{covrphi(0,0),covrphi(1,1),covrphi(2,2)}; return d;")
#d = d.Define("covrphiarr","covrphi(0,0)")
#d = d.Define("covrphiarr","covrphi.fArray")
#d = d.Define("covrphiarr","covrphi.fRep")

#d = d.Filter("abs(1.-tkchi2)<0.1")
#d = d.Filter("1./gen_curv>148 && 1./gen_curv<150. && gen_eta>-2.4 && gen_eta<-2.3 && reco_charge>-2.")
#d = d.Filter("1./reco_curv>148 && 1./reco_curv<150. && reco_eta>-2.4 && reco_eta<-2.3 && reco_charge>-2.")
#d = d.Filter("1./gen_curv>5.5 && 1./gen_curv<7. && abs(gen_eta)<2.4 && reco_charge>-2.")
#d = d.Filter("1./gen_curv>148. && 1./gen_curv<150. && gen_eta>-2.4 && gen_eta<-2.3 && reco_charge>-2. && tkHits>15")

#d = d.Filter("1./gen_curv>5.5 && 1./gen_curv<7. && abs(gen_eta)<2.4 && reco_charge>-2.")
#d = d.Define("eigv","eigval(eig)")
#d = d.Define("eigv","covrphi.first(0,0)")

#cols = ["covrphiarr"]
#arr = d.AsNumpy(cols)


#print(arr)
#print(type(arr))
#print(arr["covrphiarr"].shape)
#print(arr["covrphiarr"].dtype)

#assert()

#d = d.Define("eigvalt2","cov00")
#d = d.Define("eigv","atanh(covrphi.first(0,1)/sqrt(covrphi.first(0,0)*covrphi.first(1,1)))")
#d = d.Define("eigv","reco_charge*reco_curv/gen_curv/gen_charge")
#d = d.Filter("tkHits>15")
#d = d.Define("eigv","tkHits")
##d = d.Define("eigv","covrphi.first(0,0)/reco_curv/reco_curv")
#d = d.Define("eigv","log(covrphi.first(0,0)/reco_curv/reco_curv)")
#d = d.Define("eigv","covrphi.first(0,0)*gen_curv*gen_curv/reco_curv/reco_curv")
#d = d.Define("eigv","covrphi.first(0,0)*gen_curv*gen_curv/reco_curv/reco_curv
#d = d.Filter("tkHits>15")
#d = d.Define("eigv","(1-1./3./3.)*cov22 + (1./3./3.)*cov00 + 2.*sqrt(1-1./3./3.)*(1./3.)*cov02")
#d = d.Define("eigv","cov00-2.342*cov02")

#d = d.Filter("1./gen_curv>5.5 && 1./gen_curv<7. && gen_eta>-0.6 && gen_eta<-0.5 && reco_charge>-2.")
#d = d.Filter("1./gen_curv>6. && 1./gen_curv<7. && gen_eta>-2.4 && gen_eta<-2.3 && reco_charge>-2.")
#d = d.Filter("1./reco_curv>5.5 && 1./reco_curv<150. && reco_eta>-2.4 && reco_eta<-2.3 && reco_charge>-2.")
#d = d.Filter("1./reco_curv>5.5 && 1./reco_curv<150. && reco_eta>-2.4 && reco_eta<-2.3 && reco_charge>-2.")
#d = d.Filter("1./gen_curv>120. && gen_eta>-2.4 && gen_eta>0.1 && gen_eta<0.2")
#d = d.Filter("1./gen_curv>120. && gen_eta>-2.4 && gen_eta>0.1 && gen_eta<0.2")
#d = d.Filter("genPt>100.")
#d = d.Define("dphi","ROOT::Math::VectorUtil::Phi_mpi_pi(trackPhi-genPhi)")
#d = d.Define("dtheta","2.*atan(exp(-trackEta)) - 2.*atan(exp(-genEta))")

#dplus = d.Filter("genCharge>0.")
#dminus = d.Filter("genCharge<0.")
#d = d.Define("det", "symdeteigen(cov00,cov01,cov02,cov03,cov04,cov11,cov12,cov13,cov14,cov22,cov23,cov24,cov33,cov34,cov44)")
#d = d.Define("det", "std::get<0>(eig)(1)")

hists = []
hists2d = []

heta = d.Histo1D(("heta","",100,-2.4,2.4),"gen_eta")
hetahits = dhits.Histo1D(("hetahits","",100,-2.4,2.4),"gen_eta")

#hists.append(heta)
#hists.append(hetahits)

#print("heta", heta)

hetaratio = hetahits.Clone("hetaratio")
hetaratio.Divide(heta.GetPtr())
hists.append(hetaratio)

#heig = d.Histo1D(("heig","",100,-15.,-5.),"eigv")
#heig = d.Histo1D(("heig","",100,-1.,2.5),"eigv")
#heig = d.Histo1D(("heig","",100,0.,5e-6),"eigv")
#heig = d.Histo1D(("heig","",100,0.,5e-7),"eigv")
#heig = d.Histo1D(("heig","",100,-30.,-10.),"eigv")
#heig = d.Histo1D(("heig","",100,0.,0.002),"eigv")
#heig = d.Histo1D(("heig","",100,0.,1e-4),"eigv")
#heig = d.Histo1D(("heig","",100,0.,30e3),"eigv")
#heig = d.Histo1D(("heig","",100,0.,0.1),"eigv")
#heig = d.Histo1D(("heig","",100,-6.,-2.),"eigv")
#heigalt2 = d.Histo1D(("heigalt2","",100,0.,5e-7),"eigvalt2")

#heig = d.Histo1D(("heig","",100,0.,2e-6),"eigv")
#heig = d.Histo1D(("heig","",100,0.,2.),"eigv")
#heig = d.Histo1D(("heig","",100,0.,20.),"eigv")


#cols=["eigv"]

#arr = d.AsNumpy(cols)

#mean = d.Mean("eigv")


##median = d.Median("eigv")

#median = np.median(arr["eigv"])

#print("mean", mean.GetValue())
#print("median", median)

#c = ROOT.TCanvas()
#heig.Draw("HIST")
##heigalt.SetLineColor(ROOT.kRed)
##heigalt.Draw("HISTSAME")
#c.Update()

##c2 = ROOT.TCanvas()
##heigalt2.Draw("HIST")
##c2.Update()

#input("wait")
#assert(0)

#hists.append(heig)

#for i in range(5):
    #d = d.Define(f"w_{i}",f"std::get<1>(eig)({i},1)")
    #hw = d.Histo1D((f"hw_{i}","",100,-1.,1.),f"w_{i}")
    ##hists.append(hw)
    
    #d = d.Define(f"e_{i}",f"log(std::get<0>(eig)({i}))")
    #he = d.Histo1D((f"he_{i}","",100, -25.,-5.),f"e_{i}")
    ##hists.append(he)

    #d = d.Define(f"wqp_{i}",f"std::get<1>(eig)(0,{i})")
    #hwqp = d.Histo1D((f"hwqp_{i}","",100,-1.,1.),f"wqp_{i}")
    ##hists.append(hwqp)

    #d = d.Define(f"wphi_{i}",f"std::get<1>(eig)(2,{i})")
    #hwphi = d.Histo1D((f"hwphi_{i}","",100,-1.,1.),f"wphi_{i}")
    ##hists.append(hwphi)
    
    #hwqpphi = d.Histo2D((f"hwqpphi_{i}","",100,-1.,1.,100,-1.,1.),f"wphi_{i}",f"wqp_{i}")
    ##hists2d.append(hwqpphi)


#d = d.Define("sigma","2.*log(tkptErr) + 2.*log(reco_curv)")



#detmean = d.Mean("det")
#sigmamean = d.Mean("sigma")

#hdet = d.Histo1D(("hdet","",100,0.,1e-38), "det")
#hdet = d.Histo1D(("hdet","",100,0.,1e-9), "det")

#hsigma = d.Histo1D(("hsigma","",100,-20.,-10.), "sigma")


#print("detmean", detmean.GetValue())
#print("sigmamean", sigmamean.GetValue())

#gc = []

#c=ROOT.TCanvas()
#gc.append(c)
#hdet.Draw()
#c.Update()

#c2=ROOT.TCanvas()
#hsigma.Draw()
#c2.Update()

gc = []
for hist in hists:
    c = ROOT.TCanvas()
    gc.append(c)
    hist.Draw()
    c.Update()

for hist in hists2d:
    c = ROOT.TCanvas()
    gc.append(c)
    hist.Draw("COLZ")
    c.Update()




input("wait")

