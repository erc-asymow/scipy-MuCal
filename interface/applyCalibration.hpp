#ifndef APPLYCALIBRATION_H
#define APPLYCALIBRATION_H


#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "TH1D.h"
#include "TH2D.h"
#include "TString.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "Math/Vector4D.h"
#include "module.hpp"

using namespace ROOT::VecOps;
using RNode = ROOT::RDF::RNode;
using rvec_f = const RVec<float> &;
using rvec_i = const RVec<int> &;


class applyCalibration : public Module {

    private:

    std::vector<ROOT::RDF::RResultPtr<TH1D>> _h1List;
    std::vector<ROOT::RDF::RResultPtr<TH2D>> _h2List;
    std::vector<ROOT::RDF::RResultPtr<TH3D>> _h3List;
    
    // groups of histos
    std::vector<ROOT::RDF::RResultPtr<std::vector<TH1D>>> _h1Group;
    std::vector<ROOT::RDF::RResultPtr<std::vector<TH2D>>> _h2Group;
    std::vector<ROOT::RDF::RResultPtr<std::vector<TH3D>>> _h3Group;
    
    TH2F* _hbFieldMap;
    TH1D* _hA;
    TH1D* _he;
    TH2D* _hM;

    bool _isData;
    bool _fullCalib;

    public:
    
    applyCalibration(TH2F *hbFieldMap, TH1D *hA, TH1D *he, TH2D *hM, bool isData = false,bool fullCalib = false){

        _hbFieldMap = hbFieldMap;
        _hA = hA;
        _he = he;
        _hM = hM;
        _isData = isData;
        _fullCalib = fullCalib;
    };

    ~applyCalibration() {};
    RNode run(RNode) override;
    std::vector<ROOT::RDF::RResultPtr<TH1D>> getTH1() override;
    std::vector<ROOT::RDF::RResultPtr<TH2D>> getTH2() override;
    std::vector<ROOT::RDF::RResultPtr<TH3D>> getTH3() override;

    std::vector<ROOT::RDF::RResultPtr<std::vector<TH1D>>> getGroupTH1() override;
    std::vector<ROOT::RDF::RResultPtr<std::vector<TH2D>>> getGroupTH2() override;
    std::vector<ROOT::RDF::RResultPtr<std::vector<TH3D>>> getGroupTH3() override;

    float getCorrectedPt(float pt,float eta,float phi, int charge);
    float getCorrectedPtMag(float pt,float eta,float phi);

    void reset() override;
    
};

#endif
