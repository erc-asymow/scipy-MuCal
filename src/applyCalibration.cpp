#include "interface/applyCalibration.hpp"


float applyCalibration::getCorrectedPt(float pt,float eta1,float phi1,int charge) {

  float curvature = 1.0/getCorrectedPtMag(pt,eta1,phi1);

  int bin1D = _he->FindBin(eta1);
  int bin2D = _hM->FindBin(eta1,phi1);
  
  float a1 = _hA->GetBinContent(1);
  //float a1 =1.;
  float a2 = _hA->GetBinContent(2);

  //std::cout<< a1 << " " << a2 << std::endl;

  //std::cout << " etabin " << bin1D << std::endl;
  //std::cout << " etabin2D " << bin2D << std::endl;

  float b = _hM->GetBinContent(bin2D);

  //std::cout<< b << std::endl;
  
  float e = _he->GetBinContent(bin1D);
  float st = sin(2*atan(exp(-eta1))); 

  float magnetic = a1+a2*eta1*eta1;
  
  float material= -e*st*curvature;

  float alignment=charge*b;
  curvature = (magnetic+material)*curvature+alignment;

  //std::cout<< pt << " pt " << eta1 << " eta " << phi1 << " phi " << charge << " charge " << std::endl;
  //std::cout << magnetic << " magnetic " << material << " material" << alignment << " alignment" << std::endl; 
 
  return 1.0/curvature;
}

float applyCalibration::getCorrectedPtMag(float pt,float eta,float phi) {

  
  float magneticFactor=1.0;
  if (_isData)
    magneticFactor = magneticFactor*_hbFieldMap->GetBinContent(_hbFieldMap->GetBin(
                   _hbFieldMap->GetXaxis()->FindBin(phi),
                   _hbFieldMap->GetYaxis()->FindBin(eta)));
  float curvature = (magneticFactor)/pt;

  //std::cout << "after " << 1.0/curvature << std::endl;
  
  return 1.0/curvature;

}

RNode applyCalibration::run(RNode d){

    
  auto corrMass = [](ROOT::Math::PtEtaPhiMVector corrp1,  ROOT::Math::PtEtaPhiMVector corrp2) ->float{

    return (corrp1+corrp2).M(); 

  };

  auto lambda = [this](float pt,float eta1,float phi1,int charge){


    return getCorrectedPt(pt,eta1,phi1,charge);

  };

  auto lambdaMag = [this](float pt,float eta1,float phi1){

    return getCorrectedPtMag(pt,eta1,phi1);

  };

  if(!(_fullCalib)){
  std::cout<< "calibmap" <<std::endl;
	auto d1 = d.Define("corrpt1", lambdaMag, {"pt1", "eta1", "phi1"})
            .Define("corrpt2", lambdaMag, {"pt2", "eta2", "phi2"})
            .Define("v1corr", "ROOT::Math::PtEtaPhiMVector(corrpt1,eta1,phi1,0.105658)")
            .Define("v2corr", "ROOT::Math::PtEtaPhiMVector(corrpt2,eta2,phi2,0.105658)")
            .Define("corrMass", corrMass, {"v1corr","v2corr"});
  return d1;
}
  else{
  std::cout<< "full calib" <<std::endl;
  auto d1 = d.Define("charge1", "1").Define("charge2", "-1")
            .Define("corrpt1", lambda, {"pt1", "eta1", "phi1", "charge1"})
            .Define("corrpt2", lambda, {"pt2", "eta2", "phi2", "charge2"})
            .Define("v1corr", "ROOT::Math::PtEtaPhiMVector(corrpt1,eta1,phi1,0.105658)")
            .Define("v2corr", "ROOT::Math::PtEtaPhiMVector(corrpt2,eta2,phi2,0.105658)")
            .Define("corrMass", corrMass, {"v1corr","v2corr"});

  return d1;
  }


}

std::vector<ROOT::RDF::RResultPtr<TH1D>> applyCalibration::getTH1(){ 
  return _h1List;
}
std::vector<ROOT::RDF::RResultPtr<TH2D>> applyCalibration::getTH2(){ 
  return _h2List;
}
std::vector<ROOT::RDF::RResultPtr<TH3D>> applyCalibration::getTH3(){ 
  return _h3List;
}

std::vector<ROOT::RDF::RResultPtr<std::vector<TH1D>>> applyCalibration::getGroupTH1(){ 
  return _h1Group;
}
std::vector<ROOT::RDF::RResultPtr<std::vector<TH2D>>> applyCalibration::getGroupTH2(){ 
  return _h2Group;
}
std::vector<ROOT::RDF::RResultPtr<std::vector<TH3D>>> applyCalibration::getGroupTH3(){ 
  return _h3Group;
}

void applyCalibration::reset(){
    
    _h1List.clear();
    _h2List.clear();
    _h3List.clear();

    _h1Group.clear();
    _h2Group.clear();
    _h3Group.clear();

}

