#include <ROOT/RDataFrame.hxx>
#include "TFileInfo.h"
#include "TChain.h"


void fixgrads() {
//no multithreading here to preserve entry numbers

  const std::string treename = "gradtree";

//   const std::string filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v3/200817_213237/0000/*.root";
  
  const std::string filename = "/data/bendavid/muoncaldatalarge/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v4/200820_002243/0000/*.root";

//   const std::string filename = "/eos/cms/store/cmst3/group/wmass/bendavid/muoncal/Muplusandminus_Pt3to150-gun/MuonGunGlobalCor_v3/200817_213237/0000/*.root";

  const std::string outname = "fixedGrads.root";

  // TChain* chain = new TChain(treename.c_str());

  // chain->Add(filename);
  // chain.AddFileInfoList(TFileInfo::CreateListMatching(filename));


  ROOT::RDataFrame d(treename, filename);
  // ROOT::RDataFrame d(chain);

  auto d2 = d.Define("idx","rdfentry_%61068");

  auto d3 = d2.Snapshot(treename, outname);

}
