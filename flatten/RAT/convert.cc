#include "RAT/DSReader.hh"
#include "RAT/DS/RunStore.hh"
//#include "RAT/DS/MC.hh"
//#include "RAT/DS/Root.hh"
//#include "RAT/DS/PMT.hh"
//#include "RAT/DS/Run.hh"
//#include "RAT/DS/EV.hh"

#include "TChain.h"
#include "TFile.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int usage(const std::string& pName)
{
  std::cout << pName << " -  extracts MC PMT information from the JSNS2-RAT-PAC root file\n";
  std::cout << "\tUsage: " << pName << " <INPUT_RAT_FILE.root>\n";
  return 1;
}

int main(int argc, char* argv[])
{
  if ( argc != 3 ) return usage(argv[0]);
  for ( int i = 1; i < argc; ++i ) {
    if ( std::strcmp(argv[i], "-h") == 0 or std::strcmp(argv[i], "--help") == 0 ) {
      return usage(argv[0]);
    }
  }
  const std::string finName = argv[1];
  const std::string foutName = argv[2];

  // Load Run-level tree
  TChain runTree("runT");
  runTree.Add(finName.c_str());
  RAT::DS::RunStore::SetReadTree(&runTree);

  cout << "Loading Run..." << endl;
  auto run = RAT::DS::RunStore::Get()->GetRun(1);
  if ( run == nullptr ) {
    cout << "Cound not find run. Exiting..." << endl;
    return 2;
  }
  cout << "...OK" << endl;

  // Prepare output file
  TFile* fout = TFile::Open(foutName.c_str(), "RECREATE");

  // Load the PMT information
  auto pmtInfo = run->GetPMTInfo();
  const int nPMT = pmtInfo->GetPMTCount();
  cout << "Number of PMTs=" << nPMT << endl;
  TTree* geomT = new TTree("geomT", "PMT geometry");
  std::vector<double> b_pmt_x(nPMT), b_pmt_y(nPMT), b_pmt_z(nPMT);
  std::vector<double> b_pmt_px(nPMT), b_pmt_py(nPMT), b_pmt_pz(nPMT);
  std::vector<int> b_pmt_type(nPMT);

  geomT->Branch("pmt_x", &b_pmt_x);
  geomT->Branch("pmt_y", &b_pmt_y);
  geomT->Branch("pmt_z", &b_pmt_z);

  geomT->Branch("pmt_px", &b_pmt_px);
  geomT->Branch("pmt_py", &b_pmt_py);
  geomT->Branch("pmt_pz", &b_pmt_pz);

  geomT->Branch("pmt_type", &b_pmt_type);

  for ( int i = 0; i < nPMT; ++i ) {
    const auto pmtPos = pmtInfo->GetPosition(i);
    const auto pmtDir = pmtInfo->GetDirection(i);

    b_pmt_x[i] = pmtPos.X();
    b_pmt_y[i] = pmtPos.Y();
    b_pmt_z[i] = pmtPos.Z();

    b_pmt_px[i] = pmtDir.X();
    b_pmt_py[i] = pmtDir.Y();
    b_pmt_pz[i] = pmtDir.Z();

    b_pmt_type[i] = pmtInfo->GetType(i);
  }
  
  geomT->Fill();

  TTree* eventT = new TTree("eventT", "Event tree");
  double b_nPE = 0;
  double b_vtx_x = 0, b_vtx_y = 0, b_vtx_z = 0, b_vtx_t = 0;
  double b_vtx_px = 0, b_vtx_py = 0, b_vtx_pz = 0, b_vtx_ke = 0;
  std::vector<double> b_pmt_q;
  std::vector<double> b_pmt_t;
  b_pmt_q.resize(nPMT);
  b_pmt_t.resize(nPMT);

  eventT->Branch("nPE", &b_nPE);
  eventT->Branch("vtx_x", &b_vtx_x);
  eventT->Branch("vtx_y", &b_vtx_y);
  eventT->Branch("vtx_z", &b_vtx_z);
  eventT->Branch("vtx_t", &b_vtx_t);

  eventT->Branch("vtx_px", &b_vtx_px);
  eventT->Branch("vtx_py", &b_vtx_py);
  eventT->Branch("vtx_pz", &b_vtx_pz);
  eventT->Branch("vtx_ke", &b_vtx_ke);

  eventT->Branch("pmt_q", &b_pmt_q);
  eventT->Branch("pmt_t", &b_pmt_t);

  // Load the dataset
  auto ds = std::unique_ptr<RAT::DSReader>(new RAT::DSReader(finName.c_str()));
  const int nEvents = ds->GetTotal();
  cout << "Number of Events=" << nEvents << endl;

  for ( int iEvent = 0; iEvent < nEvents; ++iEvent ) {
    cout << "Processing event " << (iEvent+1) << "/" << nEvents << "\r";
    const auto event = ds->NextEvent();
    const auto mc = event->GetMC();
    if ( mc == nullptr ) {
      cout << "Null pointer to get MC information. Stop processing..." << endl;
      break;
    }

    if ( mc->GetMCParticleCount() == 0 ) {
      cout << "No MC particle found in this event. Skipping..." << endl;
      continue;
    }

    // Get the vertex information
    const auto vertexPos = mc->GetMCParticle(0)->GetPosition();
    b_vtx_x = vertexPos.X();
    b_vtx_y = vertexPos.Y();
    b_vtx_z = vertexPos.Z();
    b_vtx_t = mc->GetMCParticle(0)->GetTime();

    const auto vertexDir = mc->GetMCParticle(0)->GetMomentum();
    b_vtx_px = vertexDir.X();
    b_vtx_py = vertexDir.Y();
    b_vtx_pz = vertexDir.Z();
    b_vtx_ke = mc->GetMCParticle(0)->GetKE();

    // Write out event level information
    b_nPE = mc->GetNumPE();
    for ( int i = 0, n = mc->GetMCPMTCount(); i < n; ++i ) {
      b_pmt_t[i] = b_pmt_q[i] = 0;
    }

    // Get MCPMT information
    std::map<int, float> pmtId2Charge;
    std::map<int, float> pmtId2Time;
    for ( int i = 0, n = mc->GetMCPMTCount(); i < n; ++i ) {
      const auto pmt = mc->GetMCPMT(i);
      const int pmtId = pmt->GetID();
      const float pmtCharge = pmt->GetCharge();
      // Calculate average time of the PMT, weighted by the charge (nPE)
      double sumW = 0, sumWT = 0;
      for ( int j = 0, m = pmt->GetMCPhotonCount(); j < m; ++j ) {
        const auto photon = pmt->GetMCPhoton(j);
        const double w = photon->GetCharge();
        sumWT += w*photon->GetHitTime();
        sumW += w;
      }
      const double pmtTime = (sumW == 0 ? 0 : sumWT/sumW);
      
      pmtId2Charge[pmtId] = pmtCharge;
      pmtId2Time[pmtId] = static_cast<float>(pmtTime);
    }

    // Fill up tree
    for ( int i = 0; i < nPMT; ++i ) {
      auto itrQ = pmtId2Charge.find(i);
      const float pmtCharge = (itrQ == pmtId2Charge.end() ? 0 : itrQ->second);
      b_pmt_q[i] = pmtCharge;

      auto itrT = pmtId2Time.find(i);
      const float pmtTime = (itrT == pmtId2Time.end() ? 0 : itrT->second);
      b_pmt_t[i] = pmtCharge;
    }

    eventT->Fill();
  }
  cout << endl;

  geomT->Write();
  eventT->Write();
  fout->Close();

  return 0;
}
