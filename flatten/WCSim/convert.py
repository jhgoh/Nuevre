#!/usr/bin/env python
import sys, os
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} INPUT.root OUTPUT.root")
    exit(0)

fin = sys.argv[1]
fout = sys.argv[2]

if not fin.endswith(".root"):
    print(f"!!! Input file has to be .root")
    exit(1)

if not os.path.exists(fin):
    print(f"!!! Cannot find input file {fin}")
    exit(1)

if os.path.exists(fout):
    print(f"!!! Output file {fout} already exists. skip.")
    exit(1)

import ROOT
import numpy as np
import h5py
#from tqdm import tqdm
tqdm = lambda x: x

ROOT.gSystem.Load("lib/libWCSimRoot.so")
fin = ROOT.TFile(fin)

## Load the event tree
eventT = fin.Get("wcsimT")
event = ROOT.WCSimRootEvent()
eventT.SetBranchAddress("wcsimrootevent", event)
eventT.GetBranch("wcsimrootevent").SetAutoDelete(1)
eventT.GetEntry(0)
nEvents = eventT.GetEntries()

## Load the geometry
geomT = fin.Get("wcsimGeoT")
geom = ROOT.WCSimRootGeom()
geomT.SetBranchAddress("wcsimrootgeom", geom)
geomT.GetEntry(0)
nODPMTs = geom.GetODWCNumPMT()
nPMTs = geom.GetWCNumPMT()

print("--------------------")
print(f" nEvents = {nEvents}")
print(f" nPMTs   = {nPMTs}")
print(f" nODPMTs = {nODPMTs}")
print("--------------------")

out_pmt_x = np.zeros(nPMTs)
out_pmt_y = np.zeros(nPMTs)
out_pmt_z = np.zeros(nPMTs)
for iPMT in range(nPMTs):
    pmt = geom.GetPMT(iPMT)
    pmt_x, pmt_y, pmt_z = [pmt.GetPosition(i) for i in range(3)]
    out_pmt_x[iPMT] = pmt_x
    out_pmt_y[iPMT] = pmt_y
    out_pmt_z[iPMT] = pmt_z

print("@@@ Start analysing data")
out_vtx_x = np.zeros(nEvents)
out_vtx_y = np.zeros(nEvents)
out_vtx_z = np.zeros(nEvents)
out_pmt_q = np.zeros((nEvents, nPMTs))
out_pmt_t = np.zeros((nEvents, nPMTs))
for iEvent in tqdm(range(nEvents)):
    eventT.GetEvent(iEvent)
    trigger = event.GetTrigger(0)

    nVtxs = trigger.GetNvtxs()
    if nVtxs < 1: continue
    vtx_x, vtx_y, vtx_z = [trigger.GetVtx(i) for i in range(3)]
    out_vtx_x[iEvent] = vtx_x
    out_vtx_y[iEvent] = vtx_y
    out_vtx_z[iEvent] = vtx_z

    nHitsC = trigger.GetNcherenkovdigihits()
    for iHit in range(nHitsC):
        hit = trigger.GetCherenkovDigiHits().At(iHit)
        iPMT = hit.GetTubeId()-1
        q = hit.GetQ()
        t = hit.GetT()
        out_pmt_q[iEvent, iPMT] = q
        out_pmt_t[iEvent, iPMT] = t

kwargs = {'dtype':'f4', 'compression':'lzf'}
with h5py.File(fout, 'w', libver='latest') as fout:
    gGeom = fout.create_group('geom')
    gGeom.create_dataset('pmt_x', data=out_pmt_x, **kwargs)
    gGeom.create_dataset('pmt_y', data=out_pmt_y, **kwargs)
    gGeom.create_dataset('pmt_z', data=out_pmt_z, **kwargs)

    gEvent = fout.create_group('event')
    gEvent.create_dataset('vtx_x', data=out_vtx_x, **kwargs)
    gEvent.create_dataset('vtx_y', data=out_vtx_y, **kwargs)
    gEvent.create_dataset('vtx_z', data=out_vtx_z, **kwargs)

    gEvent.create_dataset('pmt_q', data=out_pmt_q, **kwargs)
    gEvent.create_dataset('pmt_t', data=out_pmt_t, **kwargs)
