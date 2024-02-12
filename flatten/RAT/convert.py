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

import uproot
import numpy as np
import h5py
#from tqdm import tqdm
tqdm = lambda x: x

fin = uproot.open(fin)
geomT = fin["geomT"]
eventT = fin["eventT"]

kwargs = {'dtype':'f4', 'compression':'lzf'}
with h5py.File(fout, 'w', libver='latest') as fout:
    gGeom = fout.create_group('geom')
    gGeom.create_dataset('pmt_x', data=geomT["pmt_x"].array()[0], **kwargs)
    gGeom.create_dataset('pmt_y', data=geomT["pmt_y"].array()[0], **kwargs)
    gGeom.create_dataset('pmt_z', data=geomT["pmt_z"].array()[0], **kwargs)

    gGeom.create_dataset('pmt_px', data=geomT["pmt_px"].array()[0], **kwargs)
    gGeom.create_dataset('pmt_py', data=geomT["pmt_py"].array()[0], **kwargs)
    gGeom.create_dataset('pmt_pz', data=geomT["pmt_pz"].array()[0], **kwargs)

    gEvent = fout.create_group('event')
    gEvent.create_dataset('vtx_x', data=eventT['vtx_x'].array(), **kwargs)
    gEvent.create_dataset('vtx_y', data=eventT['vtx_y'].array(), **kwargs)
    gEvent.create_dataset('vtx_z', data=eventT['vtx_z'].array(), **kwargs)
    gEvent.create_dataset('vtx_t', data=eventT['vtx_t'].array(), **kwargs)

    gEvent.create_dataset('vtx_px', data=eventT['vtx_px'].array(), **kwargs)
    gEvent.create_dataset('vtx_py', data=eventT['vtx_py'].array(), **kwargs)
    gEvent.create_dataset('vtx_pz', data=eventT['vtx_pz'].array(), **kwargs)
    gEvent.create_dataset('vtx_ke', data=eventT['vtx_ke'].array(), **kwargs)

    gEvent.create_dataset('pmt_q', data=eventT['pmt_q'].array(), **kwargs)
    gEvent.create_dataset('pmt_t', data=eventT['pmt_t'].array(), **kwargs)
