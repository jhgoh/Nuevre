#!/usr/bin/env python
import sys, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', action='store', type=str, required=True, help='Configuration for training')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Output directory')
## Parameters for the training
parser.add_argument('--epoch', action='store', type=int, required=True, help='Epoch')
parser.add_argument('--batch', action='store', type=int, help='Batch size')
parser.add_argument('--lr', action='store', type=float, help='Learning rate')
parser.add_argument('--seed', action='store', type=int, help='Random seed')
parser.add_argument('--shuffle', action='store', type=bool, help='Turn on shuffle')
parser.add_argument('--gpu-deterministic', action='store_true', help='Set fixed GPU random state in spite of performance')
## Parameters for the computing resource
parser.add_argument('--nthread', action='store', type=int, default=os.cpu_count(), help='Number of threads for main')
parser.add_argument('--nloader', action='store', type=int, default=min(8, os.cpu_count()), help='Number of dataLoaders')
parser.add_argument('--device', action='store', type=int, default=-1, help='Device number (-1 for CPU)')
## Other parameters
parser.add_argument('-q', '--no-progress-bar', action='store_true', help='Hide progress bar')
args = parser.parse_args()

## Load configuration files
import yaml
config_datasets = yaml.load(open("data/datasets_JSNS2.yaml").read(), Loader=yaml.FullLoader)
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

## Override options
def overrideConfig(argName, configName, astype=None):
    a = getattr(args, argName)
    expr = 'config' + "".join(["['%s']" % x for x in configName.split('/')])
    val = eval(expr) if a is None else a
    if astype is None: astype = type(a)
    if astype is type(None): astype = type(b)

    setattr(args, argName, astype(val))
    exec(expr + '= astype(val)') ## A workaround to assign value to an element of dict of dicts..

overrideConfig('epoch', 'training/epoch', int)
overrideConfig('batch', 'training/batch', int)
overrideConfig('lr', 'training/learningRate', float)
overrideConfig('seed', 'random/seed', int)
overrideConfig('shuffle', 'random/shuffle', bool)
overrideConfig('gpu_deterministic', 'random/gpu-deterministic', bool)

if not os.path.exists(args.output):
    os.makedirs(args.output)

with open(os.path.join(args.output, "config.yaml"), 'w') as fout:
    yaml.dump(config, fout)


## Setup to run pytorch
import torch

torch.set_num_threads(args.nthread)
if torch.cuda.is_available() and args.device >= 0:
    torch.cuda.set_device(args.device)
    if args.gpu_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = config['random']['seed']
if seed > 0: torch.manual_seed(seed)

## Load user modules
sys.path.append("python")

## Define dataset
from datasets.NeuEvDataset import NeuEvDataset as Dataset
from re import match
dset = Dataset()
pattern = config['dataset']['name']
if not (pattern.startswith('/') and pattern.startswith('/')):
    if '*' in pattern: pattern = pattern.replace('*', '.*')
else:
    pattern = pattern[1:-1]
for d in config_datasets['datasets']:
    #if d['name'] != dname: continue
    if not match(pattern, d['name']): continue
    paths = d['paths']
    for p in paths: dset.addSample(p)
dset.initialize()
print(dset)

##### Define dataset instance #####
lengths = [int(x*len(dset)) for x in config['dataset']['split']]
lengths[-1] = len(dset) - sum(lengths[:-1])
if len(lengths) == 2: lengths.append(0)
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)

from torch.utils.data import DataLoader
kwargs = {'num_workers':args.nloader, 'pin_memory':True}
trnLoader = DataLoader(trnDset, batch_size=args.batch, shuffle=args.shuffle, **kwargs)
valLoader = DataLoader(valDset, batch_size=args.batch, shuffle=False, **kwargs)

## Define the model
from models.BaseModels import *
model = Classica(pmt_N=dset.shape[0])

device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    device = 'cuda:%d' % args.device
    model = model.to(device)
print("Runing on", device, 'cuda_is_available=', torch.cuda.is_available())

##### Define optimizer instance #####
#optm = torch.optim.Adam(model.parameters(), lr=args.lr)
optm = torch.optim.AdamW(model.parameters(), lr=args.lr)

##### Start training #####
with open(args.output+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.write(str(model))
    fout.close()

import csv
from sklearn.metrics import accuracy_score
if not args.no_progress_bar:
    from tqdm import tqdm
else:
    tqdm = lambda x, **kwargs: x

## Note: Get the pmt position information in advance
## This is not changing.
pmt_pos = dset.pmt_pos.to(device)
pmt_dir = dset.pmt_dir.to(device)

lossF = torch.nn.MSELoss()

bestState, bestLoss = {}, 1e9
train = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
nEpoch = args.epoch
for epoch in range(nEpoch):
    model.train()
    trn_loss, trn_acc = 0., 0.
    nProcessed = 0
    optm.zero_grad()

    for i, (pmt_q, pmt_t, vtx_pos) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
        #pmt_t = pmt_t.to(device)
        #pmt_q = pmt_q.to(device)
        pmtSumQ = pmt_q.sum(dim=1)
        pmt_qFrac = (pmt_q/pmtSumQ.view(-1, 1)).to(device)
        vtx_pos = vtx_pos.to(device)

        pred = model(pmt_q, pmt_t, pmt_pos, pmt_dir, vtx_pos)
        loss = lossF(pred, pmt_qFrac)
        loss.backward()
        optm.step()
        optm.zero_grad()
#        pred = pred.detach().cpu()

        ibatch = len(pmt_q)
        nProcessed += ibatch
        trn_loss += float(loss)*ibatch
        trn_acc += (float(loss)**0.5)*ibatch
    trn_loss /= nProcessed
    trn_acc /= nProcessed
    print(f"Training: loss={trn_loss} acc={trn_acc}")

    model.eval()
    val_loss, val_acc = 0., 0.
    nProcessed = 0
    for i, (pmtQ, pmtT, vtxPos) in enumerate(tqdm(valLoader)):
        #pmt_t = pmt_t.to(device)
        #pmt_q = pmt_q.to(device)
        pmtSumQ = pmt_q.sum(dim=1)
        pmt_qFrac = (pmt_q/pmtSumQ.view(-1, 1)).to(device)
        vtx_pos = vtx_pos.to(device)

        pred = model(pmt_q, pmt_t, pmt_pos, pmt_dir, vtx_pos)
        loss = lossF(pred, pmt_qFrac)
#        pred = pred.detach().cpu()

        ibatch = len(pmt_q)
        nProcessed += ibatch
        val_loss += float(loss)*ibatch
        val_acc += (float(loss)**0.5)*ibatch
    val_loss /= nProcessed
    val_acc /= nProcessed
    print(f"Validation: loss={val_loss} acc={val_acc}")

    if bestLoss > val_loss:
        bestState = {k: v.cpu() for k, v in model.state_dict().items()}
        bestLoss = val_loss
        torch.save(bestState, os.path.join(args.output, 'weight.pth'))

    train['loss'].append(trn_loss)
    train['acc'].append(trn_acc)
    train['val_loss'].append(val_loss)
    train['val_acc'].append(val_acc)

    with open(os.path.join(args.output, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = train.keys()
        writer.writerow(keys)
        for row in zip(*[train[key] for key in keys]):
            writer.writerow(row)
