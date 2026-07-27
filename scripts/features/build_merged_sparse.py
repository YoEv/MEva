#!/usr/bin/env python3
"""Build merged sparse SAE (CSR) from per-benchmark dense shards.

Usage: build_merged_sparse.py <small|large> [data_root]
data_root defaults to $MEVA_DATA_ROOT or the current directory.
"""
import sys, os, glob
import numpy as np, pandas as pd, torch
model=sys.argv[1]
R=sys.argv[2] if len(sys.argv)>2 else os.environ.get('MEVA_DATA_ROOT','.')
sae_root=f'{R}/features/sae' if model=='small' else f'{R}/features/sae_large'
ORDER=['musicpref','aime','songeval','music_arena','musiceval']
out=f'{R}/features/sae_sparse_{model}/all_5/clean'; os.makedirs(out,exist_ok=True)
def to_csr(dense):  # [T,dim] → idx,val,off
    T=dense.shape[0]; nz=[np.nonzero(dense[t])[0] for t in range(T)]
    off=np.zeros(T+1,dtype=np.int64)
    for t in range(T): off[t+1]=off[t]+len(nz[t])
    idx=np.concatenate(nz).astype(np.int32) if T else np.zeros(0,np.int32)
    val=np.concatenate([dense[t][nz[t]] for t in range(T)]).astype(np.float32) if T else np.zeros(0,np.float16)
    return idx,val,off
for split in ['train','val','test']:
    sd=f'{out}/{split}'; os.makedirs(sd,exist_ok=True)
    for f in glob.glob(f'{sd}/*.npz'): os.remove(f)
    lengths=[]; gi=0
    for ds in ORDER:
        m=torch.load(f'{sae_root}/{ds}/clean/sae_features_{split}_meta.pt',weights_only=False)
        L=[int(x) for x in m['lengths']]; mode=m.get('mode')
        if mode=='sharded':
            sdir=m.get('shard_dir',f'{sae_root}/{ds}/clean/{split}')
            for i in range(len(L)):
                d=np.load(f'{sdir}/{i:06d}.npy'); idx,val,offs=to_csr(d)
                np.savez(f'{sd}/{gi:06d}.npz', idx=idx,val=val,off=offs); gi+=1; lengths.append(L[i])
        else:
            # monolithic: musicdiscovery 写的是 raw float32 memmap(无 .npy header), 按 meta shape 读
            HD=int(m.get('hidden_dim', 4096 if model=='small' else 8192)); tot=int(np.sum(L))
            npy=np.memmap(f'{sae_root}/{ds}/clean/sae_features_{split}.npy',dtype='float32',mode='r',shape=(tot,HD))
            cum=np.concatenate([[0],np.cumsum(L)])
            for i in range(len(L)):
                d=np.asarray(npy[cum[i]:cum[i+1]]); idx,val,offs=to_csr(d)
                np.savez(f'{sd}/{gi:06d}.npz', idx=idx,val=val,off=offs); gi+=1; lengths.append(L[i])
    hd=4096 if model=='small' else 8192
    torch.save({'lengths':lengths,'hidden_dim':hd,'mode':'sparse','shard_dir':sd,'backend':'musicdiscovery'}, f'{out}/sae_features_{split}_meta.pt')
    print(f'{model}/{split}: shards={gi} sparse dir={sd}')
