#!/usr/bin/env python3
"""从稀疏CSR还原 dense sharded npy(无GPU)。用法: <sparse_dir> <dense_out_dir> <hidden_dim>"""
import sys, os, glob, numpy as np, torch
sp_root, out_root, HD = sys.argv[1], sys.argv[2], int(sys.argv[3])
for split in ['train','val','test']:
    sd_in=f'{sp_root}/{split}'; sd_out=f'{out_root}/{split}'; os.makedirs(sd_out,exist_ok=True)
    m=torch.load(f'{sp_root}/sae_features_{split}_meta.pt',weights_only=False)
    n=len(m['lengths'])
    for i in range(n):
        z=np.load(f'{sd_in}/{i:06d}.npz'); idx=z['idx'].astype(np.int64); val=z['val'].astype(np.float32); off=z['off'].astype(np.int64)
        T=off.shape[0]-1; feat=np.zeros((T,HD),dtype=np.float32)
        rows=np.repeat(np.arange(T),np.diff(off)); feat[rows,idx]=val
        np.save(f'{sd_out}/{i:06d}.npy', feat)
    torch.save({'lengths':m['lengths'],'hidden_dim':HD,'mode':'sharded','shard_dir':sd_out,'backend':'musicdiscovery'}, f'{out_root}/sae_features_{split}_meta.pt')
    print(f'{split}: {n} dense shards → {sd_out}', flush=True)
print('DENSE_RESTORE_DONE', flush=True)
