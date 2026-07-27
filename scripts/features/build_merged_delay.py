#!/usr/bin/env python3
"""Merge per-benchmark splits + SAE shards (symlinked) into the pooled all_5 set.

Usage: build_merged_delay.py <small|large> [data_root]
data_root defaults to $MEVA_DATA_ROOT or the current directory.
"""
import sys, os, glob
import numpy as np, pandas as pd, torch
model=sys.argv[1]
R=sys.argv[2] if len(sys.argv)>2 else os.environ.get('MEVA_DATA_ROOT','.')
feat_tag=f'musicgen_{model}_delay'
sae_root=f'{R}/features/sae' if model=='small' else f'{R}/features/sae_large'
ORDER=['musicpref','aime','songeval','music_arena','musiceval']
out_split_dir=f'{R}/data/full_splits/all_5_delay_{model}'; out_sae_dir=f'{R}/features/sae_merged_{model}/all_5/clean'
os.makedirs(out_split_dir,exist_ok=True)
for split in ['train','val','test']:
    parts=[]
    for ds in ORDER:
        sp=pd.read_csv(f'{R}/features/{feat_tag}/{ds}/loss/split_{split}.csv'); sp['source']=ds; parts.append(sp)
    merged=pd.concat(parts,ignore_index=True); merged.to_csv(f'{out_split_dir}/{split}.csv',index=False)
    shard_out=f'{out_sae_dir}/{split}'; os.makedirs(shard_out,exist_ok=True)
    for f in glob.glob(f'{shard_out}/*.npy'): os.remove(f)
    lengths=[]; idx=0
    for ds in ORDER:
        m=torch.load(f'{sae_root}/{ds}/clean/sae_features_{split}_meta.pt',weights_only=False)
        L=[int(x) for x in m['lengths']]; mode=m.get('mode')
        if mode=='sharded':
            sdir=m.get('shard_dir',f'{sae_root}/{ds}/clean/{split}')
            for i in range(len(L)):
                os.symlink(os.path.join(sdir,f'{i:06d}.npy'), f'{shard_out}/{idx:06d}.npy'); idx+=1; lengths.append(L[i])
        else:
            npy=np.load(f'{sae_root}/{ds}/clean/sae_features_{split}.npy',mmap_mode='r')
            cum=np.concatenate([[0],np.cumsum(L)])
            for i in range(len(L)):
                np.save(f'{shard_out}/{idx:06d}.npy', np.asarray(npy[cum[i]:cum[i+1]])); idx+=1; lengths.append(L[i])
    hd=4096 if model=='small' else 8192
    torch.save({'lengths':lengths,'hidden_dim':hd,'mode':'sharded','shard_dir':shard_out,'backend':'musicdiscovery'}, f'{out_sae_dir}/sae_features_{split}_meta.pt')
    print(f'{model}/{split}: split={len(merged)} SAE={idx} match={len(merged)==idx}')
