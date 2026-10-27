"""Pure reconstruction of one saved aggregate fit dataset; no execution or fitting."""
from pathlib import Path
import numpy as np
import run_practical_pong_v2 as base


def load(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');theta=np.load(directory/'carry_theta.npy');rows=[]
    for path in m['cached_current_receipts']+m['new_current_receipts']:
        p=Path(path);r=base.read(p);rows.append(dict(r,feature_path=str(p.parent.parent/r['feature_file'])))
    byseed={r['seed']:r for r in rows};memories=[dict(v) for v in m['memories']];discovery=directory/'discovery'
    for w in base.read(directory/'discovery-result.json')['witnesses']:
        path=discovery/w['feature_file']
        with np.load(path) as f:length=len(f['X'])
        hits=[i for i,r in enumerate(byseed[w['seed']]['records'][:w['transition']]) if r['hits']]+[length-1]
        memories.append(dict(seed=w['seed'],phase=w['phase'],feature_path=str(path),hit_indices=hits,source='current_burst_success',target_hits=len(hits)))
    positive=[];matching=[];preservation=[]
    for memory in memories:
        with np.load(memory['feature_path']) as f:X=f['X'];actions=f['actions']
        if memory.get('slice_end'):X=X[:memory['slice_end']];actions=actions[:memory['slice_end']]
        contrast=actions!=np.argmax(X@theta.T,1);start=0
        for ordinal,end in enumerate(memory['hit_indices']):
            group=('first_' if ordinal==0 else 'later_')+memory['phase'];segment=np.arange(start,end+1);start=end+1
            for mask,target in ((contrast,positive),(~contrast,matching)):
                index=segment[mask[segment]]
                if len(index):target.append(dict(seed=memory['seed'],group=group,X=X[index],actions=actions[index]))
    for r in rows:
        with np.load(r['feature_path']) as f:X=f['X'];actions=f['actions']
        phase='left' if base.leftward(r['seed']) else 'right';start=0
        for ordinal,end in enumerate(i for i,v in enumerate(r['records']) if v['hits']):
            preservation.append(dict(seed=r['seed'],group=('first_' if ordinal==0 else 'later_')+phase,X=X[start:end+1],actions=actions[start:end+1]));start=end+1
    return theta,positive,preservation,matching,rows,memories
