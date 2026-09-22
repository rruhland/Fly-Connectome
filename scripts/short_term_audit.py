"""Offline preflight report and independent reconstruction of local targets."""
import json
from pathlib import Path
import numpy as np
import torch
from timing_transfer import load_model
from fly_connectome.sensor import Retina
from fly_connectome.data import checksum


def main():
    torch.set_num_threads(1)
    crop,m,_=load_model();g=crop['graph'];p=Path('runs/short-term-v1')
    out=Path('docs/experiments');prefix='2026-09-21-short-term-'
    results=json.loads((p/'preflight-results.json').read_text())
    injected=Retina(**crop['retina']).injected.numpy()
    ff=np.flatnonzero(np.asarray(crop['pathways'])=='feedforward')
    delays=np.asarray(crop['delays']);start=max(delays[ff],default=0)
    target=int(np.searchsorted(g.body_ids,82450));audit={}
    assert m['learning']['visual_target']=='input-arrivals-v1'
    for key,result in results.items():
        r=np.load(p/f'{key}.npz');expected=np.zeros_like(r['target'])
        # Offline reconstruction from emitted spikes and fixed anatomical delays.
        # No simulation, plasticity or model state is advanced here.
        for e in ff:
            delay=delays[e]
            expected[delay:,g.post[e]]+=r['spikes'][:-delay,g.pre[e]]*float(crop['weights'][e])*g.signs[g.pre[e]]
        expected=np.clip(expected/m['neurons']['threshold'],-1,1)
        np.testing.assert_allclose(expected[start:,~injected],r['target'][start:,~injected],rtol=1e-5,atol=1e-6)
        name,dwell=key.split('-');b=np.load(p/f'baseline-{dwell}.npz')
        difference=r['target']!=b['target'];mask=difference.any(axis=0)
        row=dict(reconstruction_max_abs_error=float(np.max(np.abs(expected[start:,~injected]-r['target'][start:,~injected]))),
            affected_classes=sorted(set(np.asarray(crop['retina']['cell_types'])[mask].tolist())),
            changed_injected_targets=int(difference[:,injected].sum()),
            changed_visual_learning_targets=int(difference[:,np.unique(g.post[np.asarray(crop['pathways'])=='predictive'])].sum()))
        if name!='baseline':
            edge=r['release_edges'];gain=r['release_gains'];observed=r['release_observed'];paths={}
            for body in [20655,26550]:
                selected=(g.body_ids[g.pre[edge]]==body)&(g.post[edge]==target)
                stats={}
                for label,condition in [('all',np.ones(len(edge),dtype=bool)),('event',observed!=0),('quiet',observed==0)]:
                    values=gain[selected&condition]
                    stats[label]=dict(count=len(values),quantiles=np.quantile(values,[0,.1,.5,.9,1]).tolist() if len(values) else [])
                paths[str(body)]=stats
            row['target_path_release']=paths
        audit[key]=row
    for suffix,value in [('preflight-results',results),('preflight-decision',json.loads((p/'preflight-decision.json').read_text())),('audit',audit),
            ('artifacts',{str(f):checksum(f) for f in p.glob('*.npz')})]:
        (out/f'{prefix}{suffix}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
