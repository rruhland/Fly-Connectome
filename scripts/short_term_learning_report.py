"""Publish fixed STP learning metrics and a comparison plot from saved results."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from timing_transfer import load_model
from fly_connectome.data import checksum


def main():
    torch.set_num_threads(1)
    p=Path('runs/short-term-v1');out=Path('docs/experiments');prefix='2026-09-21-short-term-learning-'
    def save(name,value):
        (out/f'{prefix}{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    r=json.loads((p/'evaluation.json').read_text());assert len(r)==32
    save('results',r);save('selection',json.loads((p/'selection.json').read_text()))
    save('training',{n:json.loads((p/f'{n}-training.json').read_text()) for n in ['D','F']})
    save('amended-preflight',json.loads((p/'amended-preflight.json').read_text()))
    crop,m,_=load_model();g=crop['graph'];target=int(np.searchsorted(g.body_ids,82450))
    incoming=np.flatnonzero((g.post==target)&(np.asarray(crop['pathways'])=='predictive'))
    trained={n:np.load(p/f'{n}-training.npz') for n in ['D','F']}
    baseline=np.load('runs/multitempo-v1/mixed-training.npz');weights=[]
    for j,e in enumerate(incoming):
        weights.append(dict(pre=int(g.body_ids[g.pre[e]]),post=82450,initial=float(crop['weights'][e]),
            baseline=float(baseline['weights_trained'][e]),**{n:dict(weight=float(a['weights_trained'][e]),
            proposed_updates_on_off_quiet=a['local_update_sums'][:,j].tolist()) for n,a in trained.items()}))
    save('weights',weights)
    files=list(p.glob('*training.npz'))+list(p.glob('*initial-*.npz'))+list(p.glob('*trained-*.npz'))+list(p.glob('*unit-*.npz'))
    save('artifacts',{str(f):checksum(f) for f in files})
    for d in [2,3,4,6]:
        for n in ['baseline','D','F']:
            v=r[f'{n}-trained-{d}'];print(d,n,json.dumps(v['metrics']),'pass',v['pass'],flush=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,key,title,threshold,mult in zip(axes,['on_anticipation','off_anticipation','false_alarm_fraction'],
        ['ON anticipation','OFF anticipation','Quiet false alarms (%)'],[.1,.1,5],[1,1,100]):
        for n,color in [('baseline','#636363'),('D','#0072B2'),('F','#D55E00')]:
            ax.plot([2,3,4,6],[r[f'{n}-trained-{d}']['metrics'][key]*mult for d in [2,3,4,6]],
                marker='o',color=color,label=n)
        ax.axhline(threshold,color='black',linestyle='--',linewidth=1)
        ax.axvspan(2.85,3.15,color='grey',alpha=.12);ax.grid(alpha=.2)
        ax.set(title=title,xlabel='Dwell (frames); 3 held out',xticks=[2,3,4,6])
    axes[0].legend(frameon=False);fig.suptitle('STP: frozen sensory forecasts after 200 mixed-tempo trials')
    fig.savefig(out/'assets/2026-09-21-short-term-learning.png',dpi=180)


if __name__=='__main__':main()
