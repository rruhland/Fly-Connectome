"""Offline simple-boundary diagnostic; never used by neural execution."""
import json
from pathlib import Path
import numpy as np
import torch

from local_information import collect, NeighborProbe, calibrate_threshold, detection_metrics
from controlled_visual import crop_payload
from fly_connectome.data import checksum


FAMILIES = ['E', 'H', 'total', 'difference', 'product', 'coincidence',
            'ratio', 'E_H', 'total_ratio']


def features(x, family):
    e, h = x[:, 1], -x[:, 2]
    total = e+h
    ratio = np.divide(h, total, out=np.zeros_like(h), where=total!=0)
    values = dict(E=e, H=h, total=total, difference=h-e, product=e*h,
                  coincidence=np.minimum(e,h), ratio=ratio,
                  E_H=np.c_[e,h], total_ratio=np.c_[total,ratio])
    return values[family].reshape(len(x), -1)


def branches(z, bounds):
    return np.stack([((z>=lo)&(z<=hi)).all(axis=1) for lo,hi in bounds], axis=1)


def quality(called, y):
    on, off, quiet = (float(called[y<0].mean()), float(called[y>0].mean()),
                      float(called[y==0].mean()))
    return (min(on,off), on+off, -quiet) if quiet<=.05 else None


def fit(train, calibration, family):
    z=features(train['x'],family); cal=features(calibration['x'],family)
    options=[]
    for mask in [train['y']<0, train['y']>0]:
        options.append([(np.quantile(z[mask],lo,axis=0), np.quantile(z[mask],1-hi,axis=0))
                        for lo in [0,.025,.05,.1] for hi in [0,.025,.05,.1]])
    best=None
    for on in options[0]:
        for off in options[1]:
            bounds=np.asarray([on,off])
            key=quality(branches(cal,bounds).any(axis=1),calibration['y'])
            if key is not None and (best is None or key>best[0]):
                best=(key,bounds)
    return None if best is None else best[1]


def report(data, family, bounds, x=None):
    b=branches(features(data['x'] if x is None else x,family),bounds)
    called=b.any(axis=1); y=data['y']
    m=detection_metrics(called.astype(float),y,.5)
    m.pop('auc')  # A binary boundary has no continuous ranking score.
    m['quiet_by_phase']={p:dict(count=int(mask.sum()),false_positives=int(called[mask].sum()))
        for p in np.unique(data['phase'][y==0]) if (mask:=((data['phase']==p)&(y==0))).any()}
    m['branch_calls']={p:dict(count=int(mask.sum()),on_branch=int(b[mask,0].sum()),
                            off_branch=int(b[mask,1].sum()),both=int(b[mask].all(axis=1).sum()))
                       for p,mask in [('on',y<0),('off',y>0),('quiet',y==0)]}
    return m


def main():
    torch.set_num_threads(1)
    out=Path('runs/simple-ei-boundary-v1');out.mkdir(exist_ok=False)
    data=np.load('runs/local-information-v1/recorded.npz')
    old=np.load('runs/local-information-v1/fresh.npz')
    splits={name:{k:data[k][mask] for k in ['x','y','phase']}
            for name,mask in [('train',data['trial']<25),
                              ('calibration',(data['trial']>=25)&(data['trial']<35)),
                              ('test',data['trial']>=35)]}
    train,cal=splits['train'],splits['calibration']
    models={f:fit(train,cal,f) for f in FAMILIES}
    eligible=[f for f in FAMILIES if models[f] is not None]
    selected=max(eligible,key=lambda f:(*quality(branches(features(cal['x'],f),models[f]).any(1),cal['y']),
                                        -models[f].size,-FAMILIES.index(f))) if eligible else None
    # Persist choices before generating or inspecting new evaluation data.
    (out/'frozen-boundaries.json').write_text(json.dumps(dict(selected=selected,
        models={f:None if b is None else b.tolist() for f,b in models.items()}),indent=2)+'\n')
    source=Path('runs/temporal-area-matched-excitation-v1')
    manifest=json.loads((source/'results.json').read_text())['manifest']
    payload=torch.load(manifest['source_checkpoint'],weights_only=True);metadata=payload['metadata']
    crop,_,_=crop_payload(payload,np.searchsorted(metadata['graph']['body_ids'],manifest['body_ids']))
    assert crop['graph'].identity()==manifest['graph_sha256']
    weights=torch.from_numpy(np.load(source/'training.npz')['weights_trained'])
    fresh=collect(crop,metadata,weights,np.random.default_rng(9025).integers(12,37,size=50))
    np.savez_compressed(out/'fresh.npz',**fresh)
    assert checksum(manifest['source_checkpoint'])==manifest['source_sha256']
    results={}
    for f,b in models.items():
        results[f]=None if b is None else dict(bounds=b.tolist(),
            calibration=report(cal,f,b),test=report(splits['test'],f,b),
            previously_seen_seed9024=report(old,f,b),fresh_seed9025=report(fresh,f,b))
    probe=NeighborProbe(train['x'][:,:3],train['y'])
    threshold=calibrate_threshold(probe.score(cal['x'][:,:3]),cal['y'])
    neighbor=detection_metrics(probe.score(fresh['x'][:,:3]),fresh['y'],threshold)
    sensitivity={}
    if selected:
        b=models[selected]; noisy=fresh['x'].copy()
        noisy[:,1:3]+=np.random.default_rng(9032).normal(size=noisy[:,1:3].shape)*train['x'][:,1:3].std(0)*.01
        sensitivity['current_noise_1pct']=report(fresh,selected,b,x=noisy)
        std=features(train['x'],selected).std(0)
        for name,sign in [('expanded',1),('contracted',-1)]:
            shifted=b.copy();shifted[:,0]-=sign*.01*std;shifted[:,1]+=sign*.01*std
            sensitivity[name+'_bounds_1pct']=report(fresh,selected,shifted)
    result=dict(selected_from_calibration=selected,results=results,neighbor_fresh_seed9025=neighbor,
        sensitivity=sensitivity,source_checkpoint_sha256=manifest['source_sha256'],
        trained_artifact_sha256=checksum(source/'training.npz'),fresh_artifact_sha256=checksum(out/'fresh.npz'),
        frozen_choices_sha256=checksum(out/'frozen-boundaries.json'))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(selected=selected,fresh={f:None if r is None else r['fresh_seed9025']
        for f,r in results.items()},neighbor=neighbor,sensitivity=sensitivity),indent=2))
    plot(fresh,models,selected,out)


def plot(data,models,selected,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,limit in zip(axes,[None,(.19,.06)]):
        emax,hmax=limit or (data['x'][:,1].max()*1.04,-data['x'][:,2].min()*1.04)
        e,h=np.meshgrid(np.linspace(0,emax,400),np.linspace(0,hmax,400))
        if selected:
            grid=np.c_[(e-h).ravel(),e.ravel(),-h.ravel()]
            b=branches(features(grid,selected),models[selected])
            ax.contourf(e,h,b.any(1).reshape(e.shape).astype(float),levels=[-.5,.5,1.5],
                        colors=['#eff3f8','#fff1d0'])
        for label,mask,color,marker in [('Quiet',data['y']==0,'#7b8894','.'),
                ('ON due +8 ticks',data['y']<0,'#2766b4','o'),('OFF due +8 ticks',data['y']>0,'#c24c18','^')]:
            ax.scatter(data['x'][mask,1],-data['x'][mask,2],s=16,c=color,marker=marker,
                       alpha=.4 if label=='Quiet' else .8,label=label)
        ax.set(xlim=(0,emax),ylim=(0,hmax),xlabel='Excitatory current E',ylabel='Inhibitory magnitude H',
               title='Full state plane' if limit is None else 'Low-inhibition region (zoom)')
    axes[0].legend(loc='upper left')
    fig.suptitle(f'New frozen trials: {selected} boundary (yellow = event call)')
    fig.savefig(out/'state-plane.png',dpi=150)


if __name__=='__main__':main()
