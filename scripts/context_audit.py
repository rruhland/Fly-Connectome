"""Offline cross-tempo local-state audit; model and weights stay frozen."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

from local_information import collect, NeighborProbe, detection_metrics
from timing_transfer import load_model
from fly_connectome.data import checksum


OUT=Path('runs/context-audit-v1')


def signed_prediction(probe,x,labels,threshold):
    scaled=((x-probe.mean)/probe.scale)[:,probe.keep]
    _,idx=probe.tree.query(scaled,k=probe.k)
    if probe.k==1:idx=idx[:,None]
    votes=labels[idx]
    scores=(votes!=0).mean(1)
    polarity=np.where((votes<0).sum(1)>=(votes>0).sum(1),-1,1)
    return scores,np.where(scores>=threshold,polarity,0)


def threshold_for_tempos(scores,y,tempo):
    best=None
    for threshold in np.r_[np.unique(scores),scores.max()+1e-6]:
        rows=[detection_metrics(scores[tempo==d],y[tempo==d],threshold) for d in [2,4,6]]
        worst_quiet=max(r['quiet_false_positive_rate'] for r in rows)
        if worst_quiet<=.05:
            key=(min(min(r['on_recall'],r['off_recall']) for r in rows),
                 np.mean([r['on_recall']+r['off_recall'] for r in rows]),-worst_quiet,threshold)
            if best is None or key>best[0]:best=(key,float(threshold))
    return best[1]


def metrics(scores,pred,y,phase,threshold):
    r=detection_metrics(scores,y,threshold)
    r['signed_confusion_true_rows_predicted_columns_ON_quiet_OFF']=[
        [int(((y==a)&(pred==b)).sum()) for b in [-1,0,1]] for a in [-1,0,1]]
    r['quiet_by_phase']={str(p):dict(count=int(mask.sum()),false_positives=int((pred[mask]!=0).sum()))
        for p in np.unique(phase[y==0]) if (mask:=((phase==p)&(y==0))).any()}
    return r


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
    crop,m,_=load_model();weights=torch.from_numpy(np.load('runs/multitempo-v1/mixed-training.npz')['weights_trained'])
    all_data=[]
    for dwell in [2,3,4,6]:
        d=collect(crop,m,weights,np.random.default_rng(9070).integers(12,37,size=30),dwell=dwell)
        assert (d['y']<0).sum()==(d['y']>0).sum()==90
        np.savez_compressed(OUT/f'dwell-{dwell}.npz',**d)
        all_data.append({**{k:d[k] for k in ['x','y','trial','phase']},'tempo':np.full(len(d['y']),dwell)})
        print('Collected',dwell,flush=True)
    data={k:np.concatenate([a[k] for a in all_data]) for k in all_data[0]}
    masks={'train':(data['tempo']!=3)&(data['trial']<15),
           'cal':(data['tempo']!=3)&(data['trial']>=15)&(data['trial']<20),
           'test':(data['tempo']==3)|(data['trial']>=20)}
    train,cal,test=[{k:v[mask] for k,v in data.items()} for mask in masks.values()]
    groups={'EI':np.array([1,2]),'cell_history':np.array([1,2,6,7,8,11,12,13]),
            'synaptic_history':np.r_[1,2,np.arange(14,data['x'].shape[1])],
            'all_local':np.arange(data['x'].shape[1])}
    results={};predictions={}
    for name,cols in groups.items():
        probe=NeighborProbe(train['x'][:,cols],train['y'])
        threshold=threshold_for_tempos(probe.score(cal['x'][:,cols]),cal['y'],cal['tempo'])
        scores,pred=signed_prediction(probe,test['x'][:,cols],train['y'],threshold)
        noisy=test['x'][:,cols]+np.random.default_rng(9071).normal(size=test['x'][:,cols].shape)*probe.scale*.01
        ns,npred=signed_prediction(probe,noisy,train['y'],threshold)
        results[name]=dict(threshold=threshold,features=d['feature_names'][cols].tolist(),
            test={str(t):metrics(scores[mask],pred[mask],test['y'][mask],test['phase'][mask],threshold)
                  for t in [2,3,4,6] if (mask:=test['tempo']==t).any()},
            noisy={str(t):metrics(ns[mask],npred[mask],test['y'][mask],test['phase'][mask],threshold)
                   for t in [2,3,4,6] if (mask:=test['tempo']==t).any()})
        predictions[name]=pred
    labels=np.random.default_rng(9072).permutation(train['y'])
    probe=NeighborProbe(train['x'],labels)
    threshold=threshold_for_tempos(probe.score(cal['x']),cal['y'],cal['tempo'])
    scores,pred=signed_prediction(probe,test['x'],labels,threshold)
    results['shuffled_all_local']={str(t):metrics(scores[mask],pred[mask],test['y'][mask],test['phase'][mask],threshold)
                                  for t in [2,3,4,6] if (mask:=test['tempo']==t).any()}
    # Pair without target-label access; inspect future disagreement only afterwards.
    cols=groups['EI'];mean=train['x'][:,cols].mean(0);std=train['x'][:,cols].std(0)
    ztrain=(train['x'][:,cols]-mean)/std;ztest=(test['x'][:,cols]-mean)/std
    indices=np.zeros(len(ztest),dtype=int);distance=np.zeros(len(ztest))
    for t in [2,3,4,6]:
        candidates=np.flatnonzero(train['tempo']!=t);mask=test['tempo']==t
        dist,idx=cKDTree(ztrain[candidates]).query(ztest[mask])
        indices[mask]=candidates[idx];distance[mask]=dist/np.sqrt(2)
    close=distance<=.01;disagree=close&(test['y']!=train['y'][indices])
    pairs=[]
    scale=train['x'].std(0);scale=np.where(scale>1e-8,scale,1.)
    for i in np.flatnonzero(disagree)[:20]:
        j=indices[i]
        pairs.append(dict(test_tempo=int(test['tempo'][i]),train_tempo=int(train['tempo'][j]),
            test_future=float(test['y'][i]),train_future=float(train['y'][j]),EI_distance=float(distance[i]),
            test_EI=test['x'][i,cols].tolist(),train_EI=train['x'][j,cols].tolist(),
            standardized_local_difference=((test['x'][i]-train['x'][j])/scale).tolist()))
    audit={str(t):dict(test_count=int(mask.sum()),close_pairs=int((mask&close).sum()),
        conflicting_pairs=int((mask&disagree).sum()),
        conflict_true_counts={str(y):int((mask&disagree&(test['y']==y)).sum()) for y in [-1,0,1]},
        correct_signed_future_on_conflicts={name:int(((p==test['y'])&mask&disagree).sum()) for name,p in predictions.items()})
        for t in [2,3,4,6] if (mask:=test['tempo']==t).any()}
    output=dict(results=results,matching=audit,example_pairs=pairs,feature_names=d['feature_names'].tolist(),
                mixed_weights_sha256=checksum('runs/multitempo-v1/mixed-training.npz'),
                artifacts={str(p):checksum(p) for p in OUT.glob('*.npz')})
    (OUT/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Audit saved',flush=True)


if __name__=='__main__':main()
