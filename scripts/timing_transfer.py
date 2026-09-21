"""Frozen tempo transfer followed by original-tempo-only margin calibration."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from local_information import collect, NeighborProbe, calibrate_threshold, detection_metrics
from simple_ei_boundary import features, branches, report
from controlled_visual import crop_payload
from fly_connectome.data import checksum


OUT=Path('runs/timing-transfer-v1')


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_model():
    source=Path('runs/temporal-area-matched-excitation-v1')
    manifest=json.loads((source/'results.json').read_text())['manifest']
    payload=torch.load(manifest['source_checkpoint'],weights_only=True);m=payload['metadata']
    crop,_,_=crop_payload(payload,np.searchsorted(m['graph']['body_ids'],manifest['body_ids']))
    assert crop['graph'].identity()==manifest['graph_sha256']
    assert checksum(manifest['source_checkpoint'])==manifest['source_sha256']
    weights=torch.from_numpy(np.load(source/'training.npz')['weights_trained'])
    return crop,m,weights


def original_data():
    data=np.load('runs/local-information-v1/recorded.npz')
    return [{k:data[k][mask] for k in ['x','y','phase']}
            for mask in [data['trial']<25,(data['trial']>=25)&(data['trial']<35)]]


def original_bounds():
    return np.asarray(json.loads(Path('runs/simple-ei-boundary-v1/frozen-boundaries.json').read_text())
                      ['models']['total_ratio'])


def signed_forecast(data):
    return {name:dict(count=int(mask.sum()),correct_sign=int((data['prediction'][mask]*data['y'][mask]>0).sum()),
                     mean_signed_anticipation=float((data['prediction'][mask]*data['y'][mask]).mean()))
            for name,mask in [('on',data['y']<0),('off',data['y']>0)]}


def transfer():
    OUT.mkdir(exist_ok=False)
    crop,m,weights=load_model();bounds=original_bounds();train,cal=original_data()
    probe=NeighborProbe(train['x'][:,:3],train['y'])
    threshold=calibrate_threshold(probe.score(cal['x'][:,:3]),cal['y'])
    save(OUT/'transfer-frozen.json',dict(bounds=bounds.tolist(),neighbor_threshold=threshold,
        dwell_frames=[2,4,6],blank_seed=9026,trials=25))
    results={}
    blanks=np.random.default_rng(9026).integers(12,37,size=25)
    for dwell in [2,4,6]:
        d=collect(crop,m,weights,blanks,dwell=dwell)
        assert (d['y']<0).sum()==(d['y']>0).sum()==75
        np.savez_compressed(OUT/f'dwell-{dwell}.npz',**d)
        results[str(dwell)]=dict(boundary=report(d,'total_ratio',bounds),
            neighbor=detection_metrics(probe.score(d['x'][:,:3]),d['y'],threshold),
            signed_forecast=signed_forecast(d),artifact_sha256=checksum(OUT/f'dwell-{dwell}.npz'))
        save(OUT/'transfer-results.json',results)
        print('Completed transfer dwell',dwell,json.dumps(results[str(dwell)]),flush=True)


def padded(bounds,padding,std):
    b=bounds.copy();b[:,0]-=np.asarray(padding)*std;b[:,1]+=np.asarray(padding)*std
    return b


def noisy_currents(x,std,seed):
    z=x.copy()
    z[:,1:3]+=np.random.default_rng(seed).normal(size=z[:,1:3].shape)*std*.01
    return z


def margin():
    # Padding selection reads only original train/calibration arrays, never transfer data.
    assert (OUT/'transfer-results.json').exists()
    train,cal=original_data();bounds=original_bounds()
    std=features(train['x'],'total_ratio').std(0);current_std=train['x'][:,1:3].std(0)
    conditions=[cal['x']]+[noisy_currents(cal['x'],current_std,s) for s in range(9040,9045)]
    candidates=[];best=None
    for total_pad in [0,.005,.01,.02,.04]:
        for ratio_pad in [0,.005,.01,.02,.04]:
            pad=[total_pad,ratio_pad];b=padded(bounds,pad,std)
            metrics=[detection_metrics(branches(features(x,'total_ratio'),b).any(1).astype(float),cal['y'],.5)
                     for x in conditions]
            worst_quiet=max(v['quiet_false_positive_rate'] for v in metrics)
            worst_recall=min(min(v['on_recall'],v['off_recall']) for v in metrics)
            candidates.append(dict(padding=pad,worst_quiet=worst_quiet,worst_recall=worst_recall))
            if worst_quiet<=.05:
                key=(worst_recall,np.mean([v['on_recall']+v['off_recall'] for v in metrics]),-worst_quiet,-sum(pad))
                if best is None or key>best[0]:best=(key,pad,b,metrics)
    frozen=dict(candidates=candidates,selected_padding=None if best is None else best[1],
                bounds=None if best is None else best[2].tolist(),
                calibration=None if best is None else best[3])
    assert not (OUT/'margin-frozen.json').exists()
    save(OUT/'margin-frozen.json',frozen)
    if best is None:
        print('No padding candidate satisfies calibration quiet cap',flush=True);return
    crop,m,weights=load_model()
    fresh=collect(crop,m,weights,np.random.default_rng(9027).integers(12,37,size=50))
    assert (fresh['y']<0).sum()==(fresh['y']>0).sum()==150
    np.savez_compressed(OUT/'margin-fresh.npz',**fresh)
    results={}
    for name,b in [('original',bounds),('padded',best[2])]:
        results[name]=dict(clean=report(fresh,'total_ratio',b),
            noisy=[report(fresh,'total_ratio',b,x=noisy_currents(fresh['x'],current_std,s)) for s in range(9050,9055)],
            transfer={str(dwell):report(np.load(OUT/f'dwell-{dwell}.npz'),'total_ratio',b) for dwell in [2,4,6]})
    save(OUT/'margin-results.json',dict(selected_padding=best[1],results=results,
        fresh_artifact_sha256=checksum(OUT/'margin-fresh.npz'),choices_sha256=checksum(OUT/'margin-frozen.json')))
    print('Margin evaluation complete',json.dumps({n:v['clean'] for n,v in results.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['transfer','margin'])
    args=parser.parse_args();torch.set_num_threads(1)
    (transfer if args.stage=='transfer' else margin)()
