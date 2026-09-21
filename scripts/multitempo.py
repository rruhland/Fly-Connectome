"""Actual local learning across tempos; no diagnostic readout or gate."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from timing_transfer import load_model
from controlled_visual import make_network, run_sequence, score_forecasts, passed
from temporal_visual import oscillation, recurrent_boundaries


OUT=Path('runs/multitempo-v1')
KINETICS='area-matched-excitation-v1'


def training_schedule(count,seed):
    rng=np.random.default_rng(seed)
    dwells=np.resize([2,4,6],count);rng.shuffle(dwells)
    blanks=rng.integers(12,37,size=count)
    frames=torch.cat([oscillation(int(b),dwell=int(d)) for b,d in zip(blanks,dwells)])
    return frames,blanks,dwells


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def train(adapt=False):
    crop,m,_=load_model();target=int(np.searchsorted(crop['graph'].body_ids,82450))
    if adapt:
        start=torch.from_numpy(np.load(OUT/'mixed-training.npz')['weights_trained'])
        blanks=np.random.default_rng(9062).integers(12,37,size=100)
        dwells=np.full(100,3);frames=torch.cat([oscillation(int(b)) for b in blanks]);name='adapted'
    else:
        OUT.mkdir(exist_ok=False);start=crop['weights'].clone()
        frames,blanks,dwells=training_schedule(200,9060);name='mixed'
    assert not (OUT/f'{name}-training.npz').exists()
    save_json(OUT/f'{name}-schedule.json',dict(blanks=blanks.tolist(),dwells=dwells.tolist(),frames=len(frames)))
    net=make_network(crop,m,start,predictive_kinetics=KINETICS)
    begun=time.perf_counter()
    result=run_sequence(net,crop,m,frames,[target],learning=True,
                        visual_schedule='frame-horizon-v1',deadline=begun+1200)
    np.savez_compressed(OUT/f'{name}-training.npz',**{k:v for k,v in result.items() if k!='stability'},
                        weights_initial=start.numpy(),weights_trained=net.magnitudes.numpy(),blank_frames=blanks,dwells=dwells)
    save_json(OUT/f'{name}-training.json',dict(seconds=time.perf_counter()-begun,frames=len(frames),trials=len(blanks),
              stability=result['stability'],changed_weights=int((net.magnitudes!=start).sum()),
              local_update_sums=result['local_update_sums'].sum(axis=1).tolist()))
    print(name,'training saved',flush=True)


def evaluate(adapt=False):
    crop,m,single=load_model();target=int(np.searchsorted(crop['graph'].body_ids,82450))
    variants={'adapted':torch.from_numpy(np.load(OUT/'adapted-training.npz')['weights_trained'])} if adapt else {
        'initial':crop['weights'],'single3':single,
        'mixed':torch.from_numpy(np.load(OUT/'mixed-training.npz')['weights_trained'])}
    path=OUT/'evaluation.json';results=json.loads(path.read_text()) if path.exists() else {}
    blanks=np.random.default_rng(9061).integers(12,37,size=15)
    for dwell in [2,3,4,6]:
        frames=torch.cat([oscillation(int(b),dwell=dwell) for b in blanks])
        boundary=recurrent_boundaries(blanks,dwell=dwell)
        for name,w in variants.items():
            key=f'{name}-dwell-{dwell}';artifact=OUT/(key+'.npz')
            if key in results:continue  # Resume only fully saved evaluations.
            net=make_network(crop,m,w,predictive_kinetics=KINETICS)
            r=run_sequence(net,crop,m,frames,[target],learning=False,deadline=time.perf_counter()+600)
            assert torch.equal(net.magnitudes,w)
            baseline=OUT/f'initial-dwell-{dwell}.npz'
            if name!='initial':np.testing.assert_array_equal(r['target'],np.load(baseline)['target'])
            y=r['target'][boundary+8];p=r['prediction'][boundary]
            assert (y<0).sum()==(y>0).sum()==45
            metrics=score_forecasts(r['target'],r['prediction'],boundary,8)
            metrics['correct_sign']={n:dict(count=int(mask.sum()),correct=int((p[mask]*y[mask]>0).sum()))
                                     for n,mask in [('on',y<0),('off',y>0)]}
            np.savez_compressed(artifact,**{k:v for k,v in r.items() if k!='stability'},blank_frames=blanks)
            results[key]=dict(metrics=metrics,stability=r['stability'])
            if name=='initial':
                results[f'zero-dwell-{dwell}']=dict(metrics=score_forecasts(r['target'],np.zeros_like(r['target']),boundary,8))
                results[f'persistence-dwell-{dwell}']=dict(metrics=score_forecasts(r['target'],r['target'],boundary,8))
            else:
                results[key]['controlled_task_pass']=passed(metrics,results[f'initial-dwell-{dwell}']['metrics'],
                    results[f'zero-dwell-{dwell}']['metrics'],results[f'persistence-dwell-{dwell}']['metrics'])
            save_json(path,results)
            print(key,json.dumps(metrics),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['train','evaluate','adapt','evaluate-adapted'])
    args=p.parse_args();torch.set_num_threads(1)
    if args.stage in ['train','adapt']:train(args.stage=='adapt')
    else:evaluate(args.stage=='evaluate-adapted')
