"""Fixed approved adaptation A/B experiments, staged and resumable."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from timing_transfer import load_model
from multitempo import training_schedule
from controlled_visual import make_network,run_sequence,score_forecasts,passed
from temporal_visual import oscillation,recurrent_boundaries
from fly_connectome.data import checksum


OUT=Path('runs/adaptation-context-v1')
KINDS={'baseline':'area-matched-excitation-v1','A':'adaptation-A-v1','B':'adaptation-B-v1'}


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def tensor_bytes(net):
    return sum(v.numel()*v.element_size() for v in vars(net).values() if isinstance(v,torch.Tensor))


def preflight():
    OUT.mkdir(exist_ok=False);crop,m,_=load_model()
    target=int(np.searchsorted(crop['graph'].body_ids,82450));frames=oscillation(12)
    metrics={};times={k:[] for k in KINDS};reference=None
    for repeat in range(3):
        order=list(KINDS);order=order[repeat:]+order[:repeat]
        for name in order:
            n=make_network(crop,m,predictive_kinetics=KINDS[name]);started=time.perf_counter()
            r=run_sequence(n,crop,m,frames,[target],learning=False)
            times[name].append(time.perf_counter()-started)
            metrics[name]=dict(persistent_tensor_bytes=tensor_bytes(n))
            if name=='baseline':reference=r
            else:
                np.testing.assert_array_equal(r['spikes'],reference['spikes'])
                np.testing.assert_array_equal(r['target'],reference['target'])
    # Exact neutral-gain learning control on real crop, not just a synthetic impulse.
    base=make_network(crop,m,predictive_kinetics=KINDS['baseline'])
    neutral=make_network(crop,m,predictive_kinetics=KINDS['A'])
    neutral.forecast_gains=lambda a:(torch.ones_like(a),torch.ones_like(a))
    a,b=[run_sequence(n,crop,m,frames,[target],learning=True,visual_schedule='frame-horizon-v1') for n in [base,neutral]]
    for key in ['prediction','target','spikes','eligibility','local_update_sums']:
        np.testing.assert_array_equal(a[key],b[key])
    torch.testing.assert_close(base.magnitudes,neutral.magnitudes,rtol=0,atol=0)
    for name in KINDS:
        metrics[name].update(seconds=times[name],median_seconds=float(np.median(times[name])),frames=len(frames))
    save(OUT/'preflight.json',dict(metrics=metrics,neutral_learning_bit_exact=True,
        frozen_spikes_and_targets_bit_exact=True,benchmark='3 rotated-order frozen runs, warmup excluded, full diagnostic recording'))
    print('Preflight passed',flush=True)


def train(name):
    assert (OUT/'preflight.json').exists()
    artifact=OUT/f'{name}-training.npz';assert not artifact.exists()
    crop,m,_=load_model();target=int(np.searchsorted(crop['graph'].body_ids,82450))
    frames,blanks,dwells=training_schedule(200,9060)
    old=np.load('runs/multitempo-v1/mixed-training.npz')
    np.testing.assert_array_equal(blanks,old['blank_frames']);np.testing.assert_array_equal(dwells,old['dwells'])
    np.testing.assert_array_equal(crop['weights'].numpy(),old['weights_initial'])
    n=make_network(crop,m,predictive_kinetics=KINDS[name]);started=time.perf_counter()
    r=run_sequence(n,crop,m,frames,[target],learning=True,visual_schedule='frame-horizon-v1',deadline=started+1200)
    np.savez_compressed(artifact,**{k:v for k,v in r.items() if k!='stability'},
        weights_initial=crop['weights'].numpy(),weights_trained=n.magnitudes.numpy(),blank_frames=blanks,dwells=dwells)
    save(OUT/f'{name}-training.json',dict(seconds=time.perf_counter()-started,frames=len(frames),trials=len(blanks),
        stability=r['stability'],local_update_sums=r['local_update_sums'].sum(axis=1).tolist(),
        artifact_sha256=checksum(artifact)))
    print(name,'training saved',flush=True)


def benchmark():
    """Matched learning work with zero updates; profiling is outside timing."""
    crop,m,_=load_model();m=dict(m,learning=dict(m['learning']))
    for key in ['eta_prediction','eta_reward','homeostasis_rate']:m['learning'][key]=0
    target=int(np.searchsorted(crop['graph'].body_ids,82450));frames=oscillation(12)
    times={k:[] for k in KINDS};reference=None;metrics={}
    for repeat in range(3):
        order=list(KINDS);order=order[repeat:]+order[:repeat]
        for name in order:
            n=make_network(crop,m,predictive_kinetics=KINDS[name]);started=time.perf_counter()
            r=run_sequence(n,crop,m,frames,[target],learning=True,visual_schedule='frame-horizon-v1')
            times[name].append(time.perf_counter()-started)
            assert torch.equal(n.magnitudes,crop['weights'])
            if reference is None:reference=r
            np.testing.assert_array_equal(r['spikes'],reference['spikes'])
            np.testing.assert_array_equal(r['target'],reference['target'])
    for name in KINDS:
        n=make_network(crop,m,predictive_kinetics=KINDS[name])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,record_shapes=True,with_stack=True) as prof:
            run_sequence(n,crop,m,frames[:4],[target],learning=True,visual_schedule='frame-horizon-v1')
        path=OUT/f'{name}-memory.json';prof.export_memory_timeline(str(path),device='cpu')
        _,sizes=json.loads(path.read_text())
        metrics[name]=dict(seconds=times[name],median_seconds=float(np.median(times[name])),
            frames=len(frames),profile_frames=4,tracked_cpu_tensor_peak_bytes=max(map(sum,sizes)))
    save(OUT/'benchmark.json',dict(metrics=metrics,scope='3 rotated-order learning runs, warmup excluded; '
        'zero learning rates preserve matched activity. Separate 4-frame CPU tensor profile includes '
        'temporary tensors but excludes Python/NumPy/process memory; not a full-training peak.'))
    print('Learning benchmark saved',flush=True)


def evaluate(confirm=False):
    crop,m,_=load_model();target=int(np.searchsorted(crop['graph'].body_ids,82450))
    names=['baseline','A','B']
    if confirm:
        selection=json.loads((OUT/'selection.json').read_text());names=selection['qualifying']
        if not names:return
        names=['baseline']+names
    path=OUT/('confirmation.json' if confirm else 'evaluation.json')
    results=json.loads(path.read_text()) if path.exists() else {}
    blanks=np.random.default_rng(9074 if confirm else 9073).integers(12,37,size=30 if confirm else 15)
    for dwell in [2,3,4,6]:
        frames=torch.cat([oscillation(int(b),dwell=dwell) for b in blanks]);boundary=recurrent_boundaries(blanks,dwell=dwell)
        for name in names:
            source=Path('runs/multitempo-v1/mixed-training.npz') if name=='baseline' else OUT/f'{name}-training.npz'
            weights=torch.from_numpy(np.load(source)['weights_trained'])
            for state,w in [('initial',crop['weights']),('trained',weights)]:
                key=f'{name}-{state}-{dwell}'
                if key in results:continue
                n=make_network(crop,m,w,predictive_kinetics=KINDS[name])
                r=run_sequence(n,crop,m,frames,[target],learning=False,deadline=time.perf_counter()+600)
                assert torch.equal(n.magnitudes,w)
                prefix='confirm-' if confirm else ''
                baseline=OUT/f'{prefix}baseline-initial-{dwell}.npz'
                if name!='baseline' or state!='initial':
                    old=np.load(baseline);np.testing.assert_array_equal(r['target'],old['target'])
                    if state=='initial':np.testing.assert_array_equal(r['spikes'],old['spikes'])
                np.savez_compressed(OUT/f'{prefix}{key}.npz',**{k:v for k,v in r.items() if k!='stability'})
                y=r['target'][boundary+8];p=r['prediction'][boundary]
                assert (y<0).sum()==(y>0).sum()==len(blanks)*3
                score=score_forecasts(r['target'],r['prediction'],boundary,8)
                score['correct_sign']={s:dict(count=int(mask.sum()),correct=int((p[mask]*y[mask]>0).sum()))
                    for s,mask in [('on',y<0),('off',y>0)]}
                results[key]=dict(metrics=score,stability=r['stability'])
                if name=='baseline' and state=='initial':
                    results[f'zero-{dwell}']=dict(metrics=score_forecasts(r['target'],np.zeros_like(r['target']),boundary,8))
                    results[f'persistence-{dwell}']=dict(metrics=score_forecasts(r['target'],r['target'],boundary,8))
                if state=='trained':
                    results[key]['pass']=passed(score,results[f'{name}-initial-{dwell}']['metrics'],
                                               results[f'zero-{dwell}']['metrics'],results[f'persistence-{dwell}']['metrics'])
                save(path,results);print(key,json.dumps(score),flush=True)
    qualified=[name for name in names if name!='baseline' and
        all(results[f'{name}-trained-{d}']['pass'] for d in [2,3,4,6]) and
        results[f'{name}-trained-3']['metrics']['mse']<results['baseline-trained-3']['metrics']['mse']]
    save(OUT/('confirmation-selection.json' if confirm else 'selection.json'),dict(qualifying=qualified,
        criterion='All four tempos pass own-initial/zero/persistence criteria AND held3 MSE improves over trained baseline'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','benchmark','train-A','train-B','evaluate','confirm'])
    args=p.parse_args();torch.set_num_threads(1)
    if args.stage=='preflight':preflight()
    elif args.stage=='benchmark':benchmark()
    elif args.stage.startswith('train-'):train(args.stage[-1])
    else:evaluate(args.stage=='confirm')
