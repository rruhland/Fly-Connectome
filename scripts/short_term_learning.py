"""Fixed approved STP learning runs under sensory-target comparison amendment."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from short_term_experiment import OUT,KINDS,save
from short_term import DepressionNetwork,FacilitationNetwork
from timing_transfer import load_model
from multitempo import training_schedule
from temporal_visual import oscillation,recurrent_boundaries
from controlled_visual import make_network,run_sequence,score_forecasts,passed
from fly_connectome.sensor import Retina
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.data import checksum


def compare_targets(candidate,reference,injected,target):
    np.testing.assert_equal(candidate.shape,reference.shape)
    np.testing.assert_array_equal(candidate[:,injected],reference[:,injected])
    np.testing.assert_array_equal(candidate[:,target],reference[:,target])
    return int((candidate[:,~injected]!=reference[:,~injected]).sum())


def validate_cached(row,artifact,weights,stimulus_hash):
    assert row['weights_sha256']==hashlib.sha256(weights.numpy().tobytes()).hexdigest(), 'cached weights changed'
    assert row['stimulus_sha256']==stimulus_hash, 'cached stimulus changed'
    assert row['sha256']==checksum(artifact), 'cached artifact changed'


def reconstruct(crop,m,weights,result):
    g=crop['graph'];injected=Retina(**crop['retina']).injected.numpy()
    ff=np.flatnonzero(np.asarray(crop['pathways'])=='feedforward');delays=np.asarray(crop['delays'])
    expected=np.zeros_like(result['target']);start=max(delays[ff],default=0)
    for e in ff:
        d=delays[e]
        expected[d:,g.post[e]]+=result['spikes'][:-d,g.pre[e]]*float(weights[e])*g.signs[g.pre[e]]
    expected=np.clip(expected/m['neurons']['threshold'],-1,1)
    np.testing.assert_allclose(expected[start:,~injected],result['target'][start:,~injected],rtol=1e-5,atol=1e-6)
    return float(np.max(np.abs(expected[start:,~injected]-result['target'][start:,~injected])))


def amend():
    crop,m,_=load_model();injected=Retina(**crop['retina']).injected.numpy()
    target=int(np.searchsorted(crop['graph'].body_ids,82450))
    neutral=json.loads((OUT/'neutral.json').read_text());assert all(neutral[n] for n in KINDS)
    recorded=json.loads((OUT/'preflight-results.json').read_text());evidence={}
    for d in [2,3,4,6]:
        baseline=np.load(OUT/f'baseline-{d}.npz')
        for name in KINDS:
            key=f'{name}-{d}';path=OUT/f'{key}.npz'
            assert checksum(path)==recorded[key]['sha256']
            r=np.load(path)
            evidence[key]=dict(internal_changes=compare_targets(r['target'],baseline['target'],injected,target),
                reconstruction_error=reconstruct(crop,m,crop['weights'],r),sha256=checksum(path))
    save(OUT/'amended-preflight.json',dict(passed=True,protocol='approved sensory/scored target equality',evidence=evidence))
    print('Saved amended decision; historical all-neuron stop decision preserved',flush=True)


def train(name):
    assert json.loads((OUT/'amended-preflight.json').read_text())['passed']
    path=OUT/f'{name}-training.npz';assert not path.exists()
    crop,m,_=load_model();target=int(np.searchsorted(crop['graph'].body_ids,82450))
    frames,blanks,dwells=training_schedule(200,9060)
    old=np.load('runs/multitempo-v1/mixed-training.npz')
    for a,b in [(blanks,old['blank_frames']),(dwells,old['dwells']),(crop['weights'].numpy(),old['weights_initial'])]:
        np.testing.assert_array_equal(a,b)
    n=make_network(crop,m,predictive_kinetics=KINDS[name]);started=time.perf_counter()
    r=run_sequence(n,crop,m,frames,[target],learning=True,visual_schedule='frame-horizon-v1',deadline=started+1200)
    assert torch.isfinite(n.release_state).all() and ((n.release_state>=0)&(n.release_state<=1)).all()
    np.savez_compressed(path,**{k:v for k,v in r.items() if k!='stability'},weights_initial=crop['weights'].numpy(),
        weights_trained=n.magnitudes.numpy(),blank_frames=blanks,dwells=dwells)
    save(OUT/f'{name}-training.json',dict(seconds=time.perf_counter()-started,frames=len(frames),trials=200,
        stability=r['stability'],sha256=checksum(path),stimulus_sha256=hashlib.sha256(frames.numpy().tobytes()).hexdigest()))
    print(name,'training complete',flush=True)


def build(crop,m,weights,name,unit=False):
    if not unit:return make_network(crop,m,weights,predictive_kinetics=KINDS[name])
    cls=DepressionNetwork if name=='D' else FacilitationNetwork
    n=cls(crop['graph'],crop['delays'],crop['pathways'],unit_release=True,
        config=NeuronConfig(**m['neurons']),cell_types=crop['retina']['cell_types'])
    n.magnitudes.copy_(weights)
    for _ in range(m['config']['warmup_steps']):n.step(torch.zeros_like(n.voltage))
    return n


def evaluate(confirm=False):
    names=list(KINDS)
    if confirm:
        qualifiers=json.loads((OUT/'selection.json').read_text())['qualifying']
        if not qualifiers:return
        names=['baseline']+qualifiers
    crop,m,_=load_model();target=int(np.searchsorted(crop['graph'].body_ids,82450))
    injected=Retina(**crop['retina']).injected.numpy()
    prefix='confirmation-' if confirm else '';path=OUT/f'{prefix}evaluation.json'
    results=json.loads(path.read_text()) if path.exists() else {}
    seed=9082 if confirm else 9081;trials=30 if confirm else 15
    blanks=np.random.default_rng(seed).integers(12,37,size=trials)
    for d in [2,3,4,6]:
        frames=torch.cat([oscillation(int(b),dwell=d) for b in blanks]);boundary=recurrent_boundaries(blanks,dwell=d)
        stimulus_hash=hashlib.sha256(frames.numpy().tobytes()).hexdigest()
        for name in names:
            source='runs/multitempo-v1/mixed-training.npz' if name=='baseline' else OUT/f'{name}-training.npz'
            weights=torch.from_numpy(np.load(source)['weights_trained'])
            states=[('initial',crop['weights']),('trained',weights)]
            if confirm and name!='baseline':states.append(('unit',weights))
            for state,w in states:
                key=f'{name}-{state}-{d}'
                artifact=OUT/f'{prefix}{key}.npz';cached=results.get(key)
                if cached is not None:
                    validate_cached(cached,artifact,w,stimulus_hash)
                    r=dict(np.load(artifact));seconds=cached['seconds'];stability=cached['stability']
                else:
                    n=build(crop,m,w,name,unit=state=='unit');started=time.perf_counter()
                    r=run_sequence(n,crop,m,frames,list(range(n.n)),learning=False,deadline=started+600)
                    seconds=time.perf_counter()-started;stability=r['stability']
                    assert torch.equal(n.magnitudes,w)
                assert hashlib.sha256(frames.numpy().tobytes()).hexdigest()==stimulus_hash
                baseline=OUT/f'{prefix}baseline-initial-{d}.npz'
                reference=r if name=='baseline' and state=='initial' else np.load(baseline)
                changes=compare_targets(r['target'],reference['target'],injected,target)
                error=reconstruct(crop,m,w,r)
                if cached is None:np.savez_compressed(artifact,**{k:v for k,v in r.items() if k!='stability'})
                y,p=r['target'][:,target:target+1],r['prediction'][:,target:target+1]
                actual,forecast=y[boundary+8],p[boundary]
                assert (actual<0).sum()==(actual>0).sum()==trials*3
                score=score_forecasts(y,p,boundary,8)
                score['correct_sign']={s:dict(count=int(mask.sum()),correct=int((forecast[mask]*actual[mask]>0).sum()))
                    for s,mask in [('on',actual<0),('off',actual>0)]}
                results[key]=dict(metrics=score,stability=stability,internal_target_changes=changes,
                    reconstruction_error=error,seconds=seconds,stimulus_sha256=stimulus_hash,sha256=checksum(artifact),
                    weights_sha256=hashlib.sha256(w.numpy().tobytes()).hexdigest())
                if name=='baseline' and state=='initial':
                    results[f'zero-{d}']=dict(metrics=score_forecasts(y,np.zeros_like(y),boundary,8))
                    results[f'persistence-{d}']=dict(metrics=score_forecasts(y,y,boundary,8))
                if state=='trained':
                    results[key]['pass']=passed(score,results[f'{name}-initial-{d}']['metrics'],
                        results[f'zero-{d}']['metrics'],results[f'persistence-{d}']['metrics'])
                save(path,results);print(key,json.dumps(score),flush=True)
    qualified=[n for n in names if n!='baseline' and all(results[f'{n}-trained-{d}']['pass'] for d in [2,3,4,6])
        and results[f'{n}-trained-3']['metrics']['mse']<results['baseline-trained-3']['metrics']['mse']]
    save(OUT/f'{prefix}selection.json',dict(qualifying=qualified,seed=seed,trials=trials,
        criterion='Every tempo passes unchanged criteria and held3 MSE improves over trained baseline'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['amend','train-D','train-F','evaluate','confirm'])
    args=p.parse_args();torch.set_num_threads(1)
    if args.stage=='amend':amend()
    elif args.stage.startswith('train-'):train(args.stage[-1])
    else:evaluate(args.stage=='confirm')
