"""Frozen forecasts by anatomical sign support; shifted scores are diagnostics only."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from audit_predictive_support import predictive_support
from fly_connectome.data import checksum
from fly_connectome.sensor import EventCamera
from fly_connectome.training import load_checkpoint


def group_scores(target,prediction,positive,negative):
    on,off=target<0,target>0
    masks=dict(all=torch.ones_like(target,dtype=torch.bool),
        on_supported=on&negative,on_unsupported=on&~negative,
        off_supported=off&positive,off_unsupported=off&~positive,quiet=target==0)
    result={}
    for name,mask in masks.items():
        if not mask.any():
            result[name]=None
            continue
        y,p=target[mask].double(),prediction[mask].double()
        active=y!=0
        result[name]=dict(samples=len(y),active_samples=int(active.sum()),
            model_mse=float((y-p).square().mean()),zero_mse=float(y.square().mean()),
            prediction_power=float(p.square().mean()),cross_mean=float((y*p).mean()),
            mean_prediction=float(p.mean()),mean_absolute_prediction=float(p.abs().mean()),
            correct_sign_fraction=float((y[active]*p[active]>0).double().mean()) if active.any() else None)
    return result


def lag_scores(targets,timeline,neural_steps,positive,negative,lags=(-16,-8,-4,-2,-1,0,1,2,4,8,16)):
    bases=torch.arange(len(targets))*neural_steps
    valid=(bases+min(lags)>=0)&(bases+max(lags)<len(timeline))
    frames=valid.nonzero().flatten()
    # Identical target frames at every lag prevent changing sample composition.
    return dict(common_frames=frames.tolist(),
        convention='0 is the preceding-tick forecast; positive offsets use post-event activity and cannot establish anticipation',
        scores={str(lag):group_scores(targets[valid],timeline[bases[valid]+lag],positive,negative)
                for lag in lags} if len(frames) else {})


@torch.no_grad()
def audit(checkpoint,steps,seeds,diagnostic=None):
    if steps<2:
        raise ValueError('at least two frames required')
    before=checksum(checkpoint)
    model=load_checkpoint(checkpoint,evaluation=True,seeds=seeds)
    if set(model.retina.spec['injection'].values())!={'contrast'}:
        raise ValueError('ON/OFF labels require signed contrast injection')
    assert model.plasticity is None
    net=model.network
    sensory=model.retina.injected
    positive,negative=(v[sensory] for v in predictive_support(net))
    targets,forecasts=[],[]
    timeline=[model.previous_predicted[:,sensory].clone()]
    original=net.step
    def record(*args,**kwargs):
        activity=original(*args,**kwargs)
        timeline.append(model.learning_config.encode(activity.predicted,net.config.threshold)[:,sensory].clone())
        return activity
    net.step=record
    camera=EventCamera(len(seeds),model.retina.spec['height'],model.retina.spec['width'])
    for _ in range(steps):
        frame=model.environment.render(model.retina.spec['height'],model.retina.spec['width'])
        targets.append(model.retina.project(camera.observe(frame))[:,sensory])
        forecasts.append(model.previous_predicted[:,sensory].clone())
        model.step('scripted')
    targets,forecasts,timeline=map(torch.stack,(targets,forecasts,timeline))
    assert torch.equal(forecasts,timeline[torch.arange(steps)*model.config.neural_steps])
    strict=group_scores(targets,forecasts,positive,negative)
    permutation=torch.randperm(steps,generator=torch.Generator().manual_seed(421))
    shuffled=group_scores(targets,forecasts[permutation],positive,negative)
    types=[t for t,keep in zip(model.retina.spec['cell_types'],sensory.tolist()) if keep]
    populations={}
    for label in ('L1','L2','L3'):
        mask=torch.tensor([t==label for t in types])
        if mask.any():
            populations[label]=group_scores(targets[:,:,mask],forecasts[:,:,mask],positive[mask],negative[mask])
    assert checksum(checkpoint)==before
    configuration=dict(neurons=asdict(net.config),learning=asdict(model.learning_config),
                       run=asdict(model.config),physics=asdict(model.environment.config),retina=model.retina.spec)
    result=dict(checkpoint_sha256=before,graph_sha256=net.graph.identity(),training_steps=model.training_step,
        configuration_sha256=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest(),
        frames=steps,seeds=seeds,neural_ticks=len(timeline)-1,dt=net.config.dt,
        strict=strict,populations=populations,shuffled=shuffled,permutation_seed=421,
        lagged=lag_scores(targets,timeline,model.config.neural_steps,positive,negative),
        interpretation='Frozen exact scripted M1A stream. No fitting, learning or parameter changes. Structural support includes zero-weight edges and does not imply active/useful sources. Strict scores include first-frame zero forecasts; lagged scores use common frames. Positive lags use post-event activity, never acceptance forecasts. Metrics use float64 accumulation of stored float32 predictions. Final evaluation seeds are not used.')
    if diagnostic is not None:
        result['additional_diagnostic']=diagnostic(targets,timeline,positive,negative,model.config.neural_steps,net.config.dt)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--seeds',default='1101,1102')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=audit(args.checkpoint,args.steps,[int(s) for s in args.seeds.split(',')])
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(training_steps=result['training_steps'],strict=result['strict'])),flush=True)
