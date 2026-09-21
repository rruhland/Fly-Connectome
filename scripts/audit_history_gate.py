"""Approved Gate A: fixed trace candidates versus causal local-history controls."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from audit_long_horizons import next_events
from audit_trace_horizons import referenced_event_counts
from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint

PAIRS=((50062,51452),(75482,71110),(42416,44723))


def local_history(targets,horizon):
    last=torch.zeros_like(targets[0])
    age=torch.full_like(last,horizon,dtype=torch.int64)
    histories,ages,strata=[],[],[]
    for event in targets:
        age=torch.where(event!=0,0,(age+1).clamp(max=horizon))
        last=torch.where(event!=0,event,last)
        strata.append(torch.where(last==0,0,1+(last>0).long()*(horizon+1)+age))
        histories.append(last.clone()); ages.append(age.clone())
    return torch.stack(strata),torch.stack(histories),torch.stack(ages)


def conditional_shuffle(labels,strata,seed):
    result=labels.clone()
    generator=torch.Generator().manual_seed(seed)
    groups=[]
    permutable_count=0
    for value in strata.unique(sorted=True).tolist():
        indices=(strata==value).nonzero().flatten()
        values,counts=labels[indices].unique(return_counts=True)
        permutable=len(values)>1
        if permutable:
            result[indices]=labels[indices[torch.randperm(len(indices),generator=generator)]]
            permutable_count+=len(indices)
        groups.append(dict(stratum=value,windows=len(indices),permutable=permutable,
                           label_counts={str(float(v)):int(c) for v,c in zip(values,counts)}))
    return result,groups,permutable_count/len(labels)


def statistics(target,trace):
    y,e=target.double(),trace.double()
    covariance=((y-y.mean())*(e-e.mean())).mean()
    scale=(y.var(unbiased=False)*e.var(unbiased=False)).sqrt()
    return dict(covariance=float(covariance),correlation=float(covariance/scale) if scale>0 else 0.,
                target_trace=float((y*e).mean()),trace_power=float(e.square().mean()))


def gate_status(statuses):
    if sum(all(s=='pass' for s in pair) for pair in statuses)>=2:
        return 'pass'
    if sum(all(s!='fail' for s in pair) for pair in statuses)>=2:
        return 'inconclusive'
    return 'fail'


@torch.no_grad()
def audit(checkpoint,steps=2000,seeds=(1103,1104),pairs=PAIRS):
    before=checksum(checkpoint)
    model=load_checkpoint(checkpoint,evaluation=True,seeds=list(seeds),warmup=False)
    assert model.plasticity is None
    assert model.learning_config.prediction_signature==('signed-current-v1','input-arrivals-v1','forecast-causal-v1')
    assert set(model.retina.spec['injection'].values())=={'contrast'}
    net=model.network
    index={int(body):i for i,body in enumerate(net.graph.body_ids)}
    posts=torch.tensor([index[post] for _,post in pairs])
    candidates=[]
    for pre,post in pairs:
        found=((net.pre==index[pre])&(net.post==index[post])&(net.pathways==1)).nonzero().flatten()
        if len(found)!=1:
            raise ValueError('fixed candidate must be exactly one measured predictive edge')
        candidates.append(int(found[0]))
    edges=((net.pathways==1)&torch.isin(net.post,posts)).nonzero().flatten()
    lookup=torch.full((net.e,),-1,dtype=torch.long)
    lookup[edges]=torch.arange(len(edges))
    candidate_slots=lookup[torch.tensor(candidates)]
    trace=torch.zeros(len(seeds),len(edges))
    targets,traces,predictions=[],[],[]
    maximum_error=0.
    original=net.step
    N=model.config.neural_steps
    def record(injection,**kwargs):
        nonlocal maximum_error
        tick=net.step_index-model.config.warmup_steps
        if tick>=0 and tick%N==0:
            targets.append((injection[:,posts]/model.config.sensory_gain).clone())
        activity=original(injection,**kwargs)
        trace.mul_(net.current_decay[net.post[edges]])
        ids=lookup[activity.arrival_edges]
        keep=ids>=0
        trace.index_put_((activity.arrival_environments[keep],ids[keep]),net.signs[activity.arrival_edges[keep]],accumulate=True)
        if tick>=0 and (tick+1)%N==0:
            traces.append(trace[:,candidate_slots].clone())
            predictions.append(model.learning_config.encode(activity.predicted,net.config.threshold)[:,posts].clone())
            reconstructed=torch.zeros_like(net.voltage)
            reconstructed.index_add_(1,net.post[edges],trace*net.magnitudes[edges])
            maximum_error=max(maximum_error,float((reconstructed[:,posts]-activity.predicted[:,posts]).abs().max()))
        return activity
    net.step=record
    model.warmup()
    model.run(steps,'scripted')
    targets,traces,predictions=map(torch.stack,(targets,traces,predictions))
    assert len(targets)==len(traces)==len(predictions)==steps
    assert maximum_error<1e-5
    assert checksum(checkpoint)==before
    horizon=round(.1/(N*net.config.dt))
    start,end=horizon,steps-horizon
    if end<=start:
        raise ValueError('complete common windows required')
    labels,waits=next_events(targets,horizon)
    counts=referenced_event_counts(waits,start,end)
    strata,last,age=local_history(targets,horizon)
    rows=[]
    for pair_index,pair in enumerate(pairs):
        results=[]
        for env,seed in enumerate(seeds):
            y=labels[start:end,env,pair_index]
            e=traces[start:end,env,pair_index]
            history=strata[start:end,env,pair_index]
            actual=statistics(y,e)
            shuffles=[]
            for shuffle_seed in range(421,431):
                shuffled,groups,fraction=conditional_shuffle(y,history,shuffle_seed)
                shuffles.append(dict(seed=shuffle_seed,**statistics(shuffled,e)))
            max_covariance=max(s['covariance'] for s in shuffles)
            enough=int(counts[env,pair_index])>=5 and fraction>0
            passes=(actual['trace_power']>1e-10 and actual['correlation']>=.05 and
                    actual['target_trace']>0 and actual['covariance']>max_covariance)
            old=last[start:end,env,pair_index]
            recent=torch.where(age[start:end,env,pair_index]<horizon,old,0.)
            controls=dict(zero=torch.zeros_like(y),model=predictions[start:end,env,pair_index],
                frame_persistence=targets[start:end,env,pair_index],last_polarity=old,opposite_last_polarity=-old,
                expiring_last_polarity=recent,expiring_opposite_polarity=-recent)
            results.append(dict(seed=seed,status=('inconclusive' if not enough else 'pass' if passes else 'fail'),
                distinct_events=int(counts[env,pair_index]),permutable_window_fraction=fraction,
                actual=actual,shuffles=shuffles,strata=groups,
                controls={name:dict(all_mse=float((y.double()-p.double()).square().mean()),
                    event_mse=float((y[y!=0].double()-p[y!=0].double()).square().mean()) if (y!=0).any() else None,
                    quiet_mse=float(p[y==0].double().square().mean()) if (y==0).any() else None)
                    for name,p in controls.items()}))
        rows.append(dict(pre_body=pair[0],post_body=pair[1],results=results))
    configuration=dict(neurons=asdict(net.config),learning=asdict(model.learning_config),run=asdict(model.config),
                       physics=asdict(model.environment.config),retina=model.retina.spec)
    return dict(checkpoint_sha256=before,graph_sha256=net.graph.identity(),training_steps=model.training_step,
        configuration_sha256=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest(),
        steps=steps,seeds=list(seeds),common_frames=[start,end],horizon_frames=horizon,
        maximum_reconstruction_error=maximum_error,pairs=rows,
        status=gate_status([[r['status'] for r in row['results']] for row in rows]),
        interpretation='Approved fixed-candidate Gate A. Frame-end traces; first future local event in next12 frames (100ms frame window). No incomplete tail windows. Unique referenced events, not repeated labels. Histories include the decision-frame event, with capped age and never-seen state. Ten within-history shuffles, separately per seed/post. Permutable fraction counts windows in groups containing multiple labels, not actual moved labels. Exploratory gate, not statistical significance. No fitting, learning or checkpoint writes; final seeds unused.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=audit(args.checkpoint)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],pairs=[dict(pre_body=p['pre_body'],post_body=p['post_body'],
        results=[{k:r[k] for k in ('seed','status','distinct_events','permutable_window_fraction','actual')} for r in p['results']])
        for p in result['pairs']])),flush=True)
