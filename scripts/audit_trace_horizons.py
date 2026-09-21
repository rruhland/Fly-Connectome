"""Frozen edge-local trace alignment, with discovery/validation development seeds."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from audit_long_horizons import next_events
from fly_connectome.data import checksum
from fly_connectome.pong import Pong
from fly_connectome.sensor import EventCamera
from fly_connectome.training import load_checkpoint


class TraceMoments:
    def __init__(self,batch,edges):
        self.sums=torch.zeros(6,batch,edges,dtype=torch.float64)
        self.count=0

    def add(self,target,trace):
        y,e=target.double(),trace.double()
        self.sums+=torch.stack((y,e,y.square(),e.square(),y*e,(y!=0).double()))
        self.count+=1

    def metrics(self):
        y,e,yy,ee,ye,events=self.sums/self.count
        covariance=ye-y*e
        scale=((yy-y.square()).clamp(min=0)*(ee-e.square()).clamp(min=0)).sqrt()
        return dict(covariance=covariance,correlation=torch.where(scale>0,covariance/scale.clamp(min=1e-30),0),
                    target_trace=ye,trace_power=ee,events=events*self.count)


def earlier_trace(ring,target_tick,lead):
    return ring[(target_tick-lead+1)%len(ring)]


def referenced_event_counts(waits,start,end):
    batch,neurons=waits.shape[1:]
    frame,env,neuron=(waits[start:end]>0).nonzero(as_tuple=True)
    event=frame+start+waits[frame+start,env,neuron]
    keys=(event*(batch*neurons)+env*neurons+neuron).unique()
    return torch.bincount(keys.remainder(batch*neurons),minlength=batch*neurons).view(batch,neurons)


def summarize(real,shuffled,event_counts,net,edges,types):
    a,b=real.metrics(),shuffled.metrics()
    eligible=(event_counts>=5)&(a['trace_power']>1e-10)
    useful=eligible&(a['correlation']>=.05)&(a['covariance']>b['covariance'])&(a['target_trace']>0)
    populations={}
    for label in ('L1','L2','L3'):
        mask=torch.tensor([types[int(p)]==label for p in net.post[edges]])
        discovery=(mask&useful[0]).nonzero().flatten()
        selected=discovery[torch.argsort(a['correlation'][0,discovery],descending=True)[:10]]
        validated=selected[useful[1,selected]]
        rows=[]
        for i in selected.tolist():
            edge=int(edges[i])
            rows.append(dict(pre_body=int(net.graph.body_ids[net.pre[edge]]),post_body=int(net.graph.body_ids[net.post[edge]]),
                pre_type=types[int(net.pre[edge])],magnitude=float(net.magnitudes[edge]),
                discovery_correlation=float(a['correlation'][0,i]),validation_correlation=float(a['correlation'][1,i]),
                discovery_shuffle_correlation=float(b['correlation'][0,i]),validation_shuffle_correlation=float(b['correlation'][1,i]),
                target_event_counts=event_counts[:,i].tolist(),validation_pass=bool(useful[1,i])))
        populations[label]=dict(measured_edges=int(mask.sum()),eligible_edges_by_seed=eligible[:,mask].sum(1).tolist(),
            discovery_candidates=len(discovery),selected=len(selected),validation_passes=len(validated),
            distinct_validated_posts=len(net.post[edges[validated]].unique()),candidates=rows)
    return populations


@torch.no_grad()
def audit(checkpoint,steps=2000,seeds=(1101,1102)):
    if len(seeds)!=2:
        raise ValueError('exactly two development seeds required')
    before=checksum(checkpoint)
    model=load_checkpoint(checkpoint,evaluation=True,seeds=list(seeds),warmup=False)
    assert model.plasticity is None
    assert model.learning_config.prediction_signature==('signed-current-v1','input-arrivals-v1','forecast-causal-v1')
    assert set(model.retina.spec['injection'].values())=={'contrast'}
    net=model.network
    dt,N=net.config.dt,model.config.neural_steps
    leads={'one_tick':1,**{str(ms):round(ms/(1000*dt)) for ms in (30,50,100)}}
    horizon=round(.1/(N*dt))
    start=(max(leads.values())+N-1)//N
    end=steps-horizon
    if end<=start:
        raise ValueError('sequence too short for complete common windows')
    sensory=model.retina.injected
    edges=((net.pathways==1)&sensory[net.post]).nonzero().flatten()
    posts=net.post[edges]
    sensory_index=torch.full((net.n,),-1,dtype=torch.long)
    sensory_index[sensory]=torch.arange(int(sensory.sum()))
    slots=sensory_index[posts]
    env=Pong(list(seeds),model.environment.config)
    camera=EventCamera(len(seeds),model.retina.spec['height'],model.retina.spec['width'])
    targets=[]
    for _ in range(steps):
        frame=env.render(model.retina.spec['height'],model.retina.spec['width'])
        targets.append(model.retina.project(camera.observe(frame))[:,sensory])
        env.step(-(env.ball[:,1]-env.body.position)*10)
    targets=torch.stack(targets)
    future,waits=next_events(targets,horizon)
    frames=torch.arange(start,end)
    permutation=frames[torch.randperm(len(frames),generator=torch.Generator().manual_seed(421))]
    moments={name:(TraceMoments(len(seeds),len(edges)),TraceMoments(len(seeds),len(edges))) for name in (*leads,'next_event')}
    trace=torch.zeros(len(seeds),len(edges))
    ring=torch.zeros(max(leads.values())+1,*trace.shape)
    lookup=torch.full((net.e,),-1,dtype=torch.long)
    lookup[edges]=torch.arange(len(edges))
    error=0.
    original=net.step
    def record(injection,**kwargs):
        nonlocal error
        f=model.step_index
        # Warmup is identified separately because it also has step_index zero.
        game_tick=net.step_index-model.config.warmup_steps
        if game_tick>=0 and game_tick%N==0:
            assert torch.equal(injection[:,sensory]/model.config.sensory_gain,targets[f])
            if start<=f<end:
                shuffled_frame=int(permutation[f-start])
                for name,lead in leads.items():
                    e=earlier_trace(ring,net.step_index,lead)
                    moments[name][0].add(targets[f,:,slots],e)
                    moments[name][1].add(targets[shuffled_frame,:,slots],e)
        activity=original(injection,**kwargs)
        trace.mul_(net.current_decay[posts])
        ids=lookup[activity.arrival_edges]
        keep=ids>=0
        trace.index_put_((activity.arrival_environments[keep],ids[keep]),net.signs[activity.arrival_edges[keep]],accumulate=True)
        ring[net.step_index%len(ring)].copy_(trace)
        if game_tick>=0 and (game_tick+1)%N==0:
            reconstructed=torch.zeros_like(net.voltage)
            reconstructed.index_add_(1,posts,trace*net.magnitudes[edges])
            error=max(error,float((reconstructed[:,sensory]-activity.predicted[:,sensory]).abs().max()))
            if start<=f<end:
                shuffled_frame=int(permutation[f-start])
                moments['next_event'][0].add(future[f,:,slots],trace)
                moments['next_event'][1].add(future[shuffled_frame,:,slots],trace)
        return activity
    net.step=record
    model.warmup()
    model.run(steps,'scripted')
    assert error<1e-5
    assert checksum(checkpoint)==before
    fixed_counts=(targets[start:end]!=0).sum(0)[:,slots]
    future_counts=referenced_event_counts(waits,start,end)[:,slots]
    configuration=dict(neurons=asdict(net.config),learning=asdict(model.learning_config),run=asdict(model.config),
                       physics=asdict(model.environment.config),retina=model.retina.spec)
    return dict(checkpoint_sha256=before,graph_sha256=net.graph.identity(),training_steps=model.training_step,
        configuration_sha256=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest(),
        steps=steps,seeds=list(seeds),common_frames=[start,end],samples_per_seed=len(frames),
        leads_ms={name:lead*dt*1000 for name,lead in leads.items()},next_event_window_frames=horizon,
        maximum_reconstruction_error=error,
        trials={name:summarize(real,shuffled,future_counts if name=='next_event' else fixed_counts,
                              net,edges,model.retina.spec['cell_types']) for name,(real,shuffled) in moments.items()},
        interpretation='Frozen physical signed traces, including warmup. Frame-boundary targets for fixed leads; frame-end traces for next-event windows. First seed selects at most10 edges/population by correlation, second checks those same edges. Require >=5 distinct target events, trace power>1e-10, correlation>=0.05, positive target-trace moment, and covariance greater than one fixed timing shuffle in each seed. Next-event counts use distinct underlying events, not repeated labels. Exploratory multiple comparisons, not significance or M1A acceptance. No fitting or parameter changes; final seeds unused.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=audit(args.checkpoint,args.steps)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:{p:{k:v[k] for k in ('discovery_candidates','validation_passes','distinct_validated_posts')}
        for p,v in pops.items()} for name,pops in result['trials'].items()}),flush=True)
