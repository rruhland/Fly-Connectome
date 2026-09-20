"""Frozen, edge-local signal audit. No fitted predictor or weight updates."""
import argparse
import json
from pathlib import Path

import torch

from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint


class SignalMoments:
    def __init__(self, edges):
        self.count = 0
        self.sums = torch.zeros(8,edges,dtype=torch.float64)

    def add(self, target, trace, prediction):
        y,e,p = target.double(),trace.double(),prediction.double()
        self.count += len(y)
        self.sums += torch.stack((y.sum(0),e.sum(0),y.square().sum(0),e.square().sum(0),
            (y*e).sum(0),(p*e).sum(0),(e.square()*(y!=0)).sum(0),(y!=0).sum(0)))

    def metrics(self):
        y,e,yy,ee,ye,pe,event_energy,events = self.sums / self.count
        covariance = ye-y*e
        scale = ((yy-y.square()).clamp(min=0)*(ee-e.square()).clamp(min=0)).sqrt()
        return dict(covariance=covariance,correlation=torch.where(scale>0,covariance/scale.clamp(min=1e-30),0),
            mean_local_update=ye-pe,mean_target_trace=ye,mean_prediction_trace=pe,
            event_trace_energy_fraction=event_energy/ee.clamp(min=1e-30),
            event_fraction=events,trace_power=ee)


@torch.no_grad()
def audit(checkpoint, steps, seeds):
    if steps<2:
        raise ValueError('at least two frames required; the first frame is excluded')
    before = checksum(checkpoint)
    model = load_checkpoint(checkpoint,evaluation=True,seeds=seeds,warmup=False)
    if model.learning_config.prediction_signature != ('signed-current-v1','input-arrivals-v1','forecast-causal-v1'):
        raise ValueError('audit requires the approved signed event/causal configuration')
    net = model.network
    selected = ((net.pathways==1)&model.retina.injected[net.post]).nonzero().flatten()
    posts = net.post[selected]
    lookup = torch.full((net.e,),-1,dtype=torch.long)
    lookup[selected] = torch.arange(len(selected))
    trace = torch.zeros(len(seeds),len(selected))
    moments = SignalMoments(len(selected))
    maximum_error = torch.zeros(())
    measured_ticks = 0
    original_step = net.step
    def record(injection,**kwargs):
        nonlocal measured_ticks
        activity = original_step(injection,**kwargs)
        # The old trace produced the preceding prediction. Skip the first game
        # frame, whose saved metric prediction is deliberately initialized to zero.
        if model.step_index>0:
            target = model.learning_config.encode(injection,model.config.sensory_gain)
            moments.add(target[:,posts],trace,model.previous_predicted[:,posts])
            measured_ticks += 1
        trace.mul_(net.current_decay[posts])
        local_edges = lookup[activity.arrival_edges]
        keep = local_edges>=0
        trace.index_put_((activity.arrival_environments[keep],local_edges[keep]),
                         net.signs[activity.arrival_edges[keep]],accumulate=True)
        reconstructed = torch.zeros_like(net.voltage)
        reconstructed.index_add_(1,posts,trace*net.magnitudes[selected])
        error = (reconstructed[:,model.retina.injected]-activity.predicted[:,model.retina.injected]).abs().max()
        maximum_error.copy_(torch.maximum(maximum_error,error))
        return activity
    net.step = record
    model.warmup()
    model.run(steps,'scripted')
    assert maximum_error<1e-5, 'edge traces did not reconstruct frozen predictive current'
    assert checksum(checkpoint)==before, 'frozen audit changed source bytes'
    metrics = moments.metrics()
    types = model.retina.spec['cell_types']
    populations = {}
    for label in ('L1','L2','L3'):
        mask = torch.tensor([types[int(post)]==label for post in posts],dtype=torch.bool)
        active = mask&(metrics['trace_power']>1e-12)
        ids = active.nonzero().flatten()
        strongest = ids[torch.argsort(metrics['correlation'][ids],descending=True)[:5]]
        populations[label] = dict(measured_edges=int(mask.sum()),active_edges=int(active.sum()),
            positive_covariance_edges=int((active&(metrics['covariance']>0)).sum()),
            positive_mean_update_edges=int((active&(metrics['mean_local_update']>0)).sum()),
            summed_target_trace=float(metrics['mean_target_trace'][mask].sum()),
            summed_prediction_trace=float(metrics['mean_prediction_trace'][mask].sum()),
            strongest_correlations=[dict(pre_body=int(net.graph.body_ids[net.pre[selected[i]]]),
                post_body=int(net.graph.body_ids[posts[i]]),pre_type=types[int(net.pre[selected[i]])],
                magnitude=float(net.magnitudes[selected[i]]),
                **{name:float(value[i]) for name,value in metrics.items()}) for i in strongest])
    return dict(checkpoint_sha256=before,graph_sha256=net.graph.identity(),training_steps=model.training_step,
        steps=steps,seeds=seeds,measured_ticks=measured_ticks,maximum_reconstruction_error=float(maximum_error),
        populations=populations,
        interpretation='Frozen physical-current moments including warmup traces, not exact initial plasticity state. Positive covariance is descriptive, not held-out predictive success. Mean updates omit eta; they do not change weights. No artificial predictor is fitted.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--seeds',default='1101,1102')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    report=audit(args.checkpoint,args.steps,[int(s) for s in args.seeds.split(',')])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
