"""Static signed-feedback support and exact scripted-event coverage; no fitting."""
import argparse
import json
from pathlib import Path
import torch
from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint


def predictive_support(network):
    positive=torch.zeros(network.n,dtype=torch.bool,device=network.post.device)
    negative=torch.zeros_like(positive)
    predictive=network.pathways==1
    positive[network.post[predictive&(network.signs>0)]]=True
    negative[network.post[predictive&(network.signs<0)]]=True
    return positive,negative


def unreachable_events(target,positive,negative):
    return ((target>0)&~positive)|((target<0)&~negative)


def error_bounds(unreachable,events,samples):
    return dict(event_conditioned_mse_lower_bound=unreachable/events if events else None,
                all_sample_mse_lower_bound=unreachable/samples if samples else None)


@torch.no_grad()
def audit(checkpoint,steps,seeds):
    identity=checksum(checkpoint)
    model=load_checkpoint(checkpoint,evaluation=True,seeds=seeds,warmup=False)
    if set(model.retina.spec['injection'].values())!={'contrast'}:
        raise ValueError('this audit requires signed contrast injection')
    positive,negative=predictive_support(model.network)
    on=torch.zeros_like(model.network.voltage,dtype=torch.int64)
    off=torch.zeros_like(on)
    unreachable=torch.zeros_like(on)
    # Scripted M1A observations are independent of neural computation. Count the
    # same event stream; no neuron/weight updates or replacement predictor.
    for _ in range(steps):
        frame=model.environment.render(model.retina.spec['height'],model.retina.spec['width'])
        target=model.retina.project(model.camera.observe(frame))
        on+=target<0
        off+=target>0
        unreachable+=unreachable_events(target,positive,negative)
        model.environment.step(-(model.environment.ball[:,1]-model.environment.body.position)*10)
    populations={}
    for label in ('L1','L2','L3'):
        mask=torch.tensor([t==label for t in model.retina.spec['cell_types']],dtype=torch.bool)
        events=int((on[:,mask]+off[:,mask]).sum())
        impossible=int(unreachable[:,mask].sum())
        populations[label]=dict(neurons=int(mask.sum()),
            positive_only=int((mask&positive&~negative).sum()),
            negative_only=int((mask&negative&~positive).sum()),
            both_signs=int((mask&positive&negative).sum()),
            neither_sign=int((mask&~positive&~negative).sum()),
            on_events=int(on[:,mask].sum()),off_events=int(off[:,mask].sum()),
            sign_unreachable_events=impossible,
            **error_bounds(impossible,events,steps*len(seeds)*int(mask.sum())))
    assert checksum(checkpoint)==identity
    return dict(checkpoint_sha256=identity,graph_sha256=model.network.graph.identity(),
        frames=steps,seeds=seeds,populations=populations,
        interpretation='Structural sign support uses all existing predictive edges, including zero-weight edges. A missing sign cannot be produced by this current sum from zero initial current and nonnegative magnitudes. The bound counts unit signed events that lack structural support; it is an optimistic expressivity bound, not an achievable learner, timing audit, or M1A acceptance result. First-frame events are included.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--seeds',default='1101,1102')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.steps<1:
        parser.error('positive steps required')
    torch.set_num_threads(1)
    result=audit(args.checkpoint,args.steps,[int(s) for s in args.seeds.split(',')])
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
