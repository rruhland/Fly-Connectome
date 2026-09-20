"""Replay the fixed scripted M1A observation stream without neural computation."""
import argparse
import json
from pathlib import Path

import torch

from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint


@torch.no_grad()
def exposure(model, steps):
    counts=torch.zeros_like(model.network.voltage,dtype=torch.int64)
    for _ in range(steps):
        events=model.camera.observe(model.environment.render(model.retina.spec['height'],model.retina.spec['width']))
        counts += model.retina.project(events)!=0
        # Exactly the approved scripted M1A control, not an SNN observation.
        model.environment.step(-(model.environment.ball[:,1]-model.environment.body.position)*10)
    return counts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps',default='2000,10000')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    model=load_checkpoint(args.checkpoint,evaluation=True,seeds=[args.seed],warmup=False)
    visible=torch.isin(model.retina.neuron_columns,model.retina.pixel_bins.unique())&model.retina.injected
    counts=torch.zeros_like(model.network.voltage,dtype=torch.int64)
    previous=0
    report=dict(checkpoint_sha256=checksum(args.checkpoint),seed=args.seed,measurements=[])
    for steps in sorted(set(int(s) for s in args.steps.split(','))):
        counts += exposure(model,steps-previous)
        populations={}
        for label in ('L1','L2','L3'):
            mask=torch.tensor([t==label for t in model.retina.spec['cell_types']])
            values=counts[:,mask].flatten().float()
            if len(values):
                visible_values=counts[:,mask&visible].flatten().float()
                populations[label]=dict(neurons=len(values),never_stimulated=int((values==0).sum()),
                    fewer_than_ten_events=int((values<10).sum()),median_events=float(values.median()),
                    maximum_events=float(values.max()),total_events=int(values.sum()),
                    neurons_with_render_pixels=len(visible_values),
                    mapped_neurons_never_stimulated=int((visible_values==0).sum()),
                    mapped_neurons_fewer_than_ten_events=int((visible_values<10).sum()),
                    mapped_median_events=float(visible_values.median()) if len(visible_values) else None)
        report['measurements'].append(dict(steps=steps,seconds=steps*model.environment.config.dt,populations=populations))
        previous=steps
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
