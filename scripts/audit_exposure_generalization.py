"""Frozen prediction errors grouped by prior nonzero sensory injection exposure."""
import argparse
import json
from pathlib import Path

import torch

from audit_pong_exposure import exposure
from audit_prediction import audit
from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--initial',required=True)
    parser.add_argument('--exposure-steps',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=load_checkpoint(args.initial,evaluation=True,seeds=[1],warmup=False)
    counts=exposure(source,args.exposure_steps).sum(0)
    sensory=source.retina.injected
    visible=torch.isin(source.retina.neuron_columns,source.retina.pixel_bins.unique())&sensory
    groups=dict(unmapped=sensory&~visible,unseen=visible&(counts==0),
                one_to_nine=visible&(counts>0)&(counts<10),ten_or_more=visible&(counts>=10))
    groups={name:mask for name,mask in groups.items() if mask.any()}
    graph=source.network.graph.identity()
    del source
    report=audit(args.checkpoint,500,[1101,1102],groups)
    if report['graph_sha256']!=graph:
        raise ValueError('exposure and evaluated checkpoints must use the same graph')
    report['exposure']=dict(initial_sha256=checksum(args.initial),training_seed=1,
        training_steps=args.exposure_steps,neurons={name:int(mask.sum()) for name,mask in groups.items()})
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(exposure=report['exposure'],event_groups=report['event_groups'])),flush=True)
