"""Apply a fixed Pong-informed readout after an independent change of embodiment."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_effects import transmission
from .tt24_recurrence_arm import EMBODIMENT
from .tt24_remap_game import run_policy
from .tt24_retinal_map import calibrate, decoder


DIR=Path('artifacts/tt24')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=60158723)
    p.add_argument('--held',type=int,default=24)
    args=p.parse_args()
    with gzip.open(DIR/'recurrence-arm.json.gz','rt') as f:
        fitted=json.load(f)
    with gzip.open(EMBODIMENT,'rt') as f:
        original=json.load(f)
    with (DIR/'retinal-mixed.json').open() as f:
        earlier=json.load(f)
    source=np.array(earlier['source'])
    rng=np.random.default_rng(args.seed)
    wire=transmission(rng)
    starts=rng.integers(1,2**31-1,args.held).tolist()
    arena=BounceArena(view='wrist',width=64,height=48,arm='alternate')
    results={}
    try:
        target,cost=calibrate(arena,wire,args.seed+541,'changed')
        learned=decoder(source,target,.6)
        identity=np.eye(5)
        for mode in ('pong','cold','shuffled','warm','uncorrected_pong','original_wiring_pong','hold','blind_pong'):
            key=mode if mode in fitted['policies'] else 'pong'
            record=dict(weights=fitted['policies'][key],motor_effects=original['motor_effects'],
                        stride=2,readout='rate',retinal='absolute')
            physical=identity if mode=='original_wiring_pong' else wire
            inverse=identity if mode in ('uncorrected_pong','original_wiring_pong') else learned
            scores=[run_policy(arena,s,record,inverse,physical,regime='changed',
                               hold=mode=='hold',blind=mode=='blind_pong') for s in starts]
            results[mode]=scores
            print(mode,sum(x['rebounds'] for x in scores),max(x['consecutive'] for x in scores),flush=True)
        receipt=dict(starts=starts,source_checkpoint='recurrence-arm.json.gz',
                     nominal_body_data='retinal-mixed.json',target_cost=cost,source_cost=earlier['source_cost'],
                     wire=wire.tolist(),decoder=learned.tolist(),results=results)
        args.output.write_text(json.dumps(receipt,separators=(',',':'))+'\n')
    finally:
        arena.close()


if __name__=='__main__':
    main()
