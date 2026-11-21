"""Select a native actuator identification schedule from independently observed Pong scores."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from tt3_numeric_transfer import PortableMemory
from tt3_pong import (_VisualActionEffects, cases_from_seeds, evaluate_pong, summarize_pong,
                      paired_pong)
from tt3_visual import RoleModel


class ShorterDwell(_VisualActionEffects):
    dwell_frames = 3
    minimum_uses = 4
    minimum_observations = 12


class EarlyEvidence(_VisualActionEffects):
    minimum_uses = 3
    minimum_observations = 10


class QuickSequence(_VisualActionEffects):
    dwell_frames = 3
    minimum_uses = 3
    minimum_observations = 10


CANDIDATES = (_VisualActionEffects, EarlyEvidence, ShorterDwell, QuickSequence)


def evaluate(cases, role_state, memory_state, factory, points):
    memory = PortableMemory.restore(memory_state)
    rows = evaluate_pong(cases, RoleModel(**role_state), memory, max_ticks=1440,
                         points_per_match=points, motor_factory=factory)
    return dict(rows=rows, summary=summarize_pong(rows), payoff=sum(row['score'] for row in rows),
                opening_probe_frames=float(np.mean([r['visual_probe_frames'] for r in rows if r['point']==1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roles',type=Path)
    parser.add_argument('memory',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--seed',type=int,default=82170613)
    parser.add_argument('--starts',type=int,default=22)
    parser.add_argument('--points',type=int,default=2)
    parser.add_argument('--target',type=Path)
    args=parser.parse_args()
    roles=json.loads(args.roles.read_text())['role_model']
    memory=json.loads(args.memory.read_text())
    torch.set_num_threads(1)
    rng=np.random.default_rng(args.seed)
    source=cases_from_seeds(rng.choice(100_000_000,args.starts,replace=False)+1410_000_000)
    check=cases_from_seeds(rng.choice(100_000_000,40,replace=False)+1560_000_000)
    runs={}
    for factory in CANDIDATES:
        record=evaluate(source,roles,memory,factory,args.points)
        runs[factory.__name__]=record
        print(factory.__name__,record['payoff'],round(record['opening_probe_frames'],2),flush=True)
    select=max(CANDIDATES,key=lambda f:runs[f.__name__]['payoff'])
    held={}
    for name,fac in [('old',_VisualActionEffects),('selected',select)]:
        held[name]=evaluate(check,roles,memory,fac,5)
        print(name,held[name]['summary'],flush=True)
    result=dict(seed=args.seed,training=runs,chosen=select.__name__,holdout=held,
                paired=paired_pong(held['old']['rows'],held['selected']['rows']))
    with gzip.open(args.output,'wt') as f:
        json.dump(result,f,separators=(',',':'),allow_nan=False)
    if args.target:
        with gzip.open(args.target,'rt') as f: target=json.load(f)['matches'][0]['trained']
        after=evaluate(check,target['roles'],target['memory'],select,5)
        result['post_table']=dict(measured=after,paired=paired_pong(held['selected']['rows'],after['rows']))
        with gzip.open(args.output,'wt') as f:
            json.dump(result,f,separators=(',',':'),allow_nan=False)


if __name__=='__main__':
    main()
