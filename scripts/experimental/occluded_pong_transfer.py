"""Check whether learning the rate of adjustment helps when a wall flight is hidden."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from arcade_transfer_session import CHECKPOINT
from hockey_adaptation_rate import RateLearner, batched_life, select_rate


def upper_cover(frames):
    # A fixed cover hides part of the upper-left playing surface. Both paddles
    # remain visible, as does the complete path near the bottom wall.
    frames[:, :32, 5:43] = False
    return frames


def run(practice_file, seed, streams):
    torch.set_num_threads(1)
    practice = json.loads(practice_file.read_bytes())
    observed = practice['common_pong_memory']
    anchor = float(np.clip(observed['weighted_total'] / observed['precision'], 0, 1))
    learned = {label:select_rate(practice[label], anchor) for label in ('hockey', 'pong', 'varied_pong')}
    rates = {**{key:d['rate'] for key,d in learned.items()}, 'fixed_mid':.45, 'latest':1., 'frozen':0.}
    alphas = sorted(set(rates.values()))
    timing = json.loads(CHECKPOINT.read_bytes())['timing']
    rng = np.random.default_rng(seed)
    result = []
    for coefficient in (.41, .73, 1.):
        models = [RateLearner(anchor, alpha) for alpha in alphas for _ in range(streams)]
        histories = {alpha:[] for alpha in alphas}
        for bank_side in (1, -1, -1, 1, -1, -1):
            starts = rng.integers(1, 2**31-1, size=streams).tolist()
            rows = batched_life(models, timing, starts*len(alphas), coefficient, False,
                                view=upper_cover, bank_side=bank_side)
            for i,alpha in enumerate(alphas):
                histories[alpha].append(dict(bank_side=bank_side, lives=rows[i*streams:(i+1)*streams]))
        result.append(dict(coefficient=coefficient, conditions={k:histories[a] for k,a in rates.items()}))
    rollup = []
    for row in result:
        c = {}
        for key,blocks in row['conditions'].items():
            cases = [life for b in blocks if b['bank_side']==-1 for life in b['lives']]
            c[key] = dict(first_returns=sum(l['first_hit'] for l in cases),
                          first_opportunities=sum(l['opportunities']>0 for l in cases),
                          returns=sum(l['hits'] for l in cases), opportunities=sum(l['opportunities'] for l in cases),
                          wins=sum(l['score']>0 for l in cases), losses=sum(l['score']<0 for l in cases))
        rollup.append(dict(coefficient=row['coefficient'], hidden_bank=c))
    return dict(seed=seed, streams=streams, anchor=anchor, fitted=learned, rates=rates, results=result,
                summary=rollup, source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__), Path(__file__).with_name('hockey_adaptation_rate.py'), practice_file, CHECKPOINT]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('practice', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=1638485)
    parser.add_argument('--streams', type=int, default=32)
    args = parser.parse_args()
    data = run(args.practice,args.seed,args.streams)
    with args.output.open('x') as f:
        json.dump(data, f, allow_nan=False,separators=(',',':'))
    print(data['summary'])
