"""Re-enter remapped native Pong and return its online score updates to the same checkpoint."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .tt3_pong import cases_from_seeds, practice_pong, summarize_pong
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_memory import load


def practice(checkpoint, output, receipt, seed, matches, points):
    source = _read(checkpoint)
    _, prior, _, _ = load(checkpoint)
    memory = prior.span_memory
    if memory is None:
        raise ValueError('expected a player with observed response profiles')
    reference = _read(ROOT/'artifacts/tt6/pong-retention.json.gz')['schedule']
    seeds = np.random.default_rng(seed).choice(100_000_000, matches, replace=False) + 630_000_000
    cases = cases_from_seeds(seeds)
    excluded = {r['serve'] for r in reference['evaluation']}
    if any(case.serve in excluded for case in cases):
        raise ValueError('practice overlaps the held-out reference')
    before = memory.state()
    candidates = np.arange(matches) % len(memory.shared.coefficients)
    options = np.random.default_rng(seed).permutation(candidates).tolist()
    rows = practice_pong(cases, prior.roles, memory, options=options, max_ticks=reference['max_ticks'],
                         opponent_speed=reference['opponent_speed'], points_per_match=points, assignment_seed=seed)
    held_rows = _read(ROOT/'artifacts/tt6/pong-retention.json.gz')['runs']['after']['points']
    if {row['case']['serve'] for row in rows} & {row['case']['serve'] for row in held_rows}:
        raise ValueError('continuation serves overlap the reference')
    # Preserve body models, serving and all other hall-side state byte-for-byte
    # in the serialized payload. Only the observed shared and local scores move.
    source['motor_span'] = memory.state()
    source['numeric'] = [a.tolist() for a in prior.numeric]
    _write(output, source)
    _write(receipt, dict(input_sha256=_hash(checkpoint), output_sha256=_hash(output), seed=seed,
                         cases=[asdict(case) for case in cases], before=before, after=memory.state(),
                         rows=rows, summary=summarize_pong(rows)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('receipt', type=Path)
    p.add_argument('--seed', type=int, default=81439531)
    p.add_argument('--matches', type=int, default=16)
    p.add_argument('--points', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(1)
    practice(a.checkpoint, a.output, a.receipt, a.seed, a.matches, a.points)
