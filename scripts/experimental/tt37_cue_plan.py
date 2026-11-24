"""Freeze entirely new random openings and distinct random continuations for the earlier gate."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import card, opening_candidates
from .tt37_opponent_benchmark import OUT, PLAN


def prepare(feeds, output):
    original = _read(PLAN)
    if _hash(feeds) != original['feed_sha256']:
        raise ValueError('Measured feeds changed')
    forbidden = set(original['exclusions']) | {c['id'] for c in original['held_openings']}
    compatible = [s['id'] for s in opening_candidates(measured_launches(feeds))
                  if s['id'] not in forbidden][:180]
    shots = np.random.default_rng(379630).choice(compatible, 18, replace=False).tolist()
    first = [card(s, 'full', i, 497201+127*i) | dict(profile='reach', reply_seed=499301+181*i)
             for i, s in enumerate(shots)]
    later = [c | dict(reply_seed=502001+197*i, key=f'cue-earlier-reply-{i}-{c["id"]}')
             for i, c in enumerate(original['held_replies'])]
    _write(output, dict(schema='tt37-cue-earlier-plan-1', feed_sha256=original['feed_sha256'],
                        protocol='random', parent_plan_sha256=_hash(PLAN),
                        selection=('Opening draws use previously unseen compatible measured releases and new seeds. '
                                   'Seeded anchors reproduce the four useful openings by design. '
                                   'Every subsequent placement draws from a new unevaluated continuation seed.'),
                        held_openings=first, held_replies=later))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('feeds', type=Path)
    p.add_argument('--output', type=Path, default=OUT / 'cue-earlier-plan.json')
    args = p.parse_args()
    prepare(args.feeds, args.output)
