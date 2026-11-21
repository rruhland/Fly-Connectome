"""Replay commands and locate first-return failures using evaluation-only physical state."""
import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np

from .tt4_arena import TT4Arena
from .tt4_transfer_trial import Wiring, _read


def point(wiring, row):
    arena = wiring.arena
    wiring.reset(seed=row['seed'], render=False)
    previous = arena.ball - arena.paddle
    gap = None
    received = hit = landed = False
    reason = None
    for action in row['actions']:
        _, _, _, _, info = wiring.step(action, render=False)
        mujoco.mj_kinematics(arena.model, arena.data)
        difference = arena.ball - arena.paddle
        for event in info['events']:
            if event['kind'] == 'bounce' and event['side'] == 1:
                received = True
            if event['kind'] == 'contact' and event['side'] == 1:
                hit = True
            if event['kind'] == 'legal_return' and event['side'] == 1:
                landed = True
            if event['kind'] == 'point':
                reason = event['reason']
        if received and not hit and previous[1] > 0 >= difference[1] and gap is None:
            alpha = previous[1] / (previous[1] - difference[1])
            gap = ((1 - alpha) * previous + alpha * difference)[[0,2]]
        previous = difference
    if arena.point_result() != row['evaluator']:
        raise ValueError(f"command replay diverged on seed {row['seed']}")
    outcome = 'landed' if landed else ('touch_'+str(reason or 'timeout') if hit else 'miss')
    return outcome, gap


def audit(path, selection):
    doc = _read(path)
    outcomes = Counter()
    offsets = []
    for course in doc['courses']['first_person']:
        if selection not in course['runs']:
            continue
        sched = course['schedule']
        arena = TT4Arena(sched['arena_seed'], view_seed=sched['view_seed'],
                         view='first_person', **doc['geometry']['arena'])
        w = Wiring(arena,sched['arena_seed'])
        try:
            for row in course['runs'][selection]['points']:
                outcome,gap = point(w,row)
                outcomes[outcome] += 1
                if outcome == 'miss' and gap is not None:
                    offsets.append(gap)
        finally:
            arena.close()
    result = dict(outcomes=outcomes, missed_plane_crossings=len(offsets))
    if offsets:
        v = np.asarray(offsets) * 100
        result.update(median_signed_xz_cm=np.median(v,axis=0).tolist(),
                      median_abs_xz_cm=np.median(np.abs(v),axis=0).tolist(),
                      lateral_larger_than_vertical=int((np.abs(v[:,0])>np.abs(v[:,1])).sum()))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('receipt',type=Path)
    ap.add_argument('--condition',default='legacy_mini')
    args = ap.parse_args()
    print(json.dumps(audit(args.receipt,args.condition),indent=2))
