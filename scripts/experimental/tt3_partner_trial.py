"""Evaluate a block-and-placement partner with paired physical and camera-only controls."""

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from tt3_arena import TT3Arena, heldout_cases
from tt3_motor_beginner_trial import _geometric_command
from tt3_numeric_transfer import start_memory
from tt3_partner import PracticePartnerArena
from tt3_recovery_learning import DelayMemory, TimingPlayer
from tt3_recovery_trial import aggregate, play
from tt3_visual import RoleModel


def _percentiles(values):
    return np.percentile(values, (0, 10, 50, 90, 100)).tolist() if values else []


def _returns(points):
    rallies = Counter()
    terminals = Counter()
    placement_bins = Counter()
    longitudinal, widths, depths, angles = [], [], [], []
    for point in points:
        events = point['events']
        terminals[events[-1]['reason']] += 1
        ready = False
        shot = None
        for event in events:
            kind, side = event['kind'], event.get('side')
            if kind == 'legal_return' and side == 1:
                ready = True
                rallies['arm_landed'] += 1
            elif kind == 'contact' and side == -1:
                rallies['opponent_contacts'] += 1
                shot = event['outgoing']
            elif kind == 'legal_return' and side == -1:
                ready = False
                rallies['opponent_landed'] += 1
                x, y, _ = event['location']
                forward = -shot[1]
                longitudinal.append(forward)
                angles.append(math.degrees(math.atan2(shot[0], forward)))
                widths.append(x)
                depths.append(-y)
                placement_bins['left' if x < -.20 else 'right' if x > .20 else 'middle'] += 1
                shot = None
            elif kind == 'point' and ready:
                if event['reason'] == 'volley' and event['winner'] == -1:
                    rallies['opponent_interrupted_by_arm_volley'] += 1
                else:
                    phase = 'miss' if shot is None else 'invalid_shot'
                    rallies[f'opponent_{phase}_{event["reason"]}'] += 1
    return dict(rallies, terminal_reasons=dict(terminals), lateral_placements=dict(placement_bins),
                percentiles=dict(levels=(0, 10, 50, 90, 100),
                                 longitudinal_speed_mps=_percentiles(longitudinal),
                                 lateral_bounce_m=_percentiles(widths),
                                 receiver_depth_m=_percentiles(depths),
                                 horizontal_angle_degrees=_percentiles(angles)))


def _player_outcomes(points):
    """Count what ended or continued each actual opening and later chance."""
    results = {stage: Counter() for stage in ('opening', 'later')}
    for point in points:
        stage, touched = 'opening', False
        results[stage]['opportunities'] += 1
        for event in point['events']:
            kind, side = event['kind'], event.get('side')
            if kind == 'legal_return' and side == -1:
                stage, touched = 'later', False
                results[stage]['opportunities'] += 1
            elif kind == 'contact' and side == 1:
                if stage is None:
                    results['later']['before_bounce_contacts'] += 1
                else:
                    touched = True
                    results[stage]['contacts'] += 1
            elif kind == 'legal_return' and side == 1:
                results[stage]['legal_landings'] += 1
                stage = None
            elif kind == 'point' and stage is not None:
                outcome = 'fault' if touched else 'miss'
                results[stage][f'{outcome}_{event["reason"]}'] += 1
    return {stage: dict(result) for stage, result in results.items()}


def _physical(seed, points):
    cases = heldout_cases(f'{seed}/physical-qualification', points)
    results = {}
    for name, factory in (('parked', PracticePartnerArena), ('default_geometry', TT3Arena),
                          ('partner_geometry', PracticePartnerArena)):
        rows = []
        for case in cases:
            arena = factory(case.seed, view_seed=case.view_seed)
            arena.reset(render=False)
            # The same geometry check also supports prepared openings; these
            # two evaluator flags disable its corresponding serve phase here.
            arena.preparation_frames, arena._server_bounce_due = 0, False
            while not arena.done:
                action = (0., 0., 0.) if name == 'parked' else _geometric_command(arena)
                arena.step(action, render=False)
            rows.append(dict(**arena.point_result(), case=asdict(case), events=arena.events))
        summary = dict(points=len(rows), any_contact=sum(p['contacts'] > 0 for p in rows),
                       any_legal_return=sum(p['legal_returns'] > 0 for p in rows),
                       sustained=sum(p['legal_returns'] >= 2 and p['rival_returns'] >= 1 for p in rows),
                       wins=sum(p['points_for'] for p in rows), **_returns(rows),
                       arm_outcomes=_player_outcomes(rows))
        results[name] = dict(summary=summary, rows=rows)
        print(name, summary, flush=True)
    return results


class _RecordedPlayer(TimingPlayer):
    """Record actual visual-policy actions so a winning rally can be replayed."""

    def reset(self, option=None):
        super().reset(option)
        self.trace = []

    def act(self, frames):
        action = super().act(frames)
        self.trace.append(action.tolist())
        return action


def _visual(seed, matches, points, source_roles, numeric):
    trials = dict(trained=[], scratch=[])
    replays = {}
    for match in range(matches):
        started = perf_counter()
        schedule = heldout_cases(f'{seed}/visual-qualification/{match}', points)
        for name, results in trials.items():
            role_model = RoleModel(**source_roles) if name == 'trained' else RoleModel.untrained()
            memory = start_memory(name, numeric, np.random.default_rng(seed + match))
            timing = DelayMemory()
            learner = _RecordedPlayer(role_model, memory, seed=seed + match, timing_memory=timing,
                                      timing_seed=seed + match + 50593, recall=True, forecast=True,
                                      learn_roles=True, require_rebound=False)
            rows = []
            for index, case in enumerate(schedule, 1):
                arena = PracticePartnerArena(case.seed, view_seed=schedule[0].view_seed)
                row = play(arena, learner)
                row.update(case=asdict(case), view_seed=schedule[0].view_seed)
                rows.append(row)
                if (name not in replays and row['points_for'] and row['legal_returns'] >= 2
                        and row['rival_returns']):
                    replays[name] = dict(seed=case.seed, view_seed=schedule[0].view_seed,
                                         partner='block_and_placement', extra_resets=1,
                                         match=match, point=index, actions=learner.trace,
                                         expected=arena.point_result())
            results.append(dict(rows=rows, timing=timing.state(), model=asdict(learner.binder.model),
                                numeric=memory.state()))
            print(name, match, aggregate(rows), flush=True)
        print('match seconds', match, round(perf_counter() - started, 1), flush=True)
    report = {}
    for name, courses in trials.items():
        summary = {}
        for phase, start, stop in (('opening', 0, 1), ('first_eight', 0, 8), ('all', 0, points)):
            rows = [row for course in courses for row in course['rows'][start:stop]]
            summary[phase] = dict(**aggregate(rows), partner=_returns(rows),
                                  arm_outcomes=_player_outcomes(rows),
                                  probes=sum(row['probes'] for row in rows))
        report[name] = dict(summary=summary, courses=courses, replay=replays.get(name))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--physical', type=int, default=0)
    parser.add_argument('--matches', type=int, default=0)
    parser.add_argument('--points', type=int, default=16)
    parser.add_argument('--roles', type=Path, default=Path('artifacts/tt3/pong-source.json'))
    parser.add_argument('--numeric', type=Path, default=Path('artifacts/tt3/numeric-pong-source.json'))
    args = parser.parse_args()
    payload = dict(schema='tt3.practice-partner.v1', seed=args.seed,
                   geometry_cases=args.physical, matches=args.matches, points=args.points)
    if args.physical:
        payload['controls'] = _physical(args.seed, args.physical)
    if args.matches:
        roles = json.loads(args.roles.read_text())['role_model']
        numeric = json.loads(args.numeric.read_text())
        payload.update(roles=roles, numeric=numeric)
        payload['visual'] = _visual(args.seed, args.matches, args.points, roles, numeric)
    files = ('tt3_arena.py', 'tt3_partner.py', 'tt3_partner_trial.py', 'tt3_recovery_learning.py',
             'tt3_recovery_trial.py', 'tt3_visual.py', 'tt3_numeric_transfer.py')
    payload['implementation_sha256'] = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                          for name in files}
    library = os.environ.get('TT3_COMPONENT_LIBRARY')
    if library:
        payload['component_library_sha256'] = sha256(Path(library).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))


if __name__ == '__main__':
    main()
