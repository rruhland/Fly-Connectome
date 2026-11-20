"""Evaluate visual command alignment on paired first reaches with equal startup probes."""

import argparse
from dataclasses import asdict, replace
from functools import partial
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from body_effect_learner import BodyLearner
from body_motor_alignment import (BodyMotorAlignment, startup_commands, tracking_command,
                                  visual_effects)
from body_reaching_arena import BODY_SPECS, ReachingArena, _arm_configuration, sample_case
from run_body_transfer import MAX_STEPS, bootstrap, reach, rng_for


CONDITIONS = (('arm', 'arm_remapped'), ('arm', 'arm_geometry'), ('carriage', 'arm'))
COMPARATORS = ('fresh', 'unchanged', 'local_alignment')
_GEOMETRY = replace(BODY_SPECS['arm'], links=(.53, .37),
                    drive=((1.42, .10), (-.16, 2.14)), response=(.70, .65))


class GeometryReachingArena(ReachingArena):
    """Keep the matched visual task and change link lengths and actuator response."""

    def __init__(self, case):
        super().__init__('arm', case, max_steps=MAX_STEPS)
        self.body = 'arm_geometry'
        self._spec = _GEOMETRY
        self._configuration = _arm_configuration(case.start, self._spec.links)
        self._trace = [self._tip_position()]
        self._initial_error = math.dist(self.ground_truth_position, case.target)
        self._error = self._initial_error


def _rollout(body, case, model, predict, probes, rng):
    arena = (GeometryReachingArena(case) if body == 'arm_geometry'
             else ReachingArena(body, case, max_steps=MAX_STEPS))
    position, target = arena.observation()
    actions = sha256()
    probe_path = [position.tolist()]
    error_fraction_at_8 = None
    model.reset_episode()
    while not arena.done:
        probing = arena.steps < len(probes)
        if probing:
            action = probes[arena.steps]
        else:
            effects, drift = predict(position)
            action = tracking_command(position, target, effects, drift, rng)
        next_position, target = arena.step(action)
        model.observe(position, action, next_position)
        position = next_position
        actions.update(np.asarray(action, dtype='<f8').tobytes())
        if probing:
            probe_path.append(position.tolist())
        if arena.steps == 8:
            error_fraction_at_8 = arena.normalized_error
    return dict(success=arena.success, commands=arena.steps, probe_path=probe_path,
                updates=model.updates, error_fraction_at_8=(arena.normalized_error
                    if error_fraction_at_8 is None else error_fraction_at_8),
                final_error=arena.error, weighted_normalized_error=arena.weighted_normalized_error,
                actions_sha256=actions.hexdigest())


def _comparison(body, case, prior, probes, seed, index):
    aligned = BodyMotorAlignment(prior)
    local = BodyMotorAlignment(prior, spatial=False)
    fresh = BodyLearner()
    unchanged = BodyLearner(prior=prior, updates_enabled=False)
    models = {
        'aligned': (aligned, aligned.visual_effects),
        'local_alignment': (local, local.visual_effects),
        'fresh': (fresh, partial(visual_effects, fresh)),
        'unchanged': (unchanged, partial(visual_effects, unchanged)),
    }
    rows = {name: _rollout(body, case, model, predict, probes, rng_for(seed, index, 4))
            for name, (model, predict) in models.items()}
    prefix = rows['aligned']['probe_path']
    if len(prefix) != len(probes) + 1 or any(row['probe_path'] != prefix for row in rows.values()):
        raise AssertionError('all comparators must receive the same complete probe trajectory')
    if any(model.export_prior() != prior for model in (aligned.source, local.source, unchanged)):
        raise AssertionError('the carried source field changed')
    rows['aligned']['final_adapter'] = aligned.adapter.tolist()
    rows['local_alignment']['final_adapter'] = local.adapter.tolist()
    return rows


def _summarize(streams):
    summary = {}
    for source, target in CONDITIONS:
        condition = f'{source}_to_{target}'
        paired = [row['conditions'][condition] for row in streams]
        comparisons = {}
        for name in ('aligned', *COMPARATORS):
            rows = [group[name] for group in paired]
            comparisons[name] = dict(
                successes=sum(row['success'] for row in rows),
                mean_capped_commands=float(np.mean([row['commands'] for row in rows])),
                mean_error_fraction_at_8=float(np.mean([row['error_fraction_at_8'] for row in rows])),
                mean_weighted_normalized_error=float(np.mean([row['weighted_normalized_error'] for row in rows])),
            )
        paired_commands_saved = {}
        for name in COMPARATORS:
            saved = np.array([group[name]['commands'] - group['aligned']['commands'] for group in paired])
            paired_commands_saved[name] = dict(mean=float(saved.mean()), ci95=bootstrap(saved),
                ahead=int((saved > 0).sum()), tied=int((saved == 0).sum()), behind=int((saved < 0).sum()))
        summary[condition] = dict(policies=comparisons, aligned_commands_saved=paired_commands_saved)
    return summary


def run(seed, starts, practice):
    streams = []
    for index in range(starts):
        course_rng = rng_for(seed, index, 1)
        course = [sample_case(course_rng) for _ in range(practice)]
        policy_seed = int(rng_for(seed, index, 2).integers(2**31))
        checkpoints = {}
        source_results = {}
        for source in ('arm', 'carriage'):
            learner = BodyLearner(seed=policy_seed)
            rows = [reach(learner, source, case) for case in course]
            checkpoints[source] = learner.export_prior()
            source_results[source] = dict(successes=sum(row['success'] for row in rows),
                                          commands=sum(row['commands'] for row in rows))
        case = sample_case(rng_for(seed, index, 3))
        probes = startup_commands(rng_for(seed, index, 5))
        conditions = {f'{source}_to_{target}': _comparison(target, case, checkpoints[source], probes, seed, index)
                      for source, target in CONDITIONS}
        streams.append(dict(index=index, case=asdict(case), probes=probes.tolist(),
                            practice=source_results, source_checkpoints=checkpoints, conditions=conditions))
    path = Path(__file__).parent
    files = ('body_effect_learner.py', 'body_motor_alignment.py', 'body_reaching_arena.py',
             'run_body_motor_alignment.py', 'run_body_transfer.py')
    return dict(schema='paired-body-motor-alignment-v1', seed=seed, starts=starts, practice=practice,
                max_steps=MAX_STEPS, geometry=asdict(_GEOMETRY),
                source_sha256={name: sha256((path / name).read_bytes()).hexdigest() for name in files},
                summary=_summarize(streams), streams=streams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--starts', type=int, default=32)
    parser.add_argument('--practice', type=int, default=24)
    args = parser.parse_args()
    if min(args.starts, args.practice) < 1 or args.seed < 0:
        parser.error('provide a non-negative seed and positive start and practice counts')
    results = run(args.seed, args.starts, args.practice)
    with args.output.open('x') as handle:
        json.dump(results, handle, separators=(',', ':'), allow_nan=False)
    print(json.dumps(results['summary'], indent=2))
