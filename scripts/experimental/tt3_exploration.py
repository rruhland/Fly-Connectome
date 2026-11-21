"""Learn an exploration dwell in native Pong and evaluate it in table tennis."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experimental import tt3_arena, tt3_learning, tt3_pong, tt3_visual


PROBE_FRAMES = 16
DWELLS = (1, 2, 4)
NATIVE_CAMERA_TICKS = 2
SOURCE_TRAIN_SEED = 782144393
SOURCE_CHECK_SEED = 716801570
TARGET_SEED = 929817065
ARMS = ('prior_dwell', 'scratch_iid', 'fixed_orthogonal', 'fixed_two_frame', 'fixed_default')


def _balanced_actions(seed, dwell, frames=PROBE_FRAMES):
    rng = np.random.default_rng(seed)
    blocks = []
    while len(blocks) * dwell < frames:
        group = rng.permutation(tt3_pong.ACTIONS).tolist()
        if blocks and group[0] == blocks[-1]:
            group = group[1:] + group[:1]
        blocks.extend(group)
    return np.repeat(blocks, dwell)[:frames].tolist()


class _ProbePractice:
    def __init__(self, seed, dwell):
        self.requests = iter(_balanced_actions(seed, dwell))
        self.scene = tt3_visual._SceneHistory()
        self.witness_frames = []
        self.contrasts = []

    def choose(self):
        return next(self.requests)

    def observe(self, image, preceding_action, reward):
        tracks = self.scene.update(image, preceding_action)
        _, effects = tt3_visual._effects(self.scene, tracks)
        contrasts = sorted((item.features[0] * item.features[1] for item in effects), reverse=True)
        if len(contrasts) < 2:
            return
        best, next_best = contrasts[:2]
        self.contrasts.append(float(best - next_best))
        if best > .035 and best > 1.8 * next_best:
            self.witness_frames.append(self.scene.tick)

    def result(self):
        return dict(first_witness=next(iter(self.witness_frames), None),
                    witness_frames=self.witness_frames, mean_contrast=float(
                        sum(self.contrasts) / max(1, len(self.contrasts))))


def _source_cases(seed, count):
    rng = np.random.default_rng(seed)
    return tt3_pong.cases_from_seeds(rng.choice(2_147_483_646, count, replace=False) + 1)


def _source_comparison(cases):
    runs = []
    for dwell in DWELLS:
        agents = [_ProbePractice(case.exploration, dwell) for case in cases]
        outcomes = tt3_pong._native_rallies(cases, agents,
            max_ticks=PROBE_FRAMES * NATIVE_CAMERA_TICKS, opponent_speed=.35,
            camera_ticks=NATIVE_CAMERA_TICKS)
        for outcome, agent in zip(outcomes, agents, strict=True):
            runs.append(dict(**outcome, **agent.result(), dwell=dwell))
    scores = []
    for dwell in DWELLS:
        subset = [run for run in runs if run['dwell'] == dwell]
        first = [run['first_witness'] or PROBE_FRAMES + 1 for run in subset]
        scores.append(dict(dwell=dwell, witnesses=sum(value <= PROBE_FRAMES for value in first),
                           mean_cameras_to_witness_or_miss=float(np.mean(first)),
                           mean_contrast=float(np.mean([run['mean_contrast'] for run in subset]))))
    return dict(scores=scores, runs=runs)


def learn_probe_dwell(train_count=24, check_count=16):
    """Fit a single temporal exploration parameter on native image-action histories."""
    training = _source_comparison(_source_cases(SOURCE_TRAIN_SEED, train_count))
    chosen = min(training['scores'], key=lambda entry: (
        entry['mean_cameras_to_witness_or_miss'], -entry['mean_contrast'], entry['dwell']))
    check = _source_comparison(_source_cases(SOURCE_CHECK_SEED, check_count))
    return dict(chosen_dwell=chosen['dwell'], training=training, check=check,
                train_seed=SOURCE_TRAIN_SEED, check_seed=SOURCE_CHECK_SEED,
                native_camera_ticks=NATIVE_CAMERA_TICKS, probe_frames=PROBE_FRAMES)


def probe_actions(arm, seed, *, learned_dwell):
    """Draw balanced joint probes without using observed object IDs or coordinates."""
    if arm == 'fixed_default':
        return None
    rng = np.random.default_rng(seed)
    if arm == 'scratch_iid':
        rows = rng.choice((-1., 1.), (PROBE_FRAMES, 3))
    else:
        dwell = {'prior_dwell': learned_dwell, 'fixed_orthogonal': 1,
                 'fixed_two_frame': 2}[arm]
        if dwell not in DWELLS:
            raise ValueError('dwell must be a fitted or fixed candidate')
        orthogonal = np.array(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)))
        orthogonal = orthogonal[:, rng.permutation(3)] * rng.choice((-1, 1), 3)
        blocks = np.concatenate([orthogonal[rng.permutation(4)]
                                 for _ in range(PROBE_FRAMES // (4 * dwell))])
        rows = np.repeat(blocks, dwell, axis=0)
    return rows / np.sqrt(3.)


def _visible_position(world_position):
    """Convert physical coordinates to the tracker convention for scoring only."""
    width, height = tt3_arena.IMAGE_WIDTH, tt3_arena.IMAGE_HEIGHT
    horizontal, top = tt3_arena._project(world_position, 0)
    _, side = tt3_arena._project(world_position, 1)
    return np.asarray(((horizontal - width / 2) / width, (top - height / 2) / width,
                       (side - height / 2) / width))


class _AuditedArena(tt3_arena.TT3Arena):
    def step(self, action):
        # Record the objects at the camera time used to select the outgoing action.
        self.observed_roles = (_visible_position(self.paddle), _visible_position(self.ball))
        return super().step(action)


def _target_point(case, role_model, arm, learned_dwell, *, stop_probing=None, probe_arm=None):
    arena = _AuditedArena(case.seed, view_seed=case.view_seed, variant=case.variant)
    learner = tt3_learning.TableLearner(role_model, seed=case.seed, probe_frames=PROBE_FRAMES,
        probes=probe_actions(probe_arm or arm, case.seed, learned_dwell=learned_dwell),
        stop_probing=stop_probing)
    correct = []
    control_correct = []

    def capture(_frames, _action, _reward):
        positions = learner.history[-1].positions
        ids = (learner.binder._control_id, learner.binder._relevant_id)
        threshold = 2.5 / tt3_arena.IMAGE_WIDTH
        matches = [item in positions and np.linalg.norm(positions[item] - position) <= threshold
                   for item, position in zip(ids, arena.observed_roles, strict=True)]
        index = learner.frame - 1
        if matches[0]:
            control_correct.append(index)
        if all(matches):
            correct.append(index)

    # The common return relation stays fixed to isolate probe scheduling.
    point = tt3_learning.play_point(arena, learner,
        option=tt3_learning.RELATIONS.index('project'), capture=capture)
    events = point.pop('events')
    contact = next((event['frame'] for event in events if event['kind'] == 'contact'
                    and event.get('side') == 1), None)
    return dict(case=asdict(case), arm=arm, **point,
                first_correct_control=next(iter(control_correct), None),
                first_correct_both=next(iter(correct), None),
                correct_control_probe_frames=sum(frame <= PROBE_FRAMES for frame in control_correct),
                correct_both_probe_frames=sum(frame <= PROBE_FRAMES for frame in correct),
                first_contact=contact,
                legal_return_frames=[event['frame'] for event in events if event['kind'] == 'legal_return'
                                     and event.get('side') == 1],
                reason=arena.point_result()['reason'])


def _before_probe_end(frame):
    return frame if frame is not None and frame <= PROBE_FRAMES else PROBE_FRAMES + 1


def _target_summary(rows, arms=ARMS):
    def arm_summary(subset):
        timings = [_before_probe_end(row['first_correct_control']) for row in subset]
        return dict(points=len(subset), correct_control_by_probe_end=sum(
            time <= PROBE_FRAMES for time in timings),
            correct_both_by_probe_end=sum(_before_probe_end(row['first_correct_both']) <= PROBE_FRAMES
                                          for row in subset),
            mean_cameras_to_correct_or_miss=float(np.mean(timings)),
            mean_probe_count=float(np.mean([row['probes'] for row in subset])),
            contacts=sum(row['contacts'] for row in subset),
            points_with_legal_return=sum(row['legal_returns'] > 0 for row in subset),
            legal_returns=sum(row['legal_returns'] for row in subset),
            repeated_legal_returns=sum(max(0, row['legal_returns'] - 1) for row in subset),
            points_won=sum(row['points_for'] for row in subset),
            end_reasons=dict(Counter(row['reason'] for row in subset)))

    return {arm: arm_summary([row for row in rows if row['arm'] == arm]) for arm in arms}


def _paired(rows, arms=ARMS):
    rng = np.random.default_rng(650519353)
    learned = [row for row in rows if row['arm'] == arms[0]]
    sample = rng.integers(0, len(learned), size=(4000, len(learned)))
    comparisons = {}
    for arm in arms[1:]:
        comparator = [row for row in rows if row['arm'] == arm]
        binding, returns, probes = [], [], []
        for prior, other in zip(learned, comparator, strict=True):
            if prior['case'] != other['case']:
                raise ValueError('expected matched target cases')
            binding.append(_before_probe_end(other['first_correct_control']) -
                           _before_probe_end(prior['first_correct_control']))
            returns.append(int(prior['legal_returns'] > 0) - int(other['legal_returns'] > 0))
            probes.append(other['probes'] - prior['probes'])

        def summarize(values):
            values = np.asarray(values)
            bootstrapped = values[sample].mean(axis=1)
            return dict(mean=float(values.mean()), ci95=np.percentile(bootstrapped, (2.5, 97.5)).tolist(),
                        gains=int(np.sum(values > 0)), losses=int(np.sum(values < 0)))

        comparisons[arm] = dict(cameras_to_binding_saved=summarize(binding),
                                exploratory_commands_saved=summarize(probes),
                                legal_return_gain=summarize(returns))
    return comparisons


def run(output, role_model=None, target_count=8):
    torch.set_num_threads(1)
    sources = [Path(__file__)] + [Path(module.__file__) for module in
                                 (tt3_arena, tt3_learning, tt3_pong, tt3_visual)]
    source_sha256 = {path.name: sha256(path.read_bytes()).hexdigest() for path in sources}
    report = dict(source=learn_probe_dwell(), probe_frame_budget=PROBE_FRAMES,
                  source_sha256=source_sha256, target_seed=TARGET_SEED,
                  target_cases_per_variant=target_count)
    if role_model is not None:
        report['target_role_model'] = asdict(role_model)
        cases = [case for variant in tt3_arena.VARIANTS for case in
                 tt3_arena.heldout_cases(TARGET_SEED, target_count, variant=variant)]
        rows = [_target_point(case, role_model, arm, report['source']['chosen_dwell'])
                for case in cases for arm in ARMS]
        report.update(target=_target_summary(rows), target_compared=_paired(rows),
                      target_by_variant={variant: _target_summary([row for row in rows
                          if row['case']['variant'] == variant]) for variant in tt3_arena.VARIANTS},
                      target_runs=rows)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--role-model', type=Path)
    parser.add_argument('--target-count', type=int, default=8)
    arguments = parser.parse_args()
    model = None
    if arguments.role_model is not None:
        state = json.loads(arguments.role_model.read_text())
        model = tt3_visual.RoleModel(**state.get('role_model', state))
    report = run(arguments.output, model, arguments.target_count)
    print(json.dumps(dict(source_dwell=report['source']['chosen_dwell'], target=report.get('target'))))
