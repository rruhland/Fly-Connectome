"""Test whether an archived RGB reward model reduces strict SO-101 practice needs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .tt4_transfer_trial import ROOT, _read, _seeds, _write
from .tt7_flight import FlightMemory
from .tt19_factory import FactoryModel, FactoryPlayer
from .tt19_factory_trial import BASE, HIGH, LOW, SCAFFOLD, SPREAD, JointSearch, verify as verify_physics
from .tt19_flight import FlightLaw
from .tt19_practice_camera import BallEyes, PracticeCamera
from .tt19_utility import outgoing, train_source
from .tt19_view_trial import SO101ViewArena


TT19 = ROOT / 'artifacts/tt19'
SEED = 872419701
BUDGETS = (12, 24)
HELD_PER_ROLE = 12
MODES = ('no_source', 'gantry_reward', 'shuffled_reward')
SOURCE_WEIGHT = 1.5


class CameraInput:
    def __init__(self, eyes):
        self.eyes = eyes
        self.fast = True

    def measure(self, images):
        return self.eyes.position(images, self.fast)


def play(arena, player, renderer, seed, role, knobs, *, native):
    arena.reset(seed=seed, server=role, render=False)
    player.reset(role, knobs)
    player.tracker.fast = not native
    actions, encoders = [], []
    image_hash = sha256()
    for tick in range(arena.max_steps):
        images = None
        if tick % 2 == 0:
            images = arena.render_cameras() if native else renderer.capture()
            for image in images.values():
                image_hash.update(image)
        encoder = arena.joints
        requested = player.choose(images, encoder)
        encoders.append(encoder.tolist())
        actions.append(requested.tolist())
        _, reward, done, capped, _ = arena.step(requested, render=False)
        if done or capped:
            break
    trace = [(t, point.tolist()) for t, point in player.history if point is not None]
    features = outgoing(trace, role, arena.table_half_width, arena.table_half_length)
    optical_credit = 0. if features is None else .30 + .65 * features[-1] / 2.5
    return dict(seed=seed, server=role, native=native, knobs=knobs.tolist(), actions=actions,
                encoder_frames=encoders, rgb_sha256=image_hash.hexdigest(), reward=float(reward),
                action_sha256=sha256(np.asarray(actions, dtype='<f8').tobytes()).hexdigest(),
                evaluator=arena.point_result(), events=arena.events.copy(),
                optical=trace, feature=None if features is None else features.tolist(),
                local_utility=float(reward + optical_credit))


def best(bank, role, model):
    search = JointSearch(np.random.default_rng(0), role)
    ranked = []
    for index, row in enumerate(bank):
        if row['server'] != role:
            continue
        feature = row['feature']
        prediction = None if feature is None or model is None else model.predict(feature)
        source_credit = 0. if prediction is None else SOURCE_WEIGHT * (prediction - model.rate)
        score = row['local_utility'] + source_credit
        search.rows.append((row['knobs'], score))
        ranked.append(dict(index=index, prediction=prediction, source_credit=source_credit, utility=score))
    picked = search.best()
    return np.asarray(picked), ranked


def summaries(rows):
    result = dict(points=len(rows), control_frames=sum(len(row['actions']) for row in rows),
                  points_won=sum(row['reward'] > 0. for row in rows), outgoing_seen=sum(
                      row['feature'] is not None for row in rows))
    for role, key, statistic in ((-1, 'receive', 'legal_returns'), (1, 'serve', 'legal_serve')):
        chosen = [row for row in rows if row['server'] == role]
        result[key] = dict(starts=len(chosen), first_legal=sum(row['evaluator'][statistic] > 0 for row in chosen),
                           legal_total=sum(int(row['evaluator'][statistic]) for row in chosen),
                           wins=sum(row['reward'] > 0. for row in chosen))
    return result


def paired(before, after):
    if [(r['seed'], r['server']) for r in before] != [(r['seed'], r['server']) for r in after]:
        raise ValueError('held starts differ across ranking methods')
    result = dict(changed_action_tapes=sum(a['action_sha256'] != b['action_sha256']
                                          for a, b in zip(before, after, strict=True)))
    for name, role, field in (('legal_returns', -1, 'legal_returns'), ('legal_serves', 1, 'legal_serve'),
                              ('points_won', None, 'reward')):
        rows = [(a, b) for a, b in zip(before, after, strict=True) if role is None or a['server'] == role]
        first = [a['reward'] > 0 if role is None else a['evaluator'][field] > 0 for a, _ in rows]
        second = [b['reward'] > 0 if role is None else b['evaluator'][field] > 0 for _, b in rows]
        diff = np.asarray(second, dtype=int) - first
        gain, loss = int((diff > 0).sum()), int((diff < 0).sum())
        result[name] = dict(starts=len(rows), gains=gain, losses=loss, fraction_delta=float(diff.mean()),
                            exact_two_sided_p=1. if gain + loss == 0 else float(binomtest(gain, gain + loss).pvalue))
    return result


def run(output):
    models, source = train_source()
    settings = dict(SCAFFOLD, seed=SEED, max_steps=100)
    arena = SO101ViewArena(**settings)
    random = np.random.default_rng([SEED, 52015])
    roles = (-1, 1) * (BUDGETS[1] // 2)
    practice = _seeds(SEED, 71314, 0, BUDGETS[1])['tests']
    held_roles = (-1, 1) * HELD_PER_ROLE
    held = _seeds(SEED, 90130, 0, len(held_roles))['tests']
    if len(set(practice + held)) != len(practice) + len(held):
        raise ValueError('the held SO-101 starts overlap practice')
    noise = random.normal(size=(BUDGETS[1], 4))
    common, extra, reports, replayed = [], {mode: [] for mode in MODES}, {}, []
    held_cache = {}
    try:
        calibration = arena.reset(seed=SEED, server=-1)
        eyes = BallEyes(calibration)
        renderer = PracticeCamera(arena)
        player = FactoryPlayer(FactoryModel(), CameraInput(eyes), FlightLaw(), FlightMemory('impulse', frozen=True))
        for index in range(BUDGETS[0]):
            role = roles[index]
            proposal = BASE[role] if index < 2 else np.clip(BASE[role] + SPREAD[role] * noise[index], LOW, HIGH)
            row = play(arena, player, renderer, practice[index], role, proposal, native=False)
            common.append(row)
            replayed.append(row)

        for budget in BUDGETS:
            report = {}
            for mode in MODES:
                bank = [*common, *extra[mode]]
                estimators = models.get(mode, {})
                policy, rankings = {}, {}
                for role in (-1, 1):
                    policy[role], rankings[str(role)] = best(bank, role, estimators.get(role))
                rows = []
                for seed, role in zip(held, held_roles, strict=True):
                    knobs = policy[role]
                    key = (seed, role, tuple(knobs))
                    if key not in held_cache:
                        row = play(arena, player, renderer, seed, role, knobs, native=True)
                        held_cache[key] = row
                        replayed.append(row)
                    rows.append(held_cache[key])
                report[mode] = dict(policy={str(key): value.tolist() for key, value in policy.items()},
                                    rankings=rankings, metrics=summaries(rows), held=rows)
                print(budget, mode, report[mode]['policy'], report[mode]['metrics'], flush=True)
            controls = {mode: paired(report[mode]['held'], report['gantry_reward']['held'])
                        for mode in ('no_source', 'shuffled_reward')}
            reports[str(budget)] = dict(modes=report, gantry_vs_controls=controls)
            if budget == BUDGETS[0]:
                for index in range(BUDGETS[0], BUDGETS[1]):
                    role = roles[index]
                    for mode in MODES:
                        bank = [*common, *extra[mode]]
                        center, _ = best(bank, role, models.get(mode, {}).get(role))
                        proposal = (center if index < BUDGETS[0] + 2
                                    else np.clip(center + .75 * SPREAD[role] * noise[index], LOW, HIGH))
                        row = play(arena, player, renderer, practice[index], role, proposal, native=False)
                        extra[mode].append(row)
                        replayed.append(row)
    finally:
        arena.close()
    physical = json.loads(json.dumps(dict(arena=settings, learning=replayed)))
    verified = verify_physics(physical)
    result = dict(schema='tt19-gantry-rgb-reward-transfer-v1', seed=SEED, arena=settings, budgets=BUDGETS,
                  target_role_supplied=True, held_starts=held, practice_starts=practice,
                  camera_stride=2, practice_camera='RGB practice renderer', held_camera='native fixed stereo RGB',
                  factory_model='Public SO-101 CAD kinematics with joint encoders and nominal full-court flight',
                  source_weight=SOURCE_WEIGHT, source=source, common_practice=common,
                  further_practice=extra, report=reports, unique_native_held_rollouts=len(held_cache),
                  replayed_points=verified, dry_motor_commands=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=TT19 / 'utility-transfer.json.gz')
    args = parser.parse_args()
    run(args.output)
