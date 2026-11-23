"""Transfer native Pong experience about selecting informative motor probes."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong

from .tt3_pong import _Presentation, cases_from_seeds
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_body_trial import Body
from .tt24_effects import transmission
from .tt24_source_native import Scene


ARTIFACT = Path('artifacts/tt24/probe-transfer.json.gz')
MATCHED_ARTIFACT = ARTIFACT.with_name('probe-transfer-matched.json.gz')
SOURCE_SEED = 50778114
BODY_SEED = 50778119
MATCHED_BODY_SEED = 50778123
SHUFFLE_SEED = 50778128
AMPLITUDE = .45
DRIVE_FRAMES = 12
PAUSE = 8
FRAMES = DRIVE_FRAMES + PAUSE
BUDGETS = (2, 4, 8)
TAPS = (0, 1, 3, 6, 9)
TREATMENTS = ('pong', 'shuffled', 'orthogonal', 'isotropic')
SHAPES = np.array((
    [1] * 6 + [-1] * 6,
    [1, 1, 1, -1, -1, -1] * 2,
    [1, 1, -1, -1] * 3,
    [1, -1] * 6,
), dtype=float)
WAVES = np.pad(SHAPES, ((0, 0), (0, PAUSE)))
SOURCE_PROFILES = (
    Physics(ball_speed=.37, motor_gain=.76, tau_activation=.044, drag=5.1),
    Physics(ball_speed=.37, motor_gain=1.0, tau_activation=.08, drag=4.0),
    Physics(ball_speed=.37, motor_gain=.89, tau_activation=.139, drag=3.3),
)


def design(probe):
    """Use the same causal, delayed command sums for any actuator dimension."""
    total = np.r_[np.zeros((1, probe.shape[1])), probe.cumsum(axis=0)]
    index = np.maximum(0, np.arange(1, len(probe)+1)[:, None] - TAPS)
    return total[index].reshape(len(probe), -1)


def estimate(examples):
    x, y = (np.concatenate(batch) for batch in zip(*examples, strict=True))
    ridge = .12 * np.eye(x.shape[1])
    return np.linalg.solve(x.T @ x + ridge, x.T @ y)


def evidence(x, y):
    """Dimension-free shape of the just-completed observed response."""
    velocity = np.diff(np.r_[np.zeros((1, y.shape[1])), y], axis=0)
    power = np.sum(velocity**2, axis=1)
    magnitude = np.sum(y**2)
    if not magnitude > 1.e-16:
        return np.zeros(4)
    model = estimate([(x, y)])
    residue = np.sum((x @ model-y)**2)
    late = np.sum(power[DRIVE_FRAMES:]) / max(1.e-16, power.sum())
    lag = power @ np.linspace(0, 1, len(power)) / max(1.e-16, power.sum())
    offset = np.sum(y[-1]**2) * len(y) / magnitude
    return np.array((np.sqrt(residue / magnitude), late, lag, np.sqrt(offset)))


def source_scenarios(profile, cases):
    """Branch source actions from identical native point starts and read only pixels."""
    branches = len(WAVES)
    amount = len(cases)
    rng = np.random.default_rng(sum(case.exploration for case in cases) % (2**31-1))
    lenses, scenes, tracks = [], [], []
    for case in cases:
        for _ in WAVES:
            lenses.append(_Presentation(case.presentation))
            scenes.append(Scene())
            tracks.append([])
    signs = rng.choice((-1., 1.), amount)
    delay = rng.integers(0, 4, amount)
    query = rng.integers(0, branches, amount)
    goals = np.repeat(np.array([np.r_[WAVES[0], c, WAVES[q]]
                               for q in query for c in WAVES]), 2, axis=1)
    # Hidden actuator alias sign and interface latency are plant properties.
    goals = np.concatenate([np.roll(goals[i:i+branches], 2*lag, axis=1)
                            for i, lag in zip(range(0, len(goals), branches), delay, strict=True)])
    game = Pong([case.serve for case in cases for _ in WAVES], profile)

    def observe():
        images = game.render(128, 128).numpy()
        for index, (scene, camera, frame) in enumerate(zip(scenes, lenses, images, strict=True)):
            actors, _ = scene.update(camera.frame(frame))
            tracks[index].append([a.position if a.observed else (np.nan, np.nan) for a in actors])

    observe()
    for tick, issued in enumerate(goals.T):
        drive = torch.as_tensor(AMPLITUDE * issued * np.repeat(signs, branches), dtype=torch.float32)
        outcome = game.step(drive, auto_reset=False)
        if (outcome.scores != 0).any():
            raise RuntimeError('source point ended before the balanced probe windows')
        if (tick + 1) % 2 == 0:
            observe()
    traces = np.array(tracks).reshape(amount, branches, -1, 2, 2)
    training = []
    for i, case in enumerate(cases):
        # Identify the controlled optical component by disagreement between
        # counterfactual issued packets. This uses no native position or axis.
        disagreement = np.nanmean(np.nanvar(traces[i, :, FRAMES+1:2*FRAMES+1], axis=0), axis=(0, 2))
        if not np.isfinite(disagreement).any():
            continue
        controlled = int(np.nanargmax(disagreement))
        measurements = traces[i, :, :, controlled] / .10
        seen = np.isfinite(measurements).all(axis=-1)
        if (seen.mean(axis=1) < .85).any():
            continue
        for b in range(branches):
            measured = measurements[b]
            for k in range(measured.shape[-1]):
                t = np.arange(len(measured))
                measured[:, k] = np.interp(t, t[seen[b]], measured[seen[b], k])
        x0 = design(AMPLITUDE * WAVES[0, :, None])
        xq = design(AMPLITUDE * WAVES[query[i], :, None])

        labels = []
        for j in range(branches):
            path = measurements[j]
            y0 = path[1:FRAMES+1] - path[0]
            y1 = path[FRAMES+1:2*FRAMES+1] - path[FRAMES]
            yq = path[2*FRAMES+1:] - path[2*FRAMES]
            x1 = design(AMPLITUDE * WAVES[j, :, None])
            before = xq @ estimate([(x0, y0)])
            after = xq @ estimate([(x0, y0), (x1, y1)])
            baseline = np.mean((before-yq)**2)
            difference = (baseline - np.mean((after-yq)**2)) / (baseline + .009**2)
            labels.append(float(np.clip(difference, -2., 1.)))
        features = evidence(x0, measurements[0, 1:FRAMES+1] - measurements[0, 0])
        training.append(dict(case=asdict(case), controls=branches, activator=controlled,
                             cues=features.tolist(), prediction_gains=labels, query=int(query[i]),
                             delay_frames=int(delay[i]), motion=measurements.tolist()))
    return training


class Choice:
    """A small ridge estimator of the next probe's optical identification gain."""

    def __init__(self, rows, *, shuffle=False):
        x = np.asarray([r['cues'] for r in rows])
        y = np.asarray([r['prediction_gains'] for r in rows])
        self.center = x.mean(axis=0)
        self.scale = np.maximum(.05, x.std(axis=0))
        bounded = self.features(x)
        if shuffle:
            # Preserve each context's set of measured gains and destroy the
            # correspondence between gain and the replayed probe envelope.
            rng = np.random.default_rng(SHUFFLE_SEED)
            y = np.array([rng.permutation(row) for row in y])
        self.weight = np.linalg.solve(bounded.T @ bounded + 3.*np.eye(bounded.shape[1]), bounded.T @ y)

    def features(self, x):
        x = np.atleast_2d(np.clip((x-self.center)/self.scale, -2., 2.))
        return np.column_stack((np.ones(len(x)), x))

    def select(self, cue):
        return int(np.argmax((self.features(cue) @ self.weight)[0]))

    @classmethod
    def from_record(cls, record):
        result = cls.__new__(cls)
        result.center, result.scale, result.weight = (np.asarray(record[key])
                                                      for key in ('center', 'scale', 'weight'))
        return result


def native_source():
    rng = np.random.default_rng(SOURCE_SEED)
    seeds = rng.choice(2**31-1, 33, replace=False)
    practice, held = [], []
    for profile, spec in enumerate(SOURCE_PROFILES):
        use = cases_from_seeds(seeds[11*profile:11*(profile+1)])
        practice += source_scenarios(spec, use[:7])
        held += source_scenarios(spec, use[7:])
    if len(practice) < 12 or len(held) < 6:
        raise RuntimeError('insufficient chronological source observations')
    learned = Choice(practice)
    shuffled = Choice(practice, shuffle=True)
    choices = {'pong': [learned.select(r['cues']) for r in held],
               'shuffled': [shuffled.select(r['cues']) for r in held], 'orthogonal': [0] * len(held)}
    source_result = {k: dict(gain=float(np.mean([r['prediction_gains'][b] for r, b in zip(held, v, strict=True)])),
                             chosen=v) for k, v in choices.items()}
    print('native held prediction gains', source_result, flush=True)
    return {'practice': practice, 'held': held, 'native_points': int(len(seeds) * len(WAVES)),
            'native_frames_per_point': FRAMES * 3, 'native_ticks_per_frame': 2,
            'source_schedules': seeds.tolist(), 'profiles': [asdict(x) for x in SOURCE_PROFILES],
            'source_held': source_result, 'frozen': {k: dict(center=m.center.tolist(), scale=m.scale.tolist(),
                                                           weight=m.weight.tolist())
                                                     for k, m in [('pong', learned), ('shuffled', shuffled)]}}, learned, shuffled


def directions(rng, count, *, orthogonal):
    if orthogonal:
        return np.concatenate([np.linalg.qr(rng.normal(size=(5, 5)))[0].T
                               for _ in range((count+4)//5)])[:count]
    x = rng.normal(size=(count, 5))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def body_response(plant, direction, shape):
    command = AMPLITUDE * WAVES[shape, :, None] * direction
    origin = plant.observe()
    offset = np.asarray([plant.advance(action, 1) - origin for action in command])
    return design(command), offset


def body_trial(plant, seed, learned, shuffled):
    rng = np.random.default_rng(seed)
    wiring = transmission(rng)
    mass = rng.uniform((.85, 1.55), (1.68, 3.8))
    delay = int(rng.integers(0, 4))
    shared = directions(rng, max(BUDGETS), orthogonal=True)
    random = directions(rng, max(BUDGETS), orthogonal=False)
    testers = directions(rng, 16, orthogonal=False)
    query_shapes = rng.choice(len(WAVES), len(testers))

    def restart():
        plant.reset()
        plant.variant(wiring, mass, delay)

    # New independent directions and no motion/reward feedback to the fitter.
    # Separate settled test initializations provide comparable validation probes.
    challenge = []
    for direction, wave in zip(testers, query_shapes, strict=True):
        restart()
        challenge.append(body_response(plant, direction, wave))
    challenge_x, challenge_y = (np.asarray(v) for v in zip(*challenge, strict=True))
    answer = {}
    for name in TREATMENTS:
        restart()
        probes, rows, results = [], [], {}
        selector = {'pong': learned, 'shuffled': shuffled}.get(name)
        basis = random if name == 'isotropic' else shared
        for index, direction in enumerate(basis):
            shape = 0 if selector is None or index == 0 else selector.select(evidence(*rows[-1]))
            example = body_response(plant, direction, shape)
            rows.append(example)
            probes.append(dict(shape=shape, direction=direction.tolist(), response=example[1].tolist()))
            if index + 1 in BUDGETS:
                fit = estimate(rows)
                error = challenge_x @ fit - challenge_y
                results[index + 1] = dict(rmse_rad=float(np.sqrt(np.mean(error**2))),
                                          axis_rmse_rad=np.sqrt(np.mean(error**2, axis=(0, 1))).tolist())
        answer[name] = dict(probes=probes, errors=results)
    return dict(seed=int(seed), wiring=wiring.tolist(), mass_scale=mass.tolist(), delay_frames=delay,
                challenges=[dict(shape=int(k), direction=d.tolist(), response=y.tolist())
                            for k, d, y in zip(query_shapes, testers, challenge_y, strict=True)],
                outcomes=answer)


def summarize(rows, budget):
    error = {k: np.array([r['outcomes'][k]['errors'][budget]['rmse_rad'] for r in rows])
             for k in TREATMENTS}
    results = dict(mean_rmse_rad={k: float(v.mean()) for k, v in error.items()}, pong_minus={})
    draws = np.random.default_rng(318155 + budget).integers(len(rows), size=(5000, len(rows)))
    for key in TREATMENTS[1:]:
        change = error['pong'] - error[key]
        results['pong_minus'][key] = dict(mean_rad=float(change.mean()),
                                          interval95_rad=np.quantile(change[draws].mean(axis=1), (.025, .975)).tolist(),
                                          lower_error=int(np.count_nonzero(change < -1.e-12)),
                                          tied=int(np.count_nonzero(np.abs(change) <= 1.e-12)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bodies', type=int)
    parser.add_argument('--matched-rate-source', type=Path,
                        help='Reuse the frozen source fit on a fresh 60 Hz body cohort.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    prior = args.matched_rate_source
    output = args.output or (MATCHED_ARTIFACT if prior else ARTIFACT)
    if prior:
        if output.resolve() == prior.resolve():
            parser.error('write matched-rate trials separately from the original source receipt')
        old = _read(prior)['source']
        learned, shuffled = (Choice.from_record(old['frozen'][name]) for name in ('pong', 'shuffled'))
        source = dict(source_receipt=str(prior), source_sha256=_hash(prior), new_native_points=0)
    else:
        source, learned, shuffled = native_source()
    count = args.bodies if args.bodies is not None else (12 if prior else 20)
    if count < 1:
        parser.error('positive body cohort required')
    plant = Body()
    try:
        if prior:
            plant.substeps = round(1 / 60 / plant.arena.model.opt.timestep)
        dt = plant.substeps * plant.arena.model.opt.timestep
        seeds = np.random.default_rng(MATCHED_BODY_SEED if prior else BODY_SEED).choice(2**31-1, count, replace=False)
        rows = [body_trial(plant, s, learned, shuffled) for s in seeds]
    finally:
        plant.arena.close()
    aggregate = {k: summarize(rows, k) for k in BUDGETS}
    advance = any(all(r['pong_minus'][c]['interval95_rad'][1] <
                      -.05 * r['mean_rmse_rad'][c] for c in ('orthogonal', 'shuffled'))
                  for k, r in aggregate.items() if k in (4, 8))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, dict(schema='tt24-pong-probe-transfer-1', source=source, target=rows, summary=aggregate,
                        probes=dict(waves=WAVES.tolist(), drive_frames=DRIVE_FRAMES, pulse_amplitude=AMPLITUDE,
                                    target_budget=list(BUDGETS), estimator_taps=TAPS, duration_frames=FRAMES,
                                    target_tick_s=dt),
                        target_game_rewards_used=0, qualifies_for_wrist_reward_evaluation=bool(advance)))
    print('motor-identification qualification', aggregate, 'advance', advance, flush=True)


if __name__ == '__main__':
    main()
