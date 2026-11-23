"""Match measured stock actuator timing with a bounded causal input filter."""

import argparse

import mujoco
import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_forward_wiring import BallFree
from .tt24_posture import WidePosture
from .tt24_ready import prepare_step


TAG = 'drive-response'
PLAN = ROOT / f'{TAG}-plan.json'
BENCH = ROOT / f'{TAG}-bench.json.gz'
LOCK = ROOT / f'{TAG}-lock.json'
HELD = ROOT / f'{TAG}.json.gz'
SEED = 231858417
FRAMES, TAPS, ACCEPT = 72, 14, .90
STEP = 5.2 / 60.
REGIMES = {'stock': 'standard', 'alternate': 'changed'}
PERMUTATION = [2, 0, 4, 1, 3]
VARIANTS = ('unchanged', 'matched', 'shuffled', 'tracking')


class Lead:
    """A unity-DC input filter with a bounded response to changes in desired rate."""

    def __init__(self, weights):
        self.weights = np.asarray(weights)
        self.previous = np.zeros((3, 5))

    def command(self, desired):
        samples = np.vstack((desired, self.previous))
        shaped = desired + np.einsum('lc,lc->c', self.weights, samples[:-1] - samples[1:])
        self.previous[:] = samples[:3]
        return np.clip(shaped, -1., 1.)


def register():
    rng = np.random.default_rng(SEED)
    seeds = (rng.choice(2**31 - 2, 26, replace=False) + 1).tolist()
    pulse_seeds = (rng.choice(2**31 - 2, 16, replace=False) + 1).tolist()
    plan = dict(schema='tt24-drive-response-1', rng=SEED, train=list(zip(seeds[:8], pulse_seeds[:8])),
                bench=list(zip(seeds[8:16], pulse_seeds[8:])), gameplay={'stock': seeds[16:20], 'alternate': seeds[20:26]},
                portable_sha256=_hash(BUNDLE), sample_frames=FRAMES, response_taps=TAPS,
                controls=list(VARIANTS), shuffling=PERMUTATION, required_error_ratio=ACCEPT,
                minimum_uncompensated_difference_rad=.001,
                gate='held forward-rate prediction beats instantaneous by >=10% on both plants, and actual paired angle traces beat identity and tracking by >=10%',
                tracked_stock_oracle='actual matched stock encoder displacement under the original unmodified commands',
                gameplay_cap=240, reward_used_for_fit=False)
    plan['train'] = [list(pair) for pair in plan['train']]
    plan['bench'] = [list(pair) for pair in plan['bench']]
    if PLAN.exists():
        if _read(PLAN) != plan:
            raise ValueError('response protocol changed')
    else:
        _write(PLAN, plan)
    return plan


def excitation(seed):
    """Independent signed rate blocks, kept within 0.11 rad of each reset."""
    rng = np.random.default_rng(seed)
    data = np.empty((FRAMES, 5))
    for axis in range(5):
        cursor = 0
        while cursor < FRAMES:
            count = min(FRAMES - cursor, int(rng.integers(1, 7)))
            data[cursor:cursor + count, axis] = rng.uniform(-.37, .37)
            cursor += count
    data -= data.mean(axis=0)
    peak = np.abs(np.cumsum(data, axis=0) * STEP).max(axis=0)
    return data / np.maximum(1., peak / .11)


def bench_run(plant, regime, seed, pulse, mode='unchanged', weights=None):
    obs = plant.reset(seed=seed, split='held', regime=regime, render=False)
    start = obs['joints'].astype(float)
    target = start.copy()
    lead = Lead(weights) if weights is not None else None
    locations = [start]
    executed = []
    for wanted in pulse:
        if mode == 'tracking':
            sent = np.clip(wanted + .35 * (target - obs['joints']) / STEP, -1., 1.)
        elif lead is not None:
            sent = lead.command(wanted)
        else:
            sent = wanted
        obs, reward, done = plant.step(sent, render=False)
        if reward or (done and len(executed) != len(pulse) - 1):
            raise ValueError('a contact-free encoder probe ended unexpectedly')
        target += STEP * wanted
        executed.append(np.asarray(sent).tolist())
        locations.append(obs['joints'].astype(float))
    return dict(joints=np.asarray(locations).tolist(), actions=executed)


def paired_bench(plant, regime, seed, pulse, mode='unchanged', weights=None):
    records = [bench_run(plant, regime, seed, sign * pulse, mode, weights) for sign in (1., -1.)]
    q = (np.array(records[0]['joints']) - np.array(records[1]['joints'])) * .5
    return dict(records=records, relative=q.tolist())


def histories(pulse):
    padded = np.pad(pulse, ((TAPS - 1, 0), (0, 0)))
    indexes = np.arange(len(pulse))[:, None] + np.arange(TAPS - 1, -1, -1)[None]
    return padded[indexes]


def observed_rates(rows, plant):
    return np.concatenate([np.diff(np.asarray(row['bodies'][plant]['unchanged']['relative']), axis=0) / STEP for row in rows])


def identify(rows):
    inputs = np.concatenate([histories(np.asarray(row['requested'])) for row in rows])
    static, response = {}, {}
    for body in REGIMES:
        y = observed_rates(rows, body)
        static[body] = []
        response[body] = []
        for axis in range(5):
            x = inputs[:, :, axis]
            impulse = np.linalg.solve(x.T @ x + .035 * np.eye(TAPS), x.T @ y[:, axis])
            response[body].append(impulse.tolist())
            unit = x[:, 0]
            static[body].append(float((unit @ y[:, axis]) / (unit @ unit)))
    return static, response


def matching_filter(response):
    # Three successive finite differences grant causal phase lead or damping.
    # Every proposed correction sums to zero, preserving the DC command gain.
    delta = np.eye(4)[:-1] - np.eye(4)[1:]
    result = []
    for stock, alternate in zip(response['stock'], response['alternate'], strict=True):
        a = np.array(alternate)
        difference_response = np.stack([np.convolve(a, q) for q in delta], axis=1)
        wanted = np.pad(np.asarray(stock) - a, (0, 3))
        coefficients = np.linalg.solve(difference_response.T @ difference_response + .012 * np.eye(3),
                                       difference_response.T @ wanted)
        coefficients = np.clip(coefficients, -.45, .45)
        result.append(coefficients / max(1., float(np.abs(coefficients).sum()) / .75))
    return np.asarray(result).T


def probe_cases(plants, roster, filters=None):
    rows = []
    for start, seed in roster:
        pulse = excitation(seed)
        bodies = {}
        for body, plant in plants.items():
            modes = {'unchanged': None}
            if filters is not None and body == 'alternate':
                modes |= {'tracking': None} | filters
            bodies[body] = {name: paired_bench(plant, REGIMES[body], start, pulse, name, data) for name, data in modes.items()}
        rows.append(dict(seed=start, pulse_seed=seed, requested=pulse.tolist(), bodies=bodies))
    return rows


def rmse(error):
    return dict(all_axes=float(np.sqrt(np.mean(error**2))), per_axis=np.sqrt(np.mean(error**2, axis=0)).tolist())


def audit(rows, response, instant):
    inputs = np.concatenate([histories(np.asarray(row['requested'])) for row in rows])
    metrics = {}
    for body in REGIMES:
        actual = observed_rates(rows, body)
        forward = np.einsum('nta,at->na', inputs, np.asarray(response[body]))
        static = np.asarray(instant[body]) * inputs[:, 0]
        metrics[body] = dict(fir=rmse(actual - forward), instantaneous=rmse(actual - static))
    reference = np.concatenate([row['bodies']['stock']['unchanged']['relative'] for row in rows])
    realism = {}
    for mode in VARIANTS:
        measured = np.concatenate([row['bodies']['alternate'][mode]['relative'] for row in rows])
        realism[mode] = rmse(measured - reference)
    return dict(held_normalized_encoder_rate_prediction=metrics, held_actual_angle_difference_rad=realism)


def bench(plan):
    if BENCH.exists() and LOCK.exists():
        locked = _read(LOCK)
        if locked['plan'] != _hash(PLAN) or locked['bench'] != _hash(BENCH):
            raise ValueError('frozen motor response differs from the bench lineage')
        print(locked, flush=True)
        return locked
    plants = {body: BallFree(body, FRAMES) for body in REGIMES}
    try:
        train = probe_cases(plants, plan['train'])
        instantaneous, response = identify(train)
        ordered = matching_filter(response)
        filters = dict(matched=ordered.tolist(), shuffled=ordered[:, PERMUTATION].tolist())
        held = probe_cases(plants, plan['bench'], filters)
    finally:
        for plant in plants.values():
            plant.close()
    checked = audit(held, response, instantaneous)
    forward = checked['held_normalized_encoder_rate_prediction']
    physical = checked['held_actual_angle_difference_rad']
    qualified_predictor = all(row['fir']['all_axes'] <= ACCEPT * row['instantaneous']['all_axes'] for row in forward.values())
    qualified_match = (physical['unchanged']['all_axes'] >= .001
                       and physical['matched']['all_axes'] <= ACCEPT * min(physical['unchanged']['all_axes'], physical['tracking']['all_axes']))
    _write(BENCH, dict(plan=_hash(PLAN), train=train, held=held, stock_and_alternate_fir=response,
                      instantaneous=instantaneous, filters=filters, metrics=checked, observed_reward_used=False))
    value = dict(plan=_hash(PLAN), bench=_hash(BENCH), filters=filters, prediction_gate=qualified_predictor,
                 physical_gate=qualified_match, launch_gameplay=qualified_predictor and qualified_match, metrics=checked)
    _write(LOCK, value)
    print('LOCKED DRIVE BENCH', value, flush=True)
    return value


def game(arena, server, seed, body, filter_weights, replay=None):
    regime = REGIMES[body]
    server.reset(seed=seed, split='held', regime=regime, render=False)
    position, velocity, spin = [x.copy() for x in (server._ball_position, server._ball_velocity, server._ball_spin)]
    lead_ticks = int(np.random.default_rng(seed ^ 11940).integers(14, 39))
    arena.reset(seed=seed, split='held', regime=regime, render=False)
    arena._ball_position[:], arena._ball_velocity[:] = position, 0.
    mujoco.mj_forward(arena._model, arena._data)
    render = replay is None
    obs = arena._observation(render)
    if render:
        carried = CarriedWristPolicy()
        carried.begin_release(obs)
        controlled = Lead(filter_weights) if body == 'alternate' and filter_weights is not None else None
    tape, desired = [], []
    for tick in range(lead_ticks + arena.max_steps):
        if render:
            action = carried.act(obs)
            desired.append(action.tolist())
            sent = controlled.command(action) if controlled is not None else action
        else:
            sent = np.asarray(replay[tick])
        tape.append(sent.tolist())
        if tick < lead_ticks:
            prepare_step(arena, sent, position)
            if tick + 1 == lead_ticks:
                arena._ball_velocity[:], arena._ball_spin[:] = velocity, spin
                mujoco.mj_forward(arena._model, arena._data)
            obs = arena._observation(render)
        else:
            obs, _, done = arena.step(sent, render=render)
            if done:
                break
    return dict(actions=tape, judge=arena.judge_metrics(), desired=desired,
                observer=carried.calibration_index if render else None, preparation_ticks=lead_ticks)


def qualification(plan, locked):
    record = dict(plan=_hash(PLAN), filter_lock=_hash(LOCK), courses={})
    for body, launches in plan['gameplay'].items():
        player = WidePosture(view='wrist', width=192, height=144, arm=body, max_steps=plan['gameplay_cap'])
        server = BounceArena(view='none', arm=body)
        modes = {'unchanged': None} if body == 'stock' else {'unchanged': None} | locked['filters']
        try:
            variants = {}
            for mode, filter_weights in modes.items():
                games = []
                for seed in launches:
                    row = game(player, server, seed, body, filter_weights)
                    replay = game(player, server, seed, body, filter_weights, row['actions'])
                    if row['judge'] != replay['judge'] or row['actions'] != replay['actions']:
                        raise ValueError('held compound drive action replay diverged')
                    games.append(row)
                variants[mode] = games
            if body == 'stock':
                stock = variants['unchanged']
                if any(row['desired'] != row['actions'] for row in stock):
                    raise ValueError('stock actuator command is not the identity')
                variants |= {'matched': stock, 'shuffled': stock}
            record['courses'][body] = variants
            print(body, {mode: [row['judge']['longest_consecutive_rebounds'] for row in games]
                         for mode, games in variants.items()}, flush=True)
            _write(HELD, record)
        finally:
            player.close()
            server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-only', action='store_true')
    args = parser.parse_args()
    plan = register()
    locked = bench(plan)
    if locked['launch_gameplay'] and not args.bench_only:
        qualification(plan, locked)


if __name__ == '__main__':
    main()
