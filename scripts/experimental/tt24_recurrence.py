"""Learn the optical consequences of short randomized native motor interventions."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .tt3_pong import ACTIONS, CAMERA_SECONDS, NATIVE_HZ, _native_rallies, cases_from_seeds
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_source_native import Collector, OPPONENT_SPEED, batches


OUTPUT = Path(__file__).resolve().parents[2] / 'artifacts' / 'tt24'
RECEIPT = OUTPUT / 'recurrence-native.json.gz'
WEIGHTS = OUTPUT / 'recurrence-weights.json'
MODES = ('pong', 'cold', 'shuffled')
RNG = {'practice': 1335474407, 'held': 410727901, 'shuffle': 1445792112}
NAMES = ('axial_velocity', 'cross_velocity', 'axial_displacement', 'cross_displacement',
         'radial_velocity', 'radial_change', 'departure', 'ball_speed', 'separation')
PULSE = 8
CAP = 840


def slope(positions, times):
    centered = times - np.mean(times)
    return centered @ positions / (centered @ centered)


def effect_features(initial_actor, initial_mover, actor, mover, *, scale, dt):
    """Describe an observed mover's local response relative to its measured approach."""
    a, b = np.asarray(initial_actor), np.asarray(initial_mover)
    if len(a) < 3 or a.shape != b.shape or not np.isfinite(np.r_[a, b]).all():
        return None
    initial_time = dt * np.arange(1 - len(a), 1)
    incoming = slope(b, initial_time)
    speed = np.linalg.norm(incoming)
    if speed < .06 * scale:
        return None
    direction = incoming / speed
    initial_range = np.linalg.norm(b[-1] - a[-1]) / scale
    mover = np.asarray(mover)
    actor = np.asarray(actor)
    available = np.isfinite(mover).all(axis=1) & np.isfinite(actor).all(axis=1)
    if available.sum() < 2:
        return None
    t = dt * np.flatnonzero(available)
    actor, mover = actor[available], mover[available]
    velocity = slope(mover, t) / speed
    relative = mover[-1] - actor[-1]
    distance = np.linalg.norm(relative) / scale
    delta = (mover[-1] - b[-1]) / scale
    axial, progress = velocity @ direction, delta @ direction
    radial = relative / (scale * max(distance, .025))
    relative_speed = slope(mover - actor, t) / speed
    values = [axial, np.linalg.norm(velocity - axial * direction), progress,
              np.linalg.norm(delta - progress * direction), relative_speed @ radial,
              distance - initial_range, velocity @ radial, np.linalg.norm(velocity), distance]
    return np.arcsinh(values)


class PulseCollector(Collector):
    """The same RGB probe and approach gate bind all branches before a signed pulse."""

    def __init__(self, seed, pulse):
        super().__init__(seed)
        self.pulse = pulse
        self.cue = None

    def qualify(self):
        if len(self.views) < 6:
            return None
        index = self.motor.identity
        recent = self.views[-5:]
        if not all(index < len(tracks) and tracks[index].observed for tracks, _ in recent):
            return None
        blade = np.array([tracks[index].position for tracks, _ in recent])
        mover = np.array([ball for _, ball in recent])
        if not np.isfinite(mover).all():
            return None
        normal = np.array((-self.motor.axis[1], self.motor.axis[0]))
        vel = slope(mover, np.arange(5.))
        offset, motion = (mover[-1] - blade[-1]) @ normal, vel @ normal
        if abs(motion) < .006 or offset * motion >= 0.:
            return None
        span = self.scene.extended[index][1]
        time = abs(offset / motion)
        if not (4 <= time <= 21 and np.linalg.norm(mover[-1] - blade[-1]) < 4.4 * span):
            return None
        digest = sha256(np.asarray([blade, mover], dtype='<f4').tobytes()).hexdigest()
        return dict(frame=len(self.actions), eta=float(time), span=float(span), prefix=digest,
                    initial_actor=blade.tolist(), initial_mover=mover.tolist())

    def choose(self):
        if self.motor.effects is None:
            command = self.motor.probe()
        else:
            if self.cue is None:
                self.cue = self.qualify()
            within = self.cue is not None and len(self.actions) - self.cue['frame'] < PULSE
            command = self.motor.effects[self.pulse + 1 if within else 1]
        self.actions.append(command)
        return command

    def extraction(self):
        if self.cue is None:
            return None
        actor, mover = self.path()
        frame = self.cue['frame'] + int(np.ceil(self.cue['eta']))
        lo, hi = frame - 1, frame + 4
        if len(mover) < hi:
            return None
        # Align to the pre-pulse encounter estimate. A missing terminal window
        # makes an episode ineligible and supplies no feature to the fit.
        return effect_features(self.cue['initial_actor'], self.cue['initial_mover'],
                               actor[lo:hi], mover[lo:hi], scale=self.cue['span'], dt=CAMERA_SECONDS)


def collection(seeds, batch_size, name):
    cases = cases_from_seeds(seeds)
    schedule = [(case, pulse) for case in cases for pulse in ACTIONS]
    rows = []
    for i, batch in enumerate(batches(schedule, batch_size)):
        agents = [PulseCollector(case.exploration, pulse) for case, pulse in batch]
        results = _native_rallies([case for case, _ in batch], agents, max_ticks=CAP, opponent_speed=OPPONENT_SPEED)
        for (case, pulse), agent, row in zip(batch, agents, results, strict=True):
            optical = agent.extraction()
            rows.append(dict(launch=case.serve, pulse=pulse, ticks=row['ticks'], action_sha256=row['action_sha256'],
                             rgb_track_sha256=agent.scene.signature.hexdigest(), cue=agent.cue,
                             features=None if optical is None else optical.tolist()))
        if i % 4 == 0:
            print(name, len(rows), sum(row['features'] is not None for row in rows), flush=True)
    return [rows[k:k + len(ACTIONS)] for k in range(0, len(rows), len(ACTIONS))]


def comparisons(groups):
    pairs = []
    for group in groups:
        if {row['launch'] for row in group} != {group[0]['launch']}:
            raise ValueError('comparison has unmatched native launch IDs')
        cue_hashes = {row['cue']['prefix'] for row in group if row['cue'] is not None}
        if len(cue_hashes) > 1:
            raise ValueError('paired visual prefixes differed before the assigned pulse')
        for i, a in enumerate(group):
            for b in group[:i]:
                if a['features'] is None or b['features'] is None:
                    continue
                delta = np.log1p(a['ticks'] / NATIVE_HZ) - np.log1p(b['ticks'] / NATIVE_HZ)
                if abs(a['ticks'] - b['ticks']) < 12:
                    continue
                x = np.asarray(a['features']) - np.asarray(b['features'])
                pairs.append((a['launch'], x, delta))
    return pairs


def fit(data):
    _, xx, yy = zip(*data, strict=True)
    x, y = np.asarray(xx), np.asarray(yy)
    scale = np.maximum(.12, x.std(axis=0))
    z = np.clip(x / scale, -4., 4.)
    regularization = z.T @ z + 3. * np.eye(z.shape[1])
    rng = np.random.default_rng(RNG['shuffle'])
    launch_signs = dict(zip(sorted({row[0] for row in data}),
                            rng.choice((-1., 1.), size=len({row[0] for row in data})), strict=True))
    shuffle = y * np.array([launch_signs[row[0]] for row in data])
    weights = {'pong': np.linalg.solve(regularization, z.T @ y).tolist(),
               'shuffled': np.linalg.solve(regularization, z.T @ shuffle).tolist(),
               'cold': [0.] * z.shape[1]}
    return dict(schema='tt24-local-recurrence-1', features=NAMES, scale=scale.tolist(), weights=weights,
                train_pairs=len(y), train_launches=len(launch_signs), shuffle_seed=RNG['shuffle'])


def contrast(a, b, state, mode):
    if a is None or b is None:
        return 0.
    change = np.clip((np.asarray(a) - b) / state['scale'], -4., 4.)
    return float(change @ state['weights'][mode])


def qualification(state, pairs):
    if not pairs:
        return {'pairs': 0, 'launches': 0, 'passed': False}
    seeds, left, outcome = (np.asarray(v) for v in zip(*pairs, strict=True))
    outcomes = {}
    for mode in MODES:
        prediction = (np.clip(left / state['scale'], -4., 4.) @ state['weights'][mode])
        signs = np.sign(outcome) * np.sign(prediction)
        outcomes[mode] = .5 + .5 * signs
    groups = [np.flatnonzero(seeds == name) for name in np.unique(seeds)]
    rng = np.random.default_rng(241718441)
    selection = rng.integers(len(groups), size=(4096, len(groups)))
    counts = np.array([len(group) for group in groups])
    gains, confidence = {}, {}
    for mode in ('cold', 'shuffled'):
        delta = outcomes['pong'] - outcomes[mode]
        accumulated = np.array([delta[group].sum() for group in groups])
        gains[mode] = float(np.mean(delta))
        confidence[mode] = np.quantile(accumulated[selection].sum(axis=1) / counts[selection].sum(axis=1),
                                       (.025, .975)).tolist()
    accuracy = {mode: float(np.mean(values)) for mode, values in outcomes.items()}
    return dict(pairs=len(seeds), launches=len(groups), accuracy=accuracy, gains=gains, intervals=confidence,
                passed=len(groups) >= 12 and accuracy['pong'] >= .65 and all(v[0] > 0. for v in confidence.values()))


def run(train, held, batch_size):
    began = perf_counter()
    counts = dict(practice=train, held=held)
    seeds = {name: np.random.default_rng(RNG[name]).choice(2147483646, size=n, replace=False).tolist()
             for name, n in counts.items()}
    if set(seeds['practice']) & set(seeds['held']):
        raise ValueError('native practice and held launches overlap')
    report = _read(RECEIPT) if RECEIPT.exists() else dict(schema='tt24-recurrence-native-1', seeds=seeds)
    if report['seeds'] != seeds:
        raise ValueError('native recurrence receipt uses a different launch schedule')
    for name in counts:
        if name not in report:
            report[name] = collection(seeds[name], batch_size, name)
            _write(RECEIPT, report)
    native = comparisons(report['practice'])
    checks = comparisons(report['held'])
    if not native:
        raise ValueError('native pulses produced no optically comparable practice effects')
    model = fit(native)
    result = qualification(model, checks)
    _write(WEIGHTS, dict(model, qualification=result))
    report.update(weights_sha256=_hash(WEIGHTS), qualification=result, seconds=perf_counter() - began)
    _write(RECEIPT, report)
    print('native local intervention', result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=192)
    parser.add_argument('--held', type=int, default=126)
    parser.add_argument('--batch', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.train, args.held, args.batch)


if __name__ == '__main__':
    main()
