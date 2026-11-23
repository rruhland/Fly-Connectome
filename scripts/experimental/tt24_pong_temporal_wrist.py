"""Test a time-normalized native visual prediction gain on physical wrist play."""

import argparse
from collections import deque

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import Native, ROOT
from .tt24_posture import WidePosture
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_ready import episode
from .tt24_wrist_egomotion import HEIGHT, WIDTH


DIAGNOSTIC = ROOT / 'pong-temporal-diagnostic.json.gz'
HELD = ROOT / 'pong-temporal-wrist.json.gz'
SOURCE = ROOT / 'pong-temporal-source.json.gz'
DIAGNOSTIC_SEED = 1575205693
HELD_SEED = 124381823
MAX_STEPS = 300
WRIST_HZ = 60
MAX_AGE_SECONDS = 4 / 30


def advance(position, velocity, age_seconds, gain):
    """Use the same physical-time gain with either optical coordinate system."""
    return position + gain * age_seconds * velocity


def packet_ticks(tick, seed, hz):
    """Public harness delivers timestamped camera frames on a variable-age schedule."""
    onset = .24 + .04 * (seed % 3)
    phase = tick / hz - onset
    rest = .86 + .08 * (seed % 4)
    if 0. <= phase < rest:
        return round((2 + seed % 3) * hz / 30)
    if rest + .42 <= phase < rest + 1.45:
        return round((2 + (seed + 1) % 3) * hz / 30)
    return 0


class DelayedWrist:
    """Separate capture-pose decoding, present proprioception and age compensation."""

    def __init__(self, seed, gain, delay):
        self.eye = load_public(FUSED_OUTPUT)
        self.decoder = self.eye.decoder
        self.seed, self.gain, self.delay = seed, gain, delay
        self.history = deque(maxlen=round(MAX_AGE_SECONDS * WRIST_HZ) + 1)
        self.last_state = None
        self.tick = -1
        self.capture = -1
        self.packet_ages = []

    @property
    def home(self):
        return self.eye.home

    @home.setter
    def home(self, value):
        self.eye.home = value

    def observe(self, live):
        self.tick += 1
        self.history.appendleft(live)
        requested = packet_ticks(self.tick, self.seed, WRIST_HZ) if self.delay else 0
        capture = max(0, self.capture, self.tick - requested)
        age = self.tick - capture
        self.packet_ages.append(age)
        if capture > self.capture:
            for skipped in range(self.capture + 1, capture):
                self.eye.read(None, self.history[self.tick - skipped]['joints'])
            self.last_state = self.eye.observe(self.history[age])
            self.capture = capture
        if not age:
            return self.last_state
        state = self.last_state.copy()
        state[:3] = advance(state[:3], state[3:6], age / WRIST_HZ, self.gain)
        body = self.decoder.body(live['joints'])
        state[6:9] = body[12:15] - self.home
        state[9:11] = body[15:17]
        return state


def starts(seed, count):
    return np.random.default_rng(seed).choice(2**31 - 2, count, replace=False).tolist()


def effects(rows, baseline):
    delta = np.asarray([a['run'] - b['run'] for a, b in zip(rows, baseline, strict=True)])
    rng = np.random.default_rng(HELD_SEED + 1)
    interval = np.quantile(rng.choice(delta, size=(16000, len(delta))).mean(axis=1), (.025, .975))
    return dict(mean=float(delta.mean()), interval=interval.tolist(), wins=int(np.sum(delta > 0)),
                ties=int(np.sum(delta == 0)), losses=int(np.sum(delta < 0)))


def run(path, seeds, modes, metadata):
    with np.load(BUNDLE, allow_pickle=False) as archive:
        weights, equilibrium, reuse = archive['weights'], archive['equilibrium'], float(archive['reuse'])
    receipt = dict(schema='tt24-pong-temporal-wrist-1', seeds=seeds, modes=modes, **metadata,
                   duration_ticks=MAX_STEPS, bundle_sha256=_hash(BUNDLE), eye_sha256=_hash(FUSED_OUTPUT),
                   exact_physical_replay=True, conditions={}, summary={})
    if path.exists():
        old = _read(path)
        if any(old[key] != value for key, value in receipt.items() if key not in ('conditions', 'summary')):
            raise ValueError('existing wrist roster has a different protocol')
        receipt['conditions'], receipt['summary'] = old['conditions'], old['summary']
    native = Native()
    arena = WidePosture(view='wrist', width=WIDTH, height=HEIGHT, max_steps=MAX_STEPS, arm='stock')
    launcher = BounceArena(view='none', arm='stock')
    try:
        for mode, spec in modes.items():
            rows = receipt['conditions'].setdefault(mode, [])
            for seed in seeds[len(rows):]:
                tracker = None if spec is None else DelayedWrist(seed, spec['gain'], spec['delay'])
                row = episode(arena, launcher, seed, native, weights, equilibrium, 'scalar', 'standard',
                              alpha=reuse, sensor_factory=None if tracker is None else lambda _: tracker)
                row['packet_ages'] = [] if tracker is None else tracker.packet_ages
                rows.append(row)
                streaks = [result['run'] for result in rows]
                receipt['summary'][mode] = dict(total=sum(streaks), streaks=streaks, best=max(streaks),
                                                at_least_ten=sum(streak >= 10 for streak in streaks))
                _write(path, receipt)
                print('wrist', mode, seed, row['run'], flush=True)
    finally:
        arena.close()
        launcher.close()
        native.working.cleanup()
    return receipt


def variant(gain, delay=True):
    return dict(gain=float(gain), delay=bool(delay))


def diagnostic():
    modes = dict(prompt=None, hold=variant(0), full=variant(1))
    receipt = run(DIAGNOSTIC, starts(DIAGNOSTIC_SEED, 6), modes,
                  dict(gate='prompt beats hold by >=6 and full beats hold by >=3 combined streak counts'))
    scores = {name: entry['total'] for name, entry in receipt['summary'].items()}
    receipt['passed'] = scores['prompt'] - scores['hold'] >= 6 and scores['full'] - scores['hold'] >= 3
    _write(DIAGNOSTIC, receipt)
    print('diagnostic', scores, 'gate', receipt['passed'], flush=True)


def held():
    pilot = _read(DIAGNOSTIC)
    if not pilot['passed']:
        raise ValueError('registered visual-latency diagnostic failed')
    source = _read(SOURCE)
    gains = source['gains']
    seeds = starts(HELD_SEED, 12)
    if set(seeds).intersection(pilot['seeds']):
        raise ValueError('held wrist roster contains a diagnostic seed')
    modes = {name: variant(gains[name]) for name in ('cold', 'always', 'shuffled', 'pong')}
    modes.update(prompt=None, prompt_pong=variant(gains['pong'], delay=False))
    receipt = run(HELD, seeds, modes, dict(source_sha256=_hash(SOURCE), diagnostic_sha256=_hash(DIAGNOSTIC),
                                          target_gain_practice=0, camera_capture_time_public=True,
                                          reported_score='longest consecutive physical rebounds'))
    rows = receipt['conditions']
    parity = [a['actions'] == b['actions'] and a['run'] == b['run'] and a['rebounds'] == b['rebounds']
              for a, b in zip(rows['prompt'], rows['prompt_pong'], strict=True)]
    if not all(parity):
        raise ValueError('prompt wrist command sequence changed with the gain enabled')
    receipt['exact_prompt_retention'] = parity
    receipt['changes_vs_pong'] = {name: effects(rows['pong'], rows[name]) for name in ('cold', 'always', 'shuffled')}
    _write(HELD, receipt)
    print('held', receipt['summary'], 'paired Pong deltas', receipt['changes_vs_pong'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument('--diagnostic', action='store_true')
    stages.add_argument('--held', action='store_true')
    args = parser.parse_args()
    (diagnostic if args.diagnostic else held)()
