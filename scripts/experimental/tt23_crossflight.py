"""Test frozen native Pong direction policies on independent physical release draws."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt20_fixture import TT20Arena
from .tt22_duration_transfer import MACRO
from .tt22_joint_practice import READS
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer import MotorPrimitive
from .tt22_motor_transfer_trial import MotorArena
from .tt22_rally_trial import GentleDraws
from .tt23_anonymous_action import OPTIONS
from .tt23_anonymous_target import target_context
from .tt23_direction import CHOICES, MODES, NATIVE, SOURCE, arm_stats
from .tt23_optical_contact import OUT

SEED = 913186435
PER_SIGN = 16
START_MAGNITUDE = (.08, .19)
THROW_MAGNITUDE = (.08, .20)
SOURCE_ARCHIVES = ('anonymous-action-arm.json.gz', 'anonymous-replication-arm.json.gz', 'smooth-jog-arm.json.gz')
PLAN = OUT / 'crossflight-plan.json.gz'
BANK = OUT / 'crossflight-arm.json.gz'
RESULT = OUT / 'crossflight-results.json.gz'


class CrossFlightDraws(GentleDraws):
    """Vary only the two native lateral coordinates, preserving the other draws."""

    def __init__(self, rng, release):
        super().__init__(rng, .45, 1.)
        self.overrides = dict(zip(((-.30, .30), (-.28, .28)), release, strict=True))
        self.used = []

    def uniform(self, low, high):
        sample = super().uniform(low, high)
        if np.ndim(sample) == 0 and (low, high) in self.overrides:
            bounds = low, high
            self.used.append(bounds)
            return self.overrides[bounds]
        return sample


class CrossFlightArena(MotorArena):
    """Keep the stock physical first-return test with independent lateral draws."""

    def __init__(self):
        self.release = (0., 0.)
        self.commands = []
        self.snapshot = None
        super().__init__()
        self.flags = mujoco.mjtState.mjSTATE_INTEGRATION
        self.integration = np.empty(mujoco.mj_stateSize(self.model, self.flags))

    def reset(self, **kwargs):
        self.commands = []
        self.snapshot = None
        return super().reset(**kwargs)

    def _opening(self):
        if self.server == 1:
            return TT20Arena._opening(self)
        rng = self._rng
        draws = CrossFlightDraws(rng, self.release)
        self._rng = draws
        try:
            TT20Arena._opening(self)
        finally:
            self._rng = rng
        if draws.used != [(-.30, .30), (-.28, .28)]:
            raise ValueError('the native lateral release draws changed')
        self.initial = dict(ball=self.ball.tolist(), velocity=self.velocity.tolist(), spin=self.spin.tolist())

    def step(self, action, *, render=True):
        output = super().step(action, render=render)
        self.commands.append(np.asarray(action).tolist())
        if len(self.commands) == READS:
            mujoco.mj_getState(self.model, self.data, self.integration, self.flags)
            self.snapshot = digest_json(self.integration.tolist())
        return output


def schedule(seed):
    seed_rng, lane_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    signs = [(x, v) for x in (-1, 1) for v in (-1, 1) for _ in range(PER_SIGN)]
    lane_rng.shuffle(signs)
    launches = []
    for launch, (x, v) in zip(seed_rng.integers(2**31, size=len(signs)), signs, strict=True):
        start = x * lane_rng.uniform(*START_MAGNITUDE)
        offset = v * lane_rng.uniform(*THROW_MAGNITUDE)
        launches.append(dict(seed=int(launch), start_x_draw=float(start), throw_offset_draw=float(offset),
                             family='same' if x == v else 'opposite', signs=[x, v]))
    return launches


def freeze(path):
    roster = schedule(SEED)
    fresh = {row['seed'] for row in roster}
    earlier = {row['seed'] for name in SOURCE_ARCHIVES for row in _read(OUT / name)['rows']}
    native = _read(NATIVE)
    if len(fresh) != len(roster) or fresh & earlier or _hash(SOURCE) != native['source_sha256']:
        raise ValueError('the frozen source or disjoint launch schedule changed')
    planned = dict(schema='tt23-crossflight-plan-1', generator=SEED, roster=roster, count=len(roster),
                   source_sha256=_hash(SOURCE), native_sha256=_hash(NATIVE), macro_sha256=_hash(MACRO),
                   options={str(i):float(OPTIONS[i]) for i in CHOICES}, no_sighting_choice=3,
                   prefix_frames=READS)
    if path.exists():
        if _read(path) != planned:
            raise ValueError('the newly generated course differs from the frozen one')
    else:
        _write(path, planned)
    return planned


def physical(arena, row, primitive=None, *, tape=None):
    arena.reset(seed=row['seed'], server=-1, render=False)
    limit = arena.max_steps if tape is None else len(tape)
    for tick in range(limit):
        if tape is None:
            action = np.zeros(6) if tick < READS else primitive.command(tick)
        else:
            action = tape[tick]
        _, reward, done, cap, info = arena.step(action, render=False)
        if done or cap:
            if tape is not None and tick != len(tape) - 1:
                raise ValueError('physical replay terminated early')
            break
    if not done and not cap:
        raise ValueError('expected a terminal physical point')
    return dict(ticks=arena.steps, contacts=info['contacts'], legal=info['legal_returns'], fault=info['reason'],
                reward=reward, events_sha256=digest_json(arena.events), prefix=arena.snapshot)


def options(arena, start, macro, ready, observed, snapshot, initial):
    outcome = [None] * len(OPTIONS)
    for index in CHOICES:
        primitive = MotorPrimitive(macro, None)
        if observed:
            primitive._translate(ready[0] + OPTIONS[index] * .848, ready)
        scored = physical(arena, start, primitive)
        actions, events = arena.commands, arena.events.copy()
        if arena.snapshot != snapshot or arena.initial != initial:
            raise ValueError('the real option changed the release or stationary prefix')
        if physical(arena, start, tape=actions) != scored:
            raise ValueError('strict physics did not reproduce the terminal point')
        outcome[index] = dict(scored, actions=actions, events=events)
    return outcome


def collect(plan_path, output):
    plan = freeze(plan_path)
    metadata = dict(schema='tt23-crossflight-arm-1', plan_sha256=_hash(plan_path), generator=SEED,
                    source_sha256=plan['source_sha256'], macro_sha256=plan['macro_sha256'])
    bank = _read(output) if output.exists() else dict(metadata=metadata, rows=[])
    if bank['metadata'] != metadata:
        raise ValueError('cannot combine different physical plans')
    for row, expected in zip(bank['rows'], plan['roster'][:len(bank['rows'])], strict=True):
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError('existing physical rows changed the frozen schedule')
    if bank.get('verified') == plan['count'] * len(CHOICES):
        return bank
    macro = np.asarray(_read(MACRO)['parameters'])
    ready = MotorPrimitive(macro, None).ready_world
    arena = CrossFlightArena()
    began = perf_counter()
    try:
        for start in plan['roster'][len(bank['rows']):]:
            arena.release = start['start_x_draw'], start['throw_offset_draw']
            sight = prefix(arena, start['seed'], 0.)
            initial, snapshot = arena.initial, arena.snapshot
            observed = target_context(sight, ready) is not None
            trials = options(arena, start, macro, ready, observed, snapshot, initial)
            bank['rows'].append(dict(start, sight=sight, observed=observed, initial=initial,
                                     prefix=snapshot, options=trials))
            if len(bank['rows']) % 4 == 0:
                _write(output, bank)
                print('confirmed physical launches', len(bank['rows']), round(perf_counter() - began, 1), flush=True)
    finally:
        arena.close()
    bank['verified'] = len(bank['rows']) * len(CHOICES)
    _write(output, bank)
    return bank


def group_quality(rows, contexts, choices):
    valid = np.array([context is not None for context in contexts])
    if not valid.any():
        return dict(detected=0, opposite_optical_motion=0)
    detected = np.asarray([row['sight']['mover'] for row, observed in zip(rows, valid, strict=True) if observed])
    optical = np.asarray([context for context in contexts if context is not None])
    direction = np.sign(np.asarray(choices)[valid] - 3)
    mixed = optical[:, 0] * optical[:, 1] < 0.
    return dict(detected=int(valid.sum()), opposite_optical_motion=int(mixed.sum()),
                mixed_choices_toward_position=int(np.sum(mixed & (direction == np.sign(optical[:, 0])))),
                mixed_choices_toward_motion=int(np.sum(mixed & (direction == np.sign(optical[:, 1])))),
                observed_lateral_median_abs_x_m=float(np.median(abs(detected[:, 0, 0]))),
                observed_lateral_median_abs_speed_mps=float(30 * np.median(abs(detected[:, 1, 0]))))


def assessments(bank_path, output):
    bank, source, native = (_read(path) for path in (bank_path, SOURCE, NATIVE))
    rows = bank['rows']
    if (bank['verified'] != 3 * 4 * PER_SIGN or len(rows) != 4 * PER_SIGN
            or bank['metadata']['source_sha256'] != _hash(SOURCE) or native['source_sha256'] != _hash(SOURCE)):
        raise ValueError('use the fully verified physical bank and original frozen source')
    ready = MotorPrimitive(_read(MACRO)['parameters'], None).ready_world
    context = [target_context(row['sight'], ready) for row in rows]
    group_names = ('same', 'opposite')
    families = {name: [i for i, row in enumerate(rows) if row['family'] == name] for name in group_names}
    experiments = {}
    for group, indices in dict(all=list(range(len(rows))), **families).items():
        subset = [rows[i] for i in indices]
        observed = [context[i] for i in indices]
        pooled_score = arm_stats(subset, observed, source['pooled'])
        experiments[group] = dict(
            pooled=pooled_score,
            independent=[dict(offset=fit['offset'], **arm_stats(subset, observed, fit)) for fit in source['sets']],
            optics=group_quality(subset, observed, pooled_score['scores']['learned']['choices']),
        )
    target = np.array([[False if trial is None else trial['legal'] > 0 for trial in row['options']] for row in rows])
    whole = experiments['all']
    held = np.arange(len(rows))
    differences = [target[held, fit['scores']['learned']['choices']].astype(int)
                   - target[held, fit['scores']['shuffled']['choices']].astype(int) for fit in whole['independent']]
    difference = np.mean(differences, axis=0)
    rng = np.random.default_rng(SEED + 1)
    resamples = np.concatenate([rng.choice(index, size=(4096, len(index)), replace=True)
                                for index in families.values()], axis=1)
    interval = np.quantile(difference[resamples].sum(axis=1), [.025, .975]).tolist()
    reproducible = sum(np.sum(diff) > 0 for diff in differences)
    pooled = {name: row['legal'] for name, row in whole['pooled']['scores'].items()}
    opposite = {name: row['legal'] for name, row in experiments['opposite']['pooled']['scores'].items()}
    gate = (native['gate'] and pooled['learned'] >= 3 + max(pooled['neutral'], pooled['shuffled'])
            and pooled['learned'] > pooled['reversed']
            and opposite['learned'] > max(opposite['neutral'], opposite['shuffled'])
            and reproducible >= 4 and interval[0] > 0.)
    report = dict(schema='tt23-crossflight-results-1', bank_sha256=_hash(bank_path), source_sha256=_hash(SOURCE),
                  native_sha256=_hash(NATIVE), groups=experiments, independent_improvements=int(reproducible),
                  mean_independent_gain=float(difference.sum()), held_bootstrap95=interval, full_point_gate=bool(gate))
    _write(output, report)
    print('pool returns', pooled, 'opposite returns', opposite, 'matched independent wins', reproducible,
          'independent gain interval', interval, 'follow-up gate', gate, flush=True)
    return report


def plot(bank_path, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    optical_ready = MotorPrimitive(_read(MACRO)['parameters'], None).ready_world[0]
    fig, panels = plt.subplots(1, 2, figsize=(9.3, 3.8), layout='constrained', sharex=True, sharey=True)
    rows = _read(bank_path)['rows']
    first_tick = min(READS - len(row['sight']['track']) for row in rows if row['observed'])
    for ax, label, title in zip(panels, ('same', 'opposite'), ('Same-sign release draws', 'Opposite-sign release draws')):
        for row in rows:
            if row['family'] != label or not row['observed']:
                continue
            path = np.asarray(row['sight']['track'])
            reflected = 1000 * row['signs'][0] * (path[:, 0] - optical_ready)
            ax.plot(np.arange(READS - len(path), READS), reflected, alpha=.32, color='#147786', marker='.', markersize=2)
        ax.axhline(0., linewidth=.6, color='#4a4a4a')
        ax.set_title(title)
        ax.set_xlabel('Camera frame in the eight-frame prefix')
        ax.set_xticks(range(first_tick, READS))
        ax.grid(axis='y', alpha=.22)
    panels[0].set_ylabel('Lateral position from blade (mm)\nReflected by launch side')
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--bank', type=Path, default=BANK)
    parser.add_argument('--output', type=Path, default=RESULT)
    parser.add_argument('--chart', type=Path)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--freeze-only', action='store_true')
    phase.add_argument('--assess-only', action='store_true')
    args = parser.parse_args()
    if args.freeze_only:
        freeze(args.plan)
        return
    if not args.assess_only:
        collect(args.plan, args.bank)
    assessments(args.bank, args.output)
    if args.chart:
        plot(args.bank, args.chart)


if __name__ == '__main__':
    main()
