"""Adapt a warm stereo receiver between live points on the solid compact court."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
from multiprocessing import get_context

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt22_rally_trial import alternating
from .tt23_coach_trial import PRACTICE, ReplyRanking
from .tt23_sequence import CONFIG
from .tt24_rally_bank import CACHE, DIMS
from .tt24_rally_improve_combined import DATA as COMBINED, PhaseExtent
from .tt24_rally_improve_optical import CACHE as OPTICAL, ExtentHistory, OpticalPolicy
from .tt24_rally_improve_phase import PhasePolicy
from .tt24_rally_improve_trial import policy_for
from .tt24_rally_policy import EYES, ViewHistory
from .tt24_rally_trial import COACHED, PublicBody, returns
from .tt25_arm_replay import NOMINAL_NO_LOAD, SolidArm


OUTPUT = ROOT / 'artifacts' / 'tt27'
MODES = ('retained', 'reset', 'frozen')
OPTIONS = {
    'combined': ('combined', 1., 1.),
    'retina': ('retina', 1., 1.),
    'optical': ('optical', 1., 1.),
    'steady': ('combined', .86, 1.),
    'early': ('combined', 1., 1.5),
}
WARM_PRIORS = {'combined': .44, 'retina': .425, 'optical': .41}
# Each successive block changes one parameter. Stock returns last.
STAGES = {
    'stock': (0., 0., 0),
    'placement': (.18, 0., 0),
    'rebound': (.18, -1.35, 0),
    'latency': (.18, -1.35, 1),
    'stock_again': (0., 0., 0),
}
SEEDS = (1366073058, 1823940580)
DEVELOPMENT = 395947315
MOVIE_SEED = 1261056994


def schedule(seed, count, *, development=False):
    rng = np.random.default_rng(seed)
    blocks = {}
    for stage, (shift, damping, latency) in STAGES.items():
        if development and stage not in ('stock', 'latency'):
            continue
        rows = []
        for i in range(count):
            width = (.18, .49) if i % 2 == 0 else (.35, .69)
            lane = shift + (-1 if i % 4 < 2 else 1)*rng.uniform(*width)
            rows.append(dict(seed=int(rng.integers(1, 2**31 - 1)), lane=float(lane)))
        blocks[stage] = rows
    return blocks


def audit(courses):
    seeds = [row['seed'] for roster in courses for block in roster.values() for row in block]
    if len(seeds) != len(set(seeds)):
        raise ValueError('assessment seed reused')
    for path in (CACHE, OPTICAL, COMBINED):
        with np.load(path) as data:
            if set(seeds).intersection(map(int, data['seed'])):
                raise ValueError('assessment seed present in demonstrations')
    literals = [str(seed).encode() for seed in seeds]
    numeric = b'0123456789.'
    scans = 0
    for number in range(22, 27):
        for path in (ROOT / 'artifacts' / f'tt{number}').glob('*.json*'):
            if path.name.endswith('.json.gz'):
                contents = gzip.decompress(path.read_bytes())
            elif path.suffix == '.json':
                contents = path.read_bytes()
            else:
                continue
            for literal in literals:
                pos = contents.find(literal)
                while pos != -1:
                    after = pos + len(literal)
                    if ((pos == 0 or contents[pos - 1] not in numeric)
                            and (after == len(contents) or contents[after] not in numeric)):
                        raise ValueError(f'assessment seed present in {path}')
                    pos = contents.find(literal, pos + 1)
            scans += 1
    return scans


class Options:
    def __init__(self):
        with np.load(CACHE) as rates, np.load(OPTICAL) as optics, np.load(COMBINED) as extra:
            retina = policy_for('retina_rate', rates)
            optical = OpticalPolicy(optics, 'interaction')
            combined = PhasePolicy(optical, OpticalPolicy(extra, 'interaction'), .88)
            sphere = optics['sphere']
        self.models = {
            'combined': (combined, lambda: PhaseExtent(sphere)),
            'retina': (retina, ViewHistory),
            'optical': (optical, lambda: ExtentHistory(sphere)),
        }

    def new(self, name):
        model, smooth, lead = OPTIONS[name]
        actor, observe = self.models[model]
        return Controller(actor, observe(), smooth, lead)


class Controller:
    """Read the last delivered pixels and their public capture age."""

    def __init__(self, actor, history, smooth, lead):
        self.actor = actor
        self.history = history
        self.smooth = smooth
        self.lead = lead
        self.previous = None

    def command(self, frames, capture, tick, joints, target):
        state = self.history.observe(frames, joints, target)
        # Only observation time is supplied. Acceleration, contact and actual
        # ball coordinates are unavailable to the receiver.
        age = self.lead*(tick-capture)
        if age:
            for eye in (0, 7):
                if state[eye+6] <= 1:
                    state[eye:eye+2] += age*state[eye+2:eye+4]
            if len(state) >= 54 and state[49] <= 1:
                advance = age*state[41:44]
                state[38:41] += advance
                state[50:53] += advance
        command = self.actor.command(state, target)
        if self.previous is not None:
            command = self.smooth*command + (1-self.smooth)*self.previous
        self.previous = command
        self.history.issued(command)
        return command


class CompactStation:
    def __init__(self, stage):
        _, _, self.latency = STAGES[stage]
        self.arena = self.make_arena(stage, width=DIMS[0], height=DIMS[1])
        self.replayer = self.make_arena(stage)
        profile = _read(COACHED)['metadata']['opening']['far_profile']
        self.rival = install_rally_opponent(self.arena, target_profile=profile)
        self.rival.confirm = ReplyRanking(self.rival, _read(PRACTICE)).confirm
        self.tape = FarTape(self.arena)
        self.axes = [self.arena.model.joint('learner_' + axis).dofadr[0] for axis in (
            'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')]

    @staticmethod
    def make_arena(stage, **dimensions):
        _, damping, _ = STAGES[stage]
        arena = SolidArm(**dict(CONFIG, **dimensions))
        for panel in arena.table_panels:
            arena.model.geom_solref[arena.model.geom(panel).id, 1] += damping
        return arena

    def close(self):
        self.arena.close()
        self.replayer.close()

    def play(self, start, controller):
        arena = self.arena
        arena.lane = start['lane']
        arena.reset(seed=start['seed'], server=-1, render=False)
        self.tape.steps.clear()
        body = PublicBody(arena)
        camera = deque(maxlen=self.latency+1)
        actions, speeds = [], []
        initial, stereo = sha256(), sha256()
        for tick in range(arena.max_steps):
            frames = arena.render_cameras()
            for eye in EYES:
                stereo.update(eye.encode())
                stereo.update(frames[eye])
                if tick == 0:
                    initial.update(frames[eye])
            camera.append((tick, frames))
            capture, observed = camera[0]
            command = controller.command(observed, capture, tick, arena.joints, body.target)
            _, reward, done, cap, _ = arena.step(command, render=False)
            body.issued(command)
            actions.append(command.tolist())
            speeds.append(arena.data.qvel[self.axes].copy())
            if done or cap:
                break
        active = np.array(actions)[:, :5]
        signs = np.where(abs(active) < .1, 0, np.sign(active))
        reversals = int(np.count_nonzero(signs[1:]*signs[:-1] < 0.))
        measured = np.abs(np.asarray(speeds))
        result = dict(start, reward=reward, outcome=arena.point_result(), longest=alternating(arena.events),
                      events=arena.events.copy(), actions=actions, rival_actions=self.tape.tape(len(actions)),
                      first_rgb=initial.hexdigest(), rgb=stereo.hexdigest(), naturalness=dict(reversals=reversals,
                      mean_rate_change=float(np.abs(np.diff(active, axis=0)).mean()),
                      peak_motor_speed=float(measured.max()), joint_speed_p95=float(np.quantile(measured, .95)),
                      over_nominal=int(np.count_nonzero(measured.max(axis=1) > NOMINAL_NO_LOAD))))
        self.verify(result)
        return result

    def verify(self, result):
        arena = self.replayer
        arena.lane = result['lane']
        arena.reset(seed=result['seed'], server=-1, render=False)
        for tick, (near, far) in enumerate(zip(result['actions'], result['rival_actions'], strict=True), 1):
            rival = np.asarray(far)
            arena._aim = arena._command_to = lambda *args, rival=rival: rival
            _, reward, done, cap, _ = arena.step(near, render=False)
            if bool(done or cap) != (tick == len(result['actions'])):
                raise ValueError('point duration changed during replay')
        if (result['reward'] != reward or result['outcome'] != arena.point_result()
                or digest_json(result['events']) != digest_json(arena.events)):
            raise ValueError('point observations changed during replay')


class RallyChoice:
    """Update an option's utility once at the end of each real point."""

    def __init__(self, options):
        self.options = tuple(options)
        self.initial = np.asarray(list(options.values()), dtype=float)
        self.reset()

    def reset(self):
        self.total = np.zeros(len(self.options))
        self.count = np.zeros(len(self.options))

    def choose(self):
        utility = (self.total + 2*self.initial)/(self.count + 2)
        bonus = .22/np.sqrt(self.count + 1)
        return int(np.argmax(utility + bonus))

    def observe(self, choice, legal):
        # Utility is capped so one unusually long exchange cannot drown the
        # much more frequent first-return successes and failures.
        self.total *= .93
        self.count *= .93
        self.total[choice] += .65*(legal > 0) + .35*min(max(legal-1, 0), 4)/4
        self.count[choice] += 1


def brief(rows):
    near = np.asarray([len(returns(row, 1)) for row in rows])
    far = np.asarray([len(returns(row, -1)) for row in rows])
    lengths = np.asarray([row['longest'] for row in rows])
    ticks = sum(len(row['actions']) for row in rows)
    return dict(points=len(rows), first=int((near >= 1).sum()), second=int((near >= 2).sum()),
                fifth=int((near >= 5).sum()), near=int(near.sum()), far=int(far.sum()),
                ten=int((lengths >= 10).sum()), longest=int(lengths.max()),
                two_own_before=int(np.argmax(near >= 2) + 1) if (near >= 2).any() else None,
                misses_before_two_own=int((near[:np.argmax(near >= 2)] == 0).sum()) if (near >= 2).any() else int((near == 0).sum()),
                reversals_per_second=30*sum(row['naturalness']['reversals'] for row in rows)/ticks,
                mean_rate_change=float(np.average([row['naturalness']['mean_rate_change'] for row in rows],
                                                 weights=[len(row['actions']) for row in rows])),
                above_nominal=100*sum(row['naturalness']['over_nominal'] for row in rows)/ticks,
                peak_motor_speed=max(row['naturalness']['peak_motor_speed'] for row in rows), ticks=ticks)


def assess(task):
    seed, count, priors, development = task
    options = Options()
    modes = tuple(OPTIONS) if development else MODES
    selectors = {mode: RallyChoice(priors) for mode in modes} if not development else {}
    results = {}
    for stage, roster in schedule(seed, count, development=development).items():
        station = CompactStation(stage)
        if not development:
            selectors['reset'].reset()
        cases = []
        try:
            for start in roster:
                branches = {}
                # Identical option on an identical physical start would give
                # identical deterministic play. Cache only within this pair.
                played = {}
                for mode in modes:
                    selector = selectors.get(mode)
                    chosen = selector.choose() if selector is not None else None
                    name = (mode if development else next(iter(priors)) if mode == 'frozen'
                            else selector.options[chosen])
                    key = OPTIONS[name] if station.latency else OPTIONS[name][:2]
                    if key not in played:
                        played[key] = station.play(start, options.new(name))
                    row = played[key]
                    branches[mode] = dict(row, option=name)
                    if selector is not None and mode != 'frozen':
                        selector.observe(chosen, len(returns(row, 1)))
                if len({row['first_rgb'] for row in branches.values()}) != 1:
                    raise ValueError('paired image at release changed')
                cases.append(branches)
                print(seed, stage, start['seed'], {mode: (row['option'], len(returns(row, 1)), row['longest'])
                                                    for mode, row in branches.items()}, flush=True)
        finally:
            station.close()
        results[stage] = cases
    return results


def summaries(results, modes):
    stages = next(iter(results.values())).keys()
    by_stage = {stage: {mode: brief([case[mode] for course in results.values() for case in course[stage]])
                        for mode in modes} for stage in stages}
    total = {mode: brief([case[mode] for course in results.values() for cases in course.values() for case in cases])
             for mode in modes}
    return dict(stages=by_stage, total=total)


def run(*, develop, workers, count, streams):
    if count < 1:
        raise ValueError('each block needs at least one point')
    pilot = OUTPUT / 'compact-development.json.gz'
    cohorts = (DEVELOPMENT,) if develop else SEEDS[:streams]
    priors = {} if develop else WARM_PRIORS
    courses = [schedule(seed, count, development=develop) for seed in cohorts]
    if not develop:
        pilot_count = _read(pilot)['metadata']['count'] if pilot.exists() else 4
        courses.append(schedule(DEVELOPMENT, pilot_count, development=True))
    meta = dict(schema='tt27-compact-1', count=count, cohorts=list(cohorts), stages=STAGES, options=OPTIONS,
                priors=priors, archives_checked=audit(courses),
                training={path.name: _hash(path) for path in (CACHE, OPTICAL, COMBINED)},
                opponent_reference_sha256=_hash(PRACTICE),
                release='fresh opposing serve in strict solid court, no pupil command prefix',
                policy_inputs=['stereo RGB', 'camera capture tick', 'current control tick', 'near measured joint positions',
                               'own issued command history', 'terminal legal near-return count for adaptive option choice'])
    OUTPUT.mkdir(exist_ok=True)
    path = pilot if develop else OUTPUT / 'compact-assessment.json.gz'
    tasks = [(seed, count, priors, develop) for seed in cohorts]
    if workers == 1 or len(tasks) == 1:
        outcomes = [assess(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
            outcomes = list(pool.map(assess, tasks))
    results = dict(zip(cohorts, outcomes, strict=True))
    modes = tuple(OPTIONS) if develop else MODES
    summary = summaries(results, modes)
    unique = sum(len({OPTIONS[row['option']] if STAGES[stage][2] else OPTIONS[row['option']][:2]
                      for row in case.values()}) for result in results.values()
                 for stage, cases in result.items() for case in cases)
    _write(path, dict(metadata=meta, cases=results, summary=summary, verified_unique_points=unique))
    print(summary, flush=True)


def movie():
    from .tt23_sequence_coach_movie import render_tape

    source = OUTPUT / 'compact-assessment.json.gz'
    data = _read(source)
    candidates = [(stage, row['retained']) for stream in data['cases'].values()
                  for stage, cases in stream.items() for row in cases]
    stage, selected = next((stage, row) for stage, row in candidates if row['seed'] == MOVIE_SEED)
    if (selected['outcome']['legal_returns'], selected['outcome']['rival_returns']) != (7, 7):
        raise ValueError('illustrated point no longer has the verified seven returns per side')
    render_tape(selected, source, OUTPUT / 'compact-live.mp4', branch='retained',
                subtitle='Solid compact court · Changed table rebound · 30 fps',
                terminal_caption='14 alternating returns · Point shown at normal speed',
                arena_factory=lambda: CompactStation.make_arena(stage, width=DIMS[0], height=DIMS[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--develop', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--streams', type=int, choices=(1, 2), default=1)
    parser.add_argument('--movie', action='store_true')
    args = parser.parse_args()
    if args.movie:
        movie()
    else:
        run(develop=args.develop, workers=args.workers, count=args.count, streams=args.streams)
