"""Fit and score an RGB-to-joint feedback policy after a shared physical opening."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_so101 import RATES, SO101_HOME
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt22_rally_trial import alternating
from .tt23_coach_trial import PRACTICE, ReplyRanking
from .tt23_sequence import CONFIG, HANDOVER, SequenceArena, verify
from .tt23_sequence_coach import OUTPUT as COACHED, STOCK
from .tt24_rally_policy import EYES, JointPolicy, ViewHistory


OUTPUT = ROOT / 'artifacts' / 'tt24'
DATA = OUTPUT / 'rally-demonstrations.npz'
RECEIPT = OUTPUT / 'rally-joint-feedback.json.gz'
STEP_SIZE = np.asarray(RATES) / 30.
CONDITIONS = ('hold', 'zero_demo', 'sight_delay', 'demo_1', 'demo_3', 'demo_full')
DEMO_SOURCE = (COACHED, STOCK)
# Split by release, never by frame or continuation from the same release.
DEVELOPMENT = (26147261, 527080401, 179271280, 1088961436, 1679068434)


def returns(row, side):
    return [event['step'] for event in row['events']
            if event['kind'] == 'legal_return' and event['side'] == side]


def contacts(row, side):
    return [event['step'] for event in row['events']
            if event['kind'] == 'contact' and event['side'] == side]


def archived_rows(path):
    for case in _read(path)['cases']:
        yield from (row for mode, row in case['results'].items()
                    if mode in ('neutral', 'duration') and row['opening_return'] and row['outcome']['rival_returns'])


def archive_schedule():
    candidates = {}
    for row in archived_rows(COACHED):
        key = row['seed']
        if key not in candidates or len(returns(row, 1)) > len(returns(candidates[key], 1)):
            candidates[key] = row
    # Select the archived opening by teacher outcome solely to build a
    # conditional post-opening course, and use its prefix for every branch.
    return [candidates[seed] for seed in candidates if seed not in DEVELOPMENT]


def demo_schedule(assessment):
    excluded = {row['seed'] for row in assessment}
    rows, seen = [], set()
    for path in DEMO_SOURCE:
        for row in archived_rows(path):
            signature = digest_json((row['seed'], row['actions'][:HANDOVER], row['rival_actions']))
            if row['seed'] not in excluded and signature not in seen and len(returns(row, 1)) >= 2:
                seen.add(signature)
                rows.append(row)
    rows.sort(key=lambda row: (DEVELOPMENT.index(row['seed']), -len(returns(row, 1))))
    return rows


class PublicBody:
    """Encode measured joints and the locally accumulated issued position."""

    def __init__(self, arena):
        sites = arena._arm_ids[1][2]
        self.lower = arena._joint_low[sites]
        self.upper = arena._joint_high[sites]
        self.target = np.asarray(SO101_HOME)

    def issued(self, command):
        self.target = np.clip(self.target + STEP_SIZE*command, self.lower, self.upper)


def render_demo(row, *, stop_at=None, first_sample=HANDOVER):
    arena = SequenceArena(**CONFIG)
    arena.lane = row['lane']
    arena.reset(seed=row['seed'], server=-1, render=False)
    history = ViewHistory()
    body = PublicBody(arena)
    features, postures, rates = [], [], []
    stop = min(len(row['actions']), returns(row, 1)[-1] + 1)
    if stop_at is not None:
        stop = min(stop, stop_at)
    try:
        for tick, (near, far) in enumerate(zip(row['actions'][:stop], row['rival_actions'][:stop], strict=True)):
            near, far = np.asarray(near), np.asarray(far)
            state = history.observe(arena.render_cameras(), arena.joints, body.target)
            arena._aim = arena._command_to = lambda *args, far=far: far
            _, _, done, capped, _ = arena.step(near, render=False)
            body.issued(near)
            history.issued(near)
            if tick >= first_sample:
                features.append(state)
                postures.append(body.target.copy())
                rates.append(near)
            if done or capped:
                break
        original = [event for event in row['events'] if event['step'] <= arena.steps]
        if digest_json(original) != digest_json(arena.events):
            raise ValueError(f'demonstration physical tape diverged for {row["seed"]}')
        print('teacher', row['seed'], len(features), len(returns(row, 1)), flush=True)
        return dict(features=np.asarray(features), postures=np.asarray(postures), rates=np.asarray(rates),
                    seed=np.full(len(features), row['seed']), frame_count=arena.steps)
    finally:
        arena.close()


def collect(path, workers):
    assessment = archive_schedule()
    schedule = demo_schedule(assessment)
    path.parent.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        blocks = list(map(render_demo, schedule))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
            blocks = list(pool.map(render_demo, schedule))
    stacked = {key: np.concatenate([block[key] for block in blocks]) for key in ('features', 'postures', 'rates', 'seed')}
    np.savez_compressed(path, **stacked, demo_seed=[row['seed'] for row in schedule],
                        demo_frames=[block['frame_count'] for block in blocks],
                        demo_returns=[len(returns(row, 1)) - 1 for row in schedule],
                        archive_hashes=[_hash(p) for p in DEMO_SOURCE])


class Station:
    def __init__(self, profile, *, render_dims=None):
        config = dict(CONFIG)
        if render_dims is not None:
            config.update(width=render_dims[0], height=render_dims[1])
        self.arena = SequenceArena(**config)
        self.rival = install_rally_opponent(self.arena, target_profile=profile)
        self.rival.confirm = ReplyRanking(self.rival, _read(PRACTICE)).confirm
        self.tape = FarTape(self.arena)

    def play(self, start, condition, policy, new_history=ViewHistory, *, execute=None, live_opening=False):
        arena = self.arena
        arena.lane = start['lane']
        arena.reset(seed=start['seed'], server=-1, render=False)
        self.tape.steps.clear()
        history = new_history()
        body = PublicBody(arena)
        actions, observed, requested = [], [], []
        prefix_rgb, rgb = sha256(), sha256()
        reward = 0
        handoff = 0 if live_opening else HANDOVER
        prefix = [] if live_opening else start['actions'][:handoff]
        for tick in range(arena.max_steps):
            frames = arena.render_cameras()
            for eye in EYES:
                rgb.update(eye.encode())
                rgb.update(frames[eye])
                if tick <= handoff:
                    prefix_rgb.update(eye.encode())
                    prefix_rgb.update(frames[eye])
            observed.append(history.observe(frames, arena.joints, body.target))
            state = observed[-1].copy()
            if condition == 'sight_delay':
                # Leave body feedback current and delay only the visual channels.
                state[:14] = observed[max(0, tick-6)][:14]
            if tick < handoff:
                command = np.asarray(prefix[tick])
            elif condition == 'hold':
                command = np.zeros(arena.action_size)
            else:
                command = policy.command(state, body.target)
            physical = command
            if execute is not None and tick >= handoff:
                physical = np.asarray(execute(command.copy()))
            _, reward, done, cap, _ = arena.step(physical, render=False)
            actions.append(physical.tolist())
            if execute is not None:
                requested.append(command.tolist())
            history.issued(command)
            body.issued(command)
            if done or cap:
                break
        row = dict(seed=start['seed'], lane=start['lane'], reward=reward, outcome=arena.point_result(),
                   longest=alternating(arena.events), events=arena.events.copy(),
                   rgb=rgb.hexdigest(), prefix_rgb=prefix_rgb.hexdigest(),
                   actions=actions, rival_actions=self.tape.tape(len(actions)))
        if execute is not None:
            row['requested_actions'] = requested
        if not live_opening:
            if prefix != actions[:handoff]:
                raise ValueError('shared teacher opening diverged')
            if not contacts(row, 1) or contacts(row, 1)[0] > handoff or not returns(row, 1):
                raise ValueError(f'shared teacher opening did not score on seed {start["seed"]}')
            validate_opening_timestamp(row, start)
        return row


def validate_opening_timestamp(row, start):
    if contacts(row, 1)[0] != contacts(start, 1)[0] or returns(row, 1)[0] != returns(start, 1)[0]:
        raise ValueError('archived opening contact or landing diverged')
    if row['rival_actions'][:HANDOVER] != start['rival_actions'][:HANDOVER]:
        raise ValueError('live far commands diverged during the teacher opening')


def make_policy(data, condition, *, neighbors=20, body=.22, mix=0.):
    n = dict(zero_demo=0, demo_1=1, demo_3=3).get(condition, len(np.unique(data['seed'])))
    seeds = list(dict.fromkeys(data['demo_seed'].tolist()))[:n]
    selected = np.isin(data['seed'], seeds)
    model = JointPolicy(data['features'][selected], data['postures'][selected], data['rates'][selected], STEP_SIZE,
                        neighbors=neighbors, body=body, mix=mix)
    return model, dict(seeds=seeds, tapes=int(np.isin(data['demo_seed'], seeds).sum()),
                       teacher_frames=int(data['demo_frames'][np.isin(data['demo_seed'], seeds)].sum()),
                       post_handoff_samples=int(selected.sum()))


def difference_count(row, other, handoff=HANDOVER):
    a, b = row['actions'][handoff:], other['actions'][handoff:]
    commands = [np.abs(np.asarray(x)-y) for x, y in zip(a, b)]
    difference = np.asarray(commands).reshape((-1, len(RATES))) > 1e-4
    return dict(overlap_frames=len(difference), changed_frames=int(np.any(difference, axis=1).sum()),
                changed_per_channel=difference.sum(axis=0).tolist())


def summary(cases, condition, handoff=HANDOVER):
    rows = [case['results'][condition] for case in cases if condition in case['results']]
    counts = [len(returns(row, 1)) for row in rows]
    def first_return(ordinal):
        frames = 0
        for index, row in enumerate(rows, 1):
            ticks = returns(row, 1)
            if len(ticks) >= ordinal:
                total = frames + ticks[ordinal - 1] - handoff
                return dict(point=index, frames=total, seconds=total/30.)
            frames += len(row['actions']) - handoff
        return None

    return dict(points=len(rows), reach_second=sum(n >= 2 for n in counts), reach_fifth=sum(n >= 5 for n in counts),
                own_returns=counts, total_own_returns=sum(counts), longest=[row['longest'] for row in rows],
                far_replies=[len(returns(row, -1)) for row in rows],
                pupil_contacts=[sum(tick > handoff for tick in contacts(row, 1)) for row in rows],
                pupil_frames=sum(len(row['actions'])-handoff for row in rows),
                first_second=first_return(2), first_fifth=first_return(5),
                reach_tenth=sum(n >= 10 for n in counts), first_tenth=first_return(10))


STATION = POLICIES = None


def prepare(profile, data_path, conditions, settings):
    global STATION, POLICIES
    STATION = Station(profile)
    with np.load(data_path) as data:
        POLICIES = {name: make_policy(data, name, **settings)[0] for name in conditions if name != 'hold'}


def play_case(index, start, conditions):
    return index, {name: STATION.play(start, name, POLICIES.get(name)) for name in conditions}


def run(path, data_path, conditions, development, n, settings, workers=2):
    data = np.load(data_path)
    if data['archive_hashes'].tolist() != [_hash(p) for p in DEMO_SOURCE]:
        raise ValueError('demonstration archive changed')
    starts = archive_schedule()
    if development:
        starts = [row for row in archived_rows(COACHED) if row['seed'] in DEVELOPMENT]
    starts = starts[:n]
    coach = _read(COACHED)
    profile = coach['metadata']['opening']['far_profile']
    exposure = {}
    for name in conditions:
        if name != 'hold':
            _, exposure[name] = make_policy(data, name, **settings)
    metadata = dict(schema='tt24-rgb-joint-feedback-1', development=development, handoff=HANDOVER,
                    pupil_observations=['fixed stereo RGB', 'measured learner joints', 'own issued rates'],
                    pupil_action='all six SO-101 normalized joint rates', axis_step_size=STEP_SIZE.tolist(),
                    demo_data_sha256=_hash(data_path), coach_sha256=_hash(PRACTICE),
                    settings=settings, conditions=list(conditions), exposure=exposure,
                    releases=[row['seed'] for row in starts])
    path.parent.mkdir(parents=True, exist_ok=True)
    report = _read(path) if path.exists() else dict(metadata=metadata, cases=[
        dict(seed=start['seed'], lane=start['lane'], reference_own_returns=returns(start, 1),
             teacher_prefix=start['actions'][:HANDOVER], results={}) for start in starts])
    if report['metadata'] != metadata:
        raise ValueError('existing rally receipt has different inputs')
    work = []
    for index, (start, case) in enumerate(zip(starts, report['cases'], strict=True)):
        order = conditions if index % 2 == 0 else conditions[::-1]
        missing = [name for name in order if name not in case['results']]
        if missing:
            work.append((index, start, missing))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(profile, data_path, conditions, settings)) as pool:
        tasks = [pool.submit(play_case, *job) for job in work]
        for task in as_completed(tasks):
            index, results = task.result()
            matched = report['cases'][index]['results']
            matched.update(results)
            if len({row['prefix_rgb'] for row in matched.values()}) != 1:
                raise ValueError('opening RGB differs across conditions')
            if 'hold' in matched:
                for row in matched.values():
                    row['vs_hold'] = difference_count(row, matched['hold'])
            _write(path, report)
            for name, row in results.items():
                print(name, row['seed'], row['longest'], returns(row, 1), row['outcome']['reason'], flush=True)
    report['verified_tapes'] = verify(report['cases'])
    report['summary'] = {name: summary(report['cases'], name) for name in conditions}
    _write(path, report)
    return report['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--neighbors', type=int, default=20)
    parser.add_argument('--body', type=float, default=.22)
    parser.add_argument('--mix', type=float, default=0.)
    args = parser.parse_args()
    if args.collect:
        collect(args.data, args.workers)
        return
    print(run(args.output, args.data, args.conditions, args.development, args.limit,
              dict(neighbors=args.neighbors, body=args.body, mix=args.mix), args.workers), flush=True)


if __name__ == '__main__':
    main()
