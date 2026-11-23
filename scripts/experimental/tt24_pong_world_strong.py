"""Extract actual optical motion from the portable, score-trained guarded Pong actor."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt24_pong_world import OUTPUT, RAW as MODEST, fit
from .tt24_pong_world_train import register, run
from .tt24_source_native import Scene


PORTABLE = ROOT / 'artifacts' / 'm1b' / '2026-11-19-score-placement-confirmation.zip'
PREFIX = 'pong-world-strong'
RAW = OUTPUT / f'{PREFIX}-native.json.gz'
CUT = OUTPUT / f'{PREFIX}-initial.json'
PLAN = OUTPUT / f'{PREFIX}-plan.json'
RECEIPT = OUTPUT / f'{PREFIX}-learning.json.gz'
SCHEDULE = {'practice': 1451016241, 'held': 720454529}
TARGET_SCHEDULE = 10077375542
SIZE = {'practice': 48, 'held': 24}
HORIZON = (12, 22)
DT = .02
AUDIT_STARTS = 3
RUNTIME = None


def from_archive():
    global RUNTIME
    torch.set_num_threads(1)
    storage = TemporaryDirectory(prefix='tt24-pong-guarded-')
    path = Path(storage.name)
    with ZipFile(PORTABLE) as archive:
        lineage = json.loads(archive.read('candidate-lineage.json'))
        checkpoints = {name: json.loads(archive.read(f'checkpoints/{name}.json')) for name in ('motor', 'timing', 'score')}
        for file, digest in lineage['qualification_runtime'].items():
            bytes_ = archive.read(f'runtime/{file}')
            if sha256(bytes_).hexdigest() != digest:
                raise ValueError('portable runtime differs from qualified archive lineage')
            (path / file).write_bytes(bytes_)
        for name, digest in lineage['evaluation_checkpoints'].items():
            if sha256(archive.read(f'checkpoints/{name}.json')).hexdigest() != digest:
                raise ValueError('portable checkpoint differs from qualified policy lineage')
    sys.path[:0] = [str(path), str(ROOT / 'scripts' / 'experimental')]
    import pong_risk_aware_placement as guarded
    import pong_score_placement as score
    import pong_transfer_arena as original

    if Path(original.__file__).parent != path:
        raise ValueError('execute the archived guarded runtime, including its earlier camera wrapper')
    RUNTIME = storage, checkpoints, guarded, score, original


def view(entities, controller, frame):
    chosen = controller.timing.inherited.bound_ids
    if chosen is None:
        return None
    observed = {row['id']: row for row in entities if row['observed']}
    if any(key not in observed for key in chosen):
        return None
    actor, mover = [np.asarray(observed[key]['normalized_position']) for key in chosen]
    rgb = np.broadcast_to(frame.numpy()[..., None], (*frame.shape, 3))
    # Component boxes use column, row. The inherited tracker uses row, column.
    blades = [(center[::-1] - .5, span) for center, span, aspect in Scene.regions(rgb) if aspect < .45]
    apparent = min(blades, key=lambda x: np.linalg.norm(x[0] - actor), default=None)
    if apparent is None or np.linalg.norm(apparent[0] - actor) > .055:
        return None
    _, span = apparent
    return actor, mover, span, int(controller.timing.axis), tuple(chosen)


def collect_observations(path):
    finite = [x for x in path if x is not None]
    if not finite:
        return []
    initial = finite[0][0]
    time = DT * (np.arange(5) - 2.)
    examples = []
    for tick in range(8, len(path) - HORIZON[1], 3):
        current = path[tick]
        past = path[tick - 4:tick + 1]
        if current is None or any(view is None or view[4] != current[4] for view in past):
            continue
        actor, mover, span, axis, identity, intent = current
        future = [view[0][axis] for view in path[tick + HORIZON[0]:tick + HORIZON[1]]
                  if view is not None and view[4][0] == identity[0]]
        depth = mover[1 - axis] - actor[1 - axis]
        if len(future) < 4 or not (.42 < abs(depth) / span < 4.5 and np.linalg.norm(mover - actor) < 5.2 * span):
            continue
        speed = time @ np.array([row[1] for row in past]) / (time @ time)
        examples.append(dict(tick=tick, tangent_gap=float(mover[axis] - actor[axis]), distance=float(abs(depth)),
                             tangent_speed=float(speed[axis]), closing_speed=float(np.sign(depth) * speed[1 - axis]),
                             current=float(actor[axis] - initial[axis]), later=float(np.mean(future) - initial[axis]),
                             apparent_span=float(span), signed_alias=int(intent)))
    return examples


def play(launch, policy_seed, opponent):
    """Mirror archived game steps and add optical measurements outside the actor."""
    _, checkpoints, guarded, score, original = RUNTIME
    frozen = score.ScorePlacement(**checkpoints['score'], updates_enabled=False)
    controller = guarded.build_risk_aware_controller(policy_seed, checkpoints['motor'], checkpoints['timing'], frozen,
                                                     timing_updates=True)
    if controller.timing.inherited.policy.require_first_hit:
        raise ValueError('the source actor must run without a private contact gate')
    arena = original.PongCamera(launch, max_episode_ticks=1440, opponent_speed=opponent)
    counts = original.ReturnCounter()
    game = arena.game
    native_step = game.step
    game.step = lambda drive, **kwargs: counts.step(game, native_step, drive, **kwargs)
    vision, encoder = original.load_default(), original.VisualStateEncoder(64, 64, DT)

    def observe(frame, prior):
        events = torch.stack((prior & ~frame, ~prior & frame)).float()
        return original.snapshot(encoder.encode_entities(*vision.track_entities(events, frame.float())))

    frame = game.render(64, 64)[0].clone()
    entities = observe(frame, torch.zeros_like(frame))
    action_tape = bytearray()
    measurements, contacts = [], []
    optical_digest = sha256()
    while True:
        action, _ = controller.choose(entities, 1)
        action_tape.append(action)
        raw = view(entities, controller, frame)
        if raw is None:
            measurements.append(None)
        else:
            order = controller.timing.effective_requests
            intent = order.index(action) - 1
            measurements.append((*raw, intent))
            optical_digest.update(np.asarray([*raw[0], *raw[1], raw[2]], dtype='<f4').tobytes())
        before = counts.opportunities
        result = arena.step(action)
        entities = observe(result.frame, frame)
        frame = result.frame
        controller.observe(public_entities=entities, own_hits=0, reward=result.reward,
                           terminated=result.terminated, truncated=result.truncated, elapsed_seconds=result.elapsed_seconds)
        if before != counts.opportunities:
            contacts.append([arena.total_ticks / 120., bool(result.hits)])
        if result.terminated or result.truncated:
            return dict(seed=launch, policy_seed=int(policy_seed), opponent=opponent, contacts=contacts,
                        hits=counts.returns, opportunities=counts.opportunities, score=int(result.reward),
                        own_hit_win=int(result.reward > 0 and counts.returns > 0), elapsed=arena.total_ticks / 120.,
                        cameras=len(action_tape), action_sha256=sha256(action_tape).hexdigest(),
                        optical_sha256=optical_digest.hexdigest(), aims=len(controller.placement.choices),
                        cancelled=len(controller.cancelled), case=dict(serve=launch),
                        cues=collect_observations(measurements))


def audit():
    pairs = []
    with ZipFile(PORTABLE) as archive:
        schedule = json.loads(archive.read('schedules.json'))
        for task, opponent in (('retention', .65), ('scoring', .35)):
            for i, launch in enumerate(schedule[task][:AUDIT_STARTS]):
                row = json.loads(archive.read(f'run/{task}/trained/{launch}.json'))
                pairs.append((row, (launch, 1245 + i, opponent)))
    from_archive()
    checks = []
    for reference, specification in pairs:
        actual = play(*specification)
        fields = ('hits', 'opportunities', 'score', 'cameras', 'action_sha256', 'elapsed', 'contacts')
        if any(actual[key] != reference[key] for key in fields):
            failing = [key for key in fields if actual[key] != reference[key]]
            raise ValueError(f'guarded archive parity failed for launch {actual["seed"]}, fields {failing}')
        checks.append(dict(seed=actual['seed'], actions=actual['cameras'], action_sha256=actual['action_sha256'],
                           cues=len(actual['cues'])))
    if sum(row['cues'] for row in checks) < 60:
        raise ValueError('valid optical associations are missing in the replayed archive points')
    return checks


def dispatch(arguments):
    return play(*arguments)


def summary(rows):
    return dict(first=sum(bool(row['contacts'] and row['contacts'][0][1]) for row in rows),
                first_opportunities=sum(bool(row['contacts']) for row in rows),
                later=sum(sum(x[1] for x in row['contacts'][1:]) for row in rows),
                later_opportunities=sum(len(row['contacts'][1:]) for row in rows),
                own_wins=sum(row['own_hit_win'] for row in rows), time_s=sum(row['elapsed'] for row in rows),
                observations=sum(len(row['cues']) for row in rows), aimed=sum(row['aims'] for row in rows),
                cancelled=sum(row['cancelled'] for row in rows))


def capture(workers):
    start = perf_counter()
    reserved = _read(MODEST)
    known = {c['serve'] for rows in reserved['metadata']['cases'].values() for c in rows}
    with ZipFile(PORTABLE) as archive:
        known.update(seed for entries in json.loads(archive.read('schedules.json')).values() for seed in entries)
        known.update(seed for entries in json.loads(archive.read('training/schedule.json')).values() for seed in entries)
        lineage = json.loads(archive.read('candidate-lineage.json'))
    seeds = {key: np.random.default_rng(SCHEDULE[key]).choice(2147483646, count, replace=False).tolist()
             for key, count in SIZE.items()}
    launches = [seed for rows in seeds.values() for seed in rows]
    if len(set(launches)) != len(launches) or set(launches) & known:
        raise ValueError('guarded collection overlaps a previous exposure')
    metadata = dict(schema='tt24-pong-world-strong-native-1', archive_sha256=_hash(PORTABLE),
                    seeds=seeds, runtime=lineage['qualification_runtime'], checkpoints=lineage['evaluation_checkpoints'],
                    camera_seconds=DT, future_horizon=list(HORIZON), fixed_scores=True, live_motor_timing=True,
                    tracking_coordinates=['row', 'column'])
    data = _read(RAW) if RAW.exists() else dict(metadata=metadata, verified_archive=audit(), rows={})
    if data['metadata'] != metadata:
        raise ValueError('cached guarded play has changed source or launch seeds')
    _write(RAW, data)
    for group, cases in seeds.items():
        existing = data['rows'].get(group, {})
        played = existing.get('episodes', [])
        tasks = [(seed, 20892001 + i, .65 if i % 2 == 0 else .35) for i, seed in enumerate(cases)]
        if len(played) < len(tasks):
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=from_archive) as pool:
                for row in pool.map(dispatch, tasks[len(played):]):
                    played.append(row)
                    if len(played) % 6 == 0 or len(played) == len(tasks):
                        data['rows'][group] = dict(episodes=played, native=summary(played))
                        _write(RAW, data)
                        print('strong live source', group, len(played), 'seconds', round(perf_counter() - start, 1), flush=True)
        data['rows'][group] = dict(episodes=played, native=summary(played))
    _write(RAW, data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        print('archived guarded source parity', audit(), flush=True)
        return
    register(PLAN, TARGET_SCHEDULE)
    result = fit(capture(args.workers), raw_path=RAW, prefix=PREFIX)
    _write(CUT, result)
    print('strong held optical prediction', result['held'], flush=True)
    run(args.workers, plan_path=PLAN, init_path=CUT, output=RECEIPT, schedule=TARGET_SCHEDULE, prefix=PREFIX)


if __name__ == '__main__':
    main()
