"""Score optical apprenticeship on disjoint archive replies and fresh starts."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt20_curriculum import OUTPUT as TT20, _sighting, apparatus, receiver, transport
from .tt20_opponent import install_opponent
from .tt21_apprentice import Apprentice, GENERIC, style_from_motor
from .tt21_coach import install_coach
from .tt21_coach_trial import following


OUTPUT = ROOT / 'artifacts/tt21'
TEACHER = OUTPUT / 'coach-replies.json.gz'
TRAIN_POINTS = (1, 7, 2, 11)
HELD_POINTS = (5, 12, 6, 15)
RIVALS = ('standard', 'strong', 'coach')
MODES = ('stock', 'generic', 'taught')
FRESH = 909553203


def lessons(report, twin):
    rows = []
    for index in TRAIN_POINTS:
        for mode in RIVALS:
            candidate = next((r for r in report['variants'][mode] if r['held_point']==index), None)
            if candidate is None or not (candidate['oracle'] or {}).get('witnessed'):
                continue
            oracle = candidate['oracle']
            contact = next(e['step'] for e in oracle['exact_events'] if e['kind']=='contact' and e['side']==1)
            commands = candidate['actions'][:oracle['tick']] + oracle['exact_actions']
            style = style_from_motor(twin, commands, contact)
            rows.append(dict(seed=candidate['seed'], rival=mode, archived_point=index, contact=contact,
                             style=style, action_sha256=sha256(np.asarray(commands, dtype='<f8').tobytes()).hexdigest()))
            break
    if len(rows) != len(GENERIC):
        raise ValueError('need three independent physical lessons and the matched generic repertoire')
    return rows


def select_archive(report):
    cases = []
    for index in HELD_POINTS:
        options = [row for mode in RIVALS for row in report['variants'][mode]
                   if row['held_point']==index and row['legal_opponent_reply'] and row['oracle']]
        if options:
            # Use one target per opening irrespective of how many rivals were measured.
            # Favor a verified physical opportunity, then the predeclared rival order.
            cases.append(max(options, key=lambda row: row['oracle']['witnessed']))
    return cases


def station(rival):
    arena = apparatus(False)
    if rival == 'strong':
        install_opponent(arena)
    elif rival == 'coach':
        install_coach(arena)
    return arena


def before_reply(events):
    first = next((i for i, e in enumerate(events) if e['kind']=='contact' and e.get('side')==1), None)
    if first is None:
        return None
    return next((e for e in events[first+1:] if e['kind']=='contact' and e.get('side')==-1), None)


def live(arena, match, apprentice, *, seed, server, styles, archive=None):
    arena.reset(seed=seed, server=server, render=False)
    match.reset(server)
    camera = transport(arena, False).tracker()
    if styles is not None:
        apprentice.reset(styles)
    actions, optical = [], []
    cutoff = 0 if archive is None else archive['oracle']['tick']
    for tick in range(arena.max_steps):
        wants_view = tick % 2 == 0 or match.phase == 'receive' or (styles is not None and apprentice.applied is not None)
        ball = None
        if wants_view:
            camera.update(arena.render_cameras(), tick=tick)
            ball = _sighting(camera.ball)
        optical.append(ball)
        if styles is not None:
            opening = bool(tick and ((archive is not None and tick >= archive['opening_frames'])
                                    or match.player.plans or (server==1 and match.phase!='opening')))
            apprentice.observe(tick, ball, opening=opening)
        position = archive['optical'][tick] if archive is not None and tick < cutoff else ball
        command = (match.command(tick, position) if styles is None or apprentice.applied is None else np.zeros(6))
        if tick < cutoff:
            command = archive['actions'][tick]
        elif styles is not None:
            command = apprentice.command(tick, command, enabled=True)
        if styles is not None:
            apprentice.advance(command)
        actions.append(np.asarray(command).tolist())
        _, reward, done, capped, _ = arena.step(command, render=False)
        if tick + 1 == cutoff:
            expected = [e for e in archive['events'] if e['step'] <= cutoff]
            if json.dumps(arena.events) != json.dumps(expected):
                raise ValueError(f'archive prefix or opponent contact changed at seed {seed}')
        if done or capped:
            if tick + 1 < cutoff:
                raise ValueError('the preserved archive prefix ended unexpectedly')
            break
    reply = before_reply(arena.events)
    legal, counter = following(arena.events, reply)
    contacted = bool(reply and any(e['kind']=='contact' and e.get('side')==1 and e['step']>reply['step']
                                   for e in arena.events))
    return dict(seed=seed, server=server, actions=actions, optical=optical, rgb_sha256=camera.digest.hexdigest(),
                events=arena.events.copy(), result=arena.point_result(), score=reward, prefix_frames=cutoff,
                legal_reply=legal, reply_contact=contacted, second_shot=counter,
                attempts=[] if styles is None else apprentice.attempts,
                departure=None if styles is None else apprentice.applied)


def score(rows):
    return dict(starts=len(rows), rival_legal=sum(row['legal_reply'] for row in rows),
                continuation_contacts=sum(row['reply_contact'] for row in rows),
                legal_continuations=sum(row['second_shot'] for row in rows),
                learner_returns=sum(row['result']['legal_returns'] for row in rows),
                two_returns=sum(row['result']['legal_returns']>=2 for row in rows),
                wins=sum(row['score']>0 for row in rows), departures=sum(row['departure'] is not None for row in rows))


def verify(rival, rows):
    arena = station(rival)
    try:
        for row in rows:
            arena.reset(seed=row['seed'], server=row['server'], render=False)
            for command in row['actions']:
                _, reward, _, _, _ = arena.step(command, render=False)
            if (arena.point_result()!=row['result'] or reward!=row['score']
                    or json.dumps(arena.events)!=json.dumps(row['events'])):
                raise ValueError(f'physical replay changed for {rival} at {row["seed"]}')
    finally:
        arena.close()
    return len(rows)


def run(output, fresh):
    snapshot = TEACHER.read_bytes()
    teacher = json.loads(gzip.decompress(snapshot))
    original = _read(TT20 / teacher['source'])
    match = receiver(False, TT20)
    match.serves.force = original['chosen']
    motor = Apprentice(match.player.law)
    sessions = {rival: station(rival) for rival in RIVALS}
    archive, normal = [], []
    branch_replay = {rival: [] for rival in RIVALS}
    try:
        training = lessons(teacher, motor.sim.twin)
        styles = {'stock': None, 'generic': GENERIC, 'taught': [row['style'] for row in training]}
        held = select_archive(teacher)
        if len(held) < 3 or set(r['seed'] for r in training).intersection(r['seed'] for r in held):
            raise ValueError('need three disjoint archived trajectories for the pilot')
        # An entirely new fresh schedule is pre-generated before seeing held outcomes.
        fresh_seeds = np.random.default_rng(FRESH).integers(2**31, size=fresh).tolist()
        seen = previous_seeds(TT20) | previous_seeds(OUTPUT) | previous_seeds(ROOT/'artifacts/tt19')
        if output.exists():
            seen.difference_update(_read(output)['prespecified_fresh_seeds'])
        if len(set(fresh_seeds)) != len(fresh_seeds) or seen.intersection(fresh_seeds):
            raise ValueError('a fresh schedule seed has prior exposure')

        def paired(rival, seed, server, source=None):
            measurements = {name: live(sessions[rival], match, motor, seed=seed, server=server,
                                       styles=styles[name], archive=source) for name in MODES}
            stock = measurements['stock']
            for name, row in measurements.items():
                horizon = row['departure'] if row['departure'] is not None else len(row['actions'])
                if stock['actions'][:horizon] != row['actions'][:horizon]:
                    raise ValueError(f'{name} changed the opener or recovery before its legitimate departure')
            branch_replay[rival].extend(measurements.values())
            print(rival, seed, {name: (row['departure'], row['reply_contact'], row['second_shot'])
                                 for name,row in measurements.items()}, flush=True)
            return dict(seed=seed, rival=rival, modes=measurements)

        for case in held:
            archive.append(dict(archived_point=case['held_point'], prior_stock=case['native_counter'],
                                physically_witnessed=case['oracle']['witnessed'],
                                **paired(case['mode'], case['seed'], case['server'], case)))
        # Stop the bounded pilot if the teacher fails to gain contact or legality.
        gained = any(any(not pair['modes']['stock'][metric] and pair['modes']['taught'][metric]
                         for metric in ('reply_contact', 'second_shot')) for pair in archive)
        if gained:
            for i, seed in enumerate(fresh_seeds):
                normal.append(paired('standard', seed, 1 if i % 3 == 2 else -1))
    finally:
        for arena in sessions.values():
            arena.close()
        match.player.close()
        motor.close()
    replayed = sum(verify(rival, rows) for rival, rows in branch_replay.items() if rows)
    data = dict(schema='tt21-apprentice-1', teacher=TEACHER.name,
                teacher_snapshot_sha256=sha256(snapshot).hexdigest(), source=teacher['source'],
                teacher_privileged=True, online_reward_updates=0, training=training, styles=styles,
                held_archive=archive, fresh=normal, prespecified_fresh_seeds=fresh_seeds,
                fresh_trigger='a held archived contact or legal continuation gained versus stock',
                replayed=replayed, summary={key: {name: score([entry['modes'][name] for entry in rows])
                                                  for name in MODES}
                                            for key, rows in (('held_archive', archive), ('fresh', normal))},
                implementation={name:_hash(Path(__file__).with_name(name)) for name in (
                    'tt21_apprentice.py','tt21_apprentice_trial.py', 'tt21_body_state.py')})
    _write(output, data)
    print(data['summary'], flush=True)
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fresh', type=int, default=6)
    p.add_argument('--output', type=Path, default=OUTPUT/'apprentice.json.gz')
    args = p.parse_args()
    run(args.output, args.fresh)
