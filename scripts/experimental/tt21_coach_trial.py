"""Matched native returns and an evaluator-only reachability audit on the compact court."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt20_curriculum import GAME_SEED, OUTPUT as TT20, _sighting, apparatus, receiver, transport
from .tt20_opponent import install_opponent
from .tt21_coach import install_coach
from .tt21_coach_oracle import Reachability, reply_released


SOURCE = TT20 / f'curriculum-frozen-{GAME_SEED}.json.gz'
OUTPUT = TT20.parent / 'tt21' / 'coach-replies.json.gz'
MODES = ('standard', 'strong', 'coach')


def first_contact(events, side):
    return next((event for event in events if event['kind'] == 'contact' and event.get('side') == side), None)


def launches(report):
    by_role = {1: [], -1: []}
    for index, row in enumerate(report['points'], 1):
        score = row['result']
        role = row['server']
        qualifying = score['legal_serve'] if role == 1 else score['legal_returns'] > 0
        if qualifying:
            by_role[role].append((index, row, first_contact(row['events'], 1)))
    return [entry for pair in zip(by_role[-1][:4], by_role[1][:4], strict=True) for entry in pair]


def following(events, strike):
    if strike is None:
        return False, False
    reply = events[events.index(strike) + 1:]
    next_rival = first_contact(reply, -1)
    if next_rival is not None:
        reply = reply[:reply.index(next_rival)]
    scored = [event['side'] for event in reply if event['kind'] == 'legal_return']
    return -1 in scored, 1 in scored


def replay(arena, rgb, player, oracle, original, strike, mode, rival):
    role, seed = original['server'], original['seed']
    horizon = strike['step']
    tracker = rgb.tracker()
    player.reset(role)
    arena.reset(seed=seed, server=role, render=False)
    actions, optics = [], []
    snapshot, reply, reward = None, None, None
    for tick in range(arena.max_steps):
        position = None
        if tick % 2 == 0 or player.phase == 'receive':
            tracker.update(arena.render_cameras())
            position = _sighting(tracker.ball)
        # Bypass differences in accumulated camera calibration on the fixed
        # opening tape. From the strike onwards the policy uses this run's RGB.
        if tick < horizon:
            position = original['optical'][tick]
        command = player.command(tick, position)
        if tick < horizon and command.tolist() != original['actions'][tick]:
            raise ValueError('the RGB controller no longer reproduces the archived opening tape')
        optics.append(position)
        actions.append(command.tolist())
        _, reward, done, timeout, info = arena.step(command, render=False)
        if tick + 1 == horizon:
            actual = first_contact(arena.events, 1)
            if actual is None or digest_json(actual) != digest_json(strike):
                raise ValueError('the fixed opener did not reproduce the original learner contact')
        if tick + 1 > horizon and reply is None:
            reply = first_contact(info['events'], -1)
        if reply is not None and snapshot is None and not (done or timeout) and reply_released(arena):
            snapshot = oracle.assess(arena)
        if done or timeout:
            break
    opponent_legal, counter = following(arena.events, reply)
    return dict(mode=mode, seed=seed, server=role, opening_frames=horizon, opening_contact=strike,
                actions=actions, optical=optics, rgb_sha256=tracker.digest.hexdigest(),
                action_sha256=sha256(np.asarray(actions, dtype='<f8').tobytes()).hexdigest(),
                reply_contact=reply, legal_opponent_reply=opponent_legal, native_counter=counter,
                oracle=snapshot, rival_search=None if rival is None else rival.decisions.copy(),
                result=arena.point_result(), events=arena.events.copy(), score=reward)


def summary(rows):
    return dict(starts=len(rows), physical_replies=sum(row['reply_contact'] is not None for row in rows),
                scored_legal_replies=sum(row['legal_opponent_reply'] for row in rows),
                corridor_opportunities=sum(bool((row['oracle'] or {}).get('corridor_samples')) for row in rows),
                oracle_contacts=sum(bool((row['oracle'] or {}).get('contact_witness')) for row in rows),
                oracle_legal=sum(bool((row['oracle'] or {}).get('witnessed')) for row in rows),
                either_legal=sum(row['native_counter'] or bool((row['oracle'] or {}).get('witnessed')) for row in rows),
                rgb_legal=sum(row['native_counter'] for row in rows),
                solver_candidates=sum((row['oracle'] or {}).get('candidates', 0) for row in rows),
                exact_trials=sum((row['oracle'] or {}).get('exact_trials', 0) for row in rows))


def run(source, output, limit):
    original = _read(source)
    selected = launches(original)
    rows = {mode: [] for mode in MODES}
    if output.exists():
        previous = _read(output)
        if previous['source_sha256'] != _hash(source):
            raise ValueError('snapshot is bound to a different held game')
        rows = previous['variants']
    began = perf_counter()
    oracle = Reachability()
    player = receiver(False, TT20)
    player.serves.force = original['chosen']
    stations = {}
    try:
        for mode in MODES:
            arena = apparatus(False)
            rival = install_opponent(arena) if mode == 'strong' else install_coach(arena) if mode == 'coach' else None
            stations[mode] = arena, transport(arena, False), rival
        for index, original_row, strike in selected[:limit]:
            for mode, (arena, optics, rival) in stations.items():
                if any(row['held_point'] == index for row in rows[mode]):
                    continue
                row = replay(arena, optics, player, oracle, original_row, strike, mode, rival)
                row['held_point'] = index
                rows[mode].append(row)
                report = dict(schema='tt21-coach-reachabilities-2', source=source.name, source_sha256=_hash(source),
                              game_seed=original['seed'], selected_points=[entry[0] for entry in selected],
                              search_privileged=True, search_seen_by_rgb=False, variants=rows,
                              summary={key: summary(value) for key, value in rows.items()},
                              wall_seconds=perf_counter()-began)
                _write(output, report)
                print(index, mode, row['result']['reason'], row['legal_opponent_reply'], row['native_counter'],
                      None if row['oracle'] is None else {k: row['oracle'][k] for k in
                            ('corridor_samples','candidates','solver_contacts','solver_legal','witnessed')}, flush=True)
    finally:
        for arena, _, _ in stations.values():
            arena.close()
        player.player.close()
        oracle.close()
    return _read(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--limit', type=int, choices=(2, 4, 6, 8), default=8)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    print(run(a.source, a.output, a.limit)['summary'])
