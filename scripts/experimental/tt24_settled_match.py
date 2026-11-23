"""Carry physically measured bounce actuator maps into a full live table match."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt23_sequence import course, verify
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_motor_settle import game
from .tt24_rally_bank import CACHE, DIMS, HOLDOUT
from .tt24_rally_improve_trial import SCHEDULES, policy_for
from .tt24_rally_remap import driver
from .tt24_rally_trial import COACHED, Station, returns


PLAN = ROOT / 'settled-match-plan.json'
REPORT = ROOT / 'settled-match-results.json.gz'
COURSE = {'new_round_a': 1741165089, 'new_round_b': 884436119}
MODES = ('calibrated', 'frozen', 'true_inverse')


def registration():
    held = course(COURSE)
    played = _read(ROOT/'bounce-to-match.json.gz')
    previous = {r['seed'] for r in played['cases']}
    previous.update(r['seed'] for r in course(HOLDOUT))
    previous.update(r['seed'] for group in _read(SCHEDULES)['schedules'].values() for r in group)
    with np.load(CACHE, allow_pickle=False) as arrays:
        previous.update(map(int, arrays['seed']))
    draws = [r['seed'] for r in held[:16]]
    if len(set(draws))!=16 or previous.intersection(draws):
        raise ValueError('settled match must use new holds')
    specs = []
    for index, stream in enumerate((0,2)):
        path = ROOT/f'motor-settle-{stream}.json.gz'
        doc = _read(path)
        if doc['body'] != 'stock':
            raise ValueError('use stock motor channels')
        for block in (1,2):
            row = doc['sessions'][block]
            fit = row['inspection']
            truth = doc['wires'][row['interface']]
            if fit['known'] is not None or fit['ticks']!=87:
                raise ValueError('need a single newly measured trial, before recall')
            i = len(specs)
            specs.append(dict(index=i, stream=stream, session=block, calibration_file=path.name,
                              calibration_sha256=_hash(path), fit=fit['fit'], truth=truth,
                              model_error=fit['map_error'], observation_frames=fit['ticks'],
                              probe_seed=fit['seed'], probe_speed_hz=60,
                              prior_solo_seed=row['games'][0]['seed'], held=held[4*i:4*i+4]))
    return dict(schema='tt24-settled-match-plan-1', calibration='archived contact-free encoder-only effects',
                states_per_stream=4, target_adaptation_episodes=0, modes=list(MODES),
                action_port='stock vendor five joint axes, with same software wiring before the mechanism',
                fit_sha256=_hash(CACHE), opponent_profile_source_sha256=_hash(COACHED),
                bridge_comparator_sha256=_hash(ROOT/'bounce-to-match.json.gz'), specs=specs)


def exchange(row):
    near, far = len(returns(row,1)), len(returns(row,-1))
    return min(near,far)


def verify_point(point):
    checked = verify([point])
    if checked != len(MODES):
        raise ValueError('incomplete paired physical replay')
    return checked


def run_map(settings):
    path = ROOT/f'settled-match-map-{settings["index"]}.json.gz'
    data = _read(path) if path.exists() else dict(settings=settings, cases=[], verified=0)
    if data['settings']!=settings:
        raise ValueError('stored physical map receipts have changed')
    truth, fit = np.array(settings['truth']), np.array(settings['fit'])
    decoders = dict(calibrated=np.linalg.inv(fit), frozen=np.eye(5), true_inverse=np.linalg.inv(truth))
    if data['verified'] != len(data['cases'])*len(MODES):
        raise ValueError('existing table points lack complete verification')
    if [r['seed'] for r in data['cases']] != [r['seed'] for r in settings['held'][:len(data['cases'])]]:
        raise ValueError('stored table point sequence differs')
    if len(data['cases']) == len(settings['held']):
        return path
    with np.load(CACHE, allow_pickle=False) as bank:
        keeper = policy_for('retina_rate', bank)
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    station = Station(profile, render_dims=DIMS)
    try:
        for i, scene in enumerate(settings['held'][len(data['cases']):], len(data['cases'])):
            numbered = settings['index']+i
            offset = numbered % len(MODES)
            modes = MODES[offset:] + MODES[:offset]
            results = {}
            for label in modes:
                d = decoders[label]
                row = station.play(scene, label, keeper, live_opening=True,
                                    execute=lambda request, fixed=d:driver(truth, fixed, request))
                results[label] = row
            if len({row['prefix_rgb'] for row in results.values()}) != 1:
                raise ValueError('paired match conditions must start from identical camera data')
            point = dict(scene, results=results)
            data['verified'] += verify_point(point)
            data['cases'].append(point)
            _write(path, data)
            print('map',settings['index'],'fresh seed',scene['seed'],
                  {label:(len(returns(row,1)),exchange(row)) for label,row in results.items()},flush=True)
    finally:
        station.arena.close()
    return path


def totals(maps):
    cases = [r for m in maps for r in m['cases']]
    values = {}
    for name in MODES:
        rows = [case['results'][name] for case in cases]
        near = [len(returns(x,1)) for x in rows]
        pairs = [exchange(x) for x in rows]
        far = [len(returns(x,-1)) for x in rows]
        skips = [row['longest'] for row in rows]
        values[name] = dict(near=sum(near), far=sum(far), paired_exchanges=sum(pairs),
                            first_points=sum(x>0 for x in near), second_points=sum(x>=2 for x in near),
                            longest_alternating=max(skips,default=0), per_point_pairs=pairs,
                            per_point_near=near, per_point_far=far, n=len(rows))
    return values


def restore_solo(settings):
    env = BounceArena(view='wrist', width=192, height=144, max_steps=420)
    policy = CarriedWristPolicy()
    proof = []
    try:
        for row in settings:
            past = _read(ROOT/row['calibration_file'])['sessions'][row['session']]['games'][0]
            truth, fit = np.array(row['truth']), np.array(row['fit'])
            current = game(env, policy, past['seed'], 'standard', truth, fit)
            old = past['conditions']['calibrated']
            if digest_json(current) != digest_json(old):
                raise ValueError('the carried solo skill no longer reproduces its earlier motor tape')
            proof.append(dict(map=row['index'], seed=past['seed'], exact=True,
                              longest=current['metrics']['longest_consecutive_rebounds'],
                              verified_tape_sha256=digest_json(current)))
    finally:
        env.close()
    return proof


def restore_pong():
    import torch
    from .tt24_source import CHECKPOINT, VisualConsequence
    from .tt24_source_native import HELD, closed_loop
    from .tt3_pong import PongCase

    torch.set_num_threads(1)
    path = ROOT/'source-retention.json.gz'
    old = _read(path)['before']
    if old['source_sha256'] != _hash(CHECKPOINT):
        raise ValueError('native Pong original policy hash changed')
    raw = _read(CHECKPOINT)
    horizon = _read(HELD)['horizon']
    modes = ('pong','cold','shuffled')
    models = {mode:VisualConsequence.from_checkpoint(mode, raw) for mode in modes}
    cases = [PongCase(**source) for source in old['schedule'][:6]]
    rows = closed_loop(cases, models, raw['plant'], 12, horizon)
    exact = {mode: sum(a==b for a,b in zip(rows[mode],old['rows'][mode][:6],strict=True)) for mode in modes}
    if any(x != len(cases) for x in exact.values()):
        raise ValueError('reloaded native Pong observations, motor tapes or scores differ')
    return dict(source_sha256=_hash(CHECKPOINT), original_receipt_hash=_hash(path),
                exact=exact, previous_starts=old['schedule'][:6], post_transfer=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit-maps', type=int, choices=(1,2,3,4), default=4)
    args = parser.parse_args()
    config = registration()
    if PLAN.exists() and _read(PLAN)!=config:
        raise ValueError('recorded transfer plan changed')
    if not PLAN.exists():
        _write(PLAN,config)
    specs=config['specs'][:args.limit_maps]
    bundle = _hash(BUNDLE)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        paths = list(pool.map(run_map, specs))
    outcomes = [_read(path) for path in paths]
    aggregate = dict(schema='tt24-settled-match-1', plan_sha256=_hash(PLAN),
                     maps=[dict(receipt=path.name, sha256=_hash(path), metrics=totals([data]))
                           for path,data in zip(paths,outcomes,strict=True)],
                     overview=totals(outcomes), verified=sum(d['verified'] for d in outcomes),
                     fresh_target_practice=0, body_calibration_seconds_per_map=87/60.,
                     carried_solo_bundle_sha256=bundle)
    if REPORT.exists():
        previous = _read(REPORT)
        if (previous['plan_sha256'] == aggregate['plan_sha256']
                and previous['maps'] == aggregate['maps']
                and previous['carried_solo_bundle_sha256'] == bundle):
            aggregate['retention'] = previous.get('retention', {})
    _write(REPORT,aggregate)
    print('strict match results',aggregate['overview'],flush=True)
    retained = aggregate.setdefault('retention', {})
    if 'solo' not in retained:
        retained['solo'] = restore_solo(specs)
        _write(REPORT,aggregate)
    if 'pong' not in retained:
        retained['pong'] = restore_pong()
    if bundle!=_hash(BUNDLE):
        raise ValueError('carried solo bundle changed during match')
    _write(REPORT,aggregate)
    print('retention',aggregate['retention']['solo'],aggregate['retention']['pong']['exact'],flush=True)


if __name__ == '__main__':
    main()
