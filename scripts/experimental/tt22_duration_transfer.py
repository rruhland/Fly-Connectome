"""Carry duration-selected Pong gains into the native physical macro."""

import argparse
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt22_joint_practice import LIMITS, PARAMETER_SCALE, READS, draw, point
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer_trial import MACRO, MotorArena


OUTPUT = ROOT/'artifacts'/'tt22'
PONG = OUTPUT/'duration-pong.json.gz'
CHECKPOINT = OUTPUT/'duration-pong-player.json'
POINT_SOURCE = ROOT/'artifacts'/'tt3'/'discovered-numeric-source.json'
RECEIPT = OUTPUT/'duration-transfer.json.gz'
ADAPT = OUTPUT/'duration-adaptation.json.gz'
SEED = 1408142838
STREAMS = {'moderate':(.17,.62),'wide':(.31,.78)}
POINTS_PER_STREAM = 24
ADAPTATION_POINTS = 8


def performance(rows):
    return dict(points=len(rows), contacts=sum(row['contact']>0 for row in rows),
                first_returns=sum(row['legal']>0 for row in rows), total_returns=sum(row['legal'] for row in rows),
                subsequent_return_points=sum(row['legal']>1 for row in rows), return_hist=[row['legal'] for row in rows],
                opposing_returns=sum(row['rival'] for row in rows), frames=sum(row['ticks'] for row in rows),
                earliest_legal=next((i for i,row in enumerate(rows,1) if row['legal']>0),None))


def matched_difference(rows, left='duration', right='point'):
    gains = losses = 0
    for pair in rows:
        diff = int(pair[left]['legal']>0)-int(pair[right]['legal']>0)
        gains += diff > 0
        losses += diff < 0
    return dict(duration_only=gains, comparison_only=losses, same=len(rows)-gains-losses)


def priors():
    saved = _read(PONG)
    if _hash(POINT_SOURCE) != saved['point_source_sha256']:
        raise ValueError('old point-trained source changed after the native comparison')
    checkpoint = _read(CHECKPOINT)
    if _hash(CHECKPOINT) != saved['checkpoint_sha256'] or checkpoint['coefficients'][0] != saved['frozen']['duration']:
        raise ValueError('duration gains changed after the native freeze')
    result = {name:np.r_[np.array(saved['frozen'][name])[1:],np.zeros(4)] for name in ('duration','point')}
    result['neutral'] = np.zeros(6)
    return result


def verify_pair(arena, macro, pairs, policy):
    count = 0
    for entry in pairs:
        for name,row in entry['results'].items():
            if point(arena,macro,policy[name],entry['optical']) != row:
                raise ValueError('physical macro replay diverged')
            count += 1
    return count


def run_frozen():
    candidates = priors()
    macro = np.array(_read(MACRO)['parameters'])
    rng = np.random.default_rng(SEED)
    starts = {name:draw(rng,POINTS_PER_STREAM,bounds) for name,bounds in STREAMS.items()}
    all_seeds = {seed for stream in starts.values() for seed,_ in stream}
    if len(all_seeds) != POINTS_PER_STREAM*len(STREAMS):
        raise ValueError('target launches overlap')
    arena = MotorArena()
    began = perf_counter()
    receipt = dict(schema='tt22-duration-transfer-1', source=CHECKPOINT.name, source_sha256=_hash(CHECKPOINT),
                   native_qualification=PONG.name, point_source_sha256=_hash(POINT_SOURCE),
                   macro_sha256=_hash(MACRO), source_winner=_read(PONG)['selected'],
                   inputs='eight unlabelled native fixed-stereo RGB images and zero commands',
                   control='existing clock macro with vendor inverse kinematics', prefix_frames=READS, seed=SEED,
                   coefficients={name:value.tolist() for name,value in candidates.items()},
                   starts=starts, streams={name:[] for name in starts}, verified=0)
    try:
        for stream, cases in starts.items():
            for i,(seed,lane) in enumerate(cases):
                sight = prefix(arena,seed,lane)
                order = tuple(candidates)
                order = order[i%len(order):]+order[:i%len(order)]
                rows = {name:point(arena,macro,candidates[name],sight) for name in order}
                receipt['streams'][stream].append(dict(optical=sight,results=rows))
                print('held',stream,i+1,seed,{name:row['legal'] for name,row in rows.items()},flush=True)
                _write(RECEIPT,receipt)
            receipt['verified'] += verify_pair(arena,macro,receipt['streams'][stream],candidates)
        both = [item for stream in receipt['streams'].values() for item in stream]
        receipt['coverage'] = {name:sum(row['optical']['mover'] is not None for row in rs)
                               for name,rs in receipt['streams'].items()}
        receipt['assessment'] = {name:dict(policies={key:performance([r['results'][key] for r in rs]) for key in candidates},
                                          point_compare=matched_difference([r['results'] for r in rs]),
                                          neutral_compare=matched_difference([r['results'] for r in rs],right='neutral'))
                                 for name,rs in dict(receipt['streams'],combined=both).items()}
        receipt['wall_seconds'] = perf_counter()-began
        _write(RECEIPT,receipt)
        print(receipt['assessment']['combined'],flush=True)
    finally:
        arena.close()


def run_adaptation():
    baseline = _read(RECEIPT)
    if baseline['macro_sha256'] != _hash(MACRO) or baseline['verified'] != 3*POINTS_PER_STREAM*len(STREAMS):
        raise ValueError('complete the frozen matched assessment first')
    macro = np.array(_read(MACRO)['parameters'])
    rng = np.random.default_rng(SEED+1)
    result = dict(schema='tt22-duration-adaptation-1', frozen_sha256=_hash(RECEIPT),
                  practice_points_per_arm=ADAPTATION_POINTS, streams={})
    arena = MotorArena()
    began = perf_counter()
    try:
        used = {x['optical']['seed'] for rows in baseline['streams'].values() for x in rows}
        for stream, hold in baseline['streams'].items():
            initial = {name:np.array(value) for name,value in baseline['coefficients'].items()}
            exposures = draw(rng,ADAPTATION_POINTS,(.14,.58) if stream=='moderate' else (.27,.64))
            distinct = {seed for seed,_ in exposures}
            if used&distinct or len(distinct)!=len(exposures):
                raise ValueError('target practice and held launches overlap')
            used.update(distinct)
            practice, observations, score = [], {name:[] for name in initial}, {name:[] for name in initial}
            for seed,lane in exposures:
                sight = prefix(arena,seed,lane)
                noise = rng.normal(size=6)*PARAMETER_SCALE
                rows = {}
                for name,center in initial.items():
                    theta = np.clip(center+noise,-LIMITS,LIMITS)
                    trial = point(arena,macro,theta,sight)
                    observations[name].append(theta)
                    score[name].append(min(3,trial['legal']))
                    rows[name] = dict(trial,params=theta.tolist())
                practice.append(dict(optical=sight,results=rows))
                print('practice',stream,seed,{name:row['legal'] for name,row in rows.items()},flush=True)
            learned = {name:(2*initial[name]+np.asarray(score[name])@np.array(observations[name]))/(2+sum(score[name]))
                       for name in initial}
            checked_practice = verify_practice(arena,macro,practice)
            held_after = [dict(optical=row['optical'],results={name:point(arena,macro,theta,row['optical'])
                                                            for name,theta in learned.items()}) for row in hold]
            verified = verify_pair(arena,macro,held_after,learned)
            result['streams'][stream] = dict(practice=practice, held=held_after, verified=verified,
                                             verified_practice=checked_practice,
                                             endpoint={name:value.tolist() for name,value in learned.items()},
                                             metrics={name:performance([row['results'][name] for row in held_after]) for name in learned})
            _write(ADAPT,result)
        result['combined'] = {name:performance([row['results'][name] for stream in result['streams'].values() for row in stream['held']])
                              for name in baseline['coefficients']}
        result['wall_seconds'] = perf_counter()-began
        _write(ADAPT,result)
        print(result['combined'],flush=True)
    finally:
        arena.close()


def verify_practice(arena,macro,practice):
    checked = 0
    for episode in practice:
        for row in episode['results'].values():
            actual = point(arena,macro,np.array(row['params']),episode['optical'])
            if actual != {key:value for key,value in row.items() if key!='params'}:
                raise ValueError('target practice reward diverged on physical replay')
            checked += 1
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapt',action='store_true')
    parser.add_argument('--verify-practice',action='store_true')
    args = parser.parse_args()
    if args.verify_practice:
        saved = _read(ADAPT)
        if saved['frozen_sha256'] != _hash(RECEIPT):
            raise ValueError('frozen transfer receipt changed after adaptation')
        macro = np.array(_read(MACRO)['parameters'])
        arena = MotorArena()
        try:
            for group in saved['streams'].values():
                group['verified_practice'] = verify_practice(arena,macro,group['practice'])
        finally:
            arena.close()
        _write(ADAPT,saved)
        print('verified',sum(group['verified_practice'] for group in saved['streams'].values()))
    elif args.adapt:
        run_adaptation()
    else:
        run_frozen()
