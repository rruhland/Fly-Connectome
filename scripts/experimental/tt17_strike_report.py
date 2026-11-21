"""Rebuild stroke choices from public features and reward; then read scored games."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt16_transfer_protocol import FIXTURES
from .tt17_strike import StrikeMemory, rally_value


def optical_evidence(folder, site):
    p = folder/f'strike-{site}-0-practice-0.json.gz'
    practice = _read(p)
    optical = _read(folder/f'strike-{site}-optical-practice.json.gz')
    if optical['practice'] != _hash(p):
        raise ValueError('different practice sequence')
    samples, count, exact = [], [], 0
    for row, trace in zip(practice['points'],optical['traces'],strict=True):
        armed, exits = False, []
        for t, y in enumerate(trace['depths']):
            if y is None:
                continue
            if y < -.65:
                armed = True
            elif armed and y > .25:
                exits.append(t)
                armed = False
        if exits != trace['far_crossings']:
            raise ValueError('optical rally cue differs from recorded depths')
        if row['strike_choice'] is not None:
            samples.append(dict(row['strike_choice'],reward=rally_value(row['public_reward'],exits)))
        count.append([len(exits), row['player_returns']])
        exact += int(len(exits) == row['player_returns'])
    if samples != optical['learned']['observations']:
        raise ValueError('optical exploration values failed reconstruction')
    return dict(starts=len(count), optical=sum(x for x,_ in count), legal=sum(y for _,y in count),
                exact=exact, clock=sum(row['steps'] for row in practice['points']),
                zero_disagreement=sum((x==0)!=(y==0) for x,y in count))


def audit(folder):
    training = {site:optical_evidence(folder,site) for site in ('gantry','five')}
    fixtures = {}
    schedule = [(fixture, 0, ('base', 'online', 'optical')) for fixture in FIXTURES]
    schedule.extend((fixture, 1, ('base', 'online')) for fixture in ('gantry-0', 'gantry-1'))
    for fixture, replica, techniques in schedule:
        site=fixture.split('-')[0]
        modes={}
        for mode in techniques:
            p=folder/f'strike-{fixture}-{mode}-{replica}.json.gz'
            record=_read(p)
            if mode=='base':
                memory=StrikeMemory()
            else:
                source=folder/(f'strike-{site}-optical-practice.json.gz' if mode=='optical'
                               else f'strike-{site}-0-practice-0.json.gz')
                if _hash(source) != record['practice_sha256']:
                    raise ValueError('training tapes differ from evaluation source')
                memory=StrikeMemory(**_read(source)['learned'])
            used=[0,0,0]
            for point in record['points']:
                decision=point['strike_choice']
                if decision is None:
                    continue
                action=memory.choose(decision['features'],mode)
                if action!=decision['action']:
                    raise ValueError('action not reproducible from past experience')
                used[action] += 1
                if mode!='base':
                    reward=(rally_value(point['public_reward'],point['visual_crossings']) if mode=='optical'
                            else float(np.sign(point['public_reward'])))
                    memory.observations.append(dict(decision,reward=reward))
            checkpoint=p.with_name(p.name.removesuffix('.json.gz')+'-player.json.gz')
            if (memory.observations != record['learned']['observations'] or _hash(checkpoint)!=record['checkpoint']
                    or _read(checkpoint)['strike_memory']!=record['learned']):
                raise ValueError('portable learner differs after finished game')
            modes[mode]=dict(score=record['score'],returns=record['metrics']['total_legal_returns'],
                             multiple=record['metrics']['two_return_points'], options=used,game_sha256=_hash(p))
        fixtures[fixture if replica == 0 else f'{fixture}-repeat']=modes
    _write(folder/'strike-summary.json.gz',dict(training=training,fixtures=fixtures))
    print(training,fixtures,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    audit(p.parse_args().folder)
