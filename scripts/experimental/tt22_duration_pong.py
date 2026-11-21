"""Select portable spatial preferences from native Pong episode duration alone."""

import argparse
from dataclasses import asdict
import json
from time import perf_counter

import numpy as np

from .tt3_numeric import NumericMemory
from .tt3_pong import ACTIONS, PongCase, _PongReturnAgent, _Presentation, _native_rallies, cases_from_seeds, summarize_pong
from .tt3_pong_probe import ShorterDwell
from .tt3_visual import RoleModel
from .tt4_transfer_trial import ROOT, _hash, _read, _write


OUTPUT = ROOT/'artifacts'/'tt22'
RECEIPT = OUTPUT/'duration-pong.json.gz'
CHECKPOINT = OUTPUT/'duration-pong-player.json'
ROLE_MODEL = ROOT/'artifacts'/'tt3'/'discovered-role-model.json'
POINT_PLAYER = ROOT/'artifacts'/'tt3'/'discovered-numeric-source.json'
DRAW = 1318250174
SOURCE_START = 1770000000
SOURCE_VALIDATION = 1810000000
HELD_START = 1850000000
OPPONENT_SPEED = .56
CAP = 1200
POPULATION = 16


class OpticalDurationAgent(_PongReturnAgent):
    def observe(self, image, preceding_action, _):
        super().observe(image, preceding_action, 0.)


class Stationary:
    """Match native zero drive under each episode's reassigned action IDs."""

    def __init__(self, case):
        self.neutral = ACTIONS[int(np.flatnonzero(_Presentation(case.presentation).order == 0)[0])]

    def observe(self, *_):
        return None

    def choose(self):
        return self.neutral


def population():
    rng = np.random.default_rng(DRAW)
    unit = np.stack([(rng.permutation(POPULATION)+rng.random(POPULATION))*2./POPULATION-1. for _ in range(3)], axis=1)
    return unit*np.array((.11, 1.4, 1.4))


def cases(count, offset):
    rng = np.random.default_rng(DRAW+offset)
    return cases_from_seeds(rng.choice(26000000, size=count, replace=False)+offset)


def play(cases, candidates, roles, batch_size):
    """Vectorized native dynamics with one anonymous optical controller per row."""
    name_to_index = {name: index for index, name in enumerate(candidates)}
    gains = [np.zeros(3) if option is None else option for option in candidates.values()]
    memory = NumericMemory(gains)
    schedule = [(case, name) for case in cases for name in candidates]
    groups = {name: [] for name in candidates}
    for start in range(0, len(schedule), batch_size):
        batch = schedule[start:start+batch_size]
        agents = [Stationary(case) if candidates[name] is None else
                  OpticalDurationAgent(roles, memory, case.exploration, option=name_to_index[name], motor_factory=ShorterDwell)
                  for case,name in batch]
        runs = _native_rallies([case for case,_ in batch], agents, max_ticks=CAP, opponent_speed=OPPONENT_SPEED)
        for (case,name), row, agent in zip(batch,runs,agents,strict=True):
            motor = getattr(agent, 'motor', None)
            row.update(relation=name_to_index[name], visual_effects=None if motor is None else motor.effects,
                       visual_probe_frames=0 if motor is None else motor.frames,
                       relation_executed=False if motor is None else agent.credited)
            groups[name].append(row)
    return groups


def duration(rows):
    return sum(row['ticks'] for row in rows) / len(rows)


def summary(rows):
    return dict(summarize_pong(rows), duration_ticks=duration(rows), duration_seconds=duration(rows)/120.,
                median_seconds=float(np.median([row['ticks'] for row in rows]))/120.,
                points_total=sum(row['score'] for row in rows),
                mean_returns=float(np.mean([sum(contact['returned'] for contact in row['contacts']) for row in rows])))


def paired(reference, challenge):
    if [r['case'] for r in reference] != [r['case'] for r in challenge]:
        raise ValueError('episodes are not matched')
    seconds = (np.array([row['ticks'] for row in challenge])-np.array([row['ticks'] for row in reference]))/120.
    bootstrap = np.random.default_rng(DRAW+3).integers(len(seconds), size=(4096,len(seconds)))
    return dict(mean_delta_s=float(seconds.mean()), median_delta_s=float(np.median(seconds)),
                longer=int(sum(seconds>0)), shorter=int(sum(seconds<0)), equal=int(sum(seconds==0)),
                bootstrap95_s=np.quantile(seconds[bootstrap].mean(axis=1),(.025,.975)).tolist())


def shuffled_gain(data):
    index = str((int(data['selected']) + POPULATION//2) % POPULATION)
    return data['candidates'][index]


def run(screen, validation, held, batch):
    import torch

    torch.set_num_threads(1)
    roles = RoleModel(**_read(ROLE_MODEL))
    original = _read(POINT_PLAYER)
    point_best = int(np.argmax(np.divide(original['totals'],original['counts'])))
    candidates = {str(i): coeff.tolist() for i,coeff in enumerate(population())}
    screen_cases, validation_cases, held_cases = (cases(n,offset) for n,offset in
        ((screen,SOURCE_START),(validation,SOURCE_VALIDATION),(held,HELD_START)))
    begins = perf_counter()
    data = dict(schema='tt22-native-duration-1', objective='mean capped native episode ticks',
                seconds_per_tick=1/120, cap=CAP, opponent_speed=OPPONENT_SPEED, draw_seed=DRAW,
                role_sha256=_hash(ROLE_MODEL), point_source_sha256=_hash(POINT_PLAYER), candidates=candidates,
                screen_cases=[asdict(case) for case in screen_cases],
                validation_cases=[asdict(case) for case in validation_cases], held_cases=[asdict(case) for case in held_cases])
    data['screen'] = play(screen_cases,candidates,roles,batch)
    order = sorted(candidates, key=lambda name: (-duration(data['screen'][name]),int(name)))
    data['screen_duration_ticks'] = {name: duration(data['screen'][name]) for name in order}
    _write(RECEIPT,data)
    print('duration-only screen',data['screen_duration_ticks'],flush=True)
    shortlist = {name:candidates[name] for name in order[:4]}
    data['validation'] = play(validation_cases,shortlist,roles,batch)
    combined = {name: data['screen'][name]+data['validation'][name] for name in shortlist}
    winner = max(combined,key=lambda name:(duration(combined[name]),-int(name)))
    checkpoint = NumericMemory([candidates[winner]],[sum(row['ticks'] for row in combined[winner])],
                               [len(combined[winner])]).state()
    CHECKPOINT.write_text(json.dumps(checkpoint,indent=2)+'\n')
    modes = {'duration':candidates[winner], 'point':original['coefficients'][point_best],
             'cold':candidates['0'], 'opposite':candidates[order[-1]], 'stationary':None}
    data.update(selected=winner, point_source_index=point_best, opposite_index=order[-1], checkpoint=checkpoint,
                checkpoint_sha256=_hash(CHECKPOINT), frozen=modes,
                combined_duration_ticks={name:duration(rows) for name,rows in combined.items()})
    modes['permuted'] = shuffled_gain(data)
    _write(RECEIPT,data)
    print('frozen',winner,checkpoint['coefficients'][0], 'validation', data['combined_duration_ticks'],flush=True)
    data['held'] = play(held_cases,modes,roles,batch)
    data['held_summary'] = {name:summary(rows) for name,rows in data['held'].items()}
    data['paired_against_duration'] = {name:paired(rows,data['held']['duration'])
                                       for name,rows in data['held'].items() if name!='duration'}
    data['wall_seconds'] = perf_counter()-begins
    _write(RECEIPT,data)
    print('held', {name:(res['duration_seconds'],res['later_returns'],res['points_total'])
                   for name,res in data['held_summary'].items()},flush=True)


def append_permutation(batch):
    import torch

    torch.set_num_threads(1)
    data = _read(RECEIPT)
    cases = [PongCase(**item) for item in data['held_cases']]
    gains = shuffled_gain(data)
    rows = play(cases,{'permuted':gains},RoleModel(**_read(ROLE_MODEL)),batch)['permuted']
    data['frozen']['permuted'] = gains
    data['held']['permuted'] = rows
    data['held_summary']['permuted'] = summary(rows)
    data['paired_against_duration']['permuted'] = paired(rows,data['held']['duration'])
    _write(RECEIPT,data)
    print('permuted', data['held_summary']['permuted'], flush=True)


def verify_held(count,batch):
    import torch

    torch.set_num_threads(1)
    data = _read(RECEIPT)
    cases = [PongCase(**item) for item in data['held_cases'][:count]]
    replay = play(cases,data['frozen'],RoleModel(**_read(ROLE_MODEL)),batch)
    def tape(row):
        return json.dumps({key:value for key,value in row.items() if key!='relation'},sort_keys=True)
    for name,rows in replay.items():
        if [tape(row) for row in rows] != [tape(row) for row in data['held'][name][:len(cases)]]:
            raise ValueError(f'native actions or physical events diverged in {name}')
    data['verified_held_episodes'] = len(cases)*len(replay)
    _write(RECEIPT,data)
    print('verified',data['verified_held_episodes'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=int,default=12)
    parser.add_argument('--validation',type=int,default=16)
    parser.add_argument('--held',type=int,default=32)
    parser.add_argument('--batch',type=int,default=48)
    parser.add_argument('--permuted-held',action='store_true')
    parser.add_argument('--verify',type=int)
    args=parser.parse_args()
    if min(args.screen,args.validation,args.held,args.batch)<=0:
        parser.error('positive episode and batch counts are required')
    if args.verify:
        verify_held(args.verify,args.batch)
    elif args.permuted_held:
        append_permutation(args.batch)
    else:
        run(args.screen,args.validation,args.held,args.batch)
