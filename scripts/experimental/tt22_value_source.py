"""Fit short optical future paths against later native Pong survival."""

import argparse
from time import perf_counter

import numpy as np
from scipy.stats import spearmanr

from .tt3_numeric import NumericMemory
from .tt3_pong import PongCase, _native_rallies
from .tt3_pong_probe import ShorterDwell
from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_pong import CAP, OPPONENT_SPEED, OUTPUT, RECEIPT as DURATIONS, ROLE_MODEL, OpticalDurationAgent
from .tt22_value import EXTENDED, FRAMES, describe, fit, score


RECEIPT = OUTPUT/'value-source.json.gz'
LONG = OUTPUT/'value-source-extended.json.gz'
SHUFFLE = 1325077421
BATCH = 48


class TraceAgent(OpticalDurationAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.trace = []

    def observe(self,image,preceding_action,reward):
        super().observe(image,preceding_action,0.)
        actor,mover = self.actor,self.binding.relevant
        a = actor.position if actor is not None and actor.observed else None
        b = mover.position if mover is not None and mover.observed and (actor is None or mover.id!=actor.id) else None
        self.trace.append([a,b])


def capture(cases,candidates):
    roles = RoleModel(**_read(ROLE_MODEL))
    memory = NumericMemory(list(candidates.values()))
    labels = list(candidates)
    schedule = [(PongCase(**item),index) for item in cases for index in range(len(candidates))]
    rows = []
    for i in range(0,len(schedule),BATCH):
        batch = schedule[i:i+BATCH]
        agents = [TraceAgent(roles,memory,case.exploration,option=index,motor_factory=ShorterDwell) for case,index in batch]
        native = _native_rallies([c for c,_ in batch],agents,max_ticks=CAP,opponent_speed=OPPONENT_SPEED)
        for (_,index),episode,agent in zip(batch,native,agents,strict=True):
            rows.append(dict(case=episode['case'],policy=labels[index],ticks=episode['ticks'],
                             action_sha256=episode['action_sha256'],trace=agent.trace))
    return rows


def complete(points,frames):
    """Interpolate short optical gaps and extend the last visible motion."""
    available = np.flatnonzero(np.isfinite(points).all(axis=1))
    if len(available)<3 or available[0]!=0 or np.max(np.diff(available))>4:
        return None
    data = points[available]
    if np.max(np.linalg.norm(np.diff(data,axis=0)/np.diff(available)[:,None],axis=1))>.085:
        return None
    time = np.arange(frames+1)
    result = np.column_stack([np.interp(time,available,column) for column in data.T])
    prior = max(0,len(data)-4)
    tangent = (data[-1]-data[prior])/(available[-1]-available[prior])
    result += np.maximum(0,time-available[-1])[:,None]*tangent
    return result


def examples(episodes,frames=FRAMES):
    rows = []
    for episode in episodes:
        missing = [np.nan,np.nan]
        actor,mover = (np.array([entry[index] if entry[index] is not None else missing for entry in episode['trace']])
                       for index in (0,1))
        last = -FRAMES
        for tick in range(4,len(mover)-2):
            if tick-last<FRAMES or not np.isfinite(np.r_[actor[tick],mover[tick],mover[tick-3]]).all():
                continue
            velocity = (mover[tick]-mover[tick-3])/3.
            relative = mover[tick]-actor[tick]
            distance = np.linalg.norm(relative)
            inward = -(velocity@relative)/max(distance,1.e-7)
            if inward<.007 or inward<.62*np.linalg.norm(velocity) or not 5.5<=distance/inward<=10.5:
                continue
            path = complete(mover[tick:tick+frames+1],frames)
            if path is None:
                continue
            last = tick
            remaining = max(0.,episode['ticks']/120.-(tick+frames)/30.)
            rows.append(dict(seed=episode['case']['serve'],policy=episode['policy'],tick=tick,
                             relative_features=describe(path[None],actor[tick],velocity)[0].tolist(),
                             remaining_seconds=remaining))
    return rows


def matrix(rows):
    return (np.array([row['relative_features'] for row in rows]),
            np.log1p([row['remaining_seconds'] for row in rows]))


def quality(model,rows):
    x,y = matrix(rows)
    predicted = score(model,x)
    rho = spearmanr(predicted,y).statistic
    return dict(examples=len(y),mae_log_seconds=float(np.mean(abs(predicted-y))),
                spearman=float(rho) if np.isfinite(rho) else None,
                negative_examples=int(sum(y==0.)),mean_label=float(y.mean()))


def extended():
    data = _read(RECEIPT)
    training,validation = examples(data['train'],EXTENDED),examples(data['held'],EXTENDED)
    x,y = matrix(training)
    permutation = np.random.default_rng(SHUFFLE).permutation(len(y))
    models = {'duration':fit(x,y),'shuffled':fit(x,y[permutation])}
    result = dict(schema='tt22-pong-value-extended-1',source_sha256=_hash(RECEIPT),source=RECEIPT.name,
                  candidates=data['candidates'],training_examples=training,held_examples=validation,frames=EXTENDED,
                  target='log(1 + observed survival seconds after the 44-frame lookahead)',models=models,
                  shuffled_order=permutation.tolist(),quality={name:quality(model,validation) for name,model in models.items()})
    _write(LONG,result)
    print('extended source',len(training),len(validation),result['quality'],flush=True)


def run(count):
    import torch

    torch.set_num_threads(1)
    archive = _read(DURATIONS)
    leaders = list(archive['screen_duration_ticks'])[:3]
    candidates = {name:archive['frozen'][name] for name in ('duration','point','cold','permuted')}
    candidates.update({f'screen_{i}':archive['candidates'][i] for i in leaders if i!=archive['selected']})
    train_cases = archive['screen_cases'] + archive['validation_cases'][:max(0,count-len(archive['screen_cases']))]
    train_cases = train_cases[:count]
    held_cases = archive['held_cases'][:12]
    started = perf_counter()
    train,held = capture(train_cases,candidates),capture(held_cases,candidates)
    if set(e['case']['serve'] for e in train)&set(e['case']['serve'] for e in held):
        raise ValueError('source training and held native episodes overlap')
    # Reused policies on the archived cases reproduce their native actions and durations.
    reference = {(name,row['case']['serve']):row for name,rows in archive['held'].items() for row in rows}
    verified = 0
    for episode in held:
        original = reference.get((episode['policy'],episode['case']['serve']))
        if original is not None:
            if any(episode[key]!=original[key] for key in ('ticks','action_sha256')):
                raise ValueError('an original RGB policy did not reproduce its native tape')
            verified += 1
    training,validation = examples(train),examples(held)
    x,y = matrix(training)
    permutation = np.random.default_rng(SHUFFLE).permutation(len(y))
    models = {'duration':fit(x,y),'shuffled':fit(x,y[permutation])}
    data = dict(schema='tt22-pong-value-1',duration_receipt_sha256=_hash(DURATIONS),role_sha256=_hash(ROLE_MODEL),
                candidates=candidates,frames=FRAMES,scale='relative initial distance and incoming speed',
                target='log(1 + observed survival seconds after the 24-frame lookahead)',
                train=train,held=held,training_examples=training,held_examples=validation,
                shuffled_order=permutation.tolist(),models=models,quality={name:quality(w,validation) for name,w in models.items()},
                verified_original_tapes=verified,wall_seconds=perf_counter()-started)
    _write(RECEIPT,data)
    print('value source',len(training),len(validation),data['quality'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=18)
    parser.add_argument('--extended',action='store_true')
    args = parser.parse_args()
    if args.extended:
        extended()
    else:
        run(args.cases)
