"""Fit a physical stroke value to the time until a point ends."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import rankdata, spearmanr

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw, optical_start
from .tt19_model_rollout import ORIGIN, observed_ball
from .tt19_shoot import HORIZON
from .tt22_flight_inverse import FlightInverse
from .tt22_rally_trial import OUTPUT as TT22
from .tt22_value import EXTENDED, describe
from .tt22_value_calibration import ARCHIVES, chosen_archive_rows, kinetics, matrix
from .tt22_value_player import TrajectoryDrySimulator
from .tt22_value_support import SupportedValue
from .tt23_coach import OUTPUT


RECEIPT = OUTPUT/'duration-value-archive.json.gz'
SHUFFLE = 1955248134
SCALES = (32.,72.,150.)
WEIGHTS = (.55,.30,.15)
SURVIVAL_THRESHOLDS = (35,60,100)
NEW_BANKS = (
    (TT22/'ordinary-game.json.gz','branches','ordinary'),
    (TT22/'after-service.json.gz','rows','after_service'),
    (OUTPUT/'coach-held.json.gz','rows','coach'),
)


def archive_points():
    sources = [(TT22/name,None,'earlier') for name in ARCHIVES]+list(NEW_BANKS)
    points, manifest = [], {}
    seen = set()
    for path,group,bank in sources:
        data = _read(path)
        verified = data.get('verified',data.get('replayed',0))
        if isinstance(verified,dict):
            verified = sum(verified.values())
        rows = chosen_archive_rows(data) if group is None else [
            row for branch in data[group].values() for row in branch]
        if verified!=len(rows):
            raise ValueError(f'incomplete source replay in {path.name}')
        before = len(points)
        for row in rows:
            near = np.ascontiguousarray(row['actions'],dtype='<f8')
            sight = np.asarray([(tick,*position) for tick,position in row['optical']],dtype='<f8')
            identity = row['seed'],row['role'],sha256(near.tobytes()+sight.tobytes()).hexdigest()
            if identity not in seen:
                points.append((bank,path.name,identity[-1],row))
                seen.add(identity)
        manifest[path.name] = dict(sha256=_hash(path),bank=bank,verified=verified,distinct=len(points)-before)
    return points,manifest


def continuation(frames):
    time = np.asarray(frames,dtype=float)
    return (1.-np.exp(-time[...,None]/SCALES))@WEIGHTS


def decode(sim,inverse,law,row):
    """Replay only issued near commands and chronological stereo observations."""
    actions = np.asarray(row['actions'])
    optical = dict(row['optical'])
    shots = {record['tick']:record for record in row['shots']}
    sim.twin.reset()
    history,examples = [],[]
    for tick,command in enumerate(actions):
        history.append((tick,np.asarray(optical[tick]) if tick in optical else None))
        if tick in shots:
            estimate = optical_start(history,law.gravity)
            if estimate is not None:
                length = int(np.clip(np.ceil(shots[tick]['eta'])+3,3,HORIZON))
                recorded = actions[tick:tick+length,:5]
                tape = np.zeros((1,HORIZON,5))
                tape[0,:len(recorded)] = recorded
                observation = observed_ball(history,estimate,law)
                fit = inverse.fit(history)
                if fit is not None:
                    _,velocity,spin,_ = fit
                    observation = replace(observation,velocity=velocity,spin=spin)
                mujoco.mj_kinematics(sim.model,sim.twin.data)
                actor = sim.twin.data.geom('learner_racket').xpos.copy()-ORIGIN
                paths = sim.paths(tape,np.array([length]),observation,EXTENDED)
                relative = describe(paths,actor,observation.velocity/30.)[0]
                local = kinetics(paths,actor)[0]
                remaining = len(actions)-tick
                examples.append(dict(tick=tick,remaining_frames=remaining,reward=float(continuation(remaining)),
                                     source_features=relative.tolist(),target_features=local.tolist(),
                                     padded_command_frames=length-len(recorded),inverse_fit=fit is not None))
        sim.twin.advance(command)
    return examples


def collect():
    points,manifest = archive_points()
    law = FlightLaw()
    sim = TrajectoryDrySimulator(law)
    inverse = FlightInverse()
    rows,seen = [],set()
    try:
        for bank,source,digest,row in points:
            for example in decode(sim,inverse,law,row):
                features = np.asarray(example['source_features']+example['target_features'],dtype='<f8')
                identity = (row['seed'],example['tick'],example['remaining_frames'],features.tobytes())
                if identity not in seen:
                    rows.append(dict(example,seed=row['seed'],role=row['role'],bank=bank,file=source,tape=digest))
                    seen.add(identity)
    finally:
        sim.close()
    return rows,manifest,sorted({row['seed'] for _,_,_,row in points}),len(points)


def models(rows,shuffle):
    counts = Counter(row['seed'] for row in rows)
    permutation = np.random.default_rng(shuffle).permutation(len(rows))
    checkpoint = dict(training=rows,shuffle_order=permutation.tolist())
    native,shuffled = (SupportedValue(checkpoint,mode) for mode in ('kernel','kernel_shuffled'))
    per_seed = len(rows)/len(counts)
    weights = np.array([per_seed/counts[row['seed']] for row in rows])
    native.weights = weights
    shuffled.weights = weights.copy()
    return native,shuffled,permutation


def predict_group(training,held,indices,shuffle):
    model,null,_ = models(training,shuffle)
    _,features,_ = matrix(held)
    baseline = float(np.average(model.labels,weights=model.weights))
    return [dict(index=int(index),duration=float(a),shuffled=float(b),constant=baseline)
            for index,a,b in zip(indices,model.predict(features),null.predict(features),strict=True)]


def held_predictions(rows,key):
    groups = np.array([row[key] for row in rows])
    result = []
    for number,group in enumerate(np.unique(groups)):
        held = np.flatnonzero(groups==group)
        test_seeds = {rows[i]['seed'] for i in held}
        train = [row for row in rows if row['seed'] not in test_seeds]
        result.extend(predict_group(train,[rows[i] for i in held],held,SHUFFLE+number))
    return sorted(result,key=lambda row:row['index'])


def quality(rows,predictions):
    selected = [rows[item['index']] for item in predictions]
    seeds = np.array([row['seed'] for row in selected])
    time = np.array([row['remaining_frames'] for row in selected])
    reward = continuation(time)
    result = dict(shots=len(time),seeds=len(set(seeds)))
    for mode in ('duration','shuffled','constant'):
        estimates = np.array([item[mode] for item in predictions])
        error = abs(estimates-reward)
        rho = spearmanr(time,estimates).statistic if len(set(estimates))>1 else None
        ranks = rankdata(estimates)
        survival = {}
        for threshold in SURVIVAL_THRESHOLDS:
            survived = time>=threshold
            n = int(survived.sum())
            auc = (ranks[survived].sum()-n*(n+1)/2)/(n*(len(time)-n)) if 0<n<len(time) else None
            survival[str(threshold)] = dict(observed=n,auc=auc)
        result[mode] = dict(spearman=rho,mean_seed_mae=float(np.mean([error[seeds==seed].mean() for seed in set(seeds)])),
                            survival=survival,score_range=[float(estimates.min()),float(estimates.max())])
    return result


def matched_choices(rows,predictions):
    """Compare issued options following the same two-arm and optical prefix."""
    points,_ = archive_points()
    source = {(row['seed'],row['role'],digest):row for bank,_,digest,row in points if bank!='earlier'}
    groups = {}
    for item in predictions:
        row = rows[item['index']]
        raw = source.get((row['seed'],row['role'],row['tape']))
        if raw is None:
            continue
        tick = row['tick']
        near = np.asarray(raw['actions'][:tick],dtype='<f8')
        far = np.asarray(raw['rival_actions'][:tick],dtype='<f8')
        optical = np.asarray([(t,*p) for t,p in raw['optical'] if t<=tick],dtype='<f8')
        digest = sha256(near.tobytes()+far.tobytes()+optical.tobytes()).hexdigest()
        key = (row['seed'],row['role'],tick,digest)
        shot = next(record for record in raw['shots'] if record['tick']==tick)
        length = int(np.clip(np.ceil(shot['eta'])+3,3,HORIZON))
        option = sha256(np.asarray(raw['actions'][tick:tick+length],dtype='<f8').tobytes()).hexdigest()
        groups.setdefault(key,{}).setdefault(option,[]).append(item)
    compared = []
    hits = dict(duration=0.,shuffled=0.)
    for (seed,role,tick,digest),options in groups.items():
        choices = [(key,[rows[item['index']]['remaining_frames'] for item in items],items[0])
                   for key,items in options.items()]
        for (left,left_time,a),(right,right_time,b) in combinations(choices,2):
            delta = float(np.mean(left_time)-np.mean(right_time))
            if abs(delta)<2:
                continue
            change = {mode:a[mode]-b[mode] for mode in hits}
            for mode,value in change.items():
                hits[mode] += 0.5 if value==0 else float(np.sign(value)==np.sign(delta))
            compared.append(dict(seed=seed,role=role,tick=tick,prefix=digest,left=left,right=right,
                                 left_remaining_frames=left_time,right_remaining_frames=right_time,
                                 learned_difference=change))
    accuracy = {mode:count/len(compared) for mode,count in hits.items()} if compared else {}
    return dict(pairs=len(compared),seeds=len({row['seed'] for row in compared}),accuracy=accuracy,rows=compared)


def train(output):
    started = perf_counter()
    rows,manifest,seeds,points = collect()
    grouped = held_predictions(rows,'seed')
    bank_held = held_predictions(rows,'bank')
    new = [i for i,row in enumerate(rows) if row['bank']!='earlier']
    new_seeds = {rows[i]['seed'] for i in new}
    old = [row for row in rows if row['bank']=='earlier' and row['seed'] not in new_seeds]
    old_only = predict_group(old,[rows[i] for i in new],new,SHUFFLE)
    seed_quality = quality(rows,grouped)
    cross_bank = quality(rows,bank_held)
    newer_quality = quality(rows,[pick for pick in grouped if rows[pick['index']]['bank']!='earlier'])
    per_bank = {bank:quality(rows,[pick for pick in grouped if rows[pick['index']]['bank']==bank])
                for bank in sorted({row['bank'] for row in rows})}
    real,shuffled,constant = (seed_quality[mode] for mode in ('duration','shuffled','constant'))
    gate = bool(real['survival']['60']['auc']>=.70 and real['spearman']>=shuffled['spearman']+.15
                and real['mean_seed_mae']<min(shuffled['mean_seed_mae'],constant['mean_seed_mae'])
                and cross_bank['duration']['spearman']>cross_bank['shuffled']['spearman'])
    fitted,null,permutation = models(rows,SHUFFLE)
    receipt = dict(schema='tt23-duration-value-archive-1',sources=manifest,seeds=seeds,distinct_points=points,
                   scales=list(SCALES),weights=list(WEIGHTS),horizon=EXTENDED,rows=rows,
                   shuffle_order=permutation.tolist(),live_gate=gate,seed_held=seed_quality,bank_held=cross_bank,
                   seed_held_by_bank=per_bank,seed_held_on_new_banks=newer_quality,early_banks_only=quality(rows,old_only),
                   matched_seed_held=matched_choices(rows,grouped),
                   seed_held_predictions=grouped,bank_held_predictions=bank_held,early_bank_predictions=old_only,
                   models={'duration':fitted.state(),'shuffled':null.state()},wall_seconds=perf_counter()-started)
    _write(output,receipt)
    print('duration archive',len(rows),'decisions',len(seeds),'seeds',points,'points',
          'seed held',seed_quality,'bank held',cross_bank,'live gate',gate,flush=True)
    return receipt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RECEIPT)
    train(parser.parse_args().output)
