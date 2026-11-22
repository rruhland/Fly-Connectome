"""Calibrate trajectory value from archived RGB, issued motor tapes and real rewards."""

import argparse
from time import perf_counter

import numpy as np
from scipy.stats import rankdata, spearmanr

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw, optical_start
from .tt19_model_rollout import ORIGIN, observed_ball
from .tt22_rally_trial import OUTPUT
from .tt22_value import EXTENDED, describe, fit, score
from .tt22_value_player import TrajectoryDrySimulator


SOURCE = OUTPUT/'value-source-extended.json.gz'
CALIBRATION = OUTPUT/'value-calibration.json.gz'
ARCHIVES = ('rally-anchor-612423769.json.gz','rally-anchor-220381117.json.gz',
            'rally-learning-anchor.json.gz','rally-anchor-8351422.json.gz',
            'imaging-live.json.gz','value-live.json.gz')
SHUFFLE = 442905153
MODES = ('hybrid','target','source','shuffled')
WEAK_PRIOR = .12


def kinetics(paths,actor):
    """Additional target moments in a local horizontal frame and gravity direction."""
    r = paths-actor
    distance = np.maximum(1.e-7,np.linalg.norm(r[:,0],axis=-1))
    axis = r[:,0,:2].copy()
    axis /= np.maximum(1.e-7,np.linalg.norm(axis,axis=-1))[:,None]
    along = np.einsum('btd,bd->bt',r[:,:,:2],axis)
    across = np.einsum('btd,bd->bt',r[:,:,:2],axis[:,::-1]*[-1.,1.])
    local = np.stack((along,across,r[:,:,2]),axis=-1)/distance[:,None,None]
    sample = local[:,[0,12,20,28,36,44]]
    positions = sample[:,2:].copy()
    positions[:,:,1] = abs(positions[:,:,1])
    velocity = np.diff(sample,axis=1)[:,1:]
    velocity[:,:,1] = abs(velocity[:,:,1])
    curvature = np.diff(velocity,axis=1)
    curvature = np.linalg.norm(curvature,axis=-1)
    return np.column_stack((positions.reshape(len(paths),-1),velocity.reshape(len(paths),-1),curvature))


def chosen_archive_rows(data):
    base = list(data.get('rows',()))+list(data.get('practice',()))
    for pair in data.get('pairs',()):
        base.extend(value for value in pair.values() if isinstance(value,dict) and 'shots' in value)
    return base


def stroke_outcomes(row):
    attempts = [record['tick'] for record in row['shots']]
    end = len(row['actions'])
    own_returns = [event['step'] for event in row['events']
                   if event['kind']=='legal_return' and event.get('side')==1]
    result = {}
    for tick,next_shot in zip(attempts,attempts[1:]+[end]):
        actual_return = sum(tick<event<=next_shot for event in own_returns)
        remaining = max(0.,(end-tick-EXTENDED)/30.)
        result[tick] = dict(actual_return=actual_return,remaining_seconds=remaining,
                            reward=float(actual_return+.30*np.log1p(remaining)))
    return result


def shot_examples(sim,row,source,law):
    actions = np.array(row['actions'])
    sight = {tick:np.asarray(position) for tick,position in row['optical']}
    shots = {record['tick']:record for record in row['shots']}
    outcomes = stroke_outcomes(row)
    sim.twin.reset()
    history, result = [],[]
    for tick,action in enumerate(actions):
        history.append((tick,sight.get(tick)))
        if tick in shots:
            estimate = optical_start(history,law.gravity)
            if estimate is not None:
                length = min(max(3,int(np.ceil(shots[tick]['eta']))+3),20,len(actions)-tick)
                tape = np.zeros((1,20,5))
                tape[0,:length] = actions[tick:tick+length,:5]
                mover = observed_ball(history,estimate,law)
                mujoco.mj_kinematics(sim.model,sim.twin.data)
                actor = sim.twin.data.geom('learner_racket').xpos.copy()-ORIGIN
                paths = sim.paths(tape,np.array([length]),mover,EXTENDED)
                relation = describe(paths,actor,mover.velocity/30.)
                geometry = kinetics(paths,actor)
                result.append(dict(seed=row['seed'],tick=tick,source=source,**outcomes[tick],
                                   optical=mover.center.tolist(),actor=actor.tolist(),
                                   source_features=relation[0].tolist(),target_features=geometry[0].tolist()))
        sim.twin.advance(action)
    return result


class CalibratedValue:
    def __init__(self,checkpoint,mode):
        self.source = checkpoint['pong_model']
        self.base = checkpoint['prior_location']
        self.scale = checkpoint['prior_scale']
        self.target = None if mode=='source' else checkpoint['models'][mode]
        self.coefficient = WEAK_PRIOR if mode=='hybrid' else 0.

    def __call__(self,source_features,paths,actor):
        upstream = score(self.source,source_features)
        if self.target is None:
            return upstream
        data = np.column_stack((source_features,kinetics(paths,actor)))
        residual = score(self.target,data)
        return residual+self.coefficient*np.clip((upstream-self.base)/self.scale,-3.,3.)


def matrix(rows):
    upstream = np.array([row['source_features'] for row in rows])
    local = np.array([row['target_features'] for row in rows])
    return upstream,np.column_stack((upstream,local)),np.array([row['reward'] for row in rows])


def quality(check,rows):
    relation,data,reward = matrix(rows)
    labels = np.array([row['actual_return']>0 for row in rows])
    n = int(sum(labels))
    baseline = score(check['pong_model'],relation)
    table = {}
    for mode in MODES:
        if mode=='source':
            prediction = baseline
        else:
            prediction = score(check['models'][mode],data)
            if mode=='hybrid':
                prediction += WEAK_PRIOR*np.clip((baseline-check['prior_location'])/check['prior_scale'],-3.,3.)
        rho = spearmanr(reward,prediction).statistic
        auc = (rankdata(prediction)[labels].sum()-n*(n+1.)/2.)/(n*(len(labels)-n)) if 0<n<len(labels) else None
        table[mode] = dict(spearman=None if not np.isfinite(rho) else float(rho),
                           return_auc=None if auc is None else float(auc),
                           mean_absolute_error=None if mode=='source' else float(np.mean(abs(prediction-reward))))
    return dict(shots=len(rows),rewarded=n,models=table)


def calibration():
    pong = _read(SOURCE)
    began = perf_counter()
    law = FlightLaw()
    sim = TrajectoryDrySimulator(law)
    rows, counts, seen = [],{},set()
    try:
        for name in ARCHIVES:
            archive = _read(OUTPUT/name)
            before = len(rows)
            for row in chosen_archive_rows(archive):
                # Some earlier paired archives deliberately re-used identical action paths.
                key = row['seed'],hash(np.array(row['actions'],dtype='<f8').tobytes())
                if key in seen:
                    continue
                seen.add(key)
                rows.extend(shot_examples(sim,row,name,law))
            counts[name] = len(rows)-before
            print('value calibration',name,counts[name],flush=True)
    finally:
        sim.close()
    seeds = np.random.default_rng(SHUFFLE).permutation(sorted({row['seed'] for row in rows})).tolist()
    check_seeds = set(seeds[:max(2,len(seeds)//5)])
    train = [row for row in rows if row['seed'] not in check_seeds]
    valid = [row for row in rows if row['seed'] in check_seeds]
    relation,data,reward = matrix(train)
    pong_model = pong['models']['duration']
    upstream = score(pong_model,relation)
    center, spread = float(np.mean(upstream)),max(.10,float(np.std(upstream)))
    correction = WEAK_PRIOR*np.clip((upstream-center)/spread,-3.,3.)
    permutation = np.random.default_rng(SHUFFLE+1).permutation(len(reward))
    result = dict(schema='tt22-value-calibration-1',source_sha256=_hash(SOURCE),pong_model=pong_model,
                  weak_prior=WEAK_PRIOR,prior_location=center,prior_scale=spread,horizon=EXTENDED,
                  source_files={name:_hash(OUTPUT/name) for name in ARCHIVES},archive_row_counts=counts,
                  descriptor='actor-relative 2D/3D path and 3D moments with physical gravity direction',
                  label='local real learner returns plus .30 log(1 + native survival seconds beyond 44 frames)',
                  archived_seeds=seeds,training=train,validation=valid,shuffle_order=permutation.tolist(),
                  models={'hybrid':fit(data,reward-correction),'target':fit(data,reward),
                          'shuffled':fit(data,reward[permutation])})
    result['quality'] = quality(result,valid)
    result['wall_seconds'] = perf_counter()-began
    _write(CALIBRATION,result)
    print('value calibration',len(train),sum(row['actual_return']>0 for row in train),result['quality'],flush=True)


if __name__=='__main__':
    args = argparse.ArgumentParser(description=__doc__).parse_args()
    calibration()
