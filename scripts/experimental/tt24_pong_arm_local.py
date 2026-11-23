"""Fit causal local actuator response from already recorded physical Pong play."""

import argparse

import numpy as np
import torch

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE,CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_motor_bank import I,STEP
from .tt24_pong_arm import bounce,historical_retention,wirings
from .tt24_pong_arm_tip import SOURCE as TIP,Joystick,plan,point,stats
from .tt4_transfer_trial import _hash,_read,_write


SOURCE = ROOT/'pong-arm-local-source.json.gz'
TARGET = ROOT/'pong-arm-local-held.json.gz'
WINDOW = 8
ALPHAS = (.28,.50,.74,1.)
HOLDOUT_SEED = 67217811


def windows(clips,alpha,*,shuffle=False):
    groups = []
    rng = np.random.default_rng(157653410)
    for clip in clips:
        observations = np.asarray(clip['encoders'])
        requested = np.asarray(clip['commands'])
        integral = np.vstack((np.zeros(5),np.cumsum(STEP*requested,axis=0)))
        smooth = np.zeros_like(integral)
        for i in range(1,len(smooth)):
            smooth[i] = smooth[i-1]+alpha*(integral[i]-smooth[i-1])
        x = smooth[WINDOW:]-smooth[:-WINDOW]
        y = observations[WINDOW:]-observations[:-WINDOW]
        total = np.vstack((np.zeros(5),np.cumsum(abs(np.diff(observations,axis=0)),axis=0)))
        activity = total[WINDOW:]-total[:-WINDOW]
        if shuffle:
            shift = int(rng.integers(max(4,len(y)//5),max(5,4*len(y)//5)))
            y = np.roll(y,shift,axis=0)
            activity = np.roll(activity,shift,axis=0)
        active = (activity>=.00065) & (np.linalg.norm(x,axis=1,keepdims=True)>.00055)
        groups.append((x,y,active))
    return groups


def samples(groups):
    return tuple(np.concatenate(column) for column in zip(*groups,strict=True))


def regress(groups,joint):
    x,y,active = samples(groups)
    visible = active[:,joint]
    x,y = x[visible],y[visible,joint]
    regularizer = 1e-6
    weights = np.ones(len(y))
    for _ in range(13):
        wx = x*weights[:,None]
        transfer = np.linalg.solve(x.T@wx+regularizer*I,wx.T@y+regularizer*I[joint])
        residual = abs(y-x@transfer)
        scale = max(.00035,1.7*float(np.median(residual)))
        weights = 1. / (1. + (residual/scale)**2)
    return transfer


def robust_error(matrix,groups,joint):
    x,y,active = samples(groups)
    seen = active[:,joint]
    delta = abs(y[seen,joint]-x[seen]@matrix)
    return float(np.quantile(delta,.70))


def fit(clips,*,shuffle=False):
    best = [None]*5
    for alpha in ALPHAS:
        data = windows(clips,alpha,shuffle=shuffle)
        groups = [data[::2],data[1::2]]
        for joint in range(5):
            validation = np.mean([robust_error(regress(a,joint),b,joint) for a,b in (groups,groups[::-1])])
            if best[joint] is None or validation<best[joint]['internal_p70_rad']:
                best[joint] = dict(alpha=alpha,internal_p70_rad=float(validation),
                                   matrix_row=regress(data,joint).tolist())
    return best


def crosscheck(model,clips):
    output = []
    for joint, estimator in enumerate(model):
        data = windows(clips,estimator['alpha'])
        x,y,active = samples(data)
        useful = active[:,joint]
        singular = np.linalg.svd(x[useful],compute_uv=False)
        predicted = x[useful]@estimator['matrix_row']
        error = abs(predicted-y[useful,joint])
        output.append(dict(available_frames=int(useful.sum()),singular_values=singular.tolist(),
                           p70_rad=float(np.quantile(error,.70)), rms_rad=float(np.sqrt(np.mean(error**2))),
                           stale_p70_rad=robust_error(I[joint],data,joint)))
    return output


def source(arena):
    parent = _read(TIP)
    known_validation = set(parent['matched']['held_source_seeds'])
    practice = [clip for clip in parent['active'] if clip['seed'] not in known_validation]
    coils = dict(matched=fit(practice),shuffled=fit(practice,shuffle=True))
    joystick = Joystick()
    dense = wirings()[1]
    starts,_ = plan(joystick,count=6,seed=HOLDOUT_SEED)
    if {seed for seed,_ in starts}.intersection(clip['seed'] for clip in parent['active']):
        raise ValueError('fresh source tests overlap earlier source experience')
    tests = []
    for seed,pose in starts:
        live = point(arena,joystick,seed,pose,dense)
        replay = point(arena,joystick,seed,pose,dense,replay=live['scalar_commands'])
        if digest_json(live)!=digest_json(replay):
            raise ValueError('fresh native source replay changed')
        tests.append(live)
        print('fresh native local',seed,live['returns'],live['score'],flush=True)
    prediction = {name:crosscheck(model,tests) for name,model in coils.items()}
    estimates = {name:np.array([o['matrix_row'] for o in row]) for name,row in coils.items()}
    physical = {name:dict(relative_error=float(np.linalg.norm(row-dense)/np.linalg.norm(dense)),
                           row_error=(np.linalg.norm(row-dense,axis=1)/np.linalg.norm(dense,axis=1)).tolist())
                for name,row in estimates.items()}
    good = all(s['p70_rad']<.0018 and s['p70_rad']<.50*s['stale_p70_rad'] and
               s['singular_values'][-1]/s['singular_values'][0]>.009 for s in prediction['matched'])
    trusted = good and physical['matched']['relative_error']<.18
    receipt = dict(tip_sha256=_hash(TIP),bundle_sha256=_hash(BUNDLE),prior_source_ordinary_points=len(practice),
                   fit_seeds=[c['seed'] for c in practice], held=tests, source_summary=stats(tests),
                   filtered_window_ticks=WINDOW,source_gate=bool(trusted),fit=coils,
                   prediction=prediction,evaluator_only=physical)
    _write(SOURCE,receipt)
    print('local source fit',physical,[(row['p70_rad'],row['stale_p70_rad'],
          row['singular_values'][-1]/row['singular_values'][0]) for row in prediction['matched']],trusted,flush=True)
    return receipt


def target(arena,receipt):
    if not receipt['source_gate']:
        return
    if receipt['bundle_sha256']!=_hash(BUNDLE):
        raise ValueError('the usual bouncing head has changed')
    policy = CarriedWristPolicy()
    first = _read(ROOT/'pong-arm-source.json.gz')
    maps = {'stale':I,'one_axis':np.asarray(first['configurations']['matched'])}
    maps.update({name:np.asarray([r['matrix_row'] for r in fit]) for name,fit in receipt['fit'].items()})
    roster = np.random.default_rng(314207562).integers(1,2**31-1,12).tolist()
    report = _read(TARGET) if TARGET.exists() else dict(source_sha256=_hash(SOURCE),rows=[])
    if report['source_sha256']!=_hash(SOURCE):
        raise ValueError('the held reads belong to a different sealed estimate')
    physical = wirings()[1]
    for seed in roster[len(report['rows']):]:
        cases = {}
        for name,estimate in maps.items():
            active = bounce(arena,policy,seed,physical,estimate)
            raw = bounce(arena,policy,seed,physical,estimate,tape=active['actions'])
            if digest_json(active)!=digest_json(raw):
                raise ValueError('dense wrist physical replay diverged')
            cases[name] = active
        report['rows'].append(dict(seed=seed,conditions=cases))
        _write(TARGET,report)
        print('local dense wrist',seed,{m:r['metrics']['longest_consecutive_rebounds'] for m,r in cases.items()},flush=True)
    classes = {name:np.asarray([r['conditions'][name]['metrics']['longest_consecutive_rebounds']
                                for r in report['rows']]) for name in maps}
    report['counts'] = {name:dict(mean=float(row.mean()),ten=int((row>=10).sum()),maximum=int(row.max()),
                                  wins=int((row>classes['stale']).sum())) for name,row in classes.items()}
    report['exact_replays'] = len(roster)*len(maps)
    if 'historical_stock_retention' not in report:
        report['historical_stock_retention'] = historical_retention()
    _write(TARGET,report)
    print(report['counts'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    arena = BounceArena(view='wrist',width=192,height=144,max_steps=300)
    try:
        receipt = _read(SOURCE) if args.held else source(arena)
        target(arena,receipt)
    finally:
        arena.close()


if __name__=='__main__':
    main()
