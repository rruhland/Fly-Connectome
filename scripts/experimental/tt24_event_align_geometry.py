"""Test a geometry-constrained, prospective wrist transition model."""

import argparse
from pathlib import Path

import numpy as np

from .tt24_bounce_ident import _INC
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import Native
from .tt24_event_align import (DATA, FIT, MODEL, OUT, NewWide, arrival_features,
                               expected_event_time, exposure, pair_features, pulse,
                               response, wrist_sensor)
from .tt4_transfer_trial import _hash, _read, _write

NEW=tuple(range(371201,371213))
HELD=OUT/'event-align-geometry-held.json.gz'
EVIDENCE=OUT/'event-align-geometry-fit.json.gz'
CHECKPOINT_OUT=OUT/'event-align-geometry.npz'
LAGS=(0,1,2,4,6)
GATE_RATIO=.92


def collect(args):
    native=Native()
    with np.load(CHECKPOINT,allow_pickle=False) as release:
        weights=np.ascontiguousarray(release['weights'])
    arena=NewWide()
    out=dict(schema='tt24-event-geometry-independent-1',seeds=list(NEW),
             thresholds=dict(max_rmse_fraction_of_null_and_shuffle=GATE_RATIO),rows=[])
    try:
        for seed in NEW:
            planned=pulse(seed)
            plus,minus=[exposure(arena,seed,sign*planned,native,weights) for sign in (1,-1)]
            row=dict(seed=seed,split='held',body=3,planned=planned.tolist(),plus=plus,minus=minus)
            out['rows'].append(row)
            _write(args.held,out)
            print('geometry held capture',seed,[r['event']['tick'] if r['event'] else None for r in (plus,minus)],flush=True)
    finally:
        arena.close()


def integrated_rates(pair):
    common=min(len(pair['plus']['actions']),len(pair['minus']['actions']))
    divergence=np.asarray(pair['plus']['actions'][:common])-np.asarray(pair['minus']['actions'][:common])
    return np.vstack((np.zeros((1,5)),_INC*np.cumsum(divergence,axis=0)))


def encoder_inputs(integral,t):
    return np.concatenate([integral[max(0,min(t-delay,len(integral)-1))] for delay in LAGS])


def encoder_fit(rows):
    x,y=[],[]
    for row in rows:
        history=integrated_rates(row)
        q=np.asarray(row['plus']['joints'][:len(history)-1])-np.asarray(row['minus']['joints'][:len(history)-1])
        cutoff=min(r['event']['tick'] for r in (row['plus'],row['minus']))
        for t in range(6,min(cutoff,len(q))):
            x.append(encoder_inputs(history,t))
            y.append(q[t])
    a,b=np.asarray(x),np.asarray(y)
    coeff=np.linalg.solve(a.T@a+.012*np.eye(a.shape[1]),a.T@b)
    return coeff,len(a)


def blade_delta(decoder,q,delta):
    return decoder.body(q+delta/2)[12:17]-decoder.body(q-delta/2)[12:17]


def geometry(row,decoder,clock,motor):
    t=expected_event_time(row['plus'],decoder,clock)
    initial=np.asarray(row['plus']['joints'][0])
    commands=integrated_rates(row)
    now=encoder_inputs(commands,t)@motor
    before=encoder_inputs(commands,t-3)@motor
    posture=blade_delta(decoder,initial,now)
    previous=blade_delta(decoder,initial,before)
    speed=20.*(posture[:2]-previous[:2])
    early=arrival_features(row['plus'],decoder)
    incoming=np.clip(-early[-1]*1.5+t*.13,.55,3.3)
    return np.column_stack((posture[3:]*incoming,speed,posture[:2]/.10))


def fit_geometry(features,target,pair_order):
    X=np.asarray(features).reshape(-1,3)
    Y=np.asarray(target)[pair_order].ravel()
    return np.linalg.solve(X.T@X+.04*np.eye(3),X.T@Y)


def compare(args):
    dataset=_read(args.donor)
    holdout=_read(args.held)
    fit_receipt=_read(FIT)
    clock_record=np.load(MODEL,allow_pickle=False)
    timing=clock_record['event_time']
    fixed=fit_receipt['fixed_elapsed_tick']
    decoder=wrist_sensor().decoder

    def eligible(rows):
        return [r for r in rows if all(r[k]['event'] for k in ('plus','minus'))
                and arrival_features(r['plus'],decoder) is not None]

    source=eligible([r for r in dataset['rows'] if r['split']=='source'])
    test=eligible(holdout['rows'])
    motor,count=encoder_fit(source)
    X=np.asarray([geometry(r,decoder,timing,motor) for r in source])
    Y=np.asarray([response(r) for r in source])
    trained=fit_geometry(X,Y,np.arange(len(X)))
    rng=np.random.default_rng(183923)
    altered=np.arange(len(X))
    source_body=np.array([r['body'] for r in source])
    for body in np.unique(source_body):
        index=np.flatnonzero(source_body==body)
        altered[index]=rng.permutation(index)
    shuffled=fit_geometry(X,Y,altered)
    np.savez_compressed(args.model,motor=motor,geometry=trained,shuffled=shuffled,clock=timing)
    if len(test)<6:
        _write(args.receipt,dict(passed=False,why='too few paired optical events',pairs=len(test)))
        return
    H=np.asarray([geometry(r,decoder,timing,motor) for r in test])
    T=np.asarray([response(r) for r in test])
    actual=np.array([all(r[k]['physical_only_audit'] for k in ('plus','minus')) for r in test])
    P=np.asarray([response(r,'physical_only_audit') for r,use in zip(test,actual,strict=True) if use])
    estimators={'geometry':H@trained,'geometry_shuffled':H@shuffled,'null':np.zeros_like(T)}
    for mode in ('first_ten','fixed','aligned','prospective'):
        Z=np.stack([pair_features(r,mode,expected_event_time(r['plus'],decoder,timing),fixed) for r in test])
        estimators[mode]=Z@clock_record[mode]
    metrics={k:dict(optical_rmse=float(np.sqrt(np.mean((v-T)**2))),
                    physical_rmse=float(np.sqrt(np.mean((v[actual]-P)**2))),
                    optical_right_sign=int(np.sum(np.sum(v*T,axis=1)>0)),predictions=v.tolist())
             for k,v in estimators.items()}
    gate=all(metrics['geometry'][space] <= GATE_RATIO*min(metrics[k][space] for k in ('null','geometry_shuffled','first_ten'))
             for space in ('optical_rmse','physical_rmse'))
    result=dict(schema='tt24-event-geometry-gate-1',passed=gate,max_rmse_fraction=GATE_RATIO,
                source_pairs=len(source),fresh_held_pairs=len(test),new_wrist_frames=sum(len(r[k]['actions'])
                    for r in holdout['rows'] for k in ('plus','minus')),encoder_samples=count,
                uses_actual_issued_commands_through_forecast_event_time=True,
                targets='visible post-reversal horizontal velocity difference',
                donor_sha256=_hash(args.donor),held_sha256=_hash(args.held),model_sha256=_hash(args.model),
                metrics=metrics,selected_seeds=[r['seed'] for r in test])
    _write(args.receipt,result)
    print({mode:{k:v for k,v in values.items() if k!='predictions'} for mode,values in metrics.items()},
          'gate passed',gate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('collect','fit'))
    parser.add_argument('--donor',type=Path,default=DATA)
    parser.add_argument('--held',type=Path,default=HELD)
    parser.add_argument('--model',type=Path,default=CHECKPOINT_OUT)
    parser.add_argument('--receipt',type=Path,default=EVIDENCE)
    args=parser.parse_args()
    (collect if args.phase=='collect' else compare)(args)
