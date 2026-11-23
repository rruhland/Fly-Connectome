"""Learn paired motor effects using a reversal inferred from wrist RGB."""

import argparse
from pathlib import Path

import numpy as np

from .tt24_bounce_ident import _ellipse, _INC
from .tt24_broad_donors import FamilyArena, NOVEL
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import DESIGNS, DonorBounce, Native
from .tt24_launch_context import draw
from .tt24_posture import WidePosture
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_wrist_egomotion import MAP
from .tt24_wrist_egomotion_trial import TransferDecoder
from .tt4_transfer_trial import _hash, _read, _write

OUT=Path('artifacts/tt24')
DATA=OUT/'event-align-source.json.gz'
FIT=OUT/'event-align-fit.json.gz'
MODEL=OUT/'event-align-model.npz'
CAP=42
LEAD=5
TRAIN=tuple(range(371001,371025))
HELD=tuple(range(371101,371117))
PERTURB=.22
BINS=((-13,-8),(-8,-4),(-4,0))


class WideDonor(DonorBounce):
    sample_drop=staticmethod(WidePosture.sample_drop)


class NewWide(FamilyArena):
    sample_drop=staticmethod(WidePosture.sample_drop)

    def __init__(self,view='wrist'):
        super().__init__(NOVEL,view=view)


def wrist_sensor():
    eye=load_public(FUSED_OUTPUT)
    with np.load(MAP,allow_pickle=False) as mapdata:
        eye.decoder=TransferDecoder(eye.decoder,eye.decoder,mapdata['source_theta'],mapdata['mount_links'])
    return eye


def pulse(seed):
    rng=np.random.default_rng(seed^72183749)
    timing=rng.integers(3,7,size=12)
    command=rng.choice((-1.,1.),(len(timing),5))
    return np.repeat(command,timing,axis=0)[:CAP]*PERTURB


def line_fit(points, ticks):
    x=np.asarray(ticks,dtype=float)
    v=np.asarray(points)
    keep=np.isfinite(v).all(axis=1)
    if keep.sum()<max(3,len(x)-2):
        return None
    x=x[keep]
    v=v[keep]
    x=x-x.mean()
    velocity=x@v/(x@x)*60.
    residual=v-v.mean(axis=0)-x[:,None]*velocity/60.
    return velocity,float(np.sqrt(np.mean(residual**2)))


def find_event(points):
    """Find the first supported change from incoming to outgoing vertical motion."""
    p=np.asarray(points,dtype=float)
    for at in range(5,len(p)-6):
        pre=line_fit(p[at-5:at],np.arange(at-5,at))
        after=line_fit(p[at+1:at+7],np.arange(at+1,at+7))
        if pre is None or after is None:
            continue
        before,noise=pre
        outbound,late_noise=after
        if noise>.022 or late_noise>.020:
            continue
        if not (-4.<before[2]<-.38 and .32<outbound[2]<4.1 and outbound[2]-before[2]>.93):
            continue
        # Two lines should meet within a ball-scale neighborhood.
        prev=p[at-3:at].copy()
        next=p[at+1:at+4].copy()
        if not np.all(np.isfinite(prev)) or not np.all(np.isfinite(next)):
            continue
        meeting=np.linalg.norm(prev.mean(axis=0)+before*2/60.-next.mean(axis=0)+outbound*2/60.)
        if meeting>.065:
            continue
        return dict(tick=at,incoming=before.tolist(),outgoing=outbound.tolist(),noise=noise+late_noise)
    return None


def exposure(arena,seed,command,native,weights):
    arena.max_steps=CAP
    obs=arena.reset(seed=seed,split='held',regime='changed')
    eye=wrist_sensor()
    issued=np.zeros(5)
    commands,perturb,points,joints = [],[],[],[]
    while True:
        tick=len(commands)
        q=obs['joints']
        mark=_ellipse(obs['rgb'])
        measured=eye.decoder.ball(mark,eye.decoder.body(q))
        points.append(np.full(3,np.nan) if measured is None else measured)
        joints.append(np.asarray(q).copy())
        state=eye.read(mark,q)
        base=draw(native,state,weights,issued)
        # The random input has no release, contact, or score labels.
        d=command[tick] if LEAD<=tick<33 else np.zeros(5)
        action=np.clip(base+d,-.78,.78)
        issued+=_INC*(action-base)
        commands.append(action.tolist())
        perturb.append((action-base).tolist())
        obs,_,done=arena.step(action)
        if done:
            break
    event=find_event(points)
    judge=arena.judge_metrics()
    obs=arena.reset(seed=seed,split='held',regime='changed',render=False)
    # Isolated physical scoring diagnostic, populated only on command replay.
    truth=[]
    for u in commands:
        truth.append(arena._ball_position.copy())
        arena.step(u,render=False)
    if arena.judge_metrics()!=judge:
        raise ValueError('replayed single encounter diverged')
    physical=None
    if judge['rebound_events']:
        at=int(np.ceil(judge['rebound_events'][0]['contact_time']*60))
        quality=line_fit(truth[at+1:at+7],np.arange(at+1,min(len(truth),at+7)))
        if quality is not None:
            physical=dict(tick=at,outgoing=quality[0].tolist())
    return dict(actions=commands,randomized_rates=perturb,joints=np.asarray(joints).tolist(),
                positions=[None if not np.all(np.isfinite(p)) else p.tolist() for p in points],
                event=event,judge=judge,physical_only_audit=physical)


def collect(args):
    native=Native()
    with np.load(CHECKPOINT,allow_pickle=False) as release:
        weights=np.ascontiguousarray(release['weights'])
    receipt=dict(schema='tt24-event-aligned-interventions-1',source_clips=list(TRAIN),held_clips=list(HELD),
                 held_body='novel family with expanded independent starting angles',physical_replay=True,
                 donor_motor_sha256=_hash(CHECKPOINT),calibration_sha256=_hash(FUSED_OUTPUT),
                 wrist_relocation_sha256=_hash(MAP),rows=[])
    arenas=[WideDonor(d,view='wrist',width=192,height=144,max_steps=CAP) for d in DESIGNS]+[NewWide()]
    try:
        for group,seeds in [('source',TRAIN),('held',HELD)]:
            for index,seed in enumerate(seeds):
                body=index%3 if group=='source' else 3
                randomized=pulse(seed)
                branches=[exposure(arenas[body],seed,d*randomized,native,weights) for d in (1,-1)]
                receipt['rows'].append(dict(seed=seed,split=group,body=body,
                                            planned=randomized.tolist(),plus=branches[0],minus=branches[1]))
                print('event effects',group,seed,[v['event']['tick'] if v['event'] else None for v in branches],
                      [v['judge']['rebounds'] for v in branches],flush=True)
                _write(args.data,receipt)
    finally:
        for arena in arenas:
            arena.close()


def summarize_action(rates,tick):
    observed=np.asarray(rates)
    blocks=[]
    for start,stop in BINS:
        ticks=np.arange(tick+start,tick+stop)
        valid=(ticks>=0)&(ticks<len(observed))
        blocks.append(observed[ticks[valid]].sum(axis=0)/len(ticks))
    return np.concatenate(blocks)


def pair_features(row,align,forecast,fixed):
    pair=[row['plus'],row['minus']]
    ticks=[r['event']['tick'] for r in pair]
    if align=='fixed':
        ticks=[fixed,fixed]
    if align=='first_ten':
        ticks=[10,10]
    if align=='prospective':
        ticks=[forecast,forecast]
    return summarize_action(pair[0]['randomized_rates'],ticks[0])-summarize_action(pair[1]['randomized_rates'],ticks[1])


def arrival_features(branch,decoder):
    """Early optical state before any signed intervention."""
    early=branch['positions'][:LEAD]
    if len(early)!=LEAD or sum(x is not None for x in early)<LEAD-1:
        return None
    p=np.asarray([x if x is not None else [np.nan]*3 for x in early])
    measured=line_fit(p,np.arange(LEAD))
    if measured is None:
        return None
    center=decoder.body(np.asarray(branch['joints'][0]))[12:15]
    relative=np.nanmedian(p,axis=0)-center
    return np.r_[1.,relative/.15,measured[0][2]/1.5]


def expected_event_time(branch,decoder,model):
    features=arrival_features(branch,decoder)
    return int(np.clip(np.rint(features@model),9,34))


def response(pair,key='event'):
    return np.asarray(pair['plus'][key]['outgoing'][:2])-np.asarray(pair['minus'][key]['outgoing'][:2])


def fit_model(args):
    data=_read(args.data)
    decoder=wrist_sensor().decoder
    paired=[row for row in data['rows'] if all(row[label]['event'] is not None for label in ('plus','minus'))]
    paired=[row for row in paired if arrival_features(row['plus'],decoder) is not None]
    train=[r for r in paired if r['split']=='source']
    test=[r for r in paired if r['split']=='held']
    if len(train)<10 or len(test)<6:
        _write(args.receipt,dict(train_pairs=len(train),held_pairs=len(test),valid=False))
        print('insufficient optical support',len(train),len(test))
        return
    Y=np.stack([response(row) for row in train])
    T=np.stack([response(row) for row in test])
    physical=np.array([all(r[k]['physical_only_audit'] is not None for k in ('plus','minus')) for r in test])
    truth=np.stack([response(r,'physical_only_audit') for r,use in zip(test,physical,strict=True) if use])
    design=np.stack([arrival_features(r['plus'],decoder) for r in train])
    event_times=np.array([np.mean([r['plus']['event']['tick'],r['minus']['event']['tick']]) for r in train])
    predictor=np.linalg.solve(design.T@design+.12*np.diag([.01,1.,1.,1.,1.]),design.T@event_times)
    fixed=int(np.rint(np.median(event_times)))
    inferred={r['seed']:expected_event_time(r['plus'],decoder,predictor) for r in paired}
    timing_error=[abs(inferred[r['seed']]-r[side]['event']['tick']) for r in test for side in ('plus','minus')]
    metrics={}
    weights={'event_time':predictor}
    receipts=[]
    train_rng=np.random.default_rng(8163402)
    groups=np.array([r['body'] for r in train])
    shuffled=np.arange(len(train))
    for body in np.unique(groups):
        index=np.flatnonzero(groups==body)
        shuffled[index]=train_rng.permutation(index)

    def report(prediction):
        dot=np.sum(prediction*T,axis=1)
        return dict(optical_rmse=float(np.sqrt(np.mean((prediction-T)**2))),
                    physical_rmse=float(np.sqrt(np.mean((prediction[physical]-truth)**2))),
                    correctly_oriented_optical=int(np.sum(dot>0)),
                    predictions=prediction.tolist())

    candidates=(.10,.32,1.,3.2,10.)
    penalties={}
    for mode in ('first_ten','fixed','aligned','shuffled','prospective'):
        align='aligned' if mode=='shuffled' else mode
        A=np.stack([pair_features(row,align,inferred[row['seed']],fixed) for row in train])
        Z=np.stack([pair_features(row,align,inferred[row['seed']],fixed) for row in test])
        target=Y[shuffled] if mode=='shuffled' else Y
        identity=np.eye(A.shape[1])
        losses=[]
        for ridge in candidates:
            held=[]
            for body in np.unique(groups):
                keep=groups!=body
                terms,values=A[keep],target[keep]
                model=np.linalg.solve(terms.T@terms+ridge*identity,terms.T@values)
                held.append(np.square(A[~keep]@model-target[~keep]).ravel())
            losses.append(np.concatenate(held).mean())
        ridge=candidates[int(np.argmin(losses))]
        penalties[mode]=ridge
        weights[mode]=np.linalg.solve(A.T@A+ridge*identity,A.T@target)
        metrics[mode]=report(Z@weights[mode])
    metrics['null']=report(np.zeros_like(T))
    for row,t,phys in zip(test,T,physical,strict=True):
        receipts.append(dict(seed=row['seed'],optical=t.tolist(),physical=response(row,'physical_only_audit').tolist() if phys else None))
    np.savez_compressed(args.model,**weights)
    coverage={}
    for split in ('source','held'):
        rows=[r for r in data['rows'] if r['split']==split]
        selected=[r for r in paired if r['split']==split]
        audit=[]
        for row in rows:
            for side in ('plus','minus'):
                trial=row[side]
                optic,phys=trial['event'],trial['physical_only_audit']
                if optic and phys:
                    audit.append(abs(optic['tick']-phys['tick']))
        coverage[split]=dict(recorded_pairs=len(rows),eligible_pairs=len(selected),true_contact_time_audits=len(audit),
                             median_absolute_event_ticks=float(np.median(audit)) if audit else None,
                             wrist_frames=sum(len(r[k]['actions']) for r in rows for k in ('plus','minus')))
    report_=dict(valid=True,schema='tt24-event-action-effect-fit-1',fit_data_sha256=_hash(args.data),
                 checkpoint_sha256=_hash(args.model),coverage=coverage,selected_ridge=penalties,fixed_elapsed_tick=fixed,
                 same_randomized_action_inputs=True,prospective_event_tick_mae=float(np.mean(timing_error)),
                 held_rows=receipts,metrics=metrics)
    _write(args.receipt,report_)
    print({k:{n:v for n,v in d.items() if n!='predictions'} for k,d in metrics.items()},coverage,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('collect','fit'))
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--receipt',type=Path,default=FIT)
    parser.add_argument('--model',type=Path,default=MODEL)
    args=parser.parse_args()
    (collect if args.phase=='collect' else fit_model)(args)
