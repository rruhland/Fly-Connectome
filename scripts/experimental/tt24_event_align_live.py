"""Use a learned visual response to propose bounded wrist-game corrections."""

import argparse
from pathlib import Path

import numpy as np

from .tt24_bounce_ident import _ellipse, _INC
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import Native
from .tt24_event_align import LEAD, OUT, NewWide, arrival_features, expected_event_time, wrist_sensor
from .tt24_event_align_geometry import CHECKPOINT_OUT, LAGS, EVIDENCE
from .tt24_launch_context import draw
from .tt4_transfer_trial import _hash, _read, _write

OUTCOME=OUT/'event-align-live.json.gz'
DEVELOPMENT=tuple(range(371301,371305))
QUALIFICATION=tuple(range(371401,371417))
MAX_STEPS=240
GAINS=(.45,.80,1.15)
MODES=('frozen','learned','shuffled','level_only','hold')
LIMIT=.31


def requested_pose_response(motor,horizon):
    motor_blocks=motor.reshape(len(LAGS),5,5)
    lead=np.maximum(horizon-np.array(LAGS),0)
    return _INC*np.einsum('b,bij->ij',lead,motor_blocks)


def scene_basis(decoder,q):
    return np.stack([(decoder.body(q+s)[12:17]-decoder.body(q-s)[12:17])/.008
                     for s in np.eye(5)*.004],axis=1)


def correction(q,decoder,motor,head,incoming,drift,horizon):
    pose=scene_basis(decoder,q)
    current=requested_pose_response(motor,horizon)@pose.T
    prior=requested_pose_response(motor,horizon-3)@pose.T
    effect=head[0]*incoming*current[:,3:]+head[1]*20.*(current[:,:2]-prior[:,:2])+head[2]*current[:,:2]/.1
    # Keep first-flight action bounded and avoid large blade translations during approach.
    hessian=effect@effect.T+.12*np.eye(5)+6.*current[:,:3]@current[:,:3].T
    return np.linalg.solve(hessian,-effect@drift)


def play(arena,native,weights,learned,seed,mode,gain):
    arena.max_steps=MAX_STEPS
    eye=wrist_sensor()
    obs=arena.reset(seed=seed,split='held',regime='changed')
    prefix=dict(positions=[],joints=[])
    expected=None
    stop=0
    adjustments=0
    arrivals=[]
    issued=np.zeros(5)
    actions=[]
    while True:
        q=np.asarray(obs['joints'])
        optical=_ellipse(obs['rgb'])
        scene=eye.decoder.body(q)
        state=eye.read(optical,obs['joints'])
        if len(actions)<LEAD:
            point=eye.decoder.ball(optical,scene)
            prefix['positions'].append(None if point is None else point.tolist())
            prefix['joints'].append(q.tolist())
        elif len(actions)==LEAD and mode in ('learned','shuffled','level_only'):
            if arrival_features(prefix,eye.decoder) is not None:
                expected=expected_event_time(prefix,eye.decoder,learned['clock'])
                # An early optical forecast bounds this first-flight intervention.
                stop=expected+1
        base=np.zeros(5) if mode=='hold' else draw(native,state,weights,issued)
        d=np.zeros(5)
        if expected is not None and len(actions)<stop and optical is not None:
            gap=state[2]-state[8]
            if .015<gap<.29 and state[5]<-.20:
                h=int(np.clip(expected-len(actions)+1,2,7))
                incoming=float(np.clip(-state[5]+.12*h,.55,3.3))
                # Use the same observed drift proxy and task objective across all selectors.
                drift=np.clip(.55*state[3:5]+2.*incoming*state[9:11]+1.4*state[:2],-.85,.85)
                head=learned['geometry'] if mode=='learned' else learned['shuffled'] if mode=='shuffled' else np.array((2.,0.,0.))
                d=np.clip(gain*correction(q,eye.decoder,learned['motor'],head,incoming,drift,h),-LIMIT,LIMIT)
                if np.any(np.abs(d)>.005):
                    adjustments+=1
                    arrivals.append([len(actions),h,d.tolist()])
        action=np.clip(base+d,-.78,.78)
        if mode!='hold':
            issued+=_INC*(action-base)
        actions.append(action.tolist())
        obs,_,done=arena.step(action)
        if done:
            break
    judge=arena.judge_metrics()
    arena.reset(seed=seed,split='held',regime='changed',render=False)
    for action in actions:
        arena.step(action,render=False)
    if arena.judge_metrics()!=judge:
        raise ValueError('physical game diverges from action replay')
    return dict(seed=seed,contact_time_forecast_tick=expected,changed_ticks=adjustments,
                changes=arrivals,judge=judge,actions=actions)


def payoff(row):
    """The development selector scores only measured scalar rebound rewards."""
    total=row['judge']['rebounds']
    return min(5,total)+int(total>=2)


def evaluate(args):
    claim=_read(EVIDENCE)
    if not claim['passed']:
        raise ValueError('held visual-response model failed the prospective criterion')
    native=Native()
    with np.load(CHECKPOINT,allow_pickle=False) as original:
        weights=np.ascontiguousarray(original['weights'])
    with np.load(CHECKPOINT_OUT,allow_pickle=False) as model:
        learned={k:v.copy() for k,v in model.items()}
    arena=NewWide()
    doc=dict(schema='tt24-event-response-reward-only-first-encounter-1',
             model_sha256=_hash(CHECKPOINT_OUT),physical_gate_sha256=_hash(EVIDENCE),
             frozen_actor_sha256=_hash(CHECKPOINT),training_starts=list(DEVELOPMENT),
             held_starts=list(QUALIFICATION),horizon=MAX_STEPS,gains=list(GAINS),
             effort_rate_limit=LIMIT,exact_judge_replay=True,development={},gain_choice={},held={})
    try:
        baseline=[play(arena,native,weights,learned,seed,'frozen',0.) for seed in DEVELOPMENT]
        baseline_score=sum(map(payoff,baseline))
        doc['development']['frozen']=baseline
        _write(args.out,doc)
        for mode in ('learned','shuffled','level_only'):
            rank={0.:baseline_score}
            for gain in GAINS:
                rows=[play(arena,native,weights,learned,seed,mode,gain) for seed in DEVELOPMENT]
                rank[gain]=sum(map(payoff,rows))
                doc['development'].setdefault(mode,{})[str(gain)]=rows
                print('scalar-only development',mode,gain,rank[gain],flush=True)
                _write(args.out,doc)
            doc['gain_choice'][mode]=max(rank,key=rank.get)
        _write(args.out,doc)
        for mode in MODES:
            gain=doc['gain_choice'].get(mode,0.)
            rows=[]
            for seed in QUALIFICATION:
                executable='frozen' if mode!='hold' and gain==0. else mode
                row=play(arena,native,weights,learned,seed,executable,gain)
                rows.append(row)
                doc['held'][mode]=rows
                print('fresh wide',mode,seed,row['judge']['rebounds'],row['changed_ticks'],flush=True)
                _write(args.out,doc)
    finally:
        arena.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUTCOME)
    evaluate(parser.parse_args())
