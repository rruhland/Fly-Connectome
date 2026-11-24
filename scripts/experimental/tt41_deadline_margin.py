"""Date the latest observed successful onset of known low-ball physical movements."""

import argparse
from pathlib import Path

import numpy as np

from . import tt41_deadline as dd
from .tt41_deadline_warm import received, reference_tape
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent_audit import CourtClearance, contact_observer, verify_clearance
from .tt40_long_low import same_trace

# Finite sparse grid, not a monotonicity assumption or an optimum.
FAR_LEADS_MS=(50,100,150,200,250,325,400,500,650,800,1000)


def trial(a,old,c,leg,onset,goals,stride,low,high,clear=None):
    a.sample_clearance(False)
    if clear is None:
        dd.replay_prefix(a,old,c,onset)
    else:
        dd.guarded_prefix(a,clear,old,c,onset)
    actions=list(old['actions'][:onset])
    ax=a._arm_ids[1][2]
    q=a.data.ctrl[ax].copy()
    u=np.asarray(actions[-1])
    r=dd.bound(goals[None,onset+1:,:],q,u,stride,low,high)[0]
    tape=np.diff(np.vstack([q,r]),axis=0)/stride
    for command in np.clip(tape,-dd.RATE,dd.RATE).tolist():
        actions.append(command)
        _,_,end,cap,_=a.exchange(command,FAR_COMMAND)
        if end or cap or (a.rival_returns>=leg+2 and leg+1<len(a.point_placements()['delivered'])):
            break
    measured=received(a,leg)
    target=old['placements']['delivered'][leg]
    matched=(same_trace(measured.get('far'),target['far_contact'])
             and same_trace(measured.get('bounce'),target['near_bounce']))
    result=dict(leg=leg,fork_tick=onset,uncapped=measured,incoming_unchanged=matched,
                planned_peak_slew=float(np.max(abs(np.diff(np.vstack([u,tape]),axis=0)))),
                end_s=float(a.data.time),events_sha256=digest_json(a.events))
    if clear is not None:
        verify_clearance(a)
        result.update(clearance=a.clearance_result(),actions=actions,events=a.events,shot=old['shot'],
                      placements=a.point_placements())
    return result


def run(output):
    source=_read(dd.SOURCE)
    ids={c['id']:i for i,c in enumerate(source['cards'])}
    proofs={p:_read(dd.OUT/(p+'.json.gz')) for p in dd.PROOFS}
    result=dict(role='shot-specific, privileged, finite timing witnesses, not a policy or deadline necessity proof',
                source_sha256=_hash(dd.SOURCE),proof_hashes={p:_hash(dd.OUT/(p+'.json.gz')) for p in proofs},
                lead_grid_ms=FAR_LEADS_MS,rate_bound=dd.RATE,slew_bound=dd.SLEW,rows=[])
    _write(output,result)
    with centered(source['ready']):
        a=dd.arena(protocol='random')
        cad=dd.arena(protocol='random',detail='cad')
        clear=CourtClearance(cad.model,articulated=True)
        step=a.control_dt
        try:
            for id in dd.ORDER:
                idx=ids[id]; c=source['cards'][idx]; old=source['rows'][idx]
                data=[(p,r) for p,d in proofs.items() for r in d['rows'] if r['id']==id and r['wins']]
                latest=[]
                for p,proof in data:
                    k=proof['leg']; win=proof['wins'][0]
                    far=old['placements']['delivered'][k]['far_contact']['time']
                    g=reference_tape(a,win,c)
                    row=dict(id=id,card=c,leg=k,index=idx,source=p,original_proof_onset=proof['fork_tick'],
                             actual_far_contact_s=far,face=win['chosen']['variant']['face'],trials=[],win=None)
                    result['rows'].append(row)
                    for lead in FAR_LEADS_MS:
                        onset=int(np.floor((far-lead/1000.)/step))
                        if onset<proof['fork_tick'] or (latest and onset<=max(latest)):
                            break
                        x=trial(a,old,c,k,onset,*g)
                        x['tested_lead_s']=far-onset*step
                        row['trials'].append(x)
                        print(id,p,round(x['tested_lead_s']*1000),x['uncapped']['contact'],x['uncapped']['legal'],
                              x['uncapped']['answered'],x['incoming_unchanged'],flush=True)
                        if x['incoming_unchanged'] and x['uncapped']['answered'] and x['planned_peak_slew']<=dd.SLEW+1e-9:
                            contacts=[]
                            with contact_observer(a,contacts):
                                y=trial(a,old,c,k,onset,*g,clear=clear)
                            assert y['events_sha256']==x['events_sha256']
                            x['clearance']=y['clearance']
                            if y['clearance']['penetration_samples']==0:
                                row['win']=dict(y,contacts=contacts)
                                latest.append(onset)
                                break
                        _write(output,result)
                    _write(output,result)
        finally:
            a.close()
            cad.close()
    result['complete']=True
    _write(output,result)


def audit_native(stock,out):
    """Check pre-bounce evaluation against already-realized clean physical suffixes."""
    src=_read(dd.SOURCE)
    base=_read(stock)
    proofs={p:_read(dd.OUT/(p+'.json.gz')) for p in dd.PROOFS}
    if not base.get('complete') or base['source_sha256']!=_hash(dd.SOURCE):
        raise ValueError('Complete matching stock receipt required')
    result=dict(role='Private comparator positive controls, after earlier proved physical preparation',
                stock_sha256=_hash(stock),rows=[])
    with centered(src['ready']):
        a=dd.arena(protocol='random')
        engine=dd.Rollout(a,1)
        try:
            for row in base['rows']:
                old=src['rows'][row['index']]
                c=row['card']
                stamp=lambda: np.r_[a.ball,a.velocity,a.spin]
                for name,doc in proofs.items():
                    for p in doc['rows']:
                        if p['id']!=row['id'] or not p['wins']:
                            continue
                        win=p['wins'][0]
                        refs,*_=reference_tape(a,win,c)
                        for gate in row['stages']:
                            tick=gate['fork_tick']
                            if tick is None:
                                continue
                            dd.replay_prefix(a,old,c,tick)
                            stock_state=stamp()
                            dd.replay_prefix(a,win,c,tick)
                            prepared_state=stamp()
                            error=float(np.max(abs(stock_state-prepared_state)))
                            if error>1e-8:
                                raise ValueError('Prepared flight not identical at tested gate')
                            f=np.zeros((1,9))
                            a.clearance_margin=.0006
                            try:
                                _,v=engine.trajectories(refs[None,tick+1:],dd.AIM,.70,feedback=f)
                            finally:
                                a.clearance_margin=0.
                            result['rows'].append(dict(id=c['id'],proof=name,gate=gate['name'],tick=tick,
                                                       ball_state_max_abs_difference=error,touched=bool(f[0,1]),
                                                       legal=bool(np.isfinite(v[0,0])),feedback=f[0].tolist()))
        finally:
            a.close()
    result['complete']=True
    _write(out,result)


if __name__=='__main__':
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--output',type=Path,default=dd.DEST/'deadline-onset.json.gz')
    arg.add_argument('--audit-native',type=Path)
    args=arg.parse_args()
    if args.audit_native:
        audit_native(args.audit_native,args.output)
    else:
        run(args.output)
