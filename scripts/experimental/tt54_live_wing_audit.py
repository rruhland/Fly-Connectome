"""Independently replay physical wing evidence and recheck court and clearance."""

import argparse
from pathlib import Path

from . import tt54_live_wing as wing
from .tt37_opponent_audit import CourtClearance,guarded
from .tt40_long_low import same_trace
from .tt40_low_feasibility import replay_prefix

rp=wing.rp;mpc=wing.mpc;prior=rp.pf.prior


def exact(path,output):
    d=rp._read(path);old=rp._read(prior.ARCHIVE);case,original=prior.terminal(old)
    p=rp._read(prior.PLAN);start=p['anchor'];end=p['keep_old_commands_through']
    if not d['complete'] or d['case']!=case['id'] or d['stream']!=case['card']['reply_seed']:
        raise ValueError('Wrong exact native context')
    out=dict(source_sha256=rp._hash(path),first_changed_tick=d['first_changed_tick'],rows=[])
    with mpc.centered(old['ready']):
        a=mpc.arena(width=64,height=48);cad=mpc.arena(width=64,height=48,detail='cad')
        check=CourtClearance(cad.model,articulated=True)
        try:
            out['previous_native_prefix_sha256']=replay_prefix(a,original,case['card'],start)
            saved=prior.State(a)
            for r in d['rows']:
                if r['commands'][:end-start+1]!=original['actions'][start:end+1]:
                    raise ValueError('Protected earlier stroke action changed')
                saved.restore(a);guarded(a,check)
                for u in r['commands']:a.exchange(u,mpc.FAR_COMMAND)
                o=a.point_placements()['delivered'][case['leg']]
                if prior.digest_json(a.events)!=r['event_sha256'] or not same_trace(o,r['offer']):
                    raise ValueError('Exact paired command replay diverged')
                out['rows'].append(dict(mode=r['mode'],legal=o['near_landed'],same_original_offer=True,
                                        opponent_answered=a.rival_returns>=case['leg']+2,clearance=a.clearance_result(),
                                        last_event_sha256=r['event_sha256']))
        finally:a.close();cad.close()
    out['complete']=True;rp._write(output,out)


def whole(path,output):
    d=rp._read(path);old=rp._read(prior.ARCHIVE)
    if not d['complete']:raise ValueError('Need complete prospective branch results')
    out=dict(source_sha256=rp._hash(path),rows=[])
    with mpc.centered(old['ready']):
        a=mpc.arena(width=64,height=48);cad=mpc.arena(width=64,height=48,detail='cad')
        check=CourtClearance(cad.model,articulated=True)
        try:
            for c,s in zip(d['cards'],d['rows'],strict=True):
                a0,b0=s['variants'].values()
                first=next((i for i,(u,v) in enumerate(zip(a0['actions'],b0['actions'])) if u!=v),None)
                if first!=s['first_changed_tick']:raise ValueError('Different first action boundary')
                common=[]
                for x,y in zip(a0['placements']['delivered'],b0['placements']['delivered']):
                    if not all(same_trace(x[k],y[k]) for k in ('far_contact','near_bounce')):break
                    common.append([x['near_landed'],y['near_landed']])
                row=dict(id=c['id'],first_changed_tick=first,common_native_offers=common,runs={})
                tally=None
                for mode,r in s['variants'].items():
                    if tally is None or a0['actions']!=b0['actions']:
                        a.launch(r['shot'],card=c);a.partner.reset(seed=c['seed']);guarded(a,check)
                        for u in r['actions']:a.exchange(u,mpc.FAR_COMMAND)
                        if not same_trace(a.point_placements(),r['placements']) or prior.digest_json(a.events)!=prior.digest_json(r['events']):
                            raise ValueError('Whole native tape diverged')
                        if any(req['target']!=list(rp.draw(c,k,'random',None)[0]) for k,req in enumerate(r['placements']['requested'])):
                            raise ValueError('Changed official draws')
                        tally=a.clearance_result()
                    elif prior.digest_json(a0['events'])!=prior.digest_json(b0['events']):
                        raise ValueError('Identical controls reported different native events')
                    row['runs'][mode]=dict(own=r['score']['learner'],opponent=r['score']['opponent'],clearance=tally)
                out['rows'].append(row);rp._write(output,out)
                print(c['id'],first,len(common),[(m,v['own']) for m,v in row['runs'].items()],flush=True)
        finally:a.close();cad.close()
    out['complete']=True;rp._write(output,out)


def record(path,output):
    from . import tt54_live_wing_record as receipt
    data=rp._read(path);bond,old,_=receipt.facts()
    if data['plan_sha256']!=rp._hash(receipt.PLAN) or not data.get('complete'):
        raise ValueError('Missing recorded full-point recovery')
    card=bond['card'];tape=data['run']
    with mpc.centered(old['ready']):
        a=mpc.arena(width=64,height=48);c=mpc.arena(width=64,height=48,detail='cad')
        try:
            a.launch(tape['shot'],card=card);a.partner.reset(seed=card['seed'])
            guarded(a,CourtClearance(c.model,articulated=True))
            for u in tape['actions']:a.exchange(u,mpc.FAR_COMMAND)
            got=a.point_placements()
            score=mpc.audit(a.events,opening_side=1)
            if prior.digest_json(a.events)!=prior.digest_json(tape['events']) or not same_trace(got,tape['placements']):
                raise ValueError('Recovered complete native tape did not independently reproduce')
            if score['learner']!=data['score'] or any(x['target']!=list(rp.draw(card,k,'random',None)[0])
                                                     for k,x in enumerate(got['requested'])):
                raise ValueError('Recorded score or randomized draw changed')
            out=dict(source_sha256=rp._hash(path),independent_native_score=True,own=score['learner'],opponent=score['opponent'],
                     physical_events_sha256=prior.digest_json(a.events),clearance=a.clearance_result(),
                     first_changed_tick=data['first_changed_tick'],prefix_exact=data['event_prefix_exact'],
                     prefix_within_roundoff=data['event_prefix_within_roundoff'])
            rp._write(output,out)
            print(out,flush=True)
        finally:a.close();c.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('exact','whole','record'))
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args();{'exact':exact,'whole':whole,'record':record}[a.phase](a.source,a.output)
