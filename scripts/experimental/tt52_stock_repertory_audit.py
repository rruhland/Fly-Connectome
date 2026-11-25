"""Check physical rewards, option attribution and strict clearance on complete stock-window points."""

import argparse
from pathlib import Path

from . import tt52_stock_repertory as rp
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance,guarded
from .tt40_long_low import same_trace


def run(path,out):
    d=rp._read(path);reg=rp._read(rp.plan_path)
    if not d.get('complete') or d['cards']!=reg['cards'][:len(d['rows'])]:raise ValueError('Unfinished or unrelated course')
    prior=rp.pf.prior;mpc=rp.mpc
    result=dict(source_sha256=rp._hash(path),n=len(d['cards']),same_native_plant=True,rows=[])
    boot=rp._read(prior.ARCHIVE)['ready']
    with mpc.centered(boot):
        a=mpc.arena(width=64,height=48);cad=mpc.arena(detail='cad',width=64,height=48)
        checking=CourtClearance(cad.model,articulated=True)
        try:
            for c,r in zip(d['cards'],d['rows'],strict=True):
                std,extra=(r['variants'][k] for k in ('prepared-stock','stock-options'))
                first=next((k for k,(u,v) in enumerate(zip(std['actions'],extra['actions'])) if u!=v),None)
                if first is None and len(std['actions'])!=len(extra['actions']):
                    raise ValueError('One variant stopped before a common physical rule ended it')
                t=(len(std['actions']) if first is None else first)*mpc.DT
                e0,e1=([e for e in z['events'] if e['time']<t-1e-7] for z in (std,extra))
                if e0!=e1:raise ValueError('Events differed before the first changed action')
                common=[]
                for x,y in zip(*(z['placements']['delivered'] for z in (std,extra))):
                    if not all(same_trace(x[k],y[k]) for k in ('far_contact','near_bounce')):break
                    common.append([bool(x['near_landed']),bool(y['near_landed'])])
                passed=[k for k in extra['repertory'] if k['accepted']]
                row=dict(id=c['id'],first_different_action=first,common_native_offer_scores=common,
                         fallbacks_considered=len(extra['repertory']),fallbacks_accepted=passed,
                         compared_to_marginal_stock=sum(k.get('stock_initial_support')==1 for k in extra['repertory']),branches={})
                cached=None
                for label,z in r['variants'].items():
                    if cached is not None and first is None:
                        if (digest_json(std['events'])!=digest_json(extra['events']) or
                            not same_trace(std['placements'],extra['placements'])):
                            raise ValueError('Identical controls did not report identical native rules')
                        clearance=cached
                    else:
                        a.launch(z['shot'],card=c);a.partner.reset(seed=c['seed']);guarded(a,checking)
                        for u in z['actions']:a.exchange(u,mpc.FAR_COMMAND)
                        real=a.point_placements()
                        if digest_json(a.events)!=digest_json(z['events']) or not same_trace(real,z['placements']):
                            raise ValueError('Independent native control tape diverged')
                        if any(p['target']!=list(rp.draw(c,j,'random',None)[0]) for j,p in enumerate(real['requested'])):
                            raise ValueError('Unexpected non-broad draw in replay')
                        clearance=a.clearance_result();cached=clearance
                    row['branches'][label]=dict(own=z['score']['learner'],opponent=z['score']['opponent'],
                                               clearance=clearance,events_sha256=digest_json(z['events']))
                result['rows'].append(row);rp._write(out,result)
                print(c['id'],first,len(common),len(passed),[v['own'] for v in row['branches'].values()],flush=True)
        finally:a.close();cad.close()
    result['complete']=True;rp._write(out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=rp.root/'live.json.gz')
    p.add_argument('--out',type=Path,default=rp.root/'live-audit.json')
    a=p.parse_args();run(a.source,a.out)
