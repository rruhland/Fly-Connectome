"""Evaluator-only anatomy of a fixed earlier-wing neutral comparison."""
import argparse
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import _read,_write,_hash


def report(source,output):
    d=_read(source);rows=[]
    for row in d['rows']:
        result=dict(stream=row['random_reply_stream'],matched_incoming=row.get('matched_incoming'),variants={})
        crossing=next(iter(row['variants'].values()))['actual_offer']['receive_line']
        if crossing is not None:
            x,y,z=crossing['xyz'];vx,vy,vz=crossing['velocity'];dt=(-1.53-y)/(.98*vy)
            result['evaluator_deep_receipt']=[x+dt*vx,z+dt*vz-4.905*dt*dt]
        for name,value in row['variants'].items():
            seen_out=False;window=None
            for observation in value['observations']:
                if observation['tick']<=4003:continue
                x=observation['x']
                if x is None:continue
                if x[4]>.9:seen_out=True
                if seen_out and x[4]<-.9 and x[1]<-.49:
                    window=observation;break
            trace=value.get('body_trace',[])
            stop=window['tick'] if window else float('inf')
            before=[v for v in trace if v['tick']<=stop]
            result['variants'][name]=dict(legal=value['legal'],answered=value['answered'],completed=value['completed'],
                stock_window=window,prep_trace_rows=len(before),
                far_evidence=[v for v in before if v['action']=='far optical evidence'],
                accepted_strokes=[v for v in value.get('opts',[]) if v['accepted']],
                clearance=value['clearance'])
        q={name:v['stock_window']['q'] for name,v in result['variants'].items() if v['stock_window']}
        if 'cold' in q and 'warm' in q:result['warm_cold_window_max_joint_difference']=float(np.max(abs(np.asarray(q['warm'])-q['cold'])))
        if 'cold' in row['variants'] and 'warm' in row['variants']:
            cold=row['variants']['cold']['controls'];warm=row['variants']['warm']['controls']
            start=row['variants']['warm']['observations'][0]['tick']
            result['first_warm_cold_action_difference']=next((start+i for i,(a,b) in enumerate(zip(cold,warm)) if a!=b),None)
            result['all_warm_cold_actions_identical']=cold==warm
            result['first_acquired_body_command']=next((s['tick'] for s in row['variants']['warm'].get('body_trace',[])
                if s['action']=='toward acquired body'),None)
        rows.append(result)
    totals={name:sum(r['variants'][name]['legal'] for r in rows) for name in ('stock','cold','warm')}
    pairs={name:dict(gains=sum(r['variants']['warm']['legal'] and not r['variants'][name]['legal'] for r in rows),
        losses=sum(not r['variants']['warm']['legal'] and r['variants'][name]['legal'] for r in rows)) for name in ('stock','cold')}
    pending=d.get('in_progress')
    censored=None if pending is None else dict(stream=pending['random_reply_stream'],
        variants={name:dict(legal=v['legal'],completed=v['completed'],answered=v['answered']) for name,v in pending['variants'].items()})
    _write(output,dict(source_sha256=_hash(source),complete=d['complete'] and d['status'].startswith('completed'),
        status=d['status'],complete_paired_rows=len(rows),requested_cases=d.get('roster_count',5),pending=censored,totals=totals,
        warm_paired=pairs,rows=rows,scope='Posthoc evaluator-only receipt projection and actual joint states. These native receipt summaries are never policy inputs.'))
    print(totals,pairs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();report(args.source,args.output)
