"""Prepare release-disjoint rosters and benchmark the live far paddle on a fixed RGB/encoder learner."""

import argparse
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np

from . import tt33_joint as joint, tt37_opponent as opponent
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import card, opening_candidates
from .tt31_ready import centered


ARTIFACTS = Path(__file__).resolve().parents[2] / 'artifacts'
OUT = ARTIFACTS / 'tt37'
BASE = ARTIFACTS / 'tt33' / 'joint-raw.json.gz'
PLAN = OUT / 'opponent-plan.json'
PROTOCOL = 'tt37-live-opponent-1'


def prepare(feeds, output):
    prior = ARTIFACTS/'tt36'
    practice_path = prior/'course-practice.json.gz'
    replies_path = prior/'course-replies.json.gz'
    practice_old = _read(practice_path)['cards']
    continuation = _read(replies_path)['cards']
    excluded = set(_read(BASE)['training_release_ids'])
    sources = [practice_path,replies_path,prior/'course-new.json.gz',prior/'course-confirm.json.gz',
               ARTIFACTS/'tt34'/'body-novel-plan.json.gz',ARTIFACTS/'tt35'/'confirmation-plan.json.gz',
               ARTIFACTS/'tt32'/'fresh-plan.json.gz',ARTIFACTS/'tt31'/'course-plan.json.gz',
               ARTIFACTS/'tt31'/'full-plan.json.gz',ARTIFACTS/'tt31'/'confirm-plan.json.gz']
    for path in sources:
        doc = _read(path)
        for key in ('cards','held','staged'):
            excluded.update(c['id'] for c in doc.get(key,[]))
    selected = opening_candidates(measured_launches(feeds))
    available = [s['id'] for s in selected if s['id'] not in excluded][:180]
    if len(available)<24:
        raise ValueError('Not enough additional measured releases to freeze the held roster')
    selected = np.random.default_rng(371004).choice(available,size=24,replace=False).tolist()
    held = [card(s,'full',i,371601+7*i)|dict(profile='reach',reply_seed=376601+17*i)
            for i,s in enumerate(selected)]
    train = [c|dict(profile='reach',opening='anchor',reply_seed=373101+17*i)
             for i,c in enumerate(practice_old)]
    repeat = [c|dict(profile='reach',opening='anchor',reply_seed=374301+17*i,
                     key=f'reply-{i}-{c["id"]}') for i,c in enumerate(continuation)]
    if set(selected) & {c['id'] for c in train+repeat} or {c['id'] for c in train}&{c['id'] for c in repeat}:
        raise ValueError('The practice and held releases overlap')
    document = dict(schema=PROTOCOL,feed_sha256=_hash(feeds),baseline_sha256=_hash(BASE),
                    source_sha256={p.relative_to(ARTIFACTS).as_posix():_hash(p) for p in sources},
                    profile_fields=opponent.PLACEMENT_FIELDS, profiles=opponent.PROFILES,
                    diagnostic_cap=opponent.DIAGNOSTIC_CAP, long_cap=opponent.LONG_CAP,
                    max_control_steps=opponent.MAX_CONTROL_STEPS,
                    practice=train, held_openings=held, held_replies=repeat,
                    selection='Draw 24 unique first releases from the first 180 compatibility-ranked, archived-release-excluded measured feeds. No outcome or actual delivery screening. Continuations keep four useful TT36 opening placements under four new deterministic seeds each.',
                    exclusions=sorted(excluded),baseline_opening_side=1)
    _write(output/'opponent-plan.json',document)
    print('roster',len(train),len(held),len(repeat),flush=True)


def physical_summary(row):
    target_reports = row['placements']['requested']
    far = [e for e in row['events'] if e['kind']=='contact' and e['side']==-1]
    hands = [e for e in row['events'] if e['kind']=='contact' and e['side']==1]
    verified = row['score']['shots']
    previous = [s for s in verified if s['side']==1]
    if row.get('stop')=='target' and verified and verified[-1]['side']==1:
        # The benchmark stopped before the opponent had another chance to return it.
        previous.pop()
    following = []
    for prior in previous:
        future_far = next((e for e in far if e['time']>prior['hit']),None)
        before_next = min((s['hit'] for s in previous if s['hit']>prior['hit']),default=float('inf'))
        delivered = next((s for s in verified if s['side']==-1 and prior['hit']<s['hit']<before_next),None)
        following.append(dict(near_landing=prior['landing_xyz'],near_hit_s=prior['hit'],near_landing_s=prior['landing'],
                             incoming_mps=prior['incoming_mps'],outgoing_mps=prior['outgoing_mps'],
                             contacted=bool(future_far is not None and future_far['time']<before_next),
                             answered=delivered is not None))
    finish = next((e for e in reversed(row['events']) if e['kind'] in ('point','timeout','let')),None)
    return dict(requests=len(target_reports), delivered=len(row['placements']['delivered']),
                far_touches=len(far),near_touches=len(hands), own=row['score']['learner'],
                returns=row['score']['returns'], later_replies=following,
                reason=None if finish is None else finish.get('reason'),
                referee_winner=None if finish is None else finish.get('winner'))


def mark(a, row, cap=None):
    row.update(simulation_end_s=float(a.data.time),launches=a.launches,
               far_plan=a.partner.reports, far_motor=dict(speed=a.partner.max_speed,
               angular_speed=a.partner.max_angular_speed,force=a.partner.max_force))
    row['stop'] = 'target' if cap is not None and row['score']['returns']>=cap else (
        'referee' if a.terminated else 'timeout' if a.truncated else 'other')
    row['physical']=physical_summary(row)
    requests = row['placements']['requested']
    if [p['leg'] for p in requests] != list(range(len(requests))) or any(
            p['reception']>=q['reception'] for p,q in zip(requests,requests[1:])):
        raise ValueError('The opponent requested twice on one live reception')
    steps = len(row['actions'])
    if not (steps-1)*a.control_dt-1e-6 <= row['simulation_end_s'] <= steps*a.control_dt+1e-6 or any(
            p['time']>q['time'] for p,q in zip(row['events'],row['events'][1:])):
        raise ValueError('Simulation reset while the point was active')


def play_raw(a, renderer, rig, raw, stride, lo, hi, shot, c, cap):
    """Freeze the nominal TT33 actor, also preserving one-tick aborted releases."""
    a.launch(shot,card=c)
    a.partner.reset(seed=c['seed'])
    actor=joint.JointPolicy(raw,stride,lo,hi)
    camera=joint.Vision(rig,a.control_dt,raw=actor.raw)
    actions=[]
    while True:
        tick=len(actions)
        q=a.joints
        x=camera.update(tick,renderer.capture() if tick%2==0 else None,q)
        action=actor.command(x,q)
        _, _, done, truncated, _ = a.exchange(action,joint.FAR_COMMAND)
        actions.append(action.tolist())
        if done or truncated or a.legal_returns+a.rival_returns-1>=cap:
            break
    return dict(shot=shot,opponent_seed=c['seed'],score=audit(a.events,opening_side=1),
                events=a.events,placements=a.point_placements(),actions=actions,motor_audit=a.motor_audit())


def run(feeds, output, protocol, *, long=False, limit=None):
    started = process_time()
    plan_path=output/'opponent-plan.json'
    plan=_read(plan_path)
    if _hash(feeds)!=plan['feed_sha256']:
        raise ValueError('Measured feeds differ from the frozen manifest')
    raw=_read(BASE)
    cap=plan['long_cap'] if long else plan['diagnostic_cap']
    pools={k:plan[k] for k in ('held_openings','held_replies')}
    shots=measured_launches(feeds)
    path=output/f'opponent-baseline-{protocol}{"-long" if long else ""}.json.gz'
    result=dict(protocol=protocol, cap=cap, ready=raw['ready'],opening_side=1, rows=[], cards=[], pools=[],
                plan_sha256=_hash(plan_path),pupil_sha256=_hash(BASE))
    with centered(raw['ready']):
        a=opponent.arena(protocol=protocol)
        renderer,rig,stride,lo,hi=joint.resources(a)
        try:
            for kind, cards in pools.items():
                for c in cards[:limit]:
                    count = a.launches
                    row=play_raw(a,renderer,rig,raw,stride,lo,hi,shots[c['id']],c,cap)
                    if a.launches!=count+1:
                        raise ValueError('Point did not start from a single launch')
                    mark(a,row,cap)
                    result['cards'].append(c)
                    result['rows'].append(row)
                    result['pools'].append(kind)
                    print(protocol,kind,c['id'],row['physical']['returns'],row['physical']['own'],row['physical']['delivered'],
                          row['physical']['referee_winner'],row['physical']['reason'],flush=True)
                    _write(path,result)
        finally:
            renderer.close()
            a.close()
    result['cpu_s']=process_time()-started
    _write(path,result)


def distribution(data, protocol):
    rows = data['rows']
    total = Counter()
    requested, later_requested, delivered = [], [], []
    replies_to_legal = []
    last_endings = Counter()
    for c,r in zip(data['cards'],rows,strict=True):
        stats = r['physical']
        replies = r['placements']['requested']
        landings = r['placements']['delivered']
        total.update({k:stats[k] for k in ('requests','delivered','far_touches','near_touches','own','returns')})
        total['openings_without_delivery'] += not landings
        total['later_opportunities'] += len(stats['later_replies'])
        total['later_returned'] += sum(s['answered'] for s in stats['later_replies'])
        total['refresh_replans'] += sum(p['stage']=='bounce_refresh' for p in r['far_plan'])
        last_endings[stats['reason'] or r['stop']] += 1
        requested.extend(p['target'] for p in replies)
        later_requested.extend(p['target'] for p in replies[1:])
        replies_to_legal.extend(stats['later_replies'])
        delivered.extend((p['target'][0],p['receive_line']['xyz'][0],p['receive_line']['velocity'][1],p['receive_line']['xyz'][2])
                         for p in landings if p['receive_line'] is not None)
        if protocol=='random':
            native_near = [e for e in r['events'] if e['kind']=='contact' and e['side']==1]
            for p in replies:
                if p['draw']=='random' and list(opponent.draw(c,p['leg'],protocol,None)[0])!=p['target']:
                    raise ValueError('A later placement differed from the independent seed')
                if p['leg'] and (not 1<=p['reception']<=len(native_near)
                                 or native_near[p['reception']-1]['time']>p['time_s']):
                    raise ValueError('A later opponent request had no new preceding physical learner hit')
    x = np.asarray(delivered).reshape(-1,4)
    req = np.asarray(requested).reshape(-1,4)
    unusual = [t for t in replies_to_legal if abs(t['near_landing'][0])>.30 or t['near_landing'][1]<.50
               or not 2.8<=t['outgoing_mps']<=6.0]
    unusual_missed = [r for r in unusual if not r['answered']]
    return dict(protocol=protocol,episodes=len(rows),**dict(total),maximum_legal=max((r['score']['returns'] for r in rows),default=0),
                requested_ranges=np.vstack([req.min(axis=0),req.max(axis=0)]).T.tolist() if len(req) else [],
                received_ranges=np.vstack([x.min(axis=0),x.max(axis=0)]).T.tolist() if len(x) else [],
                x_error_median=float(np.median(abs(x[:,0]-x[:,1]))) if len(x) else None,
                lanes=dict(Counter('left' if v < -.075 else 'right' if v > .075 else 'middle' for v in x[:,1])),
                all_requested=len(req), unique_requested=len(set(map(tuple,req))),
                later_unique_requested=len(set(map(tuple,later_requested))), terminal_reasons=dict(last_endings),
                unusual_legal_near=len(unusual), unusual_answered=len(unusual)-len(unusual_missed),
                unusual_misses=unusual_missed, cpu_s=data.get('cpu_s'))


def summarize(output):
    sets={}
    for p in sorted(output.glob('opponent-baseline-*.json.gz')):
        data=_read(p)
        if 'cpu_s' not in data:
            raise ValueError(f'Run still incomplete: {p.name}')
        sets[data['protocol']]=data
    report={}
    for mode,data in sets.items():
        report[mode]={'all':distribution(data,mode)}
        for pool in set(data['pools']):
            idx=[j for j,key in enumerate(data['pools']) if key==pool]
            subset=dict(rows=[data['rows'][j] for j in idx],cards=[data['cards'][j] for j in idx])
            report[mode][pool]=distribution(subset,mode)
    if 'random' in sets and 'tt36' in sets:
        fields=('target','far_contact','near_bounce','receive_line')
        anchored=0
        for k,a,b in zip(sets['random']['pools'],sets['random']['rows'],sets['tt36']['rows'],strict=True):
            if k!='held_replies':
                continue
            left=a['placements']['delivered']
            right=b['placements']['delivered']
            if left and right and all(digest_json(left[0][key])==digest_json(right[0][key]) for key in fields):
                anchored+=1
        report['identical_anchored_first_balls']=anchored
    _write(output/'opponent-baseline-summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('plan','baseline','summary'))
    p.add_argument('feeds',type=Path)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--protocol',choices=opponent.PROTOCOLS,default='random')
    p.add_argument('--long',action='store_true')
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=='plan': prepare(args.feeds,args.output)
    elif args.phase=='baseline': run(args.feeds,args.output,args.protocol,long=args.long,limit=args.limit)
    else:
        summarize(args.output)
