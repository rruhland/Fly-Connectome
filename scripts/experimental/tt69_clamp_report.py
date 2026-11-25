"""Separate learned practice, exact held first offers, and whole points on the new body."""

from . import tt69_clamp_policy as policy
from . import tt69_clamp_school as school
from . import tt69_clamp_repair as late
from . import tt69_clamp_exact_lows as exact
from .tt4_transfer_trial import _hash, _read, _write


def reward(row):
    d=row['placements']['delivered']; scored=bool(d and d[0]['near_landed'])
    return scored,bool(scored and len(d)>1 and d[1]['near_bounce'])


def main():
    d=_read(school.OUT); bank=_read(policy.MODEL)
    report=dict(practice_sha256=_hash(school.OUT),model_sha256=_hash(policy.MODEL),
                captured=dict(rows=len(bank['rows']),releases=len({r['id'] for r in bank['rows']}),missing=bank['missing']),
                measured_school={})
    if bank['source_practice']!=_hash(school.OUT):raise ValueError('Changed fitted practice archive')
    for leg in (0,1):
        a=[r for r in d['rows'] if r['leg']==leg]; kept=[school.paid(r) for r in a if school.paid(r)]
        report['measured_school'][str(leg)]=dict(offers=len(a),actual_scored_and_answered=len(kept),
                                            interior=sum(r['contact']['radial_m']<=.058 for r in kept))
    if policy.OUT.exists():
        q=_read(policy.OUT); rows=q['rows']; report['first_tape_sha256']=_hash(policy.OUT)
        report['first']=dict(cases=len(rows),complete=q['complete'],lo_below_1m=sum(r['grade'] is not None and r['grade']<1.0 for r in rows),modes={})
        for k in ('registered','phase','transport'):
            a=[r['arms'][k] for r in rows]; paired=[(r['card']['id'],r['arms']['registered'],r['arms'][k]) for r in rows if r['arms'][k]['exact_first_ball']]
            report['first']['modes'][k]=dict(scored=sum(reward(v)[0] for v in a),answered=sum(reward(v)[1] for v in a),
                exact_incoming=len(paired),taught_motion=sum(bool(v['engaged_ticks']) for v in a),
                matched_gains=[i for i,b,v in paired if reward(v)[0] and not reward(b)[0]],
                matched_losses=[i for i,b,v in paired if reward(b)[0] and not reward(v)[0]],
                branch_clearance_penetrations=sum(v.get('clearance',{}).get('penetration_samples',0) for v in a))
        if q['complete']:
            chosen,grade=policy.earned_promotion();report['qualified']=dict(promotion=chosen,evidence=grade)
    if policy.GAMES.exists():
        games=_read(policy.GAMES); report['whole_sha256']=_hash(policy.GAMES)
        report['whole']=dict(results={k:[r['arms'][k]['score']['learner'] for r in games['rows']] for k in ('registered',games['chosen'])},
                             exact_offers=[dict(c,id=r['card']['id']) for r in games['rows'] for c in r['exact_offers'] if c['old']!=c['new']])
    if late.OUT.exists():
        d=_read(late.OUT); rows=d['rows']; notes=[v for r in rows for v in r['arms']['learned']['late']]
        report['late_fresh']=dict(tape=_hash(late.OUT),cases=len(rows),complete=d['complete'],matched=sum(r['matched'] for r in rows),
              scores={mode:sum(reward(r['arms'][mode])[0] for r in rows) for mode in ('registered','learned')},
              candidate_trials=sum(v['paths'] for v in notes),legal_shadow=sum(v['legal'] for v in notes),issued=sum(v['issued'] for v in notes))
    if exact.OUT.exists():
        d=_read(exact.OUT); rows=d['rows']; notes=[v for r in rows for v in r['camera_model']]
        report['later_sub1']=dict(tape=_hash(exact.OUT),cases=len(rows),complete=d['complete'],matched=sum(r['same_offer'] for r in rows),
              stock_scored=sum(r['stock']['landed'] for r in rows),new_scored=sum(r['learned']['landed'] for r in rows),
              naturally_low=sum(not r['case']['stock'] for r in rows),candidate_trials=sum(v['paths'] for v in notes),
              legal_shadow=sum(v['legal'] for v in notes),issued=sum(r['issued'] for r in rows),
              same_offer_gains=[r['case']['id'] for r in rows if r['same_offer'] and r['learned']['landed'] and not r['case']['stock']],
              same_offer_losses=[r['case']['id'] for r in rows if r['same_offer'] and not r['learned']['landed'] and r['case']['stock']])
    if exact.KNOWN.exists():
        tryout=_read(exact.KNOWN); rows=tryout['rows']
        report['known_optic_check']=dict(path=_hash(exact.KNOWN),cases=len(rows),available=sum(r['available'] for r in rows),
             physics_supported=sum(r.get('recognition',{}).get('exact',{}).get('legal',False) for r in rows),
             rgb_supported=sum(r.get('recognition',{}).get('rgb',{}).get('legal',False) for r in rows))
    _write(school.ROOT/'report.json',report); print(report,flush=True)


if __name__=='__main__':main()
