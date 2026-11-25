"""Summarize measured struck balls separately from the finite static/velocity fit."""

import argparse

from . import tt64_deep_receive as deep
from . import tt64_deep_limits as limits
from .tt4_transfer_trial import _hash, _read, _write


def run(verify):
    scans, branches=map(_read,(deep.OUT,limits.OUT))
    if not scans['complete'] or not branches['complete'] or branches['protocol_sha256']!=_hash(limits.PLAN):
        raise ValueError('Expected the frozen complete deep tests')
    stats=dict(scan=_hash(deep.OUT),actual=_hash(limits.OUT),rows=[],independent_cad={})
    winners=[]
    for s,r in zip(scans['rows'],branches['rows'],strict=True):
        if s['case']!=r['case']:raise ValueError('Mismatched physical deep case')
        tape=[(q,x) for q in r['attempts'] if q['matched_prefix'] for x in q['physical']]
        reached=[(q,x) for q,x in tape if x['impact'] is not None and x['departure'] is not None]
        interior=[(q,x) for q,x in reached if x['impact']['radial_m']<=deep.REAL_RADIUS and abs(x['impact']['face_alignment'])>=.95]
        def describe(q,x):
            k=x['placements']['delivered'][s['case']['leg']]['near_contact']
            return dict(kind=q['kind'],face=q['fit']['face'],depth_m=k['location'][1],z_m=k['location'][2],
                        radius_m=x['impact']['radial_m'],alignment=x['impact']['face_alignment'],
                        departure_mps=x['departure'],fitted_joint_rad=q['fit']['joint'],
                        local_proxy_needed_mps=q['fit']['needed_normal_mps'],
                        local_proxy_ceiling_mps=q['fit']['ceiling_normal_mps'],
                        optimistic_forward_mps=q['recipe']['predicted_forward_ceiling'])
        stats['rows'].append(dict(case=r['case'],tested_depths=s['scanned_depths'],supported=s['supported_geometries'],
                  prep_s=(s['receiving']-s['fork'])*deep.mpc.DT,branches=len(tape),contact=len(reached),interior=len(interior),
                  alignments_at_least_95=sum(abs(x['impact']['face_alignment'])>=.95 for q,x in reached),
                  scored=sum(x['legal'] for q,x in tape),answered=sum(x['answered'] for q,x in tape),
                  matched=sum(q['matched_prefix'] for q in r['attempts']),
                  forward_separations=sum(x['departure'][1]>0. for q,x in reached),
                  suffix_cad_samples=sum(x['clearance']['samples'] for q,x in tape),
                  suffix_cad_penetrations=sum(x['clearance']['penetration_samples'] for q,x in tape),
                  clearances=[x['clearance']['nearest']['signed_m'] for q,x in tape if x['clearance']['nearest']],
                  closest_to_forward=None if not reached else describe(*max(reached,key=lambda l:l[1]['departure'][1])),
                  strict_interior=[describe(q,x) for q,x in interior]))
        if interior:
            winners.append((f'{s["case"]["id"]}_interior',s,interior[0][1]))
        elif reached:
            winners.append((f'{s["case"]["id"]}_closest_to_forward',s,max(reached,key=lambda l:l[1]['departure'][1])[1]))
    if verify:
        with deep.enabled():
            for tag,s,arm in winners:
                with deep.mpc.centered(s['ready']):
                    guide=deep.mpc.arena(width=64,height=48,detail='cad')
                    try:
                        proof=deep.full_verify(s['ready'],s['card'],arm,deep.CourtClearance(guide.model,articulated=True))
                        if proof['penetration_samples']:
                            raise ValueError('The exact full negative branch was not physically clear')
                        stats['independent_cad'][tag]=proof
                    finally:
                        guide.close()
    _write(deep.ROOT/'report.json',stats)
    print(stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--verify',action='store_true'); a=p.parse_args();run(a.verify)
