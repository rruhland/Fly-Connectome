"""Replay the long-preparation proof and export only its camera/encoder/joint tape."""

import argparse
from pathlib import Path
from unittest.mock import patch

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import resources
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt40_low_examples import playback

OUT=Path('artifacts/tt40')


def verify(sources,output,proof_path):
    docs=[_read(v) for v in sources]
    if any(not d.get('complete') or d['ready']!=docs[0]['ready']
           or d['source_sha256']!=docs[0]['source_sha256'] for d in docs):
        raise ValueError('Unexpected or partial physical prefix experiment')
    result=dict(schema='tt40-long-low-observable-v1',source_sha256={s.name:_hash(s) for s in sources},
                role='actual visual and own-encoder sequences selected by a privileged physical coach',
                camera_hz=30,control_hz=60,groups=[])
    proof=dict(schema='tt40-long-low-physical-audit-v1',source_sha256=result['source_sha256'],rows=[])
    with centered(docs[0]['ready']):
        a=arena(protocol='random')
        cad=arena(protocol='random',detail='cad')
        view,rig,*_=resources(a)
        clear=CourtClearance(cad.model,articulated=True)
        try:
            launch=a.launch
            def start(*args,**kwargs):
                launch(*args,**kwargs)
                guarded(a,clear)
            with patch.object(a,'launch',start):
                for source,d in zip(sources,docs,strict=True):
                    for p in d['rows']:
                        for r in p['wins']:
                            if not (r['legal'] and r['answered']):
                                continue
                            contacts=[]
                            with contact_observer(a,contacts):
                                steps=playback(a,view,rig,dict(card=p['card'],result=r))
                            verify_clearance(a)
                            quality=a.clearance_result()
                            if len(contacts)!=sum(e['kind']=='contact' for e in a.events):
                                raise ValueError('Contact audit does not match source events')
                            result['groups'].append(dict(card=p['card'],leg=p['leg'],origin=source.name,
                              command_prefix_end_tick=p['fork_tick'],preparation_end_tick=p['late_tick'],
                              physical_legal_and_answered=True,events_sha256=digest_json(a.events),
                              frames=steps))
                            proof['rows'].append(dict(id=p['id'],leg=p['leg'],origin=source.name,
                                events_sha256=digest_json(a.events),contacts=contacts,clearance=quality))
                            print(p['id'],p['leg'],len(steps),quality,flush=True)
                            _write(output,result)
                            _write(proof_path,proof)
        finally:
            view.close(); a.close(); cad.close()
    result['complete']=proof['complete']=True
    _write(output,result)
    _write(proof_path,proof)


def restrict_to_clear(optic,proof_path,output):
    a,b=_read(optic),_read(proof_path)
    if not a.get('complete') or not b.get('complete') or a['source_sha256']!=b['source_sha256']:
        raise ValueError('Optical and CAD receipts do not match')
    qualified={(g['origin'],g['events_sha256']) for g in b['rows'] if g['clearance']['penetration_samples']==0}
    a['groups']=[g for g in a['groups'] if (g['origin'],g['events_sha256']) in qualified]
    a['cad_proof_sha256']=_hash(proof_path)
    _write(output,a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sources',type=Path,nargs='+')
    p.add_argument('--output',type=Path,default=OUT/'long-low-optical.json.gz')
    p.add_argument('--proof',type=Path,default=OUT/'long-low-audit.json')
    p.add_argument('--restrict-existing-optics',type=Path)
    args=p.parse_args()
    if args.restrict_existing_optics:
        restrict_to_clear(args.restrict_existing_optics,args.proof,args.output)
    else:
        verify(args.sources,args.output,args.proof)
