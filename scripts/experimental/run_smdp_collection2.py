"""Second fixed feedback collection; reuse the unchanged validated capture protocol."""
import argparse
from pathlib import Path
import time
import numpy as np
import run_smdp_onpolicy_collection as protocol
from run_rally_option_memory import ROOT,MODEL,BANK,EXPECTED,read,write,sha,parallel_jobs
import run_practical_pong_v2 as base

DEST=ROOT/'pong-smdp-onpolicy-collection-20261029-r2'
ACTING=ROOT/'pong-smdp-closed-loop-refit-20261029-r1/final-forest.npz'
ACTING_SHA='e58a73f8ba9fe695aa7e1815fc268aaf43bef0759f8a65760fa58bcf8009f2cd'
PRIOR_MANIFEST=ACTING.parent/'data-manifest.json'
COMPARISON=ROOT/'pong-smdp-cycle-comparison-20261029-r1'


def admission_paths():
    paths=sorted((COMPARISON/'matched32/c0').glob('candidate_*.json'))+sorted((COMPARISON/'c1-screen8').glob('candidate_*.json'))+sorted((COMPARISON/'matched32/c1').glob('candidate_*.json'))
    assert len(paths)==56 and all(read(p)['complete'] for p in paths)
    return paths


def prepare():
    assert not DEST.exists() and sha(ACTING)==ACTING_SHA and sha(MODEL/'rbf-model.npz')==EXPECTED
    bank=read(BANK);prior=set(read(PRIOR_MANIFEST)['start_id_to_seed'].values());assert len(prior)==139;paths=admission_paths();admitted={read(p)['seed'] for p in paths};excluded=prior|admitted
    available={p:[e['seed'] for e in bank['bank'] if e['seed'] not in excluded and base.leftward(e['seed'])==(p=='left')] for p in ('left','right')}
    assert all(len(v)>=32 for v in available.values()),'insufficient newTRAINstarts; do not repeat or alter rule'
    rng=np.random.default_rng(2026102904);selected={p:rng.choice(v,32,replace=False).tolist() for p,v in available.items()};jobs=[]
    for i in range(32):
        for phase in ('left','right'):jobs.append(dict(index=len(jobs),seed=int(selected[phase][i]),phase=phase,action_seed=int(rng.integers(1,2**32,dtype=np.uint64))))
    assert len({j['seed'] for j in jobs})==64 and not {j['seed'] for j in jobs}&excluded
    DEST.mkdir();(DEST/'traces').mkdir();pinned={str(p):sha(p) for p in (ACTING,PRIOR_MANIFEST,BANK,MODEL/'rbf-model.npz',MODEL/'radial-cap.json')}
    for p in paths:
        r=read(p)
        for f in (p,Path(r['feature_file']),Path(r['structured_feature_file'])):pinned[str(f)]=sha(f)
    files={str(Path(protocol.__file__).with_name(n)):sha(Path(protocol.__file__).with_name(n)) for n in protocol.FILES};files[str(Path(__file__))]=sha(__file__)
    spec=dict(method='samefrozenmacrofeedbackcollection_cycle2',acting_model=str(ACTING),acting_sha256=ACTING_SHA,seconds=180,transition_cap=64000,per_start_cap=1000,workers=4,epsilon=.25,jobs=jobs,
        prior_training_starts=sorted(prior),admitted_comparison_starts=sorted(admitted),excluded_union=sorted(excluded),available_by_phase={p:len(v) for p,v in available.items()},
        admission_files=[str(p) for p in paths],admission_traces=56,admission_cameras=sum(read(p)['transitions'] for p in paths),cached_c0_eight_not_readmitted=True,
        capture_protocol=read(protocol.DEST/'spec.json')['capture'],baseline_Q_proof=read(protocol.DEST/'spec.json')['baseline_Q_proof'],
        delegate_configuration=dict(module='run_smdp_onpolicy_collection',DEST=str(DEST),C0=str(ACTING),C0_SHA=ACTING_SHA),
        source=base.source_signature(),files=files,pinned_assets=pinned,no_refit=True,no_new_algorithm_or_parameters=True,no_fresh_data=True,no_automatic_promotion=True)
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),excluded=len(excluded),available=spec['available_by_phase']),flush=True)


def task(job,deadline,spec):
    # Spawned workers receive this explicit configuration; no shared file is edited.
    # The capture implementation, epsilon rule, estimator and macro contract stay exact.
    saved=(protocol.DEST,protocol.C0,protocol.C0_SHA)
    protocol.DEST=DEST;protocol.C0=ACTING;protocol.C0_SHA=ACTING_SHA
    try:return protocol.task(job,deadline,spec)
    finally:protocol.DEST,protocol.C0,protocol.C0_SHA=saved


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists();assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    began=time.monotonic();deadline=began+180;write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=180,transition_cap=64000))
    results=parallel_jobs(task,[(j,deadline-1,spec) for j in spec['jobs']],deadline,progress=DEST/'progress.json');results.sort(key=lambda r:r['index'])
    summary=dict(status='all64complete' if len(results)==64 and all(r.get('complete',False) for r in results) else 'bounded_partial',traces=results,started=sum(r['initial_cameras']>0 for r in results),complete_games=sum(r.get('complete',False) for r in results),partial_games=sum(r['initial_cameras']>0 and not r.get('complete',False) for r in results),transition_cameras=sum(r['transitions'] for r in results),initial_cameras=sum(r['initial_cameras'] for r in results),actual_hits=sum(r['actual_hits'] for r in results),posthit_macro_boundaries=sum(r.get('posthit_macro_boundaries',0) for r in results),exploratory_boundaries=sum(r.get('exploratory_boundaries',0) for r in results),wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),source_frozen=all(sha(p)==h for p,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['pinned_assets'].items()),acting_sha256=ACTING_SHA,no_refit=True,no_fresh_data=True,no_automatic_promotion=True)
    assert summary['transition_cameras']<=64000 and summary['initial_cameras']<=64;write(DEST/'summary.json',summary);print({k:v for k,v in summary.items() if k!='traces'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    if a.operation=='prepare':prepare()
    else:run(a.approved_spec_sha256)
