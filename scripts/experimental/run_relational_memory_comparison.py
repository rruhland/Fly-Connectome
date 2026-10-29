"""Complete the fixed opening-gate x public-memory comparison, no fitting."""
import argparse
from pathlib import Path
import time
import numpy as np
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from public_entity_pong_native import native_task
from stateful_pong_comparison import comparison
from run_rally_option_memory import ROOT,MODEL,BANK,read,write,sha,parallel_jobs
from run_rally_repertoire_collection import select_starts,exclusions
from run_rally_raw_fresh_screen import paired_metrics
import run_practical_pong_v2 as base

PREVIOUS=ROOT/'pong-relational-search-20261029-r1'
OPENING=ROOT/'pong-relational-opening-20261029-r1'
DEST=ROOT/'pong-relational-memory-20261029-r1'
ORIGINAL=PREVIOUS/'models/final-policy.npz'
ORIGINAL_SHA='026ff8da73457238209708a03060bcaf10a5f94bbb83805d9f7a60c0644beedf'


def prepare():
    assert not DEST.exists() and sha(ORIGINAL)==ORIGINAL_SHA
    prior=read(PREVIOUS/'spec.json');opened=read(OPENING/'summary.json')
    assert opened['complete'] and opened['model_frozen'] and opened['source_frozen'] and opened['assets_frozen']
    seeds=[e['seed'] for e in prior['bank']];cached={};original=RelationalPolicy.load(ORIGINAL)
    DEST.mkdir();(DEST/'models').mkdir();(DEST/'fresh').mkdir()
    for arm,gate,reference in (('gated',True,PREVIOUS/'final32'),('ungated',False,OPENING/'train32')):
        (DEST/arm).mkdir();path=DEST/'models'/f'{arm}.npz'
        PersistentRelationalPolicy(original,require_first_hit=gate).save(path)
        loaded=PersistentRelationalPolicy.load(path)
        assert loaded.original.choices==original.choices and np.array_equal(loaded.original.coefficients,original.coefficients)
        assert np.array_equal(loaded.original.scales,original.scales) and loaded.require_first_hit==gate
        cached[arm]={}
        for i,seed in enumerate(seeds):
            p=reference/f'candidate_{i:02d}.json';r=read(p)
            assert r['complete'] and r['seed']==seed and r['model_sha256']==ORIGINAL_SHA and r['full_stateful_audit']['passed']
            cached[arm][str(seed)]=str(p)
    schedule=select_starts(1943906127,63345,exclusions(DEST),32);fresh=schedule.pop('TRAIN')
    schedule.update(probe=fresh[:32],unused_reserved=fresh[32:],purpose='reserved memory-branch qualification;never fitting')
    write(DEST/'fresh/schedules.json',schedule)
    names=('run_relational_memory_comparison.py','public_entity_pong_native.py','pong_relational_persistent_adapter.py',
        'pong_relational_persistent_binding.py','pong_relational_feedback.py','stateful_pong_comparison.py',
        'pong_fixed_rbf.py','pong_fixed_pairwise.py','pong_continuous_visual.py','run_rally_option_memory.py',
        'pong_option_memory.py','rally_raw_capture.py','return_metrics.py','run_rally_raw_optimizer.py',
        'run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_practical_pong_v2.py')
    files=[Path(__file__).with_name(n) for n in names]
    assets=[ORIGINAL,MODEL/'rbf-model.npz',MODEL/'radial-cap.json',BANK,PREVIOUS/'spec.json',PREVIOUS/'summary.json',
        OPENING/'spec.json',OPENING/'summary.json',DEST/'models/gated.npz',DEST/'models/ungated.npz']
    assets += [Path(e['receipt']) for e in prior['bank']]+[Path(p) for paths in cached.values() for p in paths.values()]
    spec=dict(seconds=300,transition_camera_cap=64000,initial_camera_cap=64,workers=4,seeds=seeds,
        arms=['gated','ungated'],dispatch='interleavedgated/ungatedforeachfixedTRAINseed',
        model_paths={a:str(DEST/'models'/f'{a}.npz') for a in cached},
        model_hashes={a:sha(DEST/'models'/f'{a}.npz') for a in cached},original_path=str(ORIGINAL),original_sha256=ORIGINAL_SHA,
        cached_observed=cached,baseline_bank=prior['bank'],
        public_input='exactexistingencodedentities:id,normalized_position,observed_velocity,observed,velocity_known,observation_age_seconds,velocity_age_seconds,association_strength',
        input_archive='float64sixfields+int64publicids+float64ages/strength+explicitNonevalidity;Npreactioninputs;allpolicydecisionsreplayed',
        rule='reanchorlearnedranksoneveryexactnobservedscene;carrysamepublishedIDson<n;clearselectedIDabsent/invalid/>n;existingtrackertimeonly',
        comparison='eachmemoryarmvsitsmatchingcachedobservedarmand40069;alsomemorygatedvsungated;inspectabsolutefirst/laterfractions,tails/timeoutsbeforechoosingfreshbranch',
        fresh_tag=63345,fresh_reserved_only=True,no_automatic_fresh_launch=True,
        earlier_fresh63329_unused=True,no_refit=True,no_new_numeric_or_binding_choices=True,no_automatic_promotion=True,
        source=base.source_signature(),files={str(p):sha(p) for p in files},assets={str(p):sha(p) for p in assets})
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),model_hashes=spec['model_hashes']),flush=True)


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['assets'].items())
    began=time.monotonic();deadline=began+spec['seconds']
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=spec['seconds']))
    jobs=[(spec['model_paths'][arm],str(DEST/arm),seed,i,deadline-1) for i,seed in enumerate(spec['seeds']) for arm in spec['arms']]
    rs=parallel_jobs(native_task,jobs,deadline,progress=DEST/'progress.json')
    groups={a:{r['seed']:r for r in rs if r.get('model_sha256')==spec['model_hashes'][a]} for a in spec['arms']}
    complete=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values())
    summary=dict(status='TRAIN2x2_complete' if complete else 'TRAIN2x2_partial',complete=complete,
        new_transition_cameras=sum(r['transitions'] for r in rs),new_initial_cameras=sum(r['initial_cameras'] for r in rs),arms={})
    for arm,g in groups.items():
        done=len(g)==32 and all(r['complete'] for r in g.values());result=dict(complete=done)
        if done:
            current=[g[s] for s in spec['seeds']];reference=[read(spec['cached_observed'][arm][str(s)]) for s in spec['seeds']]
            baseline=[read(e['receipt']) for e in spec['baseline_bank']]
            result.update(observed_to_memory=comparison(reference,current),baseline_to_memory=comparison(baseline,current),
                paired_to_observed=paired_metrics(reference,current),
                mode_counts={mode:sum(r['full_stateful_audit']['mode_counts'].get(mode,0) for r in current)
                    for mode in sorted({m for r in current for m in r['full_stateful_audit']['mode_counts']})},
                persistent_binding_cameras=sum(r['full_stateful_audit']['persistent_binding_cameras'] for r in current))
        write(DEST/arm/'summary.json',result);summary['arms'][arm]=result
    if complete:
        a=[groups['gated'][s] for s in spec['seeds']];b=[groups['ungated'][s] for s in spec['seeds']]
        summary.update(memory_gated_to_ungated=comparison(a,b),paired_memory_gate=paired_metrics(a,b))
    summary.update(wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0,time.monotonic()-deadline),
        original_frozen=sha(ORIGINAL)==ORIGINAL_SHA,source_frozen=all(sha(p)==h for p,h in spec['files'].items()),
        assets_frozen=all(sha(p)==h for p,h in spec['assets'].items()),no_refit=True,no_automatic_promotion=True,
        fresh63345_reserved_only=True,fresh63329_unused=True)
    assert summary['new_transition_cameras']<=spec['transition_camera_cap'] and summary['new_initial_cameras']<=spec['initial_camera_cap']
    write(DEST/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    prepare() if a.operation=='prepare' else run(a.approved_spec_sha256)
