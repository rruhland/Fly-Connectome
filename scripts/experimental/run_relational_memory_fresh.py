"""One frozen candidate-only fresh cohort, using the full public-input audit."""
import argparse
from pathlib import Path
import time
from pong_relational_persistent_binding import PersistentRelationalPolicy
from public_entity_pong_native import native_task
from return_metrics import aggregate
from run_rally_raw_optimizer import more
from run_rally_option_memory import ROOT,MODEL,read,write,sha,parallel_jobs
import run_practical_pong_v2 as base

PREVIOUS=ROOT/'pong-relational-memory-20261029-r1'
DEST=ROOT/'pong-relational-memory-fresh-20261029-r1'
ENDPOINT=PREVIOUS/'models/ungated.npz'
EXPECTED='75530410f2c985a3037e9aa264447b8d172e5e9297a9e556bbb74d2c511dc7cc'


def prepare():
    assert not DEST.exists() and sha(ENDPOINT)==EXPECTED
    result=read(PREVIOUS/'summary.json');prior=read(PREVIOUS/'spec.json')
    assert result['complete'] and result['source_frozen'] and result['assets_frozen'] and result['original_frozen']
    assert not PersistentRelationalPolicy.load(ENDPOINT).require_first_hit
    schedule=read(PREVIOUS/'fresh/schedules.json');seeds=schedule['probe'];assert len(set(seeds))==32
    DEST.mkdir();(DEST/'episodes').mkdir()
    names=('run_relational_memory_fresh.py','public_entity_pong_native.py','exact_visual_transport_copy.py',
        'pong_relational_persistent_adapter.py','pong_relational_persistent_binding.py','pong_relational_feedback.py',
        'stateful_pong_comparison.py','pong_fixed_rbf.py','pong_fixed_pairwise.py','pong_continuous_visual.py',
        'run_rally_option_memory.py','pong_option_memory.py','rally_raw_capture.py','return_metrics.py',
        'run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_practical_pong_v2.py')
    files=[Path(__file__).with_name(n) for n in names]
    assets=[ENDPOINT,MODEL/'rbf-model.npz',MODEL/'radial-cap.json',PREVIOUS/'spec.json',PREVIOUS/'summary.json',PREVIOUS/'fresh/schedules.json']
    spec=dict(seconds=180,transition_camera_cap=32000,initial_camera_cap=32,workers=4,seeds=seeds,fresh_tag=63345,
        model=str(ENDPOINT),model_sha256=EXPECTED,source=base.source_signature(),
        files={str(p):sha(p) for p in files},assets={str(p):sha(p) for p in assets},
        purpose='Absolute reliability of one frozen ungated public-memory endpoint on all 32 pre-reserved fresh starts.',
        point_targets=dict(first_return_fraction=.95,later_return_fraction=.95),
        scope='20-second games only; report exposure, tails and timeouts; this does not substitute for long-horizon independent replication.',
        candidate_only=True,no_baseline_rerun=True,no_refit=True,no_model_selection=True,no_automatic_promotion=True)
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json')),flush=True)


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['assets'].items())
    began=time.monotonic();deadline=began+spec['seconds']
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=spec['seconds']))
    jobs=[(spec['model'],str(DEST/'episodes'),seed,i,deadline-1) for i,seed in enumerate(spec['seeds'])]
    rs=parallel_jobs(native_task,jobs,deadline,progress=DEST/'progress.json')
    complete=len(rs)==32 and all(r['complete'] for r in rs)
    summary=dict(status='fresh_complete' if complete else 'fresh_partial',complete=complete,
        new_transition_cameras=sum(r['transitions'] for r in rs),new_initial_cameras=sum(r['initial_cameras'] for r in rs),
        candidate_only=True,model_sha256=EXPECTED)
    if complete:
        candidate=aggregate([r['annotation'] for r in rs]);tail=more(rs)
        summary.update(candidate=candidate,candidate_more=tail,
            all_policy_replay_rows=sum(r['full_stateful_audit']['rows'] for r in rs),
            meets_point_targets=candidate['first_incoming_return_fraction']>=.95 and tail['later_fraction'] is not None and tail['later_fraction']>=.95)
    summary.update(wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0,time.monotonic()-deadline),
        source_frozen=all(sha(p)==h for p,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['assets'].items()),
        no_refit=True,no_automatic_promotion=True)
    assert summary['new_transition_cameras']<=32000 and summary['new_initial_cameras']<=32
    write(DEST/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    prepare() if a.operation=='prepare' else run(a.approved_spec_sha256)
