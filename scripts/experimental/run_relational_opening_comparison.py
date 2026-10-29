"""One no-refit opening-gate test against cached gated policy and40069."""
import argparse
from pathlib import Path
import time
from run_rally_option_memory import ROOT,MODEL,BANK,read,write,sha,parallel_jobs
from run_rally_repertoire_collection import select_starts,exclusions
from run_rally_raw_fresh_screen import paired_metrics
from stateful_pong_comparison import native_task,comparison
import run_practical_pong_v2 as base

PREVIOUS=ROOT/'pong-relational-search-20261029-r1'
DEST=ROOT/'pong-relational-opening-20261029-r1'
ENDPOINT=PREVIOUS/'models/final-policy.npz'
ENDPOINT_SHA='026ff8da73457238209708a03060bcaf10a5f94bbb83805d9f7a60c0644beedf'
ADAPTER='pong_relational_opening_adapter'


def qualify(stats):
    a,b=stats['baseline'],stats['candidate'];x,y=stats['baseline_more'],stats['candidate_more']
    return (b['own_returns']>a['own_returns'] and y['later_returns']>x['later_returns']
        and b['return_fraction']>=a['return_fraction'] and y['later_fraction']>=x['later_fraction']
        and b['first_incoming_returns']>=a['first_incoming_returns'])


def prepare():
    assert not DEST.exists() and sha(ENDPOINT)==ENDPOINT_SHA
    previous=read(PREVIOUS/'spec.json');result=read(PREVIOUS/'summary.json')
    assert result['source_frozen'] and result['assets_frozen'] and result['fresh']['qualified']
    assert previous['source']==base.source_signature()
    cached={}
    for i,e in enumerate(previous['bank']):
        p=PREVIOUS/'final32'/f'candidate_{i:02d}.json';r=read(p)
        assert r['complete'] and r['seed']==e['seed'] and r['model_sha256']==ENDPOINT_SHA and r['full_stateful_audit']['passed']
        cached[str(r['seed'])]=str(p)
    DEST.mkdir();(DEST/'train32').mkdir();(DEST/'fresh').mkdir()
    schedule=select_starts(1943906127,63329,exclusions(DEST),32);seeds=schedule.pop('TRAIN')
    schedule.update(probe=seeds[:32],unused_reserved=seeds[32:],purpose='reserved independent opening-gate qualification;never fitting')
    write(DEST/'fresh/schedules.json',schedule)
    names=('run_relational_opening_comparison.py','pong_relational_opening_adapter.py',
        'pong_relational_feedback.py','stateful_pong_comparison.py','pong_fixed_rbf.py','pong_fixed_pairwise.py',
        'pong_continuous_visual.py','run_rally_option_memory.py','pong_option_memory.py','rally_raw_capture.py',
        'return_metrics.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_practical_pong_v2.py')
    files=[Path(__file__).with_name(n) for n in names]
    assets=[ENDPOINT,MODEL/'rbf-model.npz',MODEL/'radial-cap.json',BANK,PREVIOUS/'spec.json',PREVIOUS/'summary.json']
    assets += [Path(e['receipt']) for e in previous['bank']]+[Path(p) for p in cached.values()]
    spec=dict(seconds=180,transition_camera_cap=32000,initial_camera_cap=32,workers=4,
        model=str(ENDPOINT),model_sha256=ENDPOINT_SHA,adapter=ADAPTER,
        seeds=[e['seed'] for e in previous['bank']],baseline_bank=previous['bank'],cached_gated=cached,
        gated_reference=dict(returns=130,first=29,later_returns=101,later_opportunities=110),
        change='ONLYremovebaselineuntilfirstactualhitgate;sameobserved-onlyrelation/weights;actualpast_hitremainsactual',
        fallback='unchangedcardinality/invalidliteral40069;previous_actual_actionupdatesafterexecutedcommands',
        qualification='ungatedvsgated:total/latercountincrease;overall/laterfractionnotworse;firstreturncountnolower;reporttails/timeouts/saturation',
        fresh_tag=63329,fresh_reserved_only=True,no_automatic_fresh_launch=True,
        no_refit=True,no_new_numeric_or_binding_choices=True,no_automatic_promotion=True,
        source=base.source_signature(),files={str(p):sha(p) for p in files},assets={str(p):sha(p) for p in assets})
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json')),flush=True)


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['assets'].items())
    began=time.monotonic();deadline=began+spec['seconds']
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=spec['seconds']))
    jobs=[(ADAPTER,spec['model'],str(DEST/'train32'),seed,i,deadline-1,True) for i,seed in enumerate(spec['seeds'])]
    rs=parallel_jobs(native_task,jobs,deadline,progress=DEST/'progress.json')
    complete=len(rs)==32 and all(r['complete'] for r in rs)
    summary=dict(status='TRAIN32_complete' if complete else 'TRAIN32_partial',complete=complete,
        new_transition_cameras=sum(r['transitions'] for r in rs),new_initial_cameras=sum(r['initial_cameras'] for r in rs))
    if complete:
        byseed={r['seed']:r for r in rs};candidate=[byseed[s] for s in spec['seeds']]
        gated=[read(spec['cached_gated'][str(s)]) for s in spec['seeds']]
        baseline=[read(e['receipt']) for e in spec['baseline_bank']]
        a=comparison(gated,candidate);b=comparison(baseline,candidate)
        summary.update(gated_to_ungated=a,baseline_to_ungated=b,paired_to_gated=paired_metrics(gated,candidate),
            qualified=qualify(a),fresh_reserved_only=True)
    summary.update(wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0,time.monotonic()-deadline),
        model_frozen=sha(ENDPOINT)==ENDPOINT_SHA,source_frozen=all(sha(p)==h for p,h in spec['files'].items()),
        assets_frozen=all(sha(p)==h for p,h in spec['assets'].items()),no_refit=True,no_automatic_promotion=True)
    assert summary['new_transition_cameras']<=spec['transition_camera_cap'] and summary['new_initial_cameras']<=spec['initial_camera_cap']
    write(DEST/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    prepare() if a.operation=='prepare' else run(a.approved_spec_sha256)
