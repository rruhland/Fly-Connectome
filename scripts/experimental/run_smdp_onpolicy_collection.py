"""One fixed64TRAIN collection on frozen c0's induced states, no learning."""
import argparse
from pathlib import Path
import time
import numpy as np
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from pong_smdp_adaptive_forest import AdaptiveForest
from pong_smdp_exploration import ExplorationAdapter,Explorer,CONTRACT,EPSILON
from pong_continuous_visual import canonical_scene,save_scenes
from pong_fixed_rbf import load_readout
from fly_connectome.control.scaling import policy_features
from stateful_pong_comparison import StatefulEvaluator
from run_rally_option_memory import ROOT,MODEL,BANK,EXPECTED,read,write,sha,seed_rngs,parallel_jobs
import run_practical_pong_v2 as base

DEST=ROOT/'pong-smdp-onpolicy-collection-20261029-r1'
C0=ROOT/'pong-smdp-adaptive-forest-20261029-r1/final-forest.npz'
C0_SHA='ca99b349d1ea26a2036214e37d04e58a2d39a59cf26b44d5aeed7e18484df295'
OLD_MANIFEST=ROOT/'pong-smdp-adaptive-forest-20261029-r1/data-manifest.json'
C0_SCREEN=ROOT/'pong-smdp-native-20261029-r1/screen8'
FILES=('run_smdp_onpolicy_collection.py','pong_smdp_exploration.py','pong_smdp_adaptive_forest.py','stateful_pong_comparison.py','pong_continuous_visual.py','pong_fixed_rbf.py','pong_fixed_pairwise.py','run_rally_option_memory.py','pong_option_memory.py','rally_raw_capture.py','return_metrics.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_practical_pong_v2.py')


def prepare():
    assert not DEST.exists() and sha(C0)==C0_SHA;bank=read(BANK);old=set(read(OLD_MANIFEST)['start_id_to_seed'].values());screen_paths=sorted(C0_SCREEN.glob('candidate_*.json'));assert len(old)==75 and len(screen_paths)==8
    c0_seeds={read(p)['seed'] for p in screen_paths};excluded=old|c0_seeds;strata={phase:[e['seed'] for e in bank['bank'] if e['seed'] not in excluded and base.leftward(e['seed'])==(phase=='left')] for phase in ('left','right')};assert all(len(v)>=32 for v in strata.values()),'insufficient newTRAINstarts; no outcome-dependent replacement'
    rng=np.random.default_rng(2026102903);selected={phase:rng.choice(values,32,replace=False).tolist() for phase,values in strata.items()};jobs=[]
    for i in range(32):
        for phase in ('left','right'):jobs.append(dict(index=len(jobs),seed=int(selected[phase][i]),phase=phase,action_seed=int(rng.integers(1,2**32,dtype=np.uint64))))
    assert len({j['seed'] for j in jobs})==64 and not {j['seed'] for j in jobs}&excluded
    DEST.mkdir();(DEST/'traces').mkdir();pinned={str(p):sha(p) for p in (C0,OLD_MANIFEST,BANK,MODEL/'rbf-model.npz',MODEL/'radial-cap.json',*screen_paths)}
    spec=dict(contract=CONTRACT,c0=str(C0),c0_sha256=C0_SHA,seconds=180,transition_cap=64000,per_start_cap=1000,starts=64,workers=4,epsilon=EPSILON,jobs=jobs,
        selection='fixeduniform32perphase fromadmittedTRAIN256 excludingold75ANDall8c0screenstarts',old_training_seeds=sorted(old),c0_screen_seeds=sorted(c0_seeds),excluded_union_seeds=sorted(excluded),available_new_by_phase={p:len(v) for p,v in strata.items()},
        exploration='baselineuntilactualfirsthit; thereafter .75c0greedy+.25uniform4 options atmacroboundaries only; .8125greedy,.0625others;16holds orbaseline1; stoponactualhit/naturalend',
        capture='exactfloat64N+1 sixfieldscenes andexactbaselineQ/actions; N actualaction/hit/dt/ticks/terminal/truncated/ordinal/options/behaviorp; NaNp onlyonheldframes; no rawCSR',
        baseline_Q_proof='livefrozen40069 emits exactQ atallN+1 actualobservations; baselineaction==argmax(Q) checked; chronologicalexplorerreplayedfromsavedscene/Q anddeclaredRNGseed; this doesnotrecomputefullvisionfromscenes',
        admit_all_eight_existing_c0_TRAIN_traces=[str(p) for p in screen_paths],all_new_success_failure_censor_traces_retained=True,
        source=base.source_signature(),files={n:sha(Path(__file__).with_name(n)) for n in FILES},pinned_assets=pinned,no_refit=True,no_fresh_data=True,no_hyperparameter_change=True,no_runtime_planning=True,no_automatic_promotion=True)
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),excluded=len(excluded),available=spec['available_new_by_phase']),flush=True)


def task(job,deadline,spec):
    began=time.monotonic();seed_rngs();baseline=load_readout(MODEL/'rbf-model.npz');forest=AdaptiveForest.load(C0);config=PongEvaluationConfig(seed=job['seed'],episodes=1,max_episode_ticks=2400,opponent_speed=.65,policy_scaling=read(MODEL/'radial-cap.json'));session=StatefulEvaluator(baseline,forest,ExplorationAdapter(job['action_seed']),config)
    if time.monotonic()>=deadline:return dict(job,status='not_started',transitions=0,initial_cameras=0,actual_hits=0)
    session._begin_episode();scenes=[canonical_scene(session.latest_encoded)];records=[]
    while not session.complete and session.transitions<1000 and time.monotonic()<deadline:
        r=session.step(check_frozen=False);records.append(r);scenes.append(canonical_scene(session.latest_encoded))
    session.verify_frozen();choices=session.decisions;n=len(records);assert len(scenes)==n+1 and len(choices)==n
    final_action,final_q=baseline.choose(*policy_features(session.latest_encoded,session.scaler));Q=np.asarray([r['q'] for r in records]+[final_q.tolist()],dtype=np.float64).reshape(n+1,3);base_actions=np.argmax(Q,axis=1)
    assert int(base_actions[-1])==final_action and all(int(base_actions[i])==d['baseline_action'] for i,d in enumerate(choices))
    controller=Explorer(forest,job['action_seed'])
    for i,(scene,r,expected) in enumerate(zip(scenes,records,choices)):
        action,d=controller.choose(scene,int(base_actions[i]));assert d==expected and action==r['action'] and d['ordinal']==i;controller.observe(r['hits'],r['terminated'])
    folder=DEST/'traces';stem=folder/f"site_{job['index']:02d}";scene_path=str(stem)+'-raw-scenes.npz';data_path=str(stem)+'-transitions.npz';save_scenes(scene_path,scenes)
    payload={k:np.asarray([r[k] for r in records],dtype=dtype) for k,dtype in [('action',np.int8),('hits',np.int8),('elapsed_seconds',np.float64),('physics_ticks',np.int16),('terminated',bool),('truncated',bool)]}
    payload.update(baseline_q=Q,baseline_actions=base_actions.astype(np.int8),ordinal=np.arange(n,dtype=np.int64))
    for field,dtype in [('option_decision',bool),('selected_option',np.int8),('executed_option',np.int8),('greedy_option',np.int8)]:payload[field]=np.asarray([d[field] for d in choices],dtype=dtype)
    payload['behavior_p']=np.asarray([d['behavior_p'] if d['behavior_p'] is not None else [np.nan]*4 for d in choices],dtype=np.float64).reshape(n,4)
    np.savez_compressed(data_path,**payload)
    result=dict(job,status='environment_end' if session.complete else 'camera_cap_partial' if n>=1000 else 'deadline_partial',complete=session.complete,transitions=n,initial_cameras=session.initial_cameras,actual_hits=sum(r['hits'] for r in records),report=session.summary(),
        source_public_ordinal_start=0,scene_file=scene_path,transition_file=data_path,scene_sha256=sha(scene_path),transition_sha256=sha(data_path),scene_precision='float64',baseline_Q_precision='float64',model_sha256=C0_SHA,baseline_sha256=EXPECTED,
        decisions=choices,full_stateful_audit=dict(passed=True,rows=n,baseline_action_argmax_exact_rows=n+1,source_model_vision_frozen=True,extra_verification_cameras=0),
        macro_boundaries=sum(d['option_decision'] for d in choices),posthit_macro_boundaries=sum(d['option_decision'] and d['past_hit'] for d in choices),exploratory_boundaries=sum(d['option_decision'] and d['past_hit'] and d['selected_option']!=d['greedy_option'] for d in choices),wall_seconds=time.monotonic()-began)
    receipt_path=str(stem)+'.json';write(receipt_path,result)
    return dict({k:v for k,v in result.items() if k not in ('decisions','report')},receipt_file=receipt_path,receipt_sha256=sha(receipt_path))


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists();assert spec['source']==base.source_signature() and all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    began=time.monotonic();deadline=began+180;write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=180,transition_cap=64000))
    results=parallel_jobs(task,[(j,deadline-1,spec) for j in spec['jobs']],deadline,progress=DEST/'progress.json');results.sort(key=lambda r:r['index']);summary=dict(status='all64complete' if len(results)==64 and all(r.get('complete',False) for r in results) else 'bounded_partial',traces=results,
        started=sum(r['initial_cameras']>0 for r in results),complete_games=sum(r.get('complete',False) for r in results),partial_games=sum(r['initial_cameras']>0 and not r.get('complete',False) for r in results),transition_cameras=sum(r['transitions'] for r in results),initial_cameras=sum(r['initial_cameras'] for r in results),actual_hits=sum(r['actual_hits'] for r in results),
        posthit_macro_boundaries=sum(r.get('posthit_macro_boundaries',0) for r in results),exploratory_boundaries=sum(r.get('exploratory_boundaries',0) for r in results),wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),
        source_frozen=all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items()),assets_frozen=all(sha(p)==h for p,h in spec['pinned_assets'].items()),no_refit=True,no_fresh_data=True,no_automatic_promotion=True)
    assert summary['transition_cameras']<=64000 and summary['initial_cameras']<=64;write(DEST/'summary.json',summary);print({k:v for k,v in summary.items() if k!='traces'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    if a.operation=='prepare':prepare()
    else:run(a.approved_spec_sha256)
