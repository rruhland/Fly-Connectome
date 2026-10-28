"""One parent-authorized reward-context comparison; no acquisition or refitting."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import json
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from pong_fixed_rbf import FreshRBFEvaluator, load_readout
from pong_continuous_visual import canonical_scene, save_scenes
from pong_option_memory import load_memory
from pong_reward_context_memory import RewardContextController, RewardContextEvaluator
from rally_raw_capture import SparseRows
from return_metrics import ReturnCounter, native_count_status, aggregate
from run_rally_raw_fresh_screen import paired_metrics
from run_rally_repertoire_collection import exclusions, select_starts
from run_rally_option_memory import (ROOT, MODEL, BANK, EXPECTED, DEST as ORIGINAL,
    FILES as ORIGINAL_FILES, read, write, sha, seed_rngs, parallel_jobs, comparison)
import run_practical_pong_v2 as base

DEST = ROOT/'pong-reward-context-option-memory-20261028-r1'
FILES = (*ORIGINAL_FILES, 'pong_reward_context_memory.py', 'run_rally_reward_context_memory.py')


def prepare():
    assert not DEST.exists()
    prior = read(ORIGINAL/'spec.json'); outcome = read(ORIGINAL/'summary.json')
    assert outcome['status']=='rejected_screen8' and prior['source']==base.source_signature()
    assert all(sha(Path(__file__).with_name(n))==h for n,h in prior['files'].items())
    for site in prior['sites']:
        receipt = read(site['receipt'])
        assert any(r['hits']>0 for r in receipt['records'][:site['anchor']])
        assert site['previous_hit']<site['anchor']
    DEST.mkdir();(DEST/'fresh').mkdir()
    drawn=select_starts(1943906127,63153,exclusions(DEST),32)
    seeds=drawn.pop('TRAIN');drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],
        purpose='new paired reward-context qualification; excludes original63137 and all earlier schedules')
    write(DEST/'fresh/schedules.json',drawn)
    for name in ('option-memory.npz','rbf-model.npz','proposal.npy','radial-cap.json'):
        (DEST/name).write_bytes((ORIGINAL/name).read_bytes())
    spec=dict(amendment='parent authorized causal scalar-reward-history context; not a radius/duration/guard sweep',
        reason='both original losing games retrieved later-miss site1 before their first own hit; two later gains survived',
        policy_context='has_received_own_hit_since_scene_reset; false at reset, true only after actual preceding record hits>0',
        stored_context=[True]*8,stored_context_proof=prior['sites'],
        unchanged=['eight scenes','all paired advantages including negatives','radius','16camera earnedholds','40069 baseline and vision'],
        memory_sha256=sha(DEST/'option-memory.npz'),original_spec_sha256=sha(ORIGINAL/'spec.json'),
        original_source_commit='1903b20',original_outcome_sha256=sha(ORIGINAL/'summary.json'),
        seconds=180,new_acquisition_cameras=0,no_refitting=True,screen8=prior['screen8'],screen32=prior['screen32'],fresh_tag=63153,
        source=base.source_signature(),files={n:sha(Path(__file__).with_name(n)) for n in FILES},
        pinned_assets={**prior['pinned_assets'],str(ORIGINAL/'option-memory.npz'):sha(ORIGINAL/'option-memory.npz'),
            str(ORIGINAL/'acquisition-summary.json'):sha(ORIGINAL/'acquisition-summary.json'),
            str(ORIGINAL/'summary.json'):sha(ORIGINAL/'summary.json')},
        raw_only_episode_certificates_disabled=True,sequential_audit_context_updates_after_reward=True,
        first_actions_until_first_hit_must_match_actual40069_receipt=True,
        no_privileged_policy_inputs=True,no_expert_labels=True,no_backprop=True,no_runtime_lookahead=True,
        scaffold_caveat='reward-history-conditioned frozen memory; not a learned general temporal representation or continuously adapting controller',
        no_automatic_promotion=True,no_parameter_sweep=True,parent_native_review_required=True)
    write(DEST/'spec.json',spec)
    print(json.dumps(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),memory_sha256=spec['memory_sha256'])),flush=True)


def native_task(directory,seed,index,deadline,candidate=True):
    directory=Path(directory);began=time.monotonic();seed_rngs()
    original=load_readout(MODEL/'rbf-model.npz');memory=load_memory(DEST/'option-memory.npz') if candidate else None
    config=PongEvaluationConfig(seed=seed,episodes=1,max_episode_ticks=2400,opponent_speed=.65,
        policy_scaling=read(MODEL/'radial-cap.json'))
    session=RewardContextEvaluator(original,memory,config) if candidate else FreshRBFEvaluator(original,config)
    game=session.env.game;original_step=game.step;counter=ReturnCounter();records=[];events=[];raw=SparseRows();scenes=[]
    def instrument(drive,**kwargs):
        own=counter.opportunities;opponent=counter.opponent_opportunities;tick=int(game.rally_steps[0])+1
        out=counter.step(game,original_step,drive,**kwargs)
        if counter.opportunities>own:events.append(dict(kind='own',ordinal=counter.opportunities,success=bool(out.hits[0]),tick=tick,transition=session.transitions,completed_own_hits=counter.returns))
        if counter.opponent_opportunities>opponent:events.append(dict(kind='opponent',ordinal=counter.opponent_opportunities,success=float(game.ball_velocity[0,0])<0,tick=tick,transition=session.transitions,after_own_hits=counter.returns))
        return out
    game.step=instrument
    try:
        while not session.complete and time.monotonic()<deadline:
            if session.initial_cameras==session.episodes:session._begin_episode()
            raw.add(session);scenes.append(canonical_scene(session.latest_encoded));records.append(session.step(check_frozen=False))
    finally:del game.step
    session.verify_frozen();report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds)
    annotation=counter.summary(game,truncated=last['truncated'],terminated=last['terminated'],reward=last['reward'],seconds=last['seconds'])
    count_status=native_count_status(annotation,report)
    stem=f"{'candidate' if candidate else 'baseline'}_{index:02d}_{seed}"
    raw_path=directory/(stem+'-raw.npz');scene_path=directory/(stem+'-scenes.npz')
    X=raw.matrix();sp.save_npz(raw_path,X);save_scenes(scene_path,scenes)
    choices=session.option_choices if candidate else [];audit=dict(passed=True,rows=len(records),stateful_replay=True,new_native_cameras=0)
    if candidate:
        controller=RewardContextController(memory);changes=starts=holds=baseline=before_first=0
        for i,(scene,record,choice) in enumerate(zip(scenes,records,choices)):
            lo,hi=X.indptr[i:i+2];baseline_action,q=original.choose(X.indices[lo:hi],X.data[lo:hi])
            action,decision=controller.choose(scene,baseline_action)
            assert decision==choice and action==record['action'] and np.array_equal(q,np.asarray(record['q']))
            if not controller.has_received_own_hit:
                assert action==baseline_action and choice['mode']=='baseline_reward_context';before_first+=1
            controller.observe_own_reward(int(record['hits']))
            changes+=int(action!=baseline_action);starts+=int(choice['mode']=='earned_start')
            holds+=int(choice['mode']=='earned_hold');baseline+=int(choice['mode'].startswith('baseline'))
        assert len(choices)==len(records)
        audit.update(changed_actions=changes,earned_option_starts=starts,earned_held_cameras=holds,
            baseline_camera_decisions=baseline,pre_first_hit_baseline_rows=before_first,
            reward_context_updated_only_after_recorded_hits=True,raw_only_certificate_reuse=False)
    result=dict(seed=seed,complete=report['complete'],config=asdict(config),report=report,annotation=annotation,
        annotation_count_status=count_status,records=records,opportunity_events=events,option_choices=choices,
        feature_file=str(raw_path),structured_feature_file=str(scene_path),transitions=session.transitions,
        initial_cameras=session.initial_cameras,raw_only_episode_reuse=False,full_stateful_audit=audit,
        baseline_model_sha256=EXPECTED,memory_sha256=sha(DEST/'option-memory.npz') if candidate else None,
        policy_amendment='scalar_past_own_hit_context',wall_seconds=time.monotonic()-began)
    write(directory/(stem+'.json'),result)
    if report['complete']:assert count_status['valid_terminal_counts']
    return result


def context_comparison(baseline,candidate):
    stats=comparison(baseline,candidate);checked=0
    for original,learned in zip(baseline,candidate):
        assert original['seed']==learned['seed']
        for i,row in enumerate(learned['records']):
            if row is None:raise AssertionError('missing actual record')
            previous=original['records'][i]
            assert row['action']==previous['action'] and row['hits']==previous['hits']
            assert np.array_equal(np.asarray(row['q']),np.asarray(previous['q']))
            checked+=1
            if row['hits']>0:break
    stats['exact_saved40069_prefix_until_first_hit_rows']=checked
    stats['first_hit_prefix_audit_passed']=True
    return stats


def evaluate(approval):
    assert approval==sha(DEST/'spec.json') and not (DEST/'evaluation-clock.json').exists()
    spec=read(DEST/'spec.json');assert spec['source']==base.source_signature()
    assert all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items())
    assert all(sha(p)==h for p,h in spec['pinned_assets'].items())
    assert sha(DEST/'option-memory.npz')==spec['memory_sha256']
    began=time.monotonic();deadline=began+180
    write(DEST/'evaluation-clock.json',dict(started_monotonic=began,deadline_monotonic=deadline,started_wall_ns=time.time_ns(),seconds=180))
    bank={e['seed']:e for e in read(BANK)['bank']};results=[];summary=dict(status='evaluating',new_transition_cameras=0,new_initial_cameras=0,new_acquisition_cameras=0)
    for size in (8,32):
        directory=DEST/f'screen{size}';directory.mkdir();target=spec['screen8' if size==8 else 'screen32'];existing={r['seed']:r for r in results}
        jobs=[(str(directory),seed,i,deadline-1,True) for i,seed in enumerate(target) if seed not in existing]
        new=parallel_jobs(native_task,jobs,deadline,progress=directory/'progress.json');results+=new
        summary['new_transition_cameras']+=sum(r['transitions'] for r in new);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in new)
        byseed={r['seed']:r for r in results};complete=all(s in byseed and byseed[s]['complete'] for s in target)
        selected=[byseed[s] for s in target if s in byseed];baseline=[read(bank[r['seed']]['receipt']) for r in selected]
        stats=context_comparison(baseline,selected);stats['complete']=complete;write(directory/'summary.json',stats);summary[f'screen{size}']=stats
        if not complete:summary['status']=f'screen{size}_bounded_partial';break
        before=(stats['baseline']['own_returns'],stats['baseline_more']['later_returns'],stats['baseline']['first_incoming_returns'],stats['baseline_more']['tail']['2'])
        after=(stats['candidate']['own_returns'],stats['candidate_more']['later_returns'],stats['candidate']['first_incoming_returns'],stats['candidate_more']['tail']['2'])
        if any(a<b for a,b in zip(after,before)) or after[0]<=before[0]:summary['status']=f'rejected_screen{size}';break
        if size==32:
            gate=after[1]>before[1] and stats['candidate_more']['later_fraction']>=stats['baseline_more']['later_fraction']
            if not gate:summary['status']='rejected_screen32';break
            fresh=DEST/'fresh';seeds=read(fresh/'schedules.json')['probe']
            jobs=[(str(fresh),seed,i,deadline-1,candidate) for i,seed in enumerate(seeds) for candidate in (False,True)]
            fresh_results=parallel_jobs(native_task,jobs,deadline,progress=fresh/'progress.json')
            summary['new_transition_cameras']+=sum(r['transitions'] for r in fresh_results);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in fresh_results)
            groups={c:{r['seed']:r for r in fresh_results if (r['memory_sha256'] is not None)==c} for c in (False,True)}
            complete=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values())
            fs=dict(complete=complete,groups={str(c):aggregate([r['annotation'] for r in g.values()]) for c,g in groups.items()})
            if complete:
                a=[groups[False][s] for s in seeds];b=[groups[True][s] for s in seeds]
                fs.update(context_comparison(a,b),paired=paired_metrics(a,b))
            write(fresh/'summary.json',fs);summary.update(fresh=fs,status='fresh_complete' if complete else 'fresh_bounded_partial')
    summary.update(wall_seconds=time.monotonic()-began,memory_frozen=sha(DEST/'option-memory.npz')==spec['memory_sha256'],
        source_frozen=base.source_signature()==spec['source'] and all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items()),
        assets_unchanged=all(sha(p)==h for p,h in spec['pinned_assets'].items()),
        no_refitting=True,raw_only_episode_certificates_disabled=True,no_automatic_promotion=True,
        scaffold_caveat=spec['scaffold_caveat'])
    write(DEST/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','evaluate']);parser.add_argument('--approved-spec-sha256');args=parser.parse_args()
    if args.operation=='prepare':prepare()
    else:evaluate(args.approved_spec_sha256)
