"""One48site matched-window acquisition, one forest fit, one bounded evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import ExtraTreesRegressor
import sklearn
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from pong_action_credit import clone_session,step_action
from pong_continuous_visual import canonical_scene,load_scenes,save_scenes
from pong_fixed_rbf import FreshRBFEvaluator,load_readout
from pong_paired_option_forest import (features,OptionForest,save_forest,load_forest,ForestEvaluator,ForestController,window_end,SCHEMA,ORDER,HOLD)
from run_rally_option_memory import (ROOT,MODEL,BANK,CALIBRATION,SCREEN8,EXPECTED,read,write,sha,seed_rngs,parallel_jobs,comparison)
from run_rally_repertoire_collection import exclusions,select_starts
from run_rally_raw_optimizer import more
from run_rally_raw_fresh_screen import paired_metrics
from return_metrics import ReturnCounter,aggregate,native_count_status
from rally_raw_capture import SparseRows
import run_practical_pong_v2 as base

DEST=ROOT/'pong-paired-option-forest-20261029-r1'
FILES=('run_paired_option_forest.py','pong_paired_option_forest.py','run_rally_option_memory.py','pong_option_memory.py','pong_continuous_visual.py','pong_action_credit.py','pong_fixed_rbf.py','pong_fixed_pairwise.py','rally_raw_capture.py','return_metrics.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_practical_pong_v2.py')


def prepare():
    assert not DEST.exists() and sha(MODEL/'rbf-model.npz')==EXPECTED
    bank=read(BANK);used=set();strata={};rng=np.random.default_rng(2026102901)
    for kind in ('later_miss','later_success','uniform_posthit'):
        for phase in ('left','right'):
            selected=[]
            for entry in bank['bank']:
                if entry['seed'] in used or base.leftward(entry['seed'])!=(phase=='left'):continue
                r=read(entry['receipt']);assert r['complete'] and r['weights_sha256']==EXPECTED
                hits=[i for i,v in enumerate(r['records']) if v['hits']]
                if kind=='uniform_posthit':
                    if not hits or hits[0]+1>=min(len(r['records']),513):continue
                    selected.append((entry,r,hits))
                else:
                    candidates=[(h,hits[j-1]) for j,h in enumerate(hits) if j]
                    if kind=='later_miss':candidates=[(len(r['records'])-1,hits[-1])] if hits and r['records'][-1]['terminated'] and r['records'][-1]['reward']<0 else []
                    candidates=[(o,p) for o,p in candidates if 0<o-64<=512 and o-64>p]
                    if not candidates:continue
                    outcome,previous=candidates[0];selected.append(dict(seed=entry['seed'],phase=phase,kind=kind,anchor=outcome-64,outcome=outcome,previous_hit=previous,receipt=entry['receipt']))
                    used.add(entry['seed'])
                    if len(selected)==8:break
            if kind=='uniform_posthit':
                if len(selected)<8:raise RuntimeError(f'site shortfall {kind}/{phase}: {len(selected)}/8')
                choices=rng.choice(len(selected),8,replace=False);sites=[]
                for j in choices:
                    entry,r,hits=selected[j];anchor=int(rng.integers(hits[0]+1,min(len(r['records']),513)));previous=max(h for h in hits if h<anchor)
                    sites.append(dict(seed=entry['seed'],phase=phase,kind=kind,anchor=anchor,previous_hit=previous,receipt=entry['receipt']))
                    used.add(entry['seed'])
                selected=sites
            if len(selected)!=8:raise RuntimeError(f'site shortfall {kind}/{phase}: {len(selected)}/8')
            strata[kind+'/'+phase]=selected
    sites=[dict(strata[key][i],index=6*i+j,stratum=key) for i in range(8) for j,key in enumerate(strata)]
    assert len(sites)==len({s['seed'] for s in sites})==48 and all(0<s['anchor']<=512 for s in sites)
    paths=[CALIBRATION/f'calibration_{i:02d}-scenes.npz' for i in range(8)];qpaths=[CALIBRATION/f'calibration_{i:02d}-scores.npz' for i in range(8)];scenes=[s for p in paths for s in load_scenes(p)];assert len(scenes)==2048
    capacity=max(map(len,scenes));first=[e['seed'] for e in read(SCREEN8)['bank']];screen=first+[e['seed'] for e in bank['bank'] if e['seed'] not in first][:24]
    DEST.mkdir();(DEST/'acquisition').mkdir();(DEST/'fresh').mkdir();schedule=select_starts(1943906127,63201,exclusions(DEST),32);seeds=schedule.pop('TRAIN');schedule.update(probe=seeds[:32],unused_reserved=seeds[32:]);write(DEST/'fresh/schedules.json',schedule)
    pinned={str(p):sha(p) for p in [BANK,SCREEN8,*paths,*qpaths,MODEL/'rbf-model.npz',MODEL/'proposal.npy',MODEL/'radial-cap.json']}
    pinned.update({s['receipt']:sha(s['receipt']) for s in sites})
    spec=dict(method='direct_multioutput_paired_binary160camera_option_advantage',sites=sites,capacity=capacity,feature_dimension=7*capacity+4,canonical_sort_fields=list(ORDER),row_meaning='[y,x,vy,vx,observed,velocity_known]; horizontalfirstordering only, no IDs/roles/age',
        feature_definition='whole float64sixfieldscene,padding,presence,count,3frozen40069Q; no extra normalization; overflowunsupported',
        estimator=dict(n_estimators=64,criterion='squared_error',max_depth=5,min_samples_leaf=2,max_features=1.0,bootstrap=False,random_state=2026102902,n_jobs=1),
        acquisition_seconds=180,acquisition_transition_cap=60000,design_maximum_transitions=sum(s['anchor']+640 for s in sites),maximum_prefix=512,suffix_cap=160,hold=16,
        target='binary ownhit in actual160camera window; hit1,naturaltermination0,complete160withoutownhit0; earlier timeout/deadline censored',
        minimum_complete_distinct_sites=32,minimum_nonzero_paired_targets=1,balance='all6strata required; fixed1/6totalweight perstratum,uniformwithin; allcompleteactionbalanced sites retained',
        dense_capture='all suffix N+1 canonical scenes float64, N+1 baselineQfloat64,actions,hits,dt,ticks,terminal,truncated,forced; zeroextra cameras; no rawCSR acquisition',
        deployment='exactbaselineuntilactualownhit; positivehighestpredictedadvantage selects16hold; allzero/negative/overflowfallbackbaselineeachcamera',
        representation_supported_is_not_evidence_coverage=True,evaluation_seconds=180,workers=4,screen8=first,screen32=screen,fresh_tag=63201,
        source=base.source_signature(),files={n:sha(Path(__file__).with_name(n)) for n in FILES},pinned_assets=pinned,sklearn_version=sklearn.__version__,
        no_backprop=True,no_privileged_inputs=True,no_expert_labels=True,no_runtime_planning=True,no_sweep=True,no_automatic_promotion=True)
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',sites=len(sites),capacity=capacity,design_maximum_transitions=spec['design_maximum_transitions'],spec_sha256=sha(DEST/'spec.json')),flush=True)


def save_trace(stem,records,scenes,extra_q=None):
    if scenes is not None:save_scenes(str(stem)+'-scenes.npz',scenes)
    payload={key:np.asarray([r[key] for r in records],dtype=dtype) for key,dtype in [('action',np.int8),('hits',np.int8),('elapsed_seconds',np.float64),('physics_ticks',np.int16),('terminated',bool),('truncated',bool)]}
    payload['forced']=np.asarray(['intervention_action' in r for r in records],dtype=bool)
    if extra_q is not None:payload['baseline_q']=np.asarray([r['q'] for r in records]+[extra_q],dtype=np.float64).reshape(-1,3)
    np.savez_compressed(str(stem)+'-transitions.npz',**payload)


def acquisition_task(site,deadline,spec):
    began=time.monotonic();seed_rngs();original=load_readout(MODEL/'rbf-model.npz');receipt=read(site['receipt']);session=FreshRBFEvaluator(original,PongEvaluationConfig(**receipt['config']));prefix=[];outputs=[]
    if time.monotonic()>=deadline:return dict(site=site,status='not_started',transitions=0,initials=0,outcomes=[])
    session._begin_episode()
    while session.transitions<site['anchor'] and time.monotonic()<deadline:
        r=session.step(check_frozen=False);prefix.append(r);assert r==receipt['records'][len(prefix)-1]
    session.verify_frozen();paid=len(prefix);initials=session.initial_cameras;stem=DEST/'acquisition'/f"site_{site['index']:02d}";save_trace(str(stem)+'-prefix',prefix,None)
    if paid!=site['anchor']:
        result=dict(site=site,status='prefix_partial',transitions=paid,initials=initials,outcomes=[]);write(str(stem)+'.json',result);return result
    scene=canonical_scene(session.latest_encoded);baseline_q=original.choose(*policy_features(session.latest_encoded,session.scaler))[1]
    save_scenes(str(stem)+'-anchor-scene.npz',[scene])
    for option in (3,0,1,2):
        if time.monotonic()>=deadline:break
        branch=clone_session(session);records=[];scenes=[canonical_scene(branch.latest_encoded)];reason='deadline_censored';observed=False;value=None
        for t in range(160):
            if time.monotonic()>=deadline:break
            r=step_action(branch,option) if option!=3 and t<HOLD else branch.step(check_frozen=False);records.append(r);paid+=1;scenes.append(canonical_scene(branch.latest_encoded));ending=window_end(r,len(records))
            if ending is not None:reason,observed,value=ending;break
        branch.verify_frozen();assert branch.initial_cameras==initials
        if option==3:assert records==receipt['records'][site['anchor']:site['anchor']+len(records)]
        next_q=original.choose(*policy_features(branch.latest_encoded,branch.scaler))[1].tolist();branchstem=str(stem)+f'-option_{option}';save_trace(branchstem,records,scenes,next_q)
        outputs.append(dict(option=option,observed=observed,value=value,reason=reason,transitions=len(records),elapsed_seconds=sum(r['elapsed_seconds'] for r in records),trace_stem=branchstem))
        write(str(stem)+'-progress.json',dict(site=site,transitions=paid,initials=initials,outcomes=outputs))
    result=dict(site=site,status='complete' if len(outputs)==4 and all(o['observed'] for o in outputs) else 'censored_or_partial',transitions=paid,initials=initials,prefix_transitions=len(prefix),outcomes=outputs,scene_file=str(stem)+'-anchor-scene.npz',baseline_q=baseline_q.tolist(),wall_seconds=time.monotonic()-began)
    assert paid<=site['anchor']+640;write(str(stem)+'.json',result);return result


def validate(approved):
    assert approved==sha(DEST/'spec.json');spec=read(DEST/'spec.json');assert spec['source']==base.source_signature() and sklearn.__version__==spec['sklearn_version']
    assert all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items());return spec


def acquire(approved):
    spec=validate(approved);assert not (DEST/'acquisition-clock.json').exists();began=time.monotonic();deadline=began+180;write(DEST/'acquisition-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=180,transition_cap=60000))
    results=parallel_jobs(acquisition_task,[(s,deadline-1,spec) for s in spec['sites']],deadline,progress=DEST/'acquisition-progress.json');results.sort(key=lambda r:r['site']['index']);all_complete=[r for r in results if r['status']=='complete'];complete=[];rows=[];targets=[];overflow=[]
    for r in all_complete:
        f=features(load_scenes(r['scene_file'])[0],r['baseline_q'],spec['capacity'])
        if f is None:overflow.append(r['site']['index']);continue
        complete.append(r);outcomes={o['option']:o['value'] for o in r['outcomes']};rows.append(f);targets.append([outcomes[a]-outcomes[3] for a in range(3)])
    represented={r['site']['stratum'] for r in complete};fit_eligible=not overflow and len(complete)>=32 and len(represented)==6 and any(any(v!=0 for v in row) for row in targets)
    summary=dict(status='complete' if len(complete)==48 else 'bounded_or_censored',complete_actionbalanced_sites=len(all_complete),representation_supported_complete_sites=len(complete),overflow_sites=overflow,stratum_counts={k:sum(r['site']['stratum']==k for r in complete) for k in sorted({s['stratum'] for s in spec['sites']})},fit_eligible=fit_eligible,
        rescue_pairs=sum(v>0 for row in targets for v in row),lost_hit_pairs=sum(v<0 for row in targets for v in row),zero_pairs=sum(v==0 for row in targets for v in row),
        rescue_sites=sum(any(v>0 for v in row) for row in targets),lost_hit_sites=sum(any(v<0 for v in row) for row in targets),transition_cameras=sum(r['transitions'] for r in results),initial_cameras=sum(r['initials'] for r in results),wall_seconds=time.monotonic()-began,fit_sites=[r['site'] for r in complete],targets=targets)
    assert summary['transition_cameras']<=60000
    if complete:
        np.savez_compressed(DEST/'fit-data.npz',features=np.asarray(rows),targets=np.asarray(targets),site_indices=np.array([r['site']['index'] for r in complete]))
    write(DEST/'acquisition-summary.json',summary);print(summary,flush=True)


def native_task(directory,seed,index,deadline,candidate=True):
    directory=Path(directory);began=time.monotonic();seed_rngs();original=load_readout(MODEL/'rbf-model.npz');config=PongEvaluationConfig(seed=seed,episodes=1,max_episode_ticks=2400,opponent_speed=.65,policy_scaling=read(MODEL/'radial-cap.json'))
    forest=load_forest(DEST/'option-forest.npz') if candidate else None;session=ForestEvaluator(original,forest,config) if candidate else FreshRBFEvaluator(original,config)
    if time.monotonic()>=deadline:return dict(seed=seed,complete=False,transitions=0,initial_cameras=0,not_started=True)
    session._begin_episode();game=session.env.game;oldstep=game.step;counter=ReturnCounter();raw=SparseRows();scenes=[];records=[]
    def instrument(drive,**kwargs):return counter.step(game,oldstep,drive,**kwargs)
    game.step=instrument
    try:
        while not session.complete and time.monotonic()<deadline:
            raw.add(session);scenes.append(canonical_scene(session.latest_encoded));records.append(session.step(check_frozen=False))
    finally:del game.step
    session.verify_frozen();report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds);annotation=counter.summary(game,**{k:last[k] for k in ('truncated','terminated','reward','seconds')});stem=directory/f"{'candidate' if candidate else 'carry'}_{index:02d}";X=raw.matrix();sp.save_npz(str(stem)+'-raw.npz',X);save_scenes(str(stem)+'-scenes.npz',scenes);choices=session.option_choices if candidate else [];audit=dict(passed=True,rows=len(records),new_cameras=0)
    if candidate:
        assert len(records)==len(choices)==len(scenes)==X.shape[0]
        controller=ForestController(forest);changes=starts=holds=before=0
        for i,(scene,r,choice) in enumerate(zip(scenes,records,choices)):
            lo,hi=X.indptr[i:i+2];baseline,q=original.choose(X.indices[lo:hi],X.data[lo:hi]);action,got=controller.choose(scene,baseline,q);assert got==choice and action==r['action'] and np.array_equal(q,np.asarray(r['q']));controller.observe_reward(r['hits']);changes+=action!=baseline;starts+=choice['mode']=='earned_start';holds+=choice['mode']=='earned_hold';before+=choice['mode']=='baseline_before_hit'
        audit.update(changed_actions=int(changes),earned_option_starts=int(starts),earned_held_cameras=int(holds),baseline_before_hit_cameras=int(before))
    result=dict(seed=seed,complete=session.complete,report=report,annotation=annotation,records=records,option_choices=choices,transitions=session.transitions,initial_cameras=session.initial_cameras,full_stateful_audit=audit,transport_kind=SCHEMA if candidate else 'visual-transport-v1',forest_sha256=sha(DEST/'option-forest.npz') if candidate else None,feature_file=str(stem)+'-raw.npz',structured_feature_file=str(stem)+'-scenes.npz',wall_seconds=time.monotonic()-began)
    if report['complete']:assert native_count_status(annotation,report)['valid_terminal_counts']
    write(str(stem)+'.json',result);return result


def fit_evaluate(approved):
    spec=validate(approved);acq=read(DEST/'acquisition-summary.json');assert not (DEST/'evaluation-clock.json').exists();began=time.monotonic();deadline=began+180;write(DEST/'evaluation-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=180))
    if not acq['fit_eligible']:write(DEST/'summary.json',dict(status='acquisition_failure',acquisition=acq,no_fitting=True,no_automatic_promotion=True));return
    with np.load(DEST/'fit-data.npz') as f:F=f['features'];Y=f['targets']
    counts=acq['stratum_counts'];weights=np.array([1/(6*counts[s['stratum']]) for s in acq['fit_sites']]);assert np.isclose(weights.sum(),1)
    model=ExtraTreesRegressor(**spec['estimator']);fit_began=time.monotonic();model.fit(F,Y,sample_weight=weights);forest=OptionForest.export(model,spec['capacity']);expected=model.predict(F);actual=forest.predict(F);assert np.array_equal(actual,expected) and np.array_equal(np.argmax(actual,axis=1),np.argmax(expected,axis=1));save_forest(DEST/'option-forest.npz',forest)
    allF=[]
    for i in range(8):
        scenes=load_scenes(CALIBRATION/f'calibration_{i:02d}-scenes.npz')
        with np.load(CALIBRATION/f'calibration_{i:02d}-scores.npz') as f:q=f['q']
        allF.extend(features(s,v,spec['capacity']) for s,v in zip(scenes,q))
    ref=model.predict(np.asarray(allF));got=forest.predict(np.asarray(allF));assert np.array_equal(ref,got) and np.array_equal(np.argmax(ref,axis=1),np.argmax(got,axis=1))
    report=dict(minimum_actual_leaf_site_count=min(int(t.tree_.n_node_samples[t.tree_.children_left<0].min()) for t in model.estimators_),fit_seconds=time.monotonic()-fit_began,sites=len(F),nonzero_targets=int(np.count_nonzero(Y)),training_predictions=actual.tolist(),export_exact_rows=len(F)+len(allF),forest_sha256=sha(DEST/'option-forest.npz'),one_fit=True);write(DEST/'fit-report.json',report)
    bank={e['seed']:e for e in read(BANK)['bank']};results=[];summary=dict(status='evaluating',new_transition_cameras=0,new_initial_cameras=0,fit=report)
    for size in (8,32):
        folder=DEST/f'screen{size}';folder.mkdir();target=spec['screen8' if size==8 else 'screen32'];existing={r['seed']:r for r in results};jobs=[(str(folder),seed,i,deadline-1,True) for i,seed in enumerate(target) if seed not in existing];new=parallel_jobs(native_task,jobs,deadline,progress=folder/'progress.json');results+=new;summary['new_transition_cameras']+=sum(r['transitions'] for r in new);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in new);byseed={r['seed']:r for r in results};complete=all(s in byseed and byseed[s]['complete'] for s in target);selected=[byseed[s] for s in target if s in byseed and not byseed[s].get('not_started')];baseline=[read(bank[r['seed']]['receipt']) for r in selected];stats=comparison(baseline,selected);stats['complete']=complete;write(folder/'summary.json',stats);summary[f'screen{size}']=stats
        if not complete:summary['status']=f'screen{size}_partial';break
        a=(stats['baseline']['own_returns'],stats['baseline_more']['later_returns'],stats['baseline']['first_incoming_returns'],stats['baseline_more']['tail']['2']);b=(stats['candidate']['own_returns'],stats['candidate_more']['later_returns'],stats['candidate']['first_incoming_returns'],stats['candidate_more']['tail']['2'])
        if b[0]<=a[0] or b[1]<a[1] or b[2]<a[2] or (size==32 and (b[1]<=a[1] or b[3]<a[3] or stats['candidate_more']['later_fraction']<stats['baseline_more']['later_fraction'])):summary['status']=f'rejected_screen{size}';break
        if size==32:
            fresh=DEST/'fresh';seeds=read(fresh/'schedules.json')['probe'];jobs=[(str(fresh),seed,i,deadline-1,c) for i,seed in enumerate(seeds) for c in (False,True)];rs=parallel_jobs(native_task,jobs,deadline,progress=fresh/'progress.json');summary['new_transition_cameras']+=sum(r['transitions'] for r in rs);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in rs);groups={c:{r['seed']:r for r in rs if not r.get('not_started') and (r['forest_sha256'] is not None)==c} for c in (False,True)};complete=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values());fs=dict(complete=complete,groups={str(c):aggregate([r['annotation'] for r in g.values()]) for c,g in groups.items()})
            if complete:
                a=[groups[False][s] for s in seeds];b=[groups[True][s] for s in seeds];fs.update(comparison(a,b),paired=paired_metrics(a,b))
            write(fresh/'summary.json',fs);summary.update(fresh=fs,status='fresh_complete' if complete else 'fresh_partial')
    summary.update(wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),forest_frozen=sha(DEST/'option-forest.npz')==report['forest_sha256'],source_frozen=all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items()),no_automatic_promotion=True);write(DEST/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','acquire','fit-evaluate']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    if a.operation=='prepare':prepare()
    elif a.operation=='acquire':acquire(a.approved_spec_sha256)
    else:fit_evaluate(a.approved_spec_sha256)
