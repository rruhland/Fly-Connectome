"""One180s native comparison of one already-fitted frozen transition graph."""
import argparse
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from pong_kernel_transition_graph import KernelGraph
from pong_kernel_graph_policy import GraphController,GraphEvaluator,freeze_graph,CONTRACT
from pong_continuous_visual import canonical_scene,save_scenes
from pong_fixed_rbf import FreshRBFEvaluator,load_readout
from run_rally_option_memory import (ROOT,MODEL,BANK,SCREEN8,EXPECTED,read,write,sha,seed_rngs,parallel_jobs)
from run_rally_repertoire_collection import exclusions,select_starts
from run_rally_raw_optimizer import more
from run_rally_raw_fresh_screen import paired_metrics
from return_metrics import ReturnCounter,aggregate,native_count_status
from rally_raw_capture import SparseRows
import run_practical_pong_v2 as base

DEST=ROOT/'pong-kernel-graph-native-20261029-r1'
GRAPH=ROOT/'pong-kernel-transition-graph-20261029-r1-recovery/graph.npz'
GRAPH_SHA='4067f033ec767fa072ab4e9342d0d95b9f27b030cbcda536e16261fd7bdccb75'
FILES=('run_kernel_graph_comparison.py','pong_kernel_graph_policy.py','pong_kernel_transition_graph.py','run_rally_option_memory.py','pong_option_memory.py','pong_continuous_visual.py','pong_fixed_rbf.py','pong_fixed_pairwise.py','rally_raw_capture.py','return_metrics.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_practical_pong_v2.py')


def prepare():
    assert not DEST.exists() and sha(GRAPH)==GRAPH_SHA and sha(MODEL/'rbf-model.npz')==EXPECTED
    DEST.mkdir();(DEST/'fresh').mkdir();bank=read(BANK);first=[e['seed'] for e in read(SCREEN8)['bank']];screen=first+[e['seed'] for e in bank['bank'] if e['seed'] not in first][:24]
    schedule=select_starts(1943906127,63217,exclusions(DEST),32);seeds=schedule.pop('TRAIN');schedule.update(probe=seeds[:32],unused_reserved=seeds[32:]);write(DEST/'fresh/schedules.json',schedule)
    spec=dict(method=CONTRACT,graph=str(GRAPH),graph_sha256=GRAPH_SHA,seconds=180,workers=4,screen8=first,screen32=screen,fresh_tag=63217,
        source=base.source_signature(),files={n:sha(Path(__file__).with_name(n)) for n in FILES},pinned_assets={str(p):sha(p) for p in (GRAPH,BANK,SCREEN8,MODEL/'rbf-model.npz',MODEL/'radial-cap.json')},
        policy_inputs='canonicallegalvisualscene,actualscalarownhit-history,preactionpubliccameraordinalmod5,baseline40069action',
        selection='baselineuntilactualhit; changeonlyifbaselineandselectedalternativebothsupported andqalternative>qbaseline;camera-by-camera',
        qualification='8strict total/latergain,firstparity;32sameplussecondcount/laterfractionnotworse;conditional32freshpairs',
        no_fit=True,no_new_acquisition=True,no_sweep=True,no_runtime_simulator=True,raw_only_reuse_disabled=True,no_automatic_promotion=True)
    write(DEST/'spec.json',spec);print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json')),flush=True)


def native_task(directory,seed,index,deadline,candidate=True):
    began=time.monotonic();directory=Path(directory);seed_rngs();baseline=load_readout(MODEL/'rbf-model.npz');config=PongEvaluationConfig(seed=seed,episodes=1,max_episode_ticks=2400,opponent_speed=.65,policy_scaling=read(MODEL/'radial-cap.json'));graph=freeze_graph(KernelGraph.load(GRAPH)) if candidate else None;session=GraphEvaluator(baseline,graph,config) if candidate else FreshRBFEvaluator(baseline,config)
    if time.monotonic()>=deadline:return dict(seed=seed,complete=False,not_started=True,transitions=0,initial_cameras=0)
    session._begin_episode();game=session.env.game;oldstep=game.step;counter=ReturnCounter();records=[];raw=SparseRows();scenes=[]
    def instrument(drive,**kwargs):return counter.step(game,oldstep,drive,**kwargs)
    game.step=instrument
    try:
        while not session.complete and time.monotonic()<deadline:
            raw.add(session);scenes.append(canonical_scene(session.latest_encoded));records.append(session.step(check_frozen=False))
    finally:del game.step
    session.verify_frozen();report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds);annotation=counter.summary(game,**{k:last[k] for k in ('truncated','terminated','reward','seconds')});stem=directory/f"{'candidate' if candidate else 'carry'}_{index:02d}";X=raw.matrix();sp.save_npz(str(stem)+'-raw.npz',X);save_scenes(str(stem)+'-scenes.npz',scenes);choices=session.graph_choices if candidate else [];audit=dict(passed=True,rows=len(records),new_cameras=0)
    if candidate:
        assert len(choices)==len(records)==len(scenes)==X.shape[0];controller=GraphController(graph);changes=before=covered=0;minimum_gain=None
        for i,(scene,r,expected) in enumerate(zip(scenes,records,choices)):
            lo,hi=X.indptr[i:i+2];action,q=baseline.choose(X.indices[lo:hi],X.data[lo:hi]);got,decision=controller.choose(scene,action);assert decision==expected and got==r['action'] and np.array_equal(q,np.asarray(r['q']));assert decision['camera_ordinal']==i and decision['camera_phase']==i%5
            if got!=action:
                assert decision['past_hit'] and decision['supported'][action] and decision['supported'][got] and decision['q'][got]>decision['q'][action]
                gain=decision['q'][got]-decision['q'][action];minimum_gain=gain if minimum_gain is None else min(minimum_gain,gain)
            changes+=got!=action;before+=not decision['past_hit'];covered+=decision['past_hit'] and decision['supported'][action] and sum(decision['supported'])>=2;controller.feedback(r['hits'])
        audit.update(changed_actions=int(changes),baseline_before_hit_cameras=int(before),posthit_baseline_plus_alternative_supported_rows=int(covered),minimum_strict_chosen_gain=minimum_gain)
    result=dict(seed=seed,complete=session.complete,report=report,annotation=annotation,records=records,graph_choices=choices,full_stateful_audit=audit,transitions=session.transitions,initial_cameras=session.initial_cameras,graph_sha256=GRAPH_SHA if candidate else None,baseline_sha256=EXPECTED,transport_kind=CONTRACT if candidate else 'visual-transport-v1',feature_file=str(stem)+'-raw.npz',structured_feature_file=str(stem)+'-scenes.npz',wall_seconds=time.monotonic()-began)
    if report['complete']:assert native_count_status(annotation,report)['valid_terminal_counts']
    write(str(stem)+'.json',result);return result


def compare(baseline,candidate):
    return dict(baseline=aggregate([r['annotation'] for r in baseline]),candidate=aggregate([r['annotation'] for r in candidate]),baseline_more=more(baseline),candidate_more=more(candidate),
        sequential_audit_rows=sum(r['full_stateful_audit']['rows'] for r in candidate),changed_actions=sum(r['full_stateful_audit'].get('changed_actions',0) for r in candidate),posthit_supported_rows=sum(r['full_stateful_audit'].get('posthit_baseline_plus_alternative_supported_rows',0) for r in candidate))


def run(approved):
    spec=read(DEST/'spec.json');assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists();assert spec['source']==base.source_signature() and all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    began=time.monotonic();deadline=began+180;write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=180));bank={e['seed']:e for e in read(BANK)['bank']};results=[];summary=dict(status='evaluating',new_transition_cameras=0,new_initial_cameras=0)
    for size in (8,32):
        folder=DEST/f'screen{size}';folder.mkdir();target=spec['screen8' if size==8 else 'screen32'];existing={r['seed']:r for r in results};jobs=[(str(folder),seed,i,deadline-1,True) for i,seed in enumerate(target) if seed not in existing];new=parallel_jobs(native_task,jobs,deadline,progress=folder/'progress.json');results+=new;summary['new_transition_cameras']+=sum(r['transitions'] for r in new);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in new);byseed={r['seed']:r for r in results};complete=all(s in byseed and byseed[s]['complete'] for s in target);candidate=[byseed[s] for s in target if s in byseed and not byseed[s].get('not_started')];baseline=[read(bank[r['seed']]['receipt']) for r in candidate];s=compare(baseline,candidate);s['complete']=complete;write(folder/'summary.json',s);summary[f'screen{size}']=s
        if not complete:summary['status']=f'screen{size}_partial';break
        a=(s['baseline']['own_returns'],s['baseline_more']['later_returns'],s['baseline']['first_incoming_returns'],s['baseline_more']['tail']['2']);b=(s['candidate']['own_returns'],s['candidate_more']['later_returns'],s['candidate']['first_incoming_returns'],s['candidate_more']['tail']['2'])
        if b[0]<=a[0] or b[1]<=a[1] or b[2]!=a[2] or (size==32 and (b[3]<a[3] or s['candidate_more']['later_fraction']<s['baseline_more']['later_fraction'])):summary['status']=f'rejected_screen{size}';break
        if size==32:
            fresh=DEST/'fresh';seeds=read(fresh/'schedules.json')['probe'];jobs=[(str(fresh),seed,i,deadline-1,candidate) for i,seed in enumerate(seeds) for candidate in (False,True)];rs=parallel_jobs(native_task,jobs,deadline,progress=fresh/'progress.json');summary['new_transition_cameras']+=sum(r['transitions'] for r in rs);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in rs);groups={c:{r['seed']:r for r in rs if not r.get('not_started') and (r['graph_sha256'] is not None)==c} for c in (False,True)};complete=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values());fs=dict(complete=complete,groups={str(c):aggregate([r['annotation'] for r in g.values()]) for c,g in groups.items()})
            if complete:
                a=[groups[False][seed] for seed in seeds];b=[groups[True][seed] for seed in seeds];fs.update(compare(a,b),paired=paired_metrics(a,b))
            write(fresh/'summary.json',fs);summary.update(fresh=fs,status='fresh_complete' if complete else 'fresh_partial')
    summary.update(wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),graph_frozen=sha(GRAPH)==GRAPH_SHA,source_frozen=all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items()),baseline_assets_frozen=all(sha(p)==h for p,h in spec['pinned_assets'].items()),no_fit=True,no_automatic_promotion=True);write(DEST/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run']);p.add_argument('--approved-spec-sha256');a=p.parse_args()
    if a.operation=='prepare':prepare()
    else:run(a.approved_spec_sha256)
