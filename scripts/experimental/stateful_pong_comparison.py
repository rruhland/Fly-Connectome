"""Reusable frozen stateful legal-scene policy comparison; no learning/refitting."""
import importlib
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from fly_connectome.control.durable import fingerprint
from pong_fixed_rbf import FreshRBFEvaluator,load_readout
from pong_continuous_visual import canonical_scene,save_scenes
from run_rally_option_memory import ROOT,MODEL,BANK,SCREEN8,EXPECTED,read,write,sha,seed_rngs,parallel_jobs
from run_rally_repertoire_collection import exclusions,select_starts
from run_rally_raw_optimizer import more
from run_rally_raw_fresh_screen import paired_metrics
from return_metrics import ReturnCounter,aggregate,native_count_status
from rally_raw_capture import SparseRows
import run_practical_pong_v2 as base


class StatefulEvaluator(FreshRBFEvaluator):
    def __init__(self,baseline,model,adapter,config):
        self.stateful_model=model;self.adapter=adapter;self.controller=adapter.new_controller(model);self.decisions=[]
        super().__init__(baseline,config);self.initial_learned_hash=self.learned_hash()
    def learned_hash(self):return fingerprint((super().learned_hash(),self.stateful_model.frozen_parameters(),self.adapter.CONTRACT))
    def _begin_episode(self):self.controller=self.adapter.new_controller(self.stateful_model);super()._begin_episode()
    def _choose_policy(self):
        baseline,q=self.rbf_readout.choose(*policy_features(self.latest_encoded,self.scaler));action,choice=self.adapter.choose(self.controller,canonical_scene(self.latest_encoded),baseline,q);self.decisions.append(choice);return action,q
    def step(self,**kwargs):
        r=super().step(**kwargs);self.adapter.observe(self.controller,r);return r


def native_task(adapter_name,model_path,directory,seed,index,deadline,candidate=True):
    directory=Path(directory);began=time.monotonic();seed_rngs();adapter=importlib.import_module(adapter_name);baseline=load_readout(MODEL/'rbf-model.npz');config=PongEvaluationConfig(seed=seed,episodes=1,max_episode_ticks=2400,opponent_speed=.65,policy_scaling=read(MODEL/'radial-cap.json'));model=adapter.load_model(model_path) if candidate else None;session=StatefulEvaluator(baseline,model,adapter,config) if candidate else FreshRBFEvaluator(baseline,config)
    if time.monotonic()>=deadline:return dict(seed=seed,complete=False,not_started=True,transitions=0,initial_cameras=0)
    session._begin_episode();game=session.env.game;old=game.step;counter=ReturnCounter();records=[];raw=SparseRows();scenes=[]
    def instrument(drive,**kwargs):return counter.step(game,old,drive,**kwargs)
    game.step=instrument
    try:
        while not session.complete and time.monotonic()<deadline:
            raw.add(session);scenes.append(canonical_scene(session.latest_encoded));records.append(session.step(check_frozen=False))
    finally:del game.step
    session.verify_frozen();report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds);annotation=counter.summary(game,**{k:last[k] for k in ('truncated','terminated','reward','seconds')});stem=directory/f"{'candidate' if candidate else 'carry'}_{index:02d}";X=raw.matrix();sp.save_npz(str(stem)+'-raw.npz',X);save_scenes(str(stem)+'-scenes.npz',scenes);decisions=session.decisions if candidate else [];audit=dict(passed=True,rows=len(records),new_cameras=0)
    if candidate:
        assert len(decisions)==len(scenes)==len(records)==X.shape[0];controller=adapter.new_controller(model);changes=starts=holds=prehit=0
        for i,(scene,r,expected) in enumerate(zip(scenes,records,decisions)):
            lo,hi=X.indptr[i:i+2];action,q=baseline.choose(X.indices[lo:hi],X.data[lo:hi]);got,choice=adapter.choose(controller,scene,action,q);assert choice==expected and got==r['action'] and np.array_equal(q,np.asarray(r['q']));adapter.observe(controller,r);changes+=got!=action;starts+=choice.get('mode')=='earned_start';holds+=choice.get('mode')=='earned_hold';prehit+=choice.get('mode')=='baseline_before_hit'
        audit.update(changed_actions=int(changes),macro_starts=int(starts),held_cameras=int(holds),baseline_before_hit_cameras=int(prehit))
    result=dict(seed=seed,complete=session.complete,report=report,annotation=annotation,records=records,decisions=decisions,full_stateful_audit=audit,transitions=session.transitions,initial_cameras=session.initial_cameras,model_sha256=sha(model_path) if candidate else None,baseline_sha256=EXPECTED,transport_kind=adapter.CONTRACT if candidate else 'visual-transport-v1',feature_file=str(stem)+'-raw.npz',structured_feature_file=str(stem)+'-scenes.npz',wall_seconds=time.monotonic()-began)
    if report['complete']:assert native_count_status(annotation,report)['valid_terminal_counts']
    write(str(stem)+'.json',result);return result


def prepare(directory,model_path,adapter_name,tag,extra_sources=()):
    directory=Path(directory);model_path=Path(model_path);assert not directory.exists() and model_path.exists();directory.mkdir();(directory/'fresh').mkdir();adapter=importlib.import_module(adapter_name);bank=read(BANK);first=[e['seed'] for e in read(SCREEN8)['bank']];screen=first+[e['seed'] for e in bank['bank'] if e['seed'] not in first][:24];schedule=select_starts(1943906127,tag,exclusions(directory),32);seeds=schedule.pop('TRAIN');schedule.update(probe=seeds[:32],unused_reserved=seeds[32:]);write(directory/'fresh/schedules.json',schedule)
    names=('stateful_pong_comparison.py','pong_fixed_rbf.py','pong_fixed_pairwise.py','pong_continuous_visual.py','run_rally_option_memory.py','pong_option_memory.py','rally_raw_capture.py','return_metrics.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py','run_practical_pong_v2.py')
    sources=[Path(__file__).with_name(n) for n in names]+[Path(adapter.__file__),*map(Path,extra_sources)]
    spec=dict(contract=adapter.CONTRACT,adapter=adapter_name,model=str(model_path),model_sha256=sha(model_path),seconds=180,workers=4,screen8=first,screen32=screen,fresh_tag=tag,
        source=base.source_signature(),files={str(p):sha(p) for p in sources},pinned_assets={str(p):sha(p) for p in (model_path,BANK,SCREEN8,MODEL/'rbf-model.npz',MODEL/'radial-cap.json')},
        qualification='8strict total/latergain andfirstparity;32sameplussecondcount/laterfractionnotworse;conditional32freshpairs',no_fit=True,no_runtime_simulator=True,no_automatic_promotion=True)
    write(directory/'spec.json',spec);return sha(directory/'spec.json')


def comparison(baseline,candidate):
    return dict(baseline=aggregate([r['annotation'] for r in baseline]),candidate=aggregate([r['annotation'] for r in candidate]),baseline_more=more(baseline),candidate_more=more(candidate),sequential_audit_rows=sum(r['full_stateful_audit']['rows'] for r in candidate),changed_actions=sum(r['full_stateful_audit'].get('changed_actions',0) for r in candidate),macro_starts=sum(r['full_stateful_audit'].get('macro_starts',0) for r in candidate),held_cameras=sum(r['full_stateful_audit'].get('held_cameras',0) for r in candidate))


def run(directory,approved):
    directory=Path(directory);spec=read(directory/'spec.json');assert approved==sha(directory/'spec.json') and not (directory/'work-clock.json').exists();assert spec['source']==base.source_signature() and all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    began=time.monotonic();deadline=began+180;write(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=180));bank={e['seed']:e for e in read(BANK)['bank']};results=[];summary=dict(status='evaluating',new_transition_cameras=0,new_initial_cameras=0)
    for size in (8,32):
        folder=directory/f'screen{size}';folder.mkdir();target=spec['screen8' if size==8 else 'screen32'];existing={r['seed']:r for r in results};jobs=[(spec['adapter'],spec['model'],str(folder),seed,i,deadline-1,True) for i,seed in enumerate(target) if seed not in existing];new=parallel_jobs(native_task,jobs,deadline,progress=folder/'progress.json');results+=new;summary['new_transition_cameras']+=sum(r['transitions'] for r in new);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in new);byseed={r['seed']:r for r in results};complete=all(s in byseed and byseed[s]['complete'] for s in target);selected=[byseed[s] for s in target if s in byseed and not byseed[s].get('not_started')];baseline=[read(bank[r['seed']]['receipt']) for r in selected];stats=comparison(baseline,selected);stats['complete']=complete;write(folder/'summary.json',stats);summary[f'screen{size}']=stats
        if not complete:summary['status']=f'screen{size}_partial';break
        a=(stats['baseline']['own_returns'],stats['baseline_more']['later_returns'],stats['baseline']['first_incoming_returns'],stats['baseline_more']['tail']['2']);b=(stats['candidate']['own_returns'],stats['candidate_more']['later_returns'],stats['candidate']['first_incoming_returns'],stats['candidate_more']['tail']['2'])
        if b[0]<=a[0] or b[1]<=a[1] or b[2]!=a[2] or (size==32 and (b[3]<a[3] or stats['candidate_more']['later_fraction']<stats['baseline_more']['later_fraction'])):summary['status']=f'rejected_screen{size}';break
        if size==32:
            fresh=directory/'fresh';seeds=read(fresh/'schedules.json')['probe'];jobs=[(spec['adapter'],spec['model'],str(fresh),seed,i,deadline-1,c) for i,seed in enumerate(seeds) for c in (False,True)];rs=parallel_jobs(native_task,jobs,deadline,progress=fresh/'progress.json');summary['new_transition_cameras']+=sum(r['transitions'] for r in rs);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in rs);groups={c:{r['seed']:r for r in rs if not r.get('not_started') and (r['model_sha256'] is not None)==c} for c in (False,True)};complete=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values());fs=dict(complete=complete,groups={str(c):aggregate([r['annotation'] for r in g.values()]) for c,g in groups.items()})
            if complete:
                a=[groups[False][s] for s in seeds];b=[groups[True][s] for s in seeds];fs.update(comparison(a,b),paired=paired_metrics(a,b))
            write(fresh/'summary.json',fs);summary.update(fresh=fs,status='fresh_complete' if complete else 'fresh_partial')
    summary.update(wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),model_frozen=sha(spec['model'])==spec['model_sha256'],source_frozen=all(sha(p)==h for p,h in spec['files'].items()),baseline_assets_frozen=all(sha(p)==h for p,h in spec['pinned_assets'].items()),no_fit=True,no_automatic_promotion=True);write(directory/'summary.json',summary);print(summary,flush=True)
