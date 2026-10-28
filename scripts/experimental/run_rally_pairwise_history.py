"""One fixed-map output-delta replay pass targeting a retained second return."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pong_fixed_rbf import load_readout
from pong_fixed_pairwise import FrozenPairwiseReadout,save_readout
from run_rally_rbf_residual import fit_residual
from run_rally_earned_merge import load_case
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
import run_practical_pong_v2 as base

CARRY='eb2cbcf10aaaf88ee5fd0877cbc966b828286e230a0782bd91c4dce9ddce9152'
BASE='3ac7f87fbac23abab14e633b38cfe1e50986a8f03bb38b049bed8f4cc840bd5e'
FILES=('run_rally_pairwise_history.py','run_rally_rbf_residual.py','run_rally_earned_merge.py','pong_fixed_pairwise.py','pong_fixed_rbf.py','rally_action_equivalence.py','run_rally_guarded_curriculum.py','run_rally_fixed_bank_check.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py')


def resolve(root,r):
    p=Path(r['feature_file']);return p if p.is_absolute() else root/p


def admit(path,old_path,carry_root,fresh_root):
    assert not path.exists();path.parent.mkdir();old=base.read(old_path);authorization=fresh_root/'parent-admission-authorization.json';auth=base.read(authorization);assert auth['status']=='parent_authorized_after_completed_frozen_evaluation' and base.sha(auth['source_fresh_summary'])==auth['source_fresh_summary_sha256'] and base.sha(auth['source_schedule'])==auth['source_schedule_sha256'];entries=[];receipts=[]
    for source,role in [(carry_root,'refreshed_existing_TRAIN160'),(fresh_root/'paired32/candidate','explicitly_admitted_completed_validation32')]:
        for p in sorted((source/'native_candidate/episodes').glob('*.json')):
            r=base.read(p);assert r['complete'] and r['weights_sha256']==CARRY;entries.append(dict(seed=r['seed'],receipt=str(p),raw_feature=str(resolve(source,r)),source_root=str(source),provenance=role));receipts.append(r)
    assert len(entries)==192 and len({e['seed'] for e in entries})==192 and {e['seed'] for e in entries[160:]}==set(auth['seeds']);assert sum(r['report']['hits'] for r in receipts)==493 and sum(r['report']['hits']>0 for r in receipts)==174 and sum(r['report']['hits']>=2 for r in receipts)==113
    bank=dict(current_head=str(carry_root/'proposal.npy'),current_head_sha256=BASE,current_model_bundle=str(carry_root/'rbf-model.npz'),current_policy_sha256=CARRY,current_model_kind='fixed-centered-tensorsketch2-v1',requires_policy_aware_helpers=True,radial_cap=str(carry_root/'radial-cap.json'),source=base.source_signature(),bank=entries,previous_bank=str(old_path),previous_bank_sha256=base.sha(old_path),admitted_wall_ns=time.time_ns(),fresh_admitted=True,explicit_admitted32=auth['seeds'],admission_authorization=str(authorization),admission_authorization_sha256=base.sha(authorization),qualification_now_consumed_as_TRAIN=True,future_qualification_requires_new_seeds=True,no_feature_recalibration=True,expected=dict(episodes=192,returns=493,first=174,later=319,at_least2=113));assert base.sha(bank['current_head'])==BASE and base.sha(bank['current_model_bundle'])==CARRY;base.atomic_json(path,bank);return bank


def case_entry(e,bank,end):
    return dict(e,end=end,source_head=bank['current_head'],source_head_sha256=bank['current_head_sha256'],**(dict(source_model_bundle=bank['current_model_bundle'],source_policy_sha256=bank['current_policy_sha256']) if bank.get('current_model_bundle') else {}))


def key(X,i):
    lo,hi=X.indptr[i:i+2];return X.indices[lo:hi].astype(np.int64).tobytes(),X.data[lo:hi].astype(np.float64).tobytes()


def select_history(bank,old,rows):
    guards=[];aliases={};candidates=[]
    for e,r,X in rows:
        hits=[i for i,v in enumerate(r['records']) if v['hits']]
        if hits:
            c=load_case(case_entry(e,bank,hits[min(1,len(hits)-1)]+1));guards.append(c)
            for i,a in enumerate(c['actions']):
                k=key(c['X'],i)
                if k in aliases:assert aliases[k][0]==a
                else:aliases[k]=(int(a),e['seed'],i)
    for phase in ('left','right'):
        chosen=sorted([(e,r) for e,r,X in rows if r['report']['hits']==1 and r['report']['losses']==1 and base.leftward(e['seed'])==(phase=='left')],key=lambda v:v[0]['seed'])[:2]
        for e,r in chosen:
            oldentry=next((v for v in old['bank'] if v['seed']==e['seed']),None)
            if oldentry is None:candidates.append(dict(seed=e['seed'],phase=phase,eligible=False,reason='no_preserved3ac_receipt'));continue
            prior=base.read(oldentry['receipt']);hits=[i for i,v in enumerate(prior['records']) if v['hits']]
            if len(hits)<2:candidates.append(dict(seed=e['seed'],phase=phase,eligible=False,reason='historical_receipt_has_no_second_return',historical_hits=len(hits)));continue
            entry=case_entry(oldentry,old,hits[1]+1);c=load_case(entry);conflicts=[]
            for i,a in enumerate(c['actions']):
                previous=aliases.get(key(c['X'],i))
                if previous and previous[0]!=a:conflicts.append(dict(historical_row=i,historical_action=int(a),guard_action=previous[0],guard_seed=previous[1],guard_row=previous[2]))
            candidates.append(dict(seed=e['seed'],phase=phase,historical_hits=len(hits),current_hits=1,second_prefix_rows=len(c['actions']),conflicts=conflicts,eligible=not conflicts,entry=entry))
    eligible=[v for v in candidates if v['eligible']];chosen=min(eligible,key=lambda v:(v['second_prefix_rows'],v['seed'])) if eligible else None
    return guards,chosen,dict(candidates=candidates,selected_seed=chosen['seed'] if chosen else None,selection='two current1-hit terminal losses per phase by seed; shortest exactraw-compatible historical second prefix then seed; fixed beforefit',recovery_not_novel_acquisition=True,guard_cases=len(guards),guard_rows=sum(len(c['actions']) for c in guards),unique_guard_raw_rows=len(aliases),new_observations=0)


def run(directory,implementation_seconds):
    directory=Path(directory);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+900;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=900,implementation_seconds_separate=implementation_seconds,implementation_seconds_estimated=True));carry_root=base.EXPERIMENTS/'pong-rally-pairwise-residual-20261028-r1/full';fresh_root=base.EXPERIMENTS/'pong-rally-pairwise-frozen-fresh-20261028-r1';old_path=base.EXPERIMENTS/'pong-rally-TRAIN-bank-20261028-r4/admission.json';bank_path=base.EXPERIMENTS/'pong-rally-TRAIN-bank-20261028-r5/admission.json';bank=admit(bank_path,old_path,carry_root,fresh_root);old=base.read(old_path);carried=load_readout(bank['current_model_bundle']);basis=carried.basis;W=carried.base.weights;assert basis.output_dim==512;(directory/'proposal.npy').write_bytes(Path(bank['current_head']).read_bytes());(directory/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes());files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};manifest=dict(source=base.source_signature(),files=files,bank_path=str(bank_path),bank_sha256=base.sha(bank_path),prior_bank_sha256=base.sha(old_path),carried_bundle=str(bank['current_model_bundle']),carried_sha256=CARRY,base_sha256=BASE,seconds=900,fit_seconds=60,fixed_map_no_recalibration=True,one_historical_replay_fit=True,no_discovery_fallback=True,no_parameter_or_guard_sweep=True);base.atomic_json(directory/'manifest.json',manifest);summary=dict(status='preparing_historical_replay',new_transition_cameras=0,new_initial_cameras=0)
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},carry_unchanged=base.sha(bank['current_model_bundle'])==CARRY,base_unchanged=base.sha(bank['current_head'])==BASE,prior_bank_unchanged=base.sha(old_path)==manifest['prior_bank_sha256'],no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],cameras=summary['new_transition_cameras'],initials=summary['new_initial_cameras'],wall_seconds=summary['wall_seconds'],source_frozen=summary['source_frozen'])),flush=True)
    rows=[(e,base.read(e['receipt']),sp.load_npz(e['raw_feature']).tocsr()) for e in bank['bank']];guards,selected,screen=select_history(bank,old,rows);base.atomic_json(directory/'historical-selection.json',screen);print(json.dumps(dict(phase='historical_selection',screen=screen,elapsed=time.monotonic()-began)),flush=True)
    historical_native_root=Path(old['current_head']).parent
    if not list((historical_native_root/'native_candidate/episodes').glob('*.json')):historical_native_root=historical_native_root/'full'
    assert len(list((historical_native_root/'native_candidate/episodes').glob('*.json')))==160 and base.sha(historical_native_root/'proposal.npy')==BASE and base.read(historical_native_root/'manifest.json')['source']==base.source_signature() and base.read(historical_native_root/'completion-verification.json')['passed'];base.atomic_json(directory/'historical-native-source.json',dict(root=str(historical_native_root),head_sha256=BASE,complete_receipts=160,core_source_matches=True,new_observations=0))
    if selected is None:summary['status']='no_compatible_historical_second_prefix';finish();return
    target=load_case(selected['entry']);cases=[target]+guards;sentinel=[v['seed'] for v in screen['candidates']];byseed={e['seed']:(e,r) for e,r,X in rows}
    while len(sentinel)<32:
        phase=min(('left','right'),key=lambda p:sum(base.leftward(s)==(p=='left') for s in sentinel));remaining=[s for s in byseed if s not in sentinel and base.leftward(s)==(phase=='left')];band=(len(sentinel)-4)%4;group=[s for s in remaining if min(byseed[s][1]['report']['hits'],3)==band];sentinel.append(min(group or remaining))
    base.atomic_json(directory/'sentinel-schedule.json',dict(seeds=sentinel,selection='four predeclared one-hit losses plus phase/depth balanced TRAIN; fixed beforefit',TRAIN_not_holdout=True));G=np.zeros((512,512));episodes=0
    for e,r,X in rows:
        hits=[i for i,v in enumerate(r['records']) if v['hits']]
        if not hits or e['seed']==selected['seed']:continue
        Phi=basis.batch(X[:hits[-1]+1]);G+=Phi.T@Phi/len(Phi);episodes+=1
    assert episodes==173;G/=episodes;normalizer=float(np.trace(G)/512);G/=normalizer;G.flat[::513]+=.01;normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');R,report=fit_residual(W,basis,cases,G,normal['kappa'],min(deadline,time.monotonic()+60),FrozenPairwiseReadout,carried);base.atomic_json(directory/'fit-report.json',report);summary['fit']=report;print(json.dumps(dict(phase='fit',report=report,elapsed=time.monotonic()-began)),flush=True)
    if R is None:summary['status']='no_verified_output_delta_fit';finish();return
    save_readout(directory/'rbf-model.npz',W,basis,R);base.save_array(directory/'residual.npy',R);delta=R-carried.residual;base.save_array(directory/'residual-delta.npy',delta);base.atomic_json(directory/'score-objective.json',dict(episodes=episodes,mean_diagonal_normalizer=normalizer,ridge_fraction=.01,delta_objective=float(np.sum((delta@G)*delta)),absolute_R_not_minimized=True,map_assets_unchanged=all(np.array_equal(getattr(load_readout(directory/'rbf-model.npz').basis,n),getattr(basis,n)) for n in ('hashes','signs','scale','mean','rms')),new_observations=0))
    def check(name,seeds,cached=None,seconds=500):
        entries=[byseed[s][0] for s in seeds];p=directory/(name+'-bank.json');base.atomic_json(p,dict(bank,bank=entries));native_check(directory/name,p,directory/'proposal.npy',cached_native_root=cached,alternate_native_root=historical_native_root,model_bundle=directory/'rbf-model.npz',native_seconds=seconds,practical_later_screen=True,absolute_deadline=deadline);s=base.read(directory/name/'summary.json');summary[name]=s;summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];summary['status']=name+('_complete' if s['status']=='complete' else '_bounded_partial');return s
    actual=check('target',[selected['seed']],seconds=30);summary['actual_target_second_return']=actual['status']=='complete' and actual['candidate']['own_returns']>=2
    if not summary['actual_target_second_return']:summary['status']='no_actual_target_second_return' if actual['status']=='complete' else 'target_bounded_partial';finish();return
    s=check('sentinel',sentinel,directory/'target',150);first_ok=s['candidate']['first_incoming_returns']>=s['baseline']['first_incoming_returns'];second_ok=s['candidate_more']['tail']['2']>=s['baseline_more']['tail']['2'];summary['sentinel_retention_gate']=first_ok and second_ok
    if s['status']!='complete' or not summary['sentinel_retention_gate']:finish();return
    summary['sentinel_broad_collapse']=s['candidate']['own_returns']<.8*s['baseline']['own_returns'] and s['candidate_more']['later_returns']<.8*s['baseline_more']['later_returns']
    if summary['sentinel_broad_collapse']:summary['status']='sentinel_broad_collapse_parent_review';finish();return
    whole=check('full',[e['seed'] for e in bank['bank']],directory/'sentinel',600);qualify=whole['conditional_fresh_eligible'] and whole['candidate']['first_incoming_returns']>=whole['baseline']['first_incoming_returns'] and whole['candidate_more']['tail']['2']>whole['baseline_more']['tail']['2'];summary['full_second_return_qualification']=qualify
    if qualify and time.monotonic()<deadline:
        import run_rally_baseline_pong as schedules
        prior=set();proof={}
        def collect(v):
            if type(v) is int and 0<v<2**32:prior.add(v%base.MODULUS)
            elif isinstance(v,list):
                for a in v:collect(a)
            elif isinstance(v,dict):
                for a in v.values():collect(a)
        for p in sorted(base.EXPERIMENTS.rglob('*schedule*.json')):collect(base.read(p));proof[str(p)]=base.sha(p)
        for p in sorted(base.EXPERIMENTS.rglob('seed-exclusions.json')):collect(base.read(p).get('ids',[]));proof[str(p)]=base.sha(p)
        collect(list(base.old_engine.FINAL));collect([e['seed'] for e in bank['bank']]);schedules.ROOT=1943906127;schedules.TAG=62961;drawn=schedules.schedule();seeds=drawn.pop('train');assert not {s%base.MODULUS for s in seeds}&prior;folder=directory/'fresh';folder.mkdir();drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],exclusion_sha256=proof,selection='new paired scenes after32previous validation receipts explicitly admittedTRAIN; no fitting/admission');base.atomic_json(folder/'schedules.json',drawn);fresh_run(folder,carry_root,directory/'full',absolute_deadline=deadline);new=base.read(folder/'summary.json');summary.update(status='fresh_complete' if new['status']=='complete' else 'fresh_bounded_partial',fresh=new);summary['new_transition_cameras']+=new['fresh_transition_cameras'];summary['new_initial_cameras']+=new['initial_cameras']
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--implementation-seconds',type=float,required=True);a=p.parse_args();run(a.directory,a.implementation_seconds)
