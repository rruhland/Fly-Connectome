"""One fixed three-target existing-output replay pass; no new features or retries."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from pong_fixed_rbf import load_readout
from pong_fixed_pairwise import save_readout
from pong_joint_output import fit,undo
from run_rally_earned_merge import load_case
from run_rally_pairwise_history import case_entry,key
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
import run_practical_pong_v2 as base

SCAFFOLD='40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438'
SCAFFOLD_W='479472a37e9950246726cd90babf44f053467e3ae354783da2afbc35080ea15f'
EB='eb2cbcf10aaaf88ee5fd0877cbc966b828286e230a0782bd91c4dce9ddce9152'
SOURCE='fc003082feaacbab12f61f481167a166354f9d2ee4ccd1069948fc8e765a7093'
TARGETS={2667357792:(2,205),478956070:(2,321),1212947804:(3,375)}
METRIC='95206441276a758eb9abba5dcdb67eb9a1fb5df949957a4262f45941a9eb6a5e'
DENOM='fd284e9b14e05001ffcb66d7e295400eabb8f42f590e8f3dfb1b1a10b9334a1f'
FILES=('run_rally_joint_batch.py','pong_joint_output.py','pong_fixed_pairwise.py','pong_fixed_rbf.py','run_rally_earned_merge.py','run_rally_pairwise_history.py','run_rally_fixed_bank_check.py','run_rally_guarded_curriculum.py','rally_action_equivalence.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py')


def scaffold_bank(path,previous_path,scaffold):
    """Refresh the existing192 contexts only, keeping changed W and policy distinct."""
    assert not path.exists();old=base.read(previous_path);verification=scaffold/'completion-verification.json';assert base.read(verification)['passed']
    assert base.sha(scaffold/'proposal.npy')==SCAFFOLD_W and base.sha(scaffold/'rbf-model.npz')==SCAFFOLD
    entries=[];rows={}
    for p in sorted((scaffold/'native_candidate/episodes').glob('*.json')):
        r=base.read(p);raw=Path(r['feature_file']);raw=raw if raw.is_absolute() else scaffold/raw
        assert r['complete'] and r['weights_sha256']==SCAFFOLD and r['base_weights_sha256']==SCAFFOLD_W and raw.exists()
        entries.append(dict(seed=r['seed'],receipt=str(p),raw_feature=str(raw),source_root=str(scaffold),provenance='refreshed_existing_TRAIN192_no_new_fresh_admission'));rows[r['seed']]=r
    assert len(entries)==len(rows)==192 and set(rows)=={e['seed'] for e in old['bank']}
    assert sum(r['report']['hits'] for r in rows.values())==523 and sum(r['report']['hits']>0 for r in rows.values())==175 and sum(r['report']['hits']>=2 for r in rows.values())==123
    bank=dict(current_head=str(scaffold/'proposal.npy'),current_head_sha256=SCAFFOLD_W,current_model_bundle=str(scaffold/'rbf-model.npz'),current_policy_sha256=SCAFFOLD,current_model_kind='fixed-centered-tensorsketch2-v1',requires_policy_aware_helpers=True,radial_cap=str(scaffold/'radial-cap.json'),source=base.source_signature(),bank=entries,previous_bank=str(previous_path),previous_bank_sha256=base.sha(previous_path),source_completion_verification=str(verification),source_completion_verification_sha256=base.sha(verification),refreshed_wall_ns=time.time_ns(),fresh_admitted=False,no_new_fresh_admission=True,previously_admitted_contexts_retained=True,qualification62993_remains_evaluation_only=True,expected=dict(episodes=192,returns=523,first=175,later=348,at_least2=123),carry_role='TRAIN-proven scaffold, not fresh-qualified replacement')
    path.parent.mkdir();base.atomic_json(path,bank);return bank,rows


def run(directory,implementation_seconds):
    directory=Path(directory);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+900
    base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=900,implementation_seconds_separate_estimate=implementation_seconds))
    previous=base.EXPERIMENTS/'pong-rally-joint-output-20261028-r1';scaffold=previous/'full';eb=base.EXPERIMENTS/'pong-rally-pairwise-residual-20261028-r1/full';source_root=base.EXPERIMENTS/'pong-rally-pairwise-raw-reward-20261028-r1/direction_2_minus/full'
    assert base.sha(eb/'rbf-model.npz')==EB and base.sha(source_root/'rbf-model.npz')==SOURCE and base.read(source_root/'completion-verification.json')['passed']
    bank_path=base.EXPERIMENTS/'pong-rally-TRAIN-bank-20261028-r6/admission.json';bank,receipts=scaffold_bank(bank_path,base.EXPERIMENTS/'pong-rally-TRAIN-bank-20261028-r5/admission.json',scaffold);carried=load_readout(bank['current_model_bundle']);byseed={e['seed']:e for e in bank['bank']}
    files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};metric_path=previous/'score-metric.npy';denom_path=previous/'raw-optimizer-denominator.npy';assert base.sha(metric_path)==METRIC and base.sha(denom_path)==DENOM
    metric_info=base.read(previous/'conditioning.json');phi_denom=metric_info['phi_optimizer_denominator'];G=np.load(metric_path,mmap_mode='r');raw_denom=np.load(denom_path);assert G.shape==(4901,4901) and raw_denom.shape==(4389,)
    manifest=dict(source=base.source_signature(),files=files,bank=str(bank_path),bank_sha256=base.sha(bank_path),carried_sha256=SCAFFOLD,carried_base_sha256=SCAFFOLD_W,source_policy_sha256=SOURCE,seconds=900,fit_seconds=60,targets={str(s):dict(earned_returns=v[0],prefix_rows=v[1]) for s,v in TARGETS.items()},score_metric=str(metric_path),score_metric_sha256=METRIC,raw_optimizer_denominator=str(denom_path),raw_optimizer_denominator_sha256=DENOM,phi_optimizer_denominator=phi_denom,metric_reused_without_copy_or_recalibration=True,fresh_root_seed=1943906127,fresh_tag=63009,no_architecture_or_feature_change=True,no_sweep_or_retry=True,qualification62993_not_admitted=True)
    base.atomic_json(directory/'manifest.json',manifest);summary=dict(status='preparing_fixed_batch',new_transition_cameras=0,new_initial_cameras=0)
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},scaffold_unchanged=base.sha(scaffold/'rbf-model.npz')==SCAFFOLD and base.sha(scaffold/'proposal.npy')==SCAFFOLD_W,eb_unchanged=base.sha(eb/'rbf-model.npz')==EB,source_unchanged=base.sha(source_root/'rbf-model.npz')==SOURCE,metric_unchanged=base.sha(metric_path)==METRIC and base.sha(denom_path)==DENOM,bank_unchanged=base.sha(bank_path)==manifest['bank_sha256'],no_automatic_promotion=True)
        base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],cameras=summary['new_transition_cameras'],initials=summary['new_initial_cameras'],wall_seconds=summary['wall_seconds'],source_frozen=summary['source_frozen'])),flush=True)
    sources={base.read(p)['seed']:p for p in (source_root/'native_candidate/episodes').glob('*.json')};cases=[];contract=[];seen={};old_rows=0;replaced=0
    for e in bank['bank']:
        s=e['seed'];r=receipts[s];hits=[i for i,v in enumerate(r['records']) if v['hits']]
        if not hits:continue
        end=hits[-1]+1;old_rows+=end
        if s in TARGETS:replaced+=end;continue
        entry=case_entry(e,bank,end);c=load_case(entry);cases.append(c);contract.append(dict(entry,role='complete_scaffold_earned_prefix'))
        for i,a in enumerate(c['actions']):
            k=key(c['X'],i);old=seen.setdefault(k,(int(a),s,i));assert old[0]==a
    conflicts=[]
    for s,(required,end) in TARGETS.items():
        p=sources[s];r=base.read(p);hits=[i for i,v in enumerate(r['records']) if v['hits']];assert len(hits)>receipts[s]['report']['hits'] and hits[required-1]+1==end
        raw=Path(r['feature_file']);raw=raw if raw.is_absolute() else source_root/raw;entry=dict(seed=s,receipt=str(p),receipt_sha256=base.sha(p),raw_feature=str(raw),end=end,source_head=str(source_root/'proposal.npy'),source_head_sha256=base.sha(source_root/'proposal.npy'),source_model_bundle=str(source_root/'rbf-model.npz'),source_policy_sha256=SOURCE);c=load_case(entry);cases.append(c);contract.append(dict(entry,role='improved_ownreward_target_prefix',full_source_returns=len(hits),hard_prefix_earned_returns=required,replaced_old_earned_returns=receipts[s]['report']['hits']))
        for i,a in enumerate(c['actions']):
            k=key(c['X'],i);old=seen.get(k)
            if old and old[0]!=a:conflicts.append(dict(seed=s,row=i,action=int(a),guard_seed=old[1],guard_row=old[2],guard_action=old[0]))
            elif not old:seen[k]=(int(a),s,i)
    assert old_rows==80333 and replaced==395 and len(cases)==175 and sum(len(c['actions']) for c in cases)==80839
    base.atomic_json(directory/'hard-contract.json',dict(cases=contract,raw_exact_conflicts=conflicts,hard_rows=80839,old_earned_rows=old_rows,replaced_rows=replaced,source_validation_before_slicing=True,targets_old_prefixes_replaced=True))
    # Fixed phase/depth strata; all target cases included before outcome collection.
    rng=np.random.default_rng(1938476049);sentinel=list(TARGETS)
    for phase in ('left','right'):
        for depth in range(4):
            n=max(0,4-sum(base.leftward(s)==(phase=='left') and min(receipts[s]['report']['hits'],3)==depth for s in sentinel));eligible=sorted(s for s in byseed if s not in sentinel and base.leftward(s)==(phase=='left') and min(receipts[s]['report']['hits'],3)==depth);sentinel.extend(int(s) for s in rng.choice(eligible,min(n,len(eligible)),replace=False))
        n=16-sum(base.leftward(s)==(phase=='left') for s in sentinel);eligible=sorted(s for s in byseed if s not in sentinel and base.leftward(s)==(phase=='left'));sentinel.extend(int(s) for s in rng.choice(eligible,n,replace=False))
    assert len(sentinel)==len(set(sentinel))==32
    base.atomic_json(directory/'sentinel-schedule.json',dict(seeds=sentinel,selection='three targets plus fixed phase/depth strata0,1,2,3+ to4each where available; uniform remaining phase fill; RNG1938476049; prefit',TRAIN_not_fresh=True))
    if conflicts:summary['status']='incompatible_exact_raw_contract';finish();return
    W0,R0=undo(carried,np.zeros((2,4901)),raw_denom,phi_denom);assert np.array_equal(W0,carried.base.weights) and np.array_equal(R0,carried.residual)
    base.atomic_json(directory/'zero-update-identity.json',dict(passed=True,nonzero_carry_W_and_R_exact=True,same_deployed_arithmetic=True));L=la.cholesky(G,lower=True,check_finite=False);normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');print(json.dumps(dict(phase='batch_prepared',hard_rows=80839,elapsed=time.monotonic()-began)),flush=True)
    result,report=fit(carried,cases,raw_denom,phi_denom,L,normal['kappa'],min(deadline,time.monotonic()+60),SCAFFOLD);base.atomic_json(directory/'fit-report.json',report);summary['fit']=report;print(json.dumps(dict(phase='fit',report=report,elapsed=time.monotonic()-began)),flush=True)
    if result is None:summary['status']='no_verified_joint_batch_fit';finish();return
    W,R,theta=result;base.save_array(directory/'proposal.npy',W);save_readout(directory/'rbf-model.npz',W,carried.basis,R);base.save_array(directory/'joint-contrast-delta.npy',theta)
    check_model=load_readout(directory/'rbf-model.npz');assert np.array_equal(check_model.base.weights,np.load(directory/'proposal.npy')) and all(np.array_equal(getattr(check_model.basis,n),getattr(carried.basis,n)) for n in ('hashes','signs','scale','mean','rms'))
    base.atomic_json(directory/'candidate-coordinates.json',dict(new_base_weights_sha256=base.sha(directory/'proposal.npy'),new_bundle_sha256=base.sha(directory/'rbf-model.npz'),changed_W_norm=float(np.linalg.norm(W-carried.base.weights)),changed_R_norm=float(np.linalg.norm(R-carried.residual)),delta_metric_cost=float(np.sum((theta@G)*theta)),map_assets_unchanged=True,reference_is_nonzero_scaffold_W_and_R=True))
    def check(name,seeds,cached=None,alternate=None,seconds=150):
        p=directory/(name+'-bank.json');base.atomic_json(p,dict(bank,bank=[byseed[s] for s in seeds]));native_check(directory/name,p,directory/'proposal.npy',cached_native_root=cached,alternate_native_root=alternate,model_bundle=directory/'rbf-model.npz',native_seconds=seconds,practical_later_screen=True,absolute_deadline=deadline);s=base.read(directory/name/'summary.json');summary[name]=s;summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];summary['status']=name+('_complete' if s['status']=='complete' else '_bounded_partial');checks=[]
        for path in (directory/name/'native_candidate/episodes').glob('*.json'):
            r=base.read(path);minimum=TARGETS[r['seed']][0] if r['seed'] in TARGETS else receipts[r['seed']]['report']['hits'];checks.append(dict(seed=r['seed'],complete=r['complete'],expected_minimum_returns=minimum,actual_returns=r['report']['hits']))
        losses=[v for v in checks if v['complete'] and v['actual_returns']<v['expected_minimum_returns']];base.atomic_json(directory/name/'earned-outcome-invariants.json',dict(complete=s['status']=='complete',passed=s['status']=='complete' and not losses,checks=checks,violations=losses))
        if losses:summary.update(status='earned_prefix_context_invariant_failed',invariant_violations=losses);finish();raise AssertionError('saved receipt violates certified earned outcome')
        return s
    actual=check('targets',list(TARGETS),alternate=source_root,seconds=60)
    if actual['status']!='complete':finish();return
    s=check('sentinel',sentinel,directory/'targets')
    if not s['conditional_fresh_eligible']:finish();return
    full=check('full',list(byseed),directory/'sentinel',eb,600);summary['full_fresh_gate']=full['conditional_fresh_eligible'] and full['candidate']['first_incoming_returns']>=full['baseline']['first_incoming_returns'] and full['candidate_more']['tail']['2']>=full['baseline_more']['tail']['2']
    if summary['full_fresh_gate'] and time.monotonic()<deadline:
        import run_rally_baseline_pong as schedules
        excluded=set();proof={}
        def collect(v):
            if type(v) is int and 0<v<2**32:excluded.add(v%base.MODULUS)
            elif isinstance(v,list):
                for a in v:collect(a)
            elif isinstance(v,dict):
                for a in v.values():collect(a)
        for p in sorted(base.EXPERIMENTS.rglob('*schedule*.json')):collect(base.read(p));proof[str(p)]=base.sha(p)
        for p in sorted(base.EXPERIMENTS.rglob('seed-exclusions.json')):collect(base.read(p).get('ids',[]));proof[str(p)]=base.sha(p)
        collect(list(base.old_engine.FINAL));collect(list(byseed));schedules.ROOT=1943906127;schedules.TAG=63009;drawn=schedules.schedule();seeds=drawn.pop('train');assert not {s%base.MODULUS for s in seeds}&excluded;folder=directory/'fresh';folder.mkdir();drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],exclusion_sha256=proof,selection='new32shared ordinary games for eb,40069scaffold,andcandidate; no fitting/admission');base.atomic_json(folder/'schedules.json',drawn)
        fresh_run(folder,eb,directory/'full',absolute_deadline=deadline,other_baselines={'scaffold':scaffold});new=base.read(folder/'summary.json');summary.update(status='fresh_complete' if new['status']=='complete' else 'fresh_bounded_partial',fresh=new);summary['new_transition_cameras']+=new['fresh_transition_cameras'];summary['new_initial_cameras']+=new['initial_cameras']
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--implementation-seconds',type=float,required=True);a=p.parse_args();run(a.directory,a.implementation_seconds)
