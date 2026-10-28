"""One fixed joint-existing-output replay fit with complete earned retention."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from pong_fixed_rbf import load_readout
from pong_fixed_pairwise import save_readout
from pong_joint_output import fit,metric,undo
from run_rally_earned_merge import load_case
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
from run_rally_pairwise_history import CARRY,BASE,case_entry,resolve,key
import run_practical_pong_v2 as base

TARGET=4195433914
SOURCE='fc003082feaacbab12f61f481167a166354f9d2ee4ccd1069948fc8e765a7093'
FILES=('run_rally_joint_output.py','pong_joint_output.py','pong_fixed_pairwise.py','pong_fixed_rbf.py','run_rally_earned_merge.py','run_rally_fixed_bank_check.py','run_rally_guarded_curriculum.py','rally_action_equivalence.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py')


def run(directory,implementation_seconds):
    directory=Path(directory);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+900;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=900,implementation_seconds_separate_estimate=implementation_seconds));bank_path=base.EXPERIMENTS/'pong-rally-TRAIN-bank-20261028-r5/admission.json';bank=base.read(bank_path);assert bank['source']==base.source_signature() and base.sha(bank['current_model_bundle'])==CARRY and base.sha(bank['current_head'])==BASE;carried=load_readout(bank['current_model_bundle']);source_root=base.EXPERIMENTS/'pong-rally-pairwise-raw-reward-20261028-r1/direction_2_minus/full';assert base.sha(source_root/'rbf-model.npz')==SOURCE and base.read(source_root/'completion-verification.json')['passed'];source_path=next(p for p in (source_root/'native_candidate/episodes').glob('*.json') if base.read(p)['seed']==TARGET);source=base.read(source_path);hits=[i for i,v in enumerate(source['records']) if v['hits']];assert source['complete'] and source['weights_sha256']==SOURCE and len(hits)==6 and hits[1]+1==201;target_entry=dict(seed=TARGET,receipt=str(source_path),raw_feature=str(resolve(source_root,source)),end=201,source_head=str(source_root/'proposal.npy'),source_head_sha256=base.sha(source_root/'proposal.npy'),source_model_bundle=str(source_root/'rbf-model.npz'),source_policy_sha256=SOURCE);target=load_case(target_entry);rows=[(e,base.read(e['receipt']),sp.load_npz(e['raw_feature']).tocsr()) for e in bank['bank']];byseed={e['seed']:e for e,r,X in rows};old=base.read(base.EXPERIMENTS/'pong-rally-TRAIN-bank-20261028-r4/admission.json');old_target=next(e for e in old['bank'] if e['seed']==TARGET);assert base.read(old_target['receipt'])['report']['hits']==1 and base.read(byseed[TARGET]['receipt'])['report']['hits']==1;cases=[target];contract=[dict(target_entry,receipt_sha256=base.sha(source_path),role='new_vs_named3ac_andeb_second_return')];memory=[];seen={};full_old_rows=0;replaced_rows=0;acquisition=None
    for e,r,X in rows:
        ends=[i for i,v in enumerate(r['records']) if v['hits']]
        if not ends:continue
        end=ends[-1]+1;memory.append(X[:end]);full_old_rows+=end
        if e['seed']==TARGET:replaced_rows=end;continue
        entry=case_entry(e,bank,end);c=load_case(entry);cases.append(c);contract.append(dict(entry,receipt_sha256=base.sha(e['receipt']),role='complete_current_earned_prefix'))
        if e['seed']==104562292:assert len(ends)==5;acquisition=dict(seed=e['seed'],end=end,hits=5,receipt=e['receipt'])
        for i,a in enumerate(c['actions']):
            k=key(c['X'],i)
            if k in seen:assert seen[k][0]==a
            else:seen[k]=(int(a),e['seed'],i)
    conflicts=[]
    for i,a in enumerate(target['actions']):
        v=seen.get(key(target['X'],i))
        if v and v[0]!=a:conflicts.append(dict(target_row=i,target_action=int(a),guard_action=v[0],guard_seed=v[1],guard_row=v[2]))
    assert len(cases)==174 and full_old_rows==74464 and acquisition is not None;scale_path=base.EXPERIMENTS/'pong-rally-TRAIN-bank-20261028-r3/feature-scale.npy';scale=np.load(scale_path);files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};manifest=dict(source=base.source_signature(),files=files,bank=str(bank_path),bank_sha256=base.sha(bank_path),carried_bundle=bank['current_model_bundle'],carried_sha256=CARRY,reference_base_sha256=BASE,target=target_entry,target_source_sha256=SOURCE,target_full_receipt_hits=6,target_hard_prefix_hits=2,target_hard_prefix_rows=201,novelty_claim='newversus named3ac/eb1-hit baselines, not allhistory',complete_old_earned_rows=full_old_rows,replaced_old_target_rows=replaced_rows,retained_other_cases=173,acquisition=acquisition,seconds=900,fit_seconds=60,existing_trainable_coordinates='4389linear plus512fixedpairwise peraction; zero-initializeddelta aroundcarriedW/R',conditioning='reversibleoptimizeronly blockmean-diagonalRMS normalization overexistingfixedlinearfeaturemetric andexistingRMSPhi; fullcrossscoresecondmoment plus1%ridge',feature_metric=str(scale_path),feature_metric_sha256=base.sha(scale_path),no_architecture_or_feature_change=True,no_sweep_or_retry=True);base.atomic_json(directory/'manifest.json',manifest);base.atomic_json(directory/'hard-contract.json',dict(cases=contract,raw_exact_conflicts=conflicts,current_target_prefix_replaced_not_contradictorily_locked=True,hard_rows=sum(c['X'].shape[0] for c in cases),all_source_receipts_and_Q_validated_before_slicing=True));rng=np.random.default_rng(1938476013);sentinel=[TARGET]
    for phase in ('left','right'):
        n=16-sum(base.leftward(s)==(phase=='left') for s in sentinel);eligible=sorted(s for s in byseed if s not in sentinel and base.leftward(s)==(phase=='left'));sentinel.extend(int(s) for s in rng.choice(eligible,n,replace=False))
    assert len(sentinel)==len(set(sentinel))==32;base.atomic_json(directory/'sentinel-schedule.json',dict(seeds=sentinel,selection='targetplus uniformremainingTRAIN perphase to16/16; RNG1938476013; prefit',TRAIN_not_fresh=True));summary=dict(status='preparing_joint_fit',new_transition_cameras=0,new_initial_cameras=0)
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},carried_bundle_unchanged=base.sha(bank['current_model_bundle'])==CARRY,original3ac_unchanged=base.sha(bank['current_head'])==BASE,bank_unchanged=base.sha(bank_path)==manifest['bank_sha256'],target_source_unchanged=base.sha(source_root/'rbf-model.npz')==SOURCE,no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],cameras=summary['new_transition_cameras'],initials=summary['new_initial_cameras'],wall_seconds=summary['wall_seconds'],source_frozen=summary['source_frozen'])),flush=True)
    if conflicts:summary.update(status='incompatible_exact_raw_contract',conflicts=conflicts);finish();return
    W0,R0=undo(carried,np.zeros((2,4901)),np.ones(4389),1.);assert np.array_equal(W0,carried.base.weights) and np.array_equal(R0,carried.residual);save_readout(directory/'zero-update-model.npz',W0,carried.basis,R0);base.atomic_json(directory/'zero-update-identity.json',dict(passed=True,base_and_residual_exact=True,feature_assets_exact=True,unchanged_readout_arithmetic=True));start=time.monotonic();G,raw_denom,phi_denom,conditioning=metric(memory,carried.basis,scale);L=la.cholesky(G,lower=True,check_finite=False);base.save_array(directory/'score-metric.npy',G);base.save_array(directory/'raw-optimizer-denominator.npy',raw_denom);conditioning.update(phi_optimizer_denominator=phi_denom,score_metric_sha256=base.sha(directory/'score-metric.npy'),raw_optimizer_denominator_sha256=base.sha(directory/'raw-optimizer-denominator.npy'),wall_seconds=time.monotonic()-start,map_deployment_normalization_unchanged=True,undo='newW=carriedW+CONTRAST@theta_raw/raw_denom; newR=carriedR+CONTRAST@theta_phi/phi_denom');base.atomic_json(directory/'conditioning.json',conditioning);print(json.dumps(dict(phase='joint_prepared',hard_rows=sum(len(c['actions']) for c in cases),conditioning=conditioning,elapsed=time.monotonic()-began)),flush=True);normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');result,report=fit(carried,cases,raw_denom,phi_denom,L,normal['kappa'],min(deadline,time.monotonic()+60),CARRY);base.atomic_json(directory/'fit-report.json',report);summary['fit']=report;print(json.dumps(dict(phase='fit',report=report,elapsed=time.monotonic()-began)),flush=True)
    if result is None:summary['status']='no_verified_joint_output_fit';finish();return
    W,R,theta=result;base.save_array(directory/'proposal.npy',W);save_readout(directory/'rbf-model.npz',W,carried.basis,R);base.save_array(directory/'joint-contrast-delta.npy',theta);base.save_array(directory/'residual.npy',R);check_model=load_readout(directory/'rbf-model.npz');assert np.array_equal(check_model.base.weights,np.load(directory/'proposal.npy')) and all(np.array_equal(getattr(check_model.basis,n),getattr(carried.basis,n)) for n in ('hashes','signs','scale','mean','rms'));base.atomic_json(directory/'candidate-coordinates.json',dict(new_linear_weights_sha256=base.sha(directory/'proposal.npy'),new_bundle_sha256=base.sha(directory/'rbf-model.npz'),changed_linear_norm=float(np.linalg.norm(W-carried.base.weights)),changed_residual_norm=float(np.linalg.norm(R-carried.residual)),joint_metric_cost=float(np.sum((theta@G)*theta)),map_assets_unchanged=True,new_W_matches_bundle=True,original3ac_and_eb_unchanged=True));
    def check(name,seeds,cached=None,alternate=None,seconds=150):
        p=directory/(name+'-bank.json');base.atomic_json(p,dict(bank,bank=[byseed[s] for s in seeds]));native_check(directory/name,p,directory/'proposal.npy',cached_native_root=cached,alternate_native_root=alternate,model_bundle=directory/'rbf-model.npz',native_seconds=seconds,practical_later_screen=True,absolute_deadline=deadline);s=base.read(directory/name/'summary.json');summary[name]=s;summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];summary['status']=name+('_complete' if s['status']=='complete' else '_bounded_partial');losses=[];checks=[]
        if s['status']=='complete':
            for receipt_path in (directory/name/'native_candidate/episodes').glob('*.json'):
                r=base.read(receipt_path);expected=2 if r['seed']==TARGET else base.read(byseed[r['seed']]['receipt'])['report']['hits'];checks.append(dict(seed=r['seed'],expected_minimum_returns=expected,actual_returns=r['report']['hits']))
                if r['report']['hits']<expected:losses.append(checks[-1])
        base.atomic_json(directory/name/'earned-outcome-invariants.json',dict(complete=s['status']=='complete',passed=s['status']=='complete' and not losses,checks=checks,violations=losses,interpretation='Any completeperseedloss contradicts exactearnedprefixcertificate/context; not acceptableaggregate tradeoff'))
        if losses:summary['status']='earned_prefix_context_invariant_failed';summary['invariant_violations']=losses;finish();raise AssertionError('saved complete receipts violate earned-prefix outcome invariants')
        return s
    s=check('target',[TARGET],alternate=source_root,seconds=30);summary['actual_new_target_second_return']=s['status']=='complete' and s['candidate']['own_returns']>=2
    if not summary['actual_new_target_second_return']:finish();return
    s=check('sentinel',sentinel,directory/'target');retained=s['candidate']['first_incoming_returns']>=s['baseline']['first_incoming_returns'] and s['candidate_more']['tail']['2']>=s['baseline_more']['tail']['2']
    if not(s['conditional_fresh_eligible'] and retained):finish();return
    whole=check('full',list(byseed),directory/'sentinel',seconds=600);target104=[base.read(p) for p in (directory/'full/native_candidate/episodes').glob('*.json') if base.read(p)['seed']==104562292];skill=bool(target104 and target104[0]['complete'] and target104[0]['report']['hits']>=5);qualify=whole['conditional_fresh_eligible'] and whole['candidate']['first_incoming_returns']>=whole['baseline']['first_incoming_returns'] and whole['candidate_more']['tail']['2']>=whole['baseline_more']['tail']['2'] and skill;summary.update(full_fresh_gate=qualify,original104_five_return_skill_retained=skill)
    if qualify and time.monotonic()<deadline:
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
        collect(list(base.old_engine.FINAL));collect(list(byseed));schedules.ROOT=1943906127;schedules.TAG=62993;drawn=schedules.schedule();seeds=drawn.pop('train');assert not {s%base.MODULUS for s in seeds}&excluded;fresh=directory/'fresh';fresh.mkdir();drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],exclusion_sha256=proof,selection='unusedordinarypairedgames, nofitting/admission');base.atomic_json(fresh/'schedules.json',drawn);fresh_run(fresh,Path(bank['current_model_bundle']).parent,directory/'full',absolute_deadline=deadline);new=base.read(fresh/'summary.json');summary.update(status='fresh_complete' if new['status']=='complete' else 'fresh_bounded_partial',fresh=new);summary['new_transition_cameras']+=new['fresh_transition_cameras'];summary['new_initial_cameras']+=new['initial_cameras']
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--implementation-seconds',type=float,required=True);a=p.parse_args();run(a.directory,a.implementation_seconds)
