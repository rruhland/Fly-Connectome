"""One paired-divergence burst-credit fit with first-return and outcome retention."""
import argparse,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import scipy.linalg as la
from pong_causal_burst import task,audited_cases
from pong_fixed_rbf import load_readout
from pong_fixed_pairwise import save_readout
from pong_joint_output import fit
from run_rally_earned_merge import load_case
from run_rally_pairwise_history import case_entry,key
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
from run_rally_repertoire_collection import exclusions,select_starts
import run_practical_pong_v2 as base

FILES=('run_rally_causal_burst.py','pong_causal_burst.py','pong_joint_output.py','pong_fixed_rbf.py','pong_fixed_pairwise.py','pong_action_credit.py','run_rally_earned_merge.py','run_rally_raw_optimizer.py','run_rally_fixed_bank_check.py','run_rally_guarded_curriculum.py','rally_action_equivalence.py','run_rally_repertoire_collection.py','run_rally_raw_fresh_screen.py')


def conflicts(cases):
    seen={};bad=[]
    for c in cases:
        for i,a in enumerate(c['actions']):
            k=key(c['X'],i);old=seen.get(k)
            if old and old[0]!=a:bad.append(dict(seed=c['seed'],row=i,action=int(a),other_seed=old[1],other_row=old[2],other_action=old[0]))
            elif old is None:seen[k]=(int(a),c['seed'],i)
    return bad


def run(directory,spec_path):
    directory=Path(directory);spec_path=Path(spec_path);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+900;spec=base.read(spec_path);base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=900,implementation_seconds_separate_estimate=spec['implementation_seconds_separate_estimate']));(directory/'spec.json').write_bytes(spec_path.read_bytes());files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};source=base.source_signature();bank=base.read(spec['bank']);assert base.sha(spec['bank'])==spec['bank_sha256'];current_root=Path(spec['current_root']);donor_root=Path(spec['donor_root']);assert base.sha(current_root/'rbf-model.npz')==spec['current_policy_sha256'] and base.sha(current_root/'proposal.npy')==spec['current_W_sha256'] and base.sha(donor_root/'rbf-model.npz')==spec['donor_policy_sha256'] and base.sha(donor_root/'proposal.npy')==spec['donor_W_sha256'];assert base.read(current_root/'radial-cap.json')==base.read(donor_root/'radial-cap.json')
    for p,digest in spec['source_asset_hashes'].items():assert base.sha(p)==digest
    for entry in spec['cases']:
        for p,digest in entry['source_file_hashes'].items():assert base.sha(p)==digest
    for src,name in [(current_root/'rbf-model.npz','source-model.npz'),(donor_root/'rbf-model.npz','donor-model.npz')]:(directory/name).write_bytes(src.read_bytes())
    current=load_readout(directory/'source-model.npz');donor=load_readout(directory/'donor-model.npz');assert all(np.array_equal(getattr(current.basis,n),getattr(donor.basis,n)) for n in ('hashes','signs','scale','mean','rms'));summary=dict(status='acquiring_causal_patterns',new_transition_cameras=0,new_initial_cameras=0);manifest=dict(source=source,files=files,spec_sha256=base.sha(spec_path),bank_sha256=base.sha(spec['bank']),existing_own_reward_credit_not_new_algorithm=True,same_fixed_architecture=True,no_sweep_or_fit_retry=True)
    base.atomic_json(directory/'manifest.json',manifest)
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=source==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},models_unchanged=base.sha(directory/'source-model.npz')==spec['current_policy_sha256'] and base.sha(directory/'donor-model.npz')==spec['donor_policy_sha256'],source_data_and_assets_unchanged=all(base.sha(p)==d for p,d in spec['source_asset_hashes'].items()) and all(base.sha(p)==d for e in spec['cases'] for p,d in e['source_file_hashes'].items()),metric_unchanged=all(base.sha(p)==d for p,d in spec['metric_files'].items()),bank_unchanged=base.sha(spec['bank'])==spec['bank_sha256'],spec_unchanged=base.sha(spec_path)==manifest['spec_sha256'],no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],cameras=summary['new_transition_cameras'],initials=summary['new_initial_cameras'],wall_seconds=summary['wall_seconds'])),flush=True)
    fresh=directory/'fresh';fresh.mkdir();drawn=select_starts(spec['fresh_root'],spec['fresh_tag'],exclusions(directory),32);seeds=drawn.pop('TRAIN');drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],purpose='new paired qualification, no admission');base.atomic_json(fresh/'schedules.json',drawn)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        futures=[pool.submit(task,str(directory),c,i,min(deadline,time.monotonic()+210)) for i,c in enumerate(spec['cases'])];results=[f.result() for f in futures]
    summary['acquisition']=results;summary['new_transition_cameras']+=sum(r['transition_cameras'] for r in results);summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in results);base.atomic_json(directory/'acquisition-summary.json',dict(cases=results,transition_cameras=summary['new_transition_cameras'],initial_cameras=summary['new_initial_cameras']));print(json.dumps(dict(phase='acquired',cases=[{k:r[k] for k in ('seed','phase','complete','positive','branch_total_hits','total_hit_advantage','remaining_changed_choices') if k in r} for r in results],elapsed=time.monotonic()-began)),flush=True)
    if not any(r['complete'] and r['positive'] for r in results):summary['status']='no_verified_causal_pattern';finish();return
    firstguards={};byseed={e['seed']:e for e in bank['bank']};baseline={s:base.read(e['receipt']) for s,e in byseed.items()}
    for s,e in byseed.items():
        hits=[i for i,v in enumerate(baseline[s]['records']) if v['hits']]
        if hits:firstguards[s]=load_case(case_entry(e,bank,hits[0]+1))
    candidates=[]
    for r in results:
        if not r['complete'] or not r['positive']:continue
        cases,audit=audited_cases(r,current,donor,spec['current_policy_sha256'],spec['donor_policy_sha256']);bad=conflicts([c for s,c in firstguards.items() if s!=r['seed']]+cases);candidates.append(dict(result=r,cases=cases,audit=audit,conflicts=bad))
    selected=[]
    for phase in ('left','right'):
        eligible=[c for c in candidates if c['result']['phase']==phase and not c['conflicts']]
        if eligible:selected.append(min(eligible,key=lambda c:(c['result']['remaining_changed_choices'],c['audit']['rollin_rows'],c['result']['seed'])))
    base.atomic_json(directory/'causal-audit-selection.json',dict(candidates=[dict(result=c['result'],audit=c['audit'],conflicts=c['conflicts']) for c in candidates],selected=[c['result']['seed'] for c in selected],selection='atmostonepositive/phase, fewestchangedchoices thenrollincost thenseed; incompatiblepatternsrejectedbeforefit',joint_pattern_not_individual_causal_action_or_third_action_failure=True))
    if not selected:summary['status']='no_verified_compatible_causal_pattern';finish();return
    targets={c['result']['seed'] for c in selected};cases=[c for s,c in firstguards.items() if s not in targets]+[c for e in selected for c in e['cases']];bad=conflicts(cases);base.atomic_json(directory/'hard-contract.json',dict(guarded_first_cases=len(firstguards)-len(targets),target_old_first_prefixes_removed=sorted(targets),target_audits=[c['audit'] for c in selected],exact_input_conflicts=bad,hard_rows=sum(len(c['actions']) for c in cases),later_ordinary_actions_soft=True,required_causal_margin='fixed kappa',required_guard_margin='min(kappa*1e-4,halfsourcegap)'))
    if bad:summary['status']='selected_patterns_conflict_no_fit_retry';finish();return
    for name,digest in spec['metric_files'].items():assert base.sha(name)==digest
    G=np.load(spec['metric'],mmap_mode='r');L=la.cholesky(G,lower=True,check_finite=False);den=np.load(spec['raw_denominator']);phi=spec['phi_denominator'];kappa=base.read(spec['margin_report'])['kappa'];fitted,report=fit(current,cases,den,phi,L,kappa,min(deadline,time.monotonic()+90),spec['current_policy_sha256']);base.atomic_json(directory/'fit-report.json',report);summary['fit']=report;print(json.dumps(dict(phase='fit',report=report,elapsed=time.monotonic()-began)),flush=True)
    if fitted is None:summary['status']='no_verified_causal_fit';finish();return
    W,R,theta=fitted;base.save_array(directory/'proposal.npy',W);save_readout(directory/'rbf-model.npz',W,current.basis,R);base.save_array(directory/'joint-contrast-delta.npy',theta)
    def check(name,seeds,cached=None,seconds=150):
        p=directory/(name+'-bank.json');base.atomic_json(p,dict(bank,bank=[byseed[s] for s in seeds]));native_check(directory/name,p,directory/'proposal.npy',cached_native_root=cached,model_bundle=directory/'rbf-model.npz',native_seconds=seconds,practical_later_screen=True,absolute_deadline=deadline);s=base.read(directory/name/'summary.json');summary[name]=s;summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];summary['status']=name+('_complete' if s['status']=='complete' else '_bounded_partial');rows={}
        for p in (directory/name/'native_candidate/episodes').glob('*.json'):
            r=base.read(p);rows[r['seed']]=r
        firstloss=[r['seed'] for r in rows.values() if r['complete'] and baseline[r['seed']]['report']['hits']>0 and r['report']['hits']==0 and r['seed'] not in targets];base.atomic_json(directory/name/'first-contract-invariants.json',dict(first_return_losses_on_hard_guarded_cases=firstloss,late_perseed_tradeoffs_allowed=True));assert not firstloss
        return s,rows
    s,actual=check('targets',sorted(targets|{104562292}),seconds=60);target_ok=s['status']=='complete' and all(actual[t]['report']['hits']>baseline[t]['report']['hits'] for t in targets);summary['actual_target_gain']=target_ok
    if not target_ok:finish();return
    skill=actual[104562292]['report']['hits']>=5;summary['skill104_actual_returns']=actual[104562292]['report']['hits'];s,rows=check('sentinel',spec['sentinel'],directory/'targets');summary['sentinel_second_return_change']=s['candidate_more']['tail']['2']-s['baseline_more']['tail']['2'];useful=s['conditional_fresh_eligible'] and s['candidate']['first_incoming_returns']>=s['baseline']['first_incoming_returns']
    if not useful:finish();return
    if not skill:summary['status']='skill104_tradeoff_parent_review_before_full';finish();return
    full,rows=check('full',list(byseed),directory/'sentinel',600);qualify=full['conditional_fresh_eligible'] and full['candidate']['first_incoming_returns']>=full['baseline']['first_incoming_returns'];summary['full_fresh_gate']=qualify
    if qualify and time.monotonic()<deadline:
        fresh_run(fresh,current_root,directory/'full',absolute_deadline=deadline);new=base.read(fresh/'summary.json');summary.update(status='fresh_complete' if new['status']=='complete' else 'fresh_bounded_partial',fresh=new);summary['new_transition_cameras']+=new['fresh_transition_cameras'];summary['new_initial_cameras']+=new['initial_cameras']
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--manifest',required=True);a=p.parse_args();run(a.directory,a.manifest)
