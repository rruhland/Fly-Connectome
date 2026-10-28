"""Bounded fixed-feature joint learning with honest mixed-source earned memory."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from pong_fixed_rbf import load_readout
from pong_fixed_pairwise import save_readout
from pong_joint_output import metric,fit
from run_rally_earned_merge import load_case
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
from run_rally_repertoire_collection import exclusions,select_starts
from run_rally_pairwise_history import key
import run_practical_pong_v2 as base

FILES=('run_rally_joint_block.py','pong_joint_output.py','pong_fixed_pairwise.py','pong_fixed_rbf.py','run_rally_earned_merge.py','run_rally_fixed_bank_check.py','run_rally_guarded_curriculum.py','rally_action_equivalence.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py','run_rally_repertoire_collection.py')


def receipt_entry(path,root,Wpath,bundle):
    path=Path(path);root=Path(root);Wpath=Path(Wpath);bundle=Path(bundle);r=base.read(path);assert r['complete'] and r['weights_sha256']==base.sha(bundle) and r['base_weights_sha256']==base.sha(Wpath)
    raw=Path(r['feature_file']);raw=raw if raw.is_absolute() else root/raw;hits=[i for i,v in enumerate(r['records']) if v['hits']];assert len(hits)==r['report']['hits']
    return dict(seed=r['seed'],receipt=str(path),receipt_sha256=base.sha(path),raw_feature=str(raw),raw_feature_sha256=base.sha(raw),source_root=str(root),end=hits[-1]+1 if hits else 0,earned_returns=len(hits),source_head=str(Wpath),source_head_sha256=base.sha(Wpath),source_model_bundle=str(bundle),source_policy_sha256=base.sha(bundle))


def next_prefix(entry,current_returns):
    """Choose an actually rewarded next return, or declare source already achieved."""
    r=base.read(entry['receipt']);hits=[i for i,v in enumerate(r['records']) if v['hits']]
    if len(hits)<=current_returns:return None
    return dict(entry,end=hits[current_returns]+1,earned_returns=current_returns+1)


def memory_counts(memory):
    hits=[e['earned_returns'] for e in memory.values()]
    return dict(contexts=len(hits),returns=sum(hits),first=sum(h>0 for h in hits),later=sum(max(0,h-1) for h in hits),at_least2=sum(h>=2 for h in hits),tail={str(k):sum(h>=k for h in hits) for k in range(1,7)},guaranteed_lower_bounds_not_current_full_native_outcomes=True)


def violations(rows,memory):
    return [dict(seed=r['seed'],minimum=memory[r['seed']]['earned_returns'],actual=r['report']['hits']) for r in rows if r['complete'] and r['report']['hits']<memory[r['seed']]['earned_returns']]


def remember(memory,entry):
    old=memory[entry['seed']]
    if entry['earned_returns']>old['earned_returns']:memory[entry['seed']]=entry


def admit(spec):
    old=base.read(spec['previous_bank']);collection=Path(spec['collection']);audit=collection/'receipt-audit.json';assert base.sha(audit)==spec['collection_audit_sha256'] and base.read(audit)['passed'] and base.read(collection/'summary.json')['status']=='complete';head=spec['reference'];Wpath=Path(head['root'])/'proposal.npy';bundle=Path(head['root'])/'rbf-model.npz';assert base.sha(Wpath)==head['base_weights_sha256'] and base.sha(bundle)==head['policy_sha256'] and old['current_policy_sha256']==head['policy_sha256'] and old['current_head_sha256']==head['base_weights_sha256'] and old['source']==base.source_signature();entries=[]
    for e in old['bank']:entries.append(receipt_entry(e['receipt'],e['source_root'],Wpath,bundle))
    root=collection/'scaffold'
    for p in sorted((root/'native_candidate/episodes').glob('*.json')):entries.append(receipt_entry(p,root,Wpath,bundle))
    assert len(entries)==len({e['seed'] for e in entries})==256 and sum(e['earned_returns'] for e in entries)==662 and sum(e['earned_returns']>0 for e in entries)==233 and sum(e['earned_returns']>=2 for e in entries)==161
    path=Path(spec['bank']);assert not path.exists();path.parent.mkdir();bank=dict(old,bank=entries,previous_bank=spec['previous_bank'],previous_bank_sha256=base.sha(spec['previous_bank']),new_TRAIN_collection=str(collection),new_TRAIN_audit_sha256=base.sha(audit),admitted_new_TRAIN64=base.read(collection/'schedules.json')['TRAIN'],fresh_admitted=False,no_existing_fresh_admission=True,expected=dict(episodes=256,returns=662,first=233,later=429,at_least2=161),admission_authorization='parent explicit audited64TRAIN admission; evaluation62993/63009 excluded');base.atomic_json(path,bank);return bank,{e['seed']:e for e in entries}


def run(directory,spec_path):
    directory=Path(directory);spec_path=Path(spec_path);assert not directory.exists();directory.mkdir();began=time.monotonic();spec=base.read(spec_path);assert (spec['seconds'],spec['max_updates'],spec['targets_per_update'],spec['fit_seconds'])==(1200,3,3,90);deadline=began+1200;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=1200,implementation_seconds_separate_estimate=spec['implementation_seconds_separate_estimate']));(directory/'block-spec.json').write_bytes(spec_path.read_bytes());files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};source=base.source_signature();bank,memory=admit(spec);initial=dict(memory);pool=spec['pool'];assert len(pool)==12 and len({e['seed'] for e in pool})==12;queue=list(pool);summary=dict(status='preparing',new_transition_cameras=0,new_initial_cameras=0,accepted_updates=0,stages=[],updates=[],implementation_seconds_separate_estimate=spec['implementation_seconds_separate_estimate']);current=Path(spec['reference']['root']);current_sha=base.sha(current/'rbf-model.npz');initial_sha=current_sha;known={s:(e,initial_sha) for s,e in memory.items()};cache={}
    def case(e):
        assert base.sha(e['receipt'])==e['receipt_sha256'] and base.sha(e['raw_feature'])==e['raw_feature_sha256'] and base.sha(e['source_head'])==e['source_head_sha256'] and base.sha(e['source_model_bundle'])==e['source_policy_sha256'];k=(e['receipt_sha256'],e['source_policy_sha256'],e['end'])
        if k not in cache:cache[k]=load_case(e)
        return cache[k]
    def save_memory(name):base.atomic_json(directory/name,dict(cases=list(memory.values()),counts=memory_counts(memory),reference_model=str(current/'rbf-model.npz'),reference_policy_sha256=current_sha,source_provenance_not_relabelled=True))
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,final_reference_model=str(current/'rbf-model.npz'),final_reference_policy_sha256=current_sha,memory=memory_counts(memory),source_frozen=source==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},initial_reference_unchanged=base.sha(Path(spec['reference']['root'])/'rbf-model.npz')==initial_sha and base.sha(Path(spec['reference']['root'])/'proposal.npy')==spec['reference']['base_weights_sha256'],bank_unchanged=base.sha(spec['bank'])==manifest['bank_sha256'],spec_unchanged=base.sha(spec_path)==manifest['spec_sha256'],no_automatic_promotion=True);save_memory('final-memory.json');base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],accepted_updates=summary['accepted_updates'],cameras=summary['new_transition_cameras'],initials=summary['new_initial_cameras'],wall_seconds=summary['wall_seconds'],source_frozen=summary['source_frozen'])),flush=True)
    # Reserve the qualification schedule before any fitting; all prior and reserved IDs excluded.
    fresh=directory/'fresh';fresh.mkdir();excluded=exclusions(directory);drawn=select_starts(spec['fresh_root'],spec['fresh_tag'],excluded,32);seeds=drawn.pop('TRAIN');drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],purpose='new paired qualification only; no admission');base.atomic_json(fresh/'schedules.json',drawn)
    manifest=dict(source=source,files=files,spec_sha256=base.sha(spec_path),bank_sha256=base.sha(spec['bank']),evaluation_baseline_policy_sha256=initial_sha,initial_memory=memory_counts(memory),fresh_schedule_sha256=base.sha(fresh/'schedules.json'),fixed_optimizer_rule='equalepisode completeearnedprefix fullraw/Phi crosscovariance; optimizerblockRMS;1%ridge',no_sweep_or_retry=True,fit_failure_qualifies_last_accepted=True);base.atomic_json(directory/'manifest.json',manifest);save_memory('initial-memory.json');initial_cases=[case(e) for e in memory.values() if e['end']];reference=load_readout(current/'rbf-model.npz');scale_path=Path(spec['feature_metric']);assert base.sha(scale_path)==spec['feature_metric_sha256'] and base.sha(spec['margin_report'])==spec['margin_report_sha256'];start=time.monotonic();G,raw_denom,phi_denom,conditioning=metric([c['X'] for c in initial_cases],reference.basis,np.load(scale_path));L=la.cholesky(G,lower=True,check_finite=False);base.save_array(directory/'score-metric.npy',G);base.save_array(directory/'raw-optimizer-denominator.npy',raw_denom);conditioning.update(phi_optimizer_denominator=phi_denom,score_metric_sha256=base.sha(directory/'score-metric.npy'),raw_denominator_sha256=base.sha(directory/'raw-optimizer-denominator.npy'),feature_metric_sha256=base.sha(scale_path),wall_seconds=time.monotonic()-start);base.atomic_json(directory/'conditioning.json',conditioning);normal=base.read(spec['margin_report']);print(json.dumps(dict(phase='prepared',memory=memory_counts(memory),conditioning_seconds=conditioning['wall_seconds'],elapsed=time.monotonic()-began)),flush=True)
    def check(name,seeds,model,bounds,cached=None,seconds=150):
        path=directory/(name+'-bank.json');base.atomic_json(path,dict(bank,bank=[initial[s] for s in seeds]));native_check(directory/name,path,model/'proposal.npy',cached_native_root=cached,model_bundle=model/'rbf-model.npz',native_seconds=seconds,practical_later_screen=True,absolute_deadline=deadline);s=base.read(directory/name/'summary.json');rows=[(p,base.read(p)) for p in (directory/name/'native_candidate/episodes').glob('*.json')];bad=violations([r for p,r in rows],bounds);base.atomic_json(directory/name/'earned-outcome-invariants.json',dict(complete=s['status']=='complete',violations=bad,passed=s['status']=='complete' and not bad));summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];summary['stages'].append(dict(name=name,summary=s))
        if bad:summary.update(status='earned_prefix_context_invariant_failed',violations=bad);finish();raise AssertionError('saved actual outcomes violate earned-memory guarantees')
        entries=[]
        for p,r in rows:
            if r['complete']:entries.append(receipt_entry(p,directory/name,model/'proposal.npy',model/'rbf-model.npz'))
        return s,entries
    def compatible(candidate,selected):
        replaced={e['seed'] for e in selected+[candidate]};seen={}
        for e in list(memory.values())+selected:
            if not e['end'] or e['seed'] in replaced and e not in selected:continue
            c=case(e)
            for i,a in enumerate(c['actions']):
                k=key(c['X'],i);old=seen.setdefault(k,int(a));assert old==a
        c=case(candidate)
        return not any((v:=seen.get(key(c['X'],i))) is not None and v!=a for i,a in enumerate(c['actions']))
    last_sentinel=None
    for iteration in range(1,4):
        selected=[];selection=[]
        # Settle all current outcomes before choosing contracts: a new precheck
        # prefix can invalidate a tentative source path selected against old memory.
        missing=[e['seed'] for e in queue if known.get(e['seed'],(None,None))[1]!=current_sha]
        if missing:
            s,actual=check(f'precheck_{iteration:02d}',missing,current,memory,cached=last_sentinel,seconds=180)
            for e in actual:known[e['seed']]=(e,current_sha);remember(memory,e)
            if s['status']!='complete':summary['status']='precheck_bounded_partial';finish();return
        while queue and len(selected)<3 and time.monotonic()<deadline:
            batch=[queue.pop(0) for _ in range(min(3-len(selected),len(queue)))];missing=[e['seed'] for e in batch if known.get(e['seed'],(None,None))[1]!=current_sha]
            assert not missing
            for entry in batch:
                s=entry['seed'];actual=known[s][0]['earned_returns'];target=next_prefix(entry,actual)
                if target is None:selection.append(dict(seed=s,status='already_achieved_source_returns',actual_current_returns=actual));continue
                if not compatible(target,selected):selection.append(dict(seed=s,status='exact_raw_conflict_after_current_precheck',actual_current_returns=actual));continue
                selected.append(target);selection.append(dict(seed=s,status='selected',actual_current_returns=actual,required_next_returns=target['earned_returns'],rows=target['end']))
        base.atomic_json(directory/f'selection_{iteration:02d}.json',dict(entries=selection,pool_remaining=[e['seed'] for e in queue]))
        if not selected:summary['status']='pool_exhausted_without_new_target';break
        contract=dict(memory);contract.update({e['seed']:e for e in selected});cases=[case(e) for e in contract.values() if e['end']];step=directory/f'update_{iteration:02d}';step.mkdir();base.atomic_json(step/'hard-contract.json',dict(cases=list(contract.values()),counts=memory_counts(contract),hard_rows=sum(len(c['actions']) for c in cases),source_validation_before_slicing=True))
        if iteration==1:assert [e['seed'] for e in selected]==spec['first_targets'] and sum(len(c['actions']) for c in cases)==100586 and len(cases)==233
        reference=load_readout(current/'rbf-model.npz');result,report=fit(reference,cases,raw_denom,phi_denom,L,normal['kappa'],min(deadline,time.monotonic()+90),current_sha);base.atomic_json(step/'fit-report.json',report);summary['updates'].append(dict(iteration=iteration,selection=selection,fit=report));print(json.dumps(dict(phase='fit',iteration=iteration,report=report,elapsed=time.monotonic()-began)),flush=True)
        if result is None:summary['status']='fit_failed_preserving_last_accepted';break
        W,R,theta=result;base.save_array(step/'proposal.npy',W);save_readout(step/'rbf-model.npz',W,reference.basis,R);base.save_array(step/'joint-contrast-delta.npy',theta);candidate=load_readout(step/'rbf-model.npz');assert np.array_equal(candidate.base.weights,np.load(step/'proposal.npy')) and all(np.array_equal(getattr(candidate.basis,n),getattr(reference.basis,n)) for n in ('hashes','signs','scale','mean','rms'))
        s,target_actual=check(f'targets_{iteration:02d}',[e['seed'] for e in selected],step,contract,seconds=60)
        if s['status']!='complete':summary['status']='target_bounded_partial_not_accepted';break
        s,control_actual=check(f'sentinel_{iteration:02d}',spec['correctness_sentinel'],step,contract,seconds=90)
        if s['status']!='complete':summary['status']='sentinel_bounded_partial_not_accepted';break
        current=step;current_sha=base.sha(step/'rbf-model.npz');summary['accepted_updates']+=1;last_sentinel=directory/f'sentinel_{iteration:02d}'
        for e in target_actual+control_actual:remember(memory,e);known[e['seed']]=(e,current_sha)
        save_memory(f'memory_after_{iteration:02d}.json');print(json.dumps(dict(phase='accepted_update',iteration=iteration,memory=memory_counts(memory),elapsed=time.monotonic()-began)),flush=True)
    if not summary['accepted_updates']:finish();return
    full,actual=check('full',list(initial),current,memory,cached=last_sentinel,seconds=600);summary['full']=full
    qualify=full['conditional_fresh_eligible'] and full['candidate']['first_incoming_returns']>=full['baseline']['first_incoming_returns'] and full['candidate_more']['tail']['2']>=full['baseline_more']['tail']['2'];summary['full_fresh_gate']=qualify
    if qualify and time.monotonic()<deadline:
        fresh_run(fresh,Path(spec['reference']['root']),directory/'full',absolute_deadline=deadline);s=base.read(fresh/'summary.json');summary.update(status='fresh_complete' if s['status']=='complete' else 'fresh_bounded_partial',fresh=s);summary['new_transition_cameras']+=s['fresh_transition_cameras'];summary['new_initial_cameras']+=s['initial_cameras']
    else:summary['status']='full_complete' if full['status']=='complete' else 'full_bounded_partial'
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--manifest',required=True);a=p.parse_args();run(a.directory,a.manifest)
