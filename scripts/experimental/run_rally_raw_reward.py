"""Separate900s empirical-retention test of the exact eight saved raw requests."""
import argparse,json,time
from pathlib import Path
import numpy as np
from pong_constrained_residual import CONTRAST
from pong_fixed_rbf import load_readout
from pong_fixed_pairwise import save_readout
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
from run_rally_pairwise_history import CARRY,BASE
import run_practical_pong_v2 as base

REQUESTS='287f1495ca0324c879e258da11097dc58acbd3e9d73dbb2d053d855b8b860d6f'
FILES=('run_rally_raw_reward.py','pong_fixed_pairwise.py','pong_fixed_rbf.py','run_rally_fixed_bank_check.py','run_rally_guarded_curriculum.py','rally_action_equivalence.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py')


def useful(s):
    return s['status']=='complete' and s['candidate']['own_returns']>s['baseline']['own_returns'] and s['candidate_more']['later_returns']>s['baseline_more']['later_returns'] and s['candidate']['first_incoming_returns']>=s['baseline']['first_incoming_returns'] and s['candidate_more']['tail']['2']>=s['baseline_more']['tail']['2']


def run(directory,implementation_seconds):
    directory=Path(directory);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+900;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=900,separate_empirical_retention_followup=True,implementation_seconds_separate_estimate=implementation_seconds));previous=base.EXPERIMENTS/'pong-rally-pairwise-reward-search-20261028-r1';bank_path=base.EXPERIMENTS/'pong-rally-TRAIN-bank-20261028-r5/admission.json';bank=base.read(bank_path);assert base.sha(previous/'planned-proposals.npz')==REQUESTS and bank['source']==base.source_signature() and base.sha(bank['current_model_bundle'])==CARRY and base.sha(bank['current_head'])==BASE;carried=load_readout(bank['current_model_bundle']);order=base.read(previous/'planned-proposals.json')['candidates'];screen=base.read(previous/'screen-schedule.json')['seeds'];assert len(order)==8 and len(screen)==8 and 104562292 not in screen;byseed={e['seed']:e for e in bank['bank']};files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};models=[]
    with np.load(previous/'planned-proposals.npz') as f:
        for item in order:
            name=item['id'];folder=directory/name;folder.mkdir();center=np.asarray(f[name]);assert center.shape==(2,512) and np.isfinite(center).all()
            for source,target in [(Path(bank['current_head']),'proposal.npy'),(Path(bank['radial_cap']),'radial-cap.json')]:(folder/target).write_bytes(source.read_bytes())
            R=carried.residual+CONTRAST@center;save_readout(folder/'rbf-model.npz',carried.base.weights,carried.basis,R);loaded=load_readout(folder/'rbf-model.npz');assert np.array_equal(loaded.residual,R);models.append(dict(item,root=str(folder),bundle_sha256=base.sha(folder/'rbf-model.npz')))
    manifest=dict(source=base.source_signature(),files=files,bank=str(bank_path),bank_sha256=base.sha(bank_path),source_request_root=str(previous),source_request_sha256=REQUESTS,prior_projected_test_summary_sha256=base.sha(previous/'summary.json'),prior_projected_test_wall_seconds=base.read(previous/'summary.json')['wall_seconds'],seconds=900,separate_followup_reason='All8nearestprojections failednumericalbudget before rewardtest. Evaluate exactsame rawrequests by actual performance, not actionlocks; no claimthatprojectionprovedinfeasible.',models=models,screen_seeds=screen,skill_seed=104562292,skill_minimum_returns=5,execution_order='For eachcandidate actualskillfirst; below5 meansineligible andother8UNEVALUATED, notbad. Survivingcandidates usecommon8screen.',screen_gate='strict totalANDlaterreturngain withfirst/secondcountsretained; rankbytotal,earliercandidateonties; carrywinsallnonimprovingties',full_gate='strict totalANDlaterreturngain, first/secondretention, acquired104>=5; thenunusedfresh',no_projection=True,no_refitting=True,no_recalibration=True,no_rescaling=True,no_new_directions=True,no_ninth_candidate=True,no_automatic_promotion=True,prior_failure_context='Earlierlinear8D/actiondoses overfitTRAIN and failedfresh; new512spaceisnotaguarantee. This followuptestsnativeoutcome retention instead oftrajectorypreservation.');base.atomic_json(directory/'manifest.json',manifest);summary=dict(status='skill_first_raw_reward_screen',candidates=[],new_transition_cameras=0,new_initial_cameras=0);winner=None;best=None
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},carry_unchanged=base.sha(bank['current_model_bundle'])==CARRY,base_unchanged=base.sha(bank['current_head'])==BASE,bank_unchanged=base.sha(bank_path)==manifest['bank_sha256'],saved_requests_unchanged=base.sha(previous/'planned-proposals.npz')==REQUESTS,prior_projected_test_summary_unchanged=base.sha(previous/'summary.json')==manifest['prior_projected_test_summary_sha256'],no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],cameras=summary['new_transition_cameras'],initials=summary['new_initial_cameras'],wall_seconds=summary['wall_seconds'],source_frozen=summary['source_frozen'])),flush=True)
    def check(folder,name,seeds,cached=None,alternate=None,seconds=80):
        path=folder/(name+'-bank.json');base.atomic_json(path,dict(bank,bank=[byseed[s] for s in seeds]));native_check(folder/name,path,folder/'proposal.npy',cached_native_root=cached,alternate_native_root=alternate,model_bundle=folder/'rbf-model.npz',native_seconds=seconds,practical_later_screen=True,absolute_deadline=deadline);s=base.read(folder/name/'summary.json');summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];return s
    for model in models:
        if time.monotonic()>=deadline:summary['status']='budget_partial_candidates';finish();return
        folder=Path(model['root']);skill=check(folder,'skill',[104562292],seconds=30);item=dict(id=model['id'],bundle_sha256=model['bundle_sha256'],skill=skill);summary['candidates'].append(item)
        if skill['status']!='complete':item.update(status='skill_bounded_partial',other8='UNEVALUATED');summary['status']='budget_partial_skill';finish();return
        if skill['candidate']['own_returns']<5:item.update(status='ineligible_acquired_skill_not_retained',other8='UNEVALUATED');base.atomic_json(folder/'candidate-summary.json',item);print(json.dumps(dict(id=model['id'],status=item['status'],skill_returns=skill['candidate']['own_returns'],other8='UNEVALUATED')),flush=True);continue
        s=check(folder,'screen',screen);item.update(status='screen_complete' if s['status']=='complete' else 'screen_bounded_partial',screen=s,strict_screen_gate=useful(s));score=s['candidate']['own_returns']
        if useful(s) and (best is None or score>best):winner=folder;best=score
        base.atomic_json(folder/'candidate-summary.json',item);print(json.dumps(dict(id=model['id'],status=item['status'],strict_screen_gate=item['strict_screen_gate'],skill_returns=skill['candidate']['own_returns'],screen=s,selected_so_far=winner.name if winner else None)),flush=True)
        if s['status']!='complete':summary['status']='budget_partial_screen';finish();return
    if winner is None:summary['status']='no_eligible_strict_actual_reward_winner';finish();return
    summary['winner']=winner.name;summary['winner_bundle_sha256']=base.sha(winner/'rbf-model.npz');whole=check(winner,'full',list(byseed),cached=winner/'screen',alternate=winner/'skill',seconds=600);summary.update(status='full_complete' if whole['status']=='complete' else 'full_bounded_partial',full=whole);target=[base.read(p) for p in (winner/'full/native_candidate/episodes').glob('*.json') if base.read(p)['seed']==104562292];skill=bool(target and target[0]['complete'] and target[0]['report']['hits']>=5);summary.update(full_fresh_gate=useful(whole) and skill,acquired_skill_retained=skill)
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
        collect(list(base.old_engine.FINAL));collect(list(byseed));schedules.ROOT=1943906127;schedules.TAG=62977;drawn=schedules.schedule();seeds=drawn.pop('train');assert not {s%base.MODULUS for s in seeds}&excluded;fresh=directory/'fresh';fresh.mkdir();drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],exclusion_sha256=proof,selection='unconsumed originallyplanned62977 ordinarypairedgames; nofitting/admission');base.atomic_json(fresh/'schedules.json',drawn);fresh_run(fresh,Path(bank['current_model_bundle']).parent,winner/'full',absolute_deadline=deadline);new=base.read(fresh/'summary.json');summary.update(status='fresh_complete' if new['status']=='complete' else 'fresh_bounded_partial',fresh=new);summary['new_transition_cameras']+=new['fresh_transition_cameras'];summary['new_initial_cameras']+=new['initial_cameras']
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--implementation-seconds',type=float,required=True);a=p.parse_args();run(a.directory,a.implementation_seconds)
