"""One900s fixed8-proposal full-game reward search; no donor imitation or BP."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from pong_constrained_residual import CONTRAST
from pong_fixed_rbf import load_readout
from pong_fixed_pairwise import FrozenPairwiseReadout,save_readout
from pong_projected_reward import GuardProjector,crossing_scale
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
from run_rally_pairwise_history import CARRY,BASE
import run_practical_pong_v2 as base

SEED=2934610871
DOSE=.1
FILES=('run_rally_projected_reward.py','pong_projected_reward.py','pong_fixed_pairwise.py','pong_fixed_rbf.py','run_rally_fixed_bank_check.py','run_rally_guarded_curriculum.py','rally_action_equivalence.py','run_rally_raw_optimizer.py','run_rally_raw_fresh_screen.py')


def prepare(bank,carried):
    rows=[(e,base.read(e['receipt']),sp.load_npz(e['raw_feature']).tocsr()) for e in bank['bank']];ends={};G=np.zeros((512,512));episodes=0;guardX=[];guardA=[];guardQ=[];acquisition=None
    for e,r,X in rows:
        assert r['complete'] and r['weights_sha256']==CARRY;hits=[i for i,v in enumerate(r['records']) if v['hits']]
        end=hits[min(1,len(hits)-1)]+1 if hits else 0
        if e['seed']==104562292:
            assert len(hits)==5;end=hits[-1]+1;acquisition=dict(seed=e['seed'],receipt=e['receipt'],receipt_sha256=base.sha(e['receipt']),raw_feature=e['raw_feature'],last_earned_row=end-1,earned_hits=5,failed_or_timeout_suffix_excluded=True)
        ends[e['seed']]=end
        if end:guardX.append(X[:end]);guardA.extend(v['action'] for v in r['records'][:end]);guardQ.extend(v['q'] for v in r['records'][:end])
        if hits:
            Phi=carried.basis.batch(X[:hits[-1]+1]);G+=Phi.T@Phi/len(Phi);episodes+=1
    assert episodes==174 and acquisition is not None;G/=episodes;normalizer=float(np.trace(G)/512);G/=normalizer;G.flat[::513]+=.01;rng=np.random.default_rng(SEED);extra=set(rng.permutation(len(rows))[:2048%len(rows)]);phase_counts={p:sum(base.leftward(e['seed'])==(p=='left') for e,r,X in rows) for p in ('left','right')};sample=[];weights=[];provenance=[];sampleQ=[];sampleA=[]
    for j,(e,r,X) in enumerate(rows):
        available=np.arange(ends[e['seed']],X.shape[0]);n=2048//len(rows)+int(j in extra);assert len(available)>=n;ix=np.sort(rng.choice(available,n,replace=False));sample.append(X[ix]);phase='left' if base.leftward(e['seed']) else 'right';weights.extend([.5/(phase_counts[phase]*n)]*n);provenance.append(dict(seed=e['seed'],phase=phase,rows=ix.tolist(),eligible_rule='strictly after protected earned prefix, or wholefailed0-hit episode',weight_per_row=.5/(phase_counts[phase]*n)));sampleQ.extend(r['records'][i]['q'] for i in ix);sampleA.extend(r['records'][i]['action'] for i in ix)
    sample=sp.vstack(sample,format='csr');q=np.array(sampleQ);a=np.array(sampleA)
    for i in range(sample.shape[0]):
        lo,hi=sample.indptr[i:i+2];aa,qq=carried.choose(sample.indices[lo:hi],sample.data[lo:hi]);assert aa==a[i] and np.array_equal(qq,q[i])
    screen=[]
    for phase in ('left','right'):
        for band in range(4):
            eligible=[e['seed'] for e,r,X in rows if base.leftward(e['seed'])==(phase=='left') and min(r['report']['hits'],3)==band];assert eligible;screen.append(min(eligible))
    return rows,G,normalizer,sp.vstack(guardX,format='csr'),np.array(guardA),np.array(guardQ),acquisition,sample,q,a,np.array(weights),provenance,screen


def actual_dose(policy,X,actions,weights):
    changed=[]
    for i,a in enumerate(actions):
        lo,hi=X.indptr[i:i+2];got,_=policy.choose(X.indices[lo:hi],X.data[lo:hi]);changed.append(got!=a)
    return dict(changed_rows=int(np.sum(changed)),rows=len(changed),equal_phase_episode_weighted_fraction=float(weights@np.array(changed)),pooled_row_fraction=float(np.mean(changed)))


def run(directory,implementation_seconds):
    directory=Path(directory);assert not directory.exists();directory.mkdir();began=time.monotonic();deadline=began+900;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=900,implementation_seconds_separate_estimate=implementation_seconds));bank_path=base.EXPERIMENTS/'pong-rally-TRAIN-bank-20261028-r5/admission.json';bank=base.read(bank_path);assert bank['source']==base.source_signature() and bank['current_policy_sha256']==CARRY and base.sha(bank['current_model_bundle'])==CARRY and base.sha(bank['current_head'])==BASE;carried=load_readout(bank['current_model_bundle']);rows,G,normalizer,X,actions,q,acquisition,sample,sq,sa,weights,sample_provenance,screen=prepare(bank,carried);files={n:base.sha(Path(__file__).with_name(n)) for n in FILES};normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');manifest=dict(source=base.source_signature(),files=files,bank=str(bank_path),bank_sha256=base.sha(bank_path),carried_policy_sha256=CARRY,base_sha256=BASE,seconds=900,projection_seconds_each=12,directions=4,signs=[1,-1],seed=SEED,dose=DOSE,dose_rule='equal-phase equal-episode weighted fraction of carried actions changed, pooled across+/- at one crossing scale; exact actualdose measured before/after projection, no rescale',preconditioner='equal174successfulepisode scorecovariance, mean-diagonal normalized plus1%ridge',score_covariance_normalizer=normalizer,ridge_fraction=.01,guard_margin_rule='min(kappa*1e-4, half carried source gap); exactdeployedchoose required',kappa=normal['kappa'],guard_rows=len(actions),known_early_prefixes=174,acquisition=acquisition,screen_seeds=screen,screen_selection='lowestseed ineach initialphase x currenthitband0/1/2/3+, fixed before projection/reward',selection='strict full-game ownreturngain versus carry; first/second retained; carryonties, candidateorder fixed; no ninthaverage',prior_failure_context='Earlier8D linear search/action-dose adjustments overfitTRAIN and failedfresh; doseisnottransfer guarantee. Newfixed512map and cumulativeearnedskillretention are the tested differences.',no_donor_targets=True,no_reward_based_scale_tuning=True,no_feature_change=True,no_sweep=True);base.save_array(directory/'score-covariance.npy',G);manifest['score_covariance_sha256']=base.sha(directory/'score-covariance.npy');base.atomic_json(directory/'manifest.json',manifest);base.atomic_json(directory/'dose-sample.json',dict(episodes=sample_provenance,rows=2048,phase_episode_balanced=True));base.atomic_json(directory/'screen-schedule.json',dict(seeds=screen,TRAIN_not_holdout=True));print(json.dumps(dict(phase='prepared',guard_rows=len(actions),screen=screen,acquisition=acquisition,elapsed=time.monotonic()-began)),flush=True);projector=GuardProjector(carried,X,actions,q,G,normal['kappa']);L=la.cholesky(G,lower=True,check_finite=False);rng=np.random.default_rng(SEED);Phi=carried.basis.batch(sample);plans=[]
    for d in range(4):
        direction=la.solve_triangular(L.T,rng.normal(size=(512,2)),lower=False,check_finite=False).T;direction/=np.sqrt(np.sum((direction@G)*direction));scores=Phi@(CONTRAST@direction).T;
        try:scale=crossing_scale(sq,sa,scores,weights,DOSE);error=None
        except ValueError as exc:scale=None;error=str(exc)
        for sign in (1,-1):plans.append((f'direction_{d}_{"plus" if sign==1 else "minus"}',None if scale is None else sign*scale*direction,dict(direction=d,sign=sign,shared_pair_scale=scale,degenerate_pair_reason=error)))
    np.savez(directory/'planned-proposals.npz',**{name:center for name,center,p in plans if center is not None});base.atomic_json(directory/'planned-proposals.json',dict(candidates=[dict(id=name,**p) for name,center,p in plans],coefficients_sha256=base.sha(directory/'planned-proposals.npz'),generated_before_any_projection_or_reward=True));summary=dict(status='projecting_and_screening',candidates=[],new_transition_cameras=0,new_initial_cameras=0);byseed={e['seed']:e for e,r,Y in rows};winner=None;best=None
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},carry_unchanged=base.sha(bank['current_model_bundle'])==CARRY,base_unchanged=base.sha(bank['current_head'])==BASE,bank_unchanged=base.sha(bank_path)==manifest['bank_sha256'],no_automatic_promotion=True);base.atomic_json(directory/'summary.json',summary);print(json.dumps(dict(status=summary['status'],cameras=summary['new_transition_cameras'],initials=summary['new_initial_cameras'],wall_seconds=summary['wall_seconds'],source_frozen=summary['source_frozen'])),flush=True)
    for name,center,plan in plans:
        if time.monotonic()>=deadline:summary['status']='budget_partial_proposals';finish();return
        folder=directory/name;folder.mkdir()
        if center is None:
            item=dict(id=name,**plan,status='rejected_degenerate_crossing_pair_no_retry');summary['candidates'].append(item);base.atomic_json(folder/'proposal-report.json',item);continue
        requested=FrozenPairwiseReadout(carried.base.weights,carried.basis,carried.residual+CONTRAST@center);before=actual_dose(requested,sample,sa,weights);R,report=projector.project(center,min(deadline,time.monotonic()+12));item=dict(id=name,**plan,before_projection_actual_dose=before,projection=report);summary['candidates'].append(item)
        if R is None:item['status']='rejected_uncertified_projection';base.atomic_json(folder/'proposal-report.json',item);print(json.dumps(item),flush=True);continue
        policy=FrozenPairwiseReadout(carried.base.weights,carried.basis,R);after=actual_dose(policy,sample,sa,weights);item['after_projection_actual_dose']=after
        if not after['changed_rows']:item['status']='rejected_no_eligible_forward_action_change';base.atomic_json(folder/'proposal-report.json',item);print(json.dumps(item),flush=True);continue
        for source,target in [(Path(bank['current_head']),'proposal.npy'),(Path(bank['radial_cap']),'radial-cap.json')]:(folder/target).write_bytes(source.read_bytes())
        save_readout(folder/'rbf-model.npz',carried.base.weights,carried.basis,R);item['bundle_sha256']=base.sha(folder/'rbf-model.npz');base.atomic_json(folder/'proposal-report.json',item);sb=folder/'screen-bank.json';base.atomic_json(sb,dict(bank,bank=[byseed[s] for s in screen]));native_check(folder/'screen',sb,folder/'proposal.npy',model_bundle=folder/'rbf-model.npz',native_seconds=80,practical_later_screen=True,absolute_deadline=deadline);s=base.read(folder/'screen/summary.json');item.update(status='screen_complete' if s['status']=='complete' else 'screen_bounded_partial',screen=s);summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials'];useful=s['status']=='complete' and s['candidate']['own_returns']>s['baseline']['own_returns'] and s['candidate']['first_incoming_returns']>=s['baseline']['first_incoming_returns'] and s['candidate_more']['tail']['2']>=s['baseline_more']['tail']['2'];score=s['candidate']['own_returns']
        if useful and (best is None or score>best):winner=folder;best=score
        base.atomic_json(folder/'proposal-report.json',item);base.atomic_json(directory/'progress.json',dict(candidates_completed=len(summary['candidates']),winner=winner.name if winner else None,new_transition_cameras=summary['new_transition_cameras']));print(json.dumps(dict(phase='candidate',id=name,projection=report,actual_dose=after,screen=s,selected_so_far=winner.name if winner else None)),flush=True)
        if s['status']!='complete':summary['status']='budget_partial_screen';finish();return
    if winner is None:summary['status']='no_strict_actual_reward_screen_winner';finish();return
    summary['winner']=winner.name;summary['winner_bundle_sha256']=base.sha(winner/'rbf-model.npz');native_check(directory/'full',bank_path,winner/'proposal.npy',cached_native_root=winner/'screen',model_bundle=winner/'rbf-model.npz',native_seconds=600,practical_later_screen=True,absolute_deadline=deadline);whole=base.read(directory/'full/summary.json');summary.update(status='full_complete' if whole['status']=='complete' else 'full_bounded_partial',full=whole);summary['new_transition_cameras']+=whole['native']['new_transition_cameras'];summary['new_initial_cameras']+=whole['native']['new_initials'];target=[base.read(p) for p in (directory/'full/native_candidate/episodes').glob('*.json') if base.read(p)['seed']==104562292];skill=bool(target and target[0]['complete'] and target[0]['report']['hits']>=5);qualify=whole['conditional_fresh_eligible'] and whole['candidate']['first_incoming_returns']>=whole['baseline']['first_incoming_returns'] and whole['candidate_more']['tail']['2']>=whole['baseline_more']['tail']['2'] and skill;summary.update(acquired_five_return_skill_retained=skill,full_fresh_gate=qualify)
    if qualify and time.monotonic()<deadline:
        import run_rally_baseline_pong as schedules
        previous=set();proof={}
        def collect(v):
            if type(v) is int and 0<v<2**32:previous.add(v%base.MODULUS)
            elif isinstance(v,list):
                for a in v:collect(a)
            elif isinstance(v,dict):
                for a in v.values():collect(a)
        for p in sorted(base.EXPERIMENTS.rglob('*schedule*.json')):collect(base.read(p));proof[str(p)]=base.sha(p)
        for p in sorted(base.EXPERIMENTS.rglob('seed-exclusions.json')):collect(base.read(p).get('ids',[]));proof[str(p)]=base.sha(p)
        collect(list(base.old_engine.FINAL));collect(list(byseed));schedules.ROOT=1943906127;schedules.TAG=62977;drawn=schedules.schedule();seeds=drawn.pop('train');assert not {s%base.MODULUS for s in seeds}&previous;fresh=directory/'fresh';fresh.mkdir();drawn.update(probe=seeds[:32],unused_reserved=seeds[32:],exclusion_sha256=proof,selection='genuinely unused ordinary pairedgames, nofitting/admission');base.atomic_json(fresh/'schedules.json',drawn);fresh_run(fresh,Path(bank['current_model_bundle']).parent,directory/'full',absolute_deadline=deadline);new=base.read(fresh/'summary.json');summary.update(status='fresh_complete' if new['status']=='complete' else 'fresh_bounded_partial',fresh=new);summary['new_transition_cameras']+=new['fresh_transition_cameras'];summary['new_initial_cameras']+=new['initial_cameras']
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--implementation-seconds',type=float,required=True);a=p.parse_args();run(a.directory,a.implementation_seconds)
