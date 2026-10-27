"""One bounded later-return discovery pass; no fit or candidate deployment."""
import argparse,json,multiprocessing as mp,random,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import torch
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.fresh_evaluation import preserve_global_rngs
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from pong_action_credit import clone_session,legal_input
from pong_rally_credit import branch,query_positions,burst_plan
import run_practical_pong_v2 as base

CONFIG=dict(cases=12,control_cases=6,leads=[40,80,120],burst_camera_lengths=[10,20,40,80],workers=4,wall_seconds=300,camera_cap=64000,stop_first_reward_success=True)


def pins():return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_discovery_pong.py','pong_rally_credit.py','pong_action_credit.py')}


def stage(directory,baseline):
    directory=Path(directory);baseline=Path(baseline);m=base.read(baseline/'manifest.json')
    assert base.read(baseline/'completion-verification.json')['passed'];directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    for name in ('cases','branches','witnesses'):(directory/name).mkdir()
    for name in ('carry.npy','carry_theta.npy','old-transform.npy','extra-transform.npy','search-transform.npy','radial-cap.json'):(directory/name).write_bytes((baseline/name).read_bytes())
    records=[dict(base.read(p),receipt=str(p)) for p in sorted((baseline/'episodes').glob('*.json'))]
    rng=np.random.default_rng(1530982761);cases=[]
    for phase in ('left','right'):
        eligible=[r for r in records if r['report']['losses'] and r['report']['hits']>=1 and base.leftward(r['seed'])==(phase=='left')]
        selected=[eligible[int(i)] for i in rng.permutation(len(eligible))[:6]];assert len(selected)==6
        for i,r in enumerate(selected):
            last=max(e['transition'] for e in r['opportunity_events'] if e['kind']=='own' and e['success'])
            positions=query_positions(r['transitions'],last)
            cases.append(dict(seed=r['seed'],phase=phase,baseline_receipt=r['receipt'],baseline_feature=str(baseline/r['feature_file']),baseline_transitions=r['transitions'],previous_hits=r['report']['hits'],
                last_own_hit_transition=last,positions=positions,k1_control=i<3,burst_plan=burst_plan(r['transitions'],positions)))
    cases.sort(key=lambda c:(not c['k1_control'],c['seed']))
    reservation=sum(c['baseline_transitions']+sum(c['baseline_transitions']-p['transition'] for p in c['burst_plan'])+(sum(2*(c['baseline_transitions']-t) for t in c['positions']) if c['k1_control'] else 0) for c in cases)
    reservation+=max(cases[0]['baseline_transitions']-t for t in cases[0]['positions'])
    assert reservation<=64000
    base.atomic_json(directory/'schedules.json',dict(cases=cases,baseline_schedule=base.read(baseline/'schedules.json'),selection='six native later misses per initial serve direction; fixed random1530982761',maximum_camera_reservation=reservation))
    base.atomic_json(directory/'manifest.json',dict(config=CONFIG,files=pins(),source=base.source_signature(),baseline=str(baseline),carry_sha256=base.sha(directory/'carry.npy'),
        matrix_sha256=base.sha(directory/'search-transform.npy'),maximum_camera_reservation=reservation,
        supervision='native next-own-hit or own-loss only; prior episode hits are subtracted; opponent endings/deadlines are censored',
        assistance='exact simulator/vision prefix resets and held command search; deployed controller stays K1',
        labels='successful burst is a whole rewarded behavioral witness; not proof of first-action or per-frame causal preference',
        no_learning=True,heldout_untouched=True))
    print(json.dumps(dict(staged=True,cases=12,k1_controls=6,burst_branches=sum(len(c['burst_plan']) for c in cases),maximum_camera_reservation=reservation,new_observations=0)),flush=True)


def case_task(directory,c,index,deadline):
    directory=Path(directory);began=time.monotonic();baseline=base.read(c['baseline_receipt']);weights=np.load(directory/'carry.npy');matrix=np.load(directory/'search-transform.npy');snapshots={};results=[];witnesses=[];spent=0
    with np.load(c['baseline_feature']) as values:cached={k:values[k] for k in values.files}
    with preserve_global_rngs():
        random.seed(904271833);np.random.seed(904271833);torch.manual_seed(904271833)
        session=FreshPongEvaluator(weights,PongEvaluationConfig(**baseline['config']))
        for expected in baseline['records']:
            if time.monotonic()>=deadline:break
            if session.initial_cameras==session.episodes:session._begin_episode()
            if session.transitions in c['positions']:
                assert np.array_equal(legal_input(session,matrix),cached['X'][session.transitions]);snapshots[session.transitions]=clone_session(session)
            record=session.step(check_frozen=False);assert record==expected
        session.verify_frozen();prefix_cost=session.transitions
        terminal=fingerprint(session) if index==0 and session.complete else None
        if not session.complete:
            result=dict(seed=c['seed'],phase=c['phase'],complete=False,reason='prefix_deadline',prefix_cameras=prefix_cost,initial_cameras=session.initial_cameras,suffix_cameras=0,branches=[],witnesses=[])
            base.atomic_json(directory/'cases'/f'case_{index:02d}.json',result);return result
        assert session.episode_hits==c['previous_hits'] and session.summary()['losses']==1 and len(snapshots)==len(c['positions'])
        def execute(t,action,hold,kind):
            nonlocal spent
            receipt,data,ending=branch(snapshots[t],action,hold,matrix,deadline,c['baseline_transitions']-t)
            receipt.update(kind=kind,seed=c['seed'],phase=c['phase'],transition=t,index=len(results));spent+=receipt['suffix_transition_cameras']
            if kind=='null':
                records=[dict(v) for v in data['records']]
                for v in records:v.pop('intervention_action',None);v.pop('unmodified_greedy_action',None)
                assert receipt['outcome']==0 and receipt['previous_own_hits']>0 and records==baseline['records'][t:] and fingerprint(ending)==terminal
                receipt['all_records_and_terminal_graph_exact']=True
            elif receipt['outcome']==1:
                # Full legal roll-in prefix + forced and released suffix. No physics
                # diagnostics are copied into learner data or action targets.
                name=f'witnesses/c{index:02d}_b{len(results):03d}.npz'
                np.savez(directory/name,X=np.vstack([cached['X'][:t],data['X']]),actions=np.r_[cached['actions'][:t],data['actions']],q=np.vstack([cached['q'][:t],data['q']]),
                    forced=np.r_[np.zeros(t,dtype=bool),data['forced']],transition_indices=np.arange(t+len(data['X'])))
                receipt['feature_file']=name;receipt['contrasting_forced_actions']=int(np.sum(data['forced']&(data['actions']!=np.argmax(data['q'],axis=1))))
                witnesses.append(dict(seed=c['seed'],phase=c['phase'],feature_file=name,kind=kind,hold=hold,transition=t,contrasting_forced_actions=receipt['contrasting_forced_actions']))
            results.append(receipt);base.atomic_json(directory/'branches'/f'c{index:02d}_b{len(results)-1:03d}.json',receipt)
            base.atomic_json(directory/'cases'/f'case_{index:02d}.partial.json',dict(seed=c['seed'],phase=c['phase'],prefix_cameras=prefix_cost,suffix_cameras=spent,initial_cameras=1,branches=len(results),witnesses=witnesses))
            return receipt
        if index==0:
            t=c['positions'][0];execute(t,int(cached['actions'][t]),1,'null')
        if c['k1_control']:
            for t in c['positions']:
                for action in range(3):
                    if action==int(cached['actions'][t]):continue
                    if time.monotonic()>=deadline:break
                    execute(t,action,1,'K1')
        if not witnesses:
            for p in c['burst_plan']:
                if time.monotonic()>=deadline:break
                receipt=execute(p['transition'],p['action'],p['effective_hold'],'burst')
                if receipt['outcome']==1:break
        result=dict(seed=c['seed'],phase=c['phase'],complete=time.monotonic()<deadline,reason='witness_discovered' if witnesses else 'plan_finished' if time.monotonic()<deadline else 'deadline_censored',
            prefix_cameras=prefix_cost,initial_cameras=1,suffix_cameras=spent,branches=results,witnesses=witnesses,wall_seconds=time.monotonic()-began,k1_control=c['k1_control'])
        base.atomic_json(directory/'cases'/f'case_{index:02d}.json',result);(directory/'cases'/f'case_{index:02d}.partial.json').unlink(missing_ok=True);return result


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');s=base.read(directory/'schedules.json')
    assert m['files']==pins() and m['source']==base.source_signature() and m['config']==CONFIG and base.sha(directory/'carry.npy')==m['carry_sha256']
    assert not (directory/'work-clock.json').exists();began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));deadline=began+300;results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(s['cases']));active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:i,c=next(pending)
                except StopIteration:break
                active[pool.submit(case_task,str(directory),c,i,deadline-2)]=(i,c)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);results.append(f.result())
            base.atomic_json(directory/'progress.json',dict(cases=len(results),repaired=sum(bool(r['witnesses']) for r in results),policy_cameras=sum(r['prefix_cameras']+r['suffix_cameras'] for r in results)))
            fill()
    branches=[b for r in results for b in r['branches']];witnesses=[w for r in results for w in r['witnesses']]
    summary=dict(status='complete' if len(results)==12 and all(r['complete'] for r in results) else 'bounded_partial',cases=len(results),repaired_contexts=len({w['seed'] for w in witnesses}),witnesses=witnesses,
        K1=dict(contexts=sum(r.get('k1_control',False) for r in results),branches=sum(b['kind']=='K1' for b in branches),return_branches=sum(b['kind']=='K1' and b['outcome']==1 for b in branches),
            repaired_contexts=len({b['seed'] for b in branches if b['kind']=='K1' and b['outcome']==1})),
        burst=dict(branches=sum(b['kind']=='burst' for b in branches),return_branches=sum(b['kind']=='burst' and b['outcome']==1 for b in branches),
            repaired_contexts=len({b['seed'] for b in branches if b['kind']=='burst' and b['outcome']==1}),duration_counts={str(k):sum(b['kind']=='burst' and b['hold']==k for b in branches) for k in (10,20,40,80)}),
        censored_branches=sum(b['outcome'] is None for b in branches),prefix_policy_cameras=sum(r['prefix_cameras'] for r in results),suffix_policy_cameras=sum(r['suffix_cameras'] for r in results),
        all_policy_transition_cameras=sum(r['prefix_cameras']+r['suffix_cameras'] for r in results),initial_cameras=sum(r['initial_cameras'] for r in results),
        wall_seconds=time.monotonic()-began,no_fit_or_candidate_deployed=True,source_frozen=m['files']==pins() and m['source']==base.source_signature(),
        interpretation='TRAINsearch with fixed learned roll-in/frozen continuation; heldaction return is rewarded witness evidence, not individual-action causality; stopfirstsuccess creates conditional duration exposure')
    assert summary['all_policy_transition_cameras']<=64000
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['stage','run']);p.add_argument('directory');p.add_argument('--baseline',default='/workspace/m1b-experiments/pong-rally-baseline-20261027-r1');a=p.parse_args()
    if a.operation=='stage':stage(a.directory,a.baseline)
    else:run(a.directory)
