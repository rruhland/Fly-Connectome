"""Bounded ordinary-game TRAIN collection for sustained native Pong returns.

No fitting. Legal pre-action features are separate from diagnostic physics. The
contact logger only reads native outcomes and is removed before frozen checks.
"""
import argparse,json,multiprocessing as mp,random,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.fresh_evaluation import preserve_global_rngs
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fast_pong_feature_rollout import run_episode
from pong_action_credit import legal_input
from return_metrics import ReturnCounter,aggregate
import run_practical_pong_v2 as base

CARRY=base.EXPERIMENTS/'pong-interception-checkpoint-20261025'
EXPECTED='184dd67a67999224dbe4eb5ff3ba3b86ebdf821d15f1486eef2432a56e99fb15'
CONFIG=dict(serves=64,episode_ticks=2400,wall_seconds=300,workers=4,opponent_speed=.65,global_bookkeeping_seed=904271833)
ROOT=3648291701;TAG=62731


def pins():
    return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_baseline_pong.py','fast_pong_feature_rollout.py','pong_action_credit.py','return_metrics.py')}


def schedule():
    reserved=set(base.old_engine.FINAL)
    def collect(v):
        if type(v) is int and 0<v<2**32:reserved.add(v)
        elif isinstance(v,list):
            for item in v:collect(item)
        elif isinstance(v,dict):
            for item in v.values():collect(item)
    for p in base.EXPERIMENTS.rglob('schedules.json'):collect(base.read(p))
    for p in base.EXPERIMENTS.glob('*/seed-exclusions.json'):collect(base.read(p).get('ids',[]))
    reserved.add(ROOT);canonical={v%base.MODULUS for v in reserved};rng=np.random.default_rng(np.random.SeedSequence([ROOT,TAG]));accepted=[];rejected=[]
    while len(accepted)<CONFIG['serves']:
        seed=int(rng.integers(1,2**32,dtype=np.uint64))
        if seed%base.MODULUS in canonical:rejected.append(seed)
        else:accepted.append(seed)
        canonical.add(seed%base.MODULUS)
    return dict(train=accepted,rejected=rejected,root=ROOT,tag=TAG,selection='fresh unfiltered native ordinary starts; not heldout')


def stage(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);assert not (directory/'manifest.json').exists()
    for name in ('episodes','features'):(directory/name).mkdir()
    for source,name in [('weights.npy','carry.npy'),('theta.npy','carry_theta.npy'),('old-transform.npy','old-transform.npy'),('extra-transform.npy','extra-transform.npy'),('radial-cap.json','radial-cap.json')]:
        (directory/name).write_bytes((CARRY/source).read_bytes())
    theta=np.load(directory/'carry_theta.npy');matrix=np.vstack([np.load(directory/'old-transform.npy'),np.load(directory/'extra-transform.npy')]);base.save_array(directory/'search-transform.npy',matrix)
    assert base.sha(directory/'carry.npy')==EXPECTED
    assert np.array_equal(theta[:,:214]@matrix[:214]+theta[:,214:]@matrix[214:],np.load(directory/'carry.npy'))
    base.atomic_json(directory/'schedules.json',schedule());base.atomic_json(directory/'manifest.json',dict(source=base.source_signature(),files=pins(),config=CONFIG,
        carry=str(CARRY/'weights.npy'),weights_sha256=EXPECTED,scaling_sha256=base.sha(directory/'radial-cap.json'),projection_sha256=base.sha(directory/'search-transform.npy'),
        purpose='TRAIN coverage, no candidate learning; future heldout must be fresh',
        data_separation='features.npz contains only legal policy inputs/actions; beliefs are camera-derived; diagnostic_physics_only never supplies policy inputs or action labels',
        counterfactual_credit_warning='later-hit outcome must compare delta hits since snapshot, not total episode hits>0',
        hard_observation_cap=65000,training_camera_cap=64000,parity_camera_cap=1000,initials_separate=True,parity='firstTRAINepisode separately replayed with unchanged lean engine; all extra policy cameras charged as parity replay'))
    print(json.dumps(dict(staged=True,new_cameras=0,train_serves=64,config=CONFIG)),flush=True)


def collect_one(directory,seed,index,deadline,parity=False):
    directory=Path(directory);began=time.monotonic();weights=np.load(directory/'carry.npy');theta=np.load(directory/'carry_theta.npy');matrix=np.load(directory/'search-transform.npy')
    config=PongEvaluationConfig(seed=seed,episodes=1,max_episode_ticks=CONFIG['episode_ticks'],opponent_speed=.65,policy_scaling=base.read(directory/'radial-cap.json'))
    records=[];xs=[];beliefs=[];events=[];diagnostics=[];counter=ReturnCounter();error=0.
    with preserve_global_rngs():
        random.seed(CONFIG['global_bookkeeping_seed']);np.random.seed(CONFIG['global_bookkeeping_seed']);torch.manual_seed(CONFIG['global_bookkeeping_seed'])
        session=FreshPongEvaluator(weights,config);game=session.env.game;original=game.step
        def instrument(drive,**kwargs):
            own=counter.opportunities;opponent=counter.opponent_opportunities;tick=int(game.rally_steps[0])+1
            out=counter.step(game,original,drive,**kwargs)
            if counter.opportunities>own:events.append(dict(kind='own',ordinal=counter.opportunities,success=bool(out.hits[0]),tick=tick,transition=session.transitions,completed_own_hits=counter.returns))
            if counter.opponent_opportunities>opponent:events.append(dict(kind='opponent',ordinal=counter.opponent_opportunities,success=float(game.ball_velocity[0,0])<0,tick=tick,transition=session.transitions,after_own_hits=counter.returns))
            return out
        game.step=instrument
        try:
            while not session.complete and time.monotonic()<deadline:
                if session.initial_cameras==session.episodes:session._begin_episode()
                x=legal_input(session,matrix);xs.append(x)
                encoded=session.latest_encoded
                beliefs.append([dict(id=int(e['id']),position=e['position'].tolist(),observed=bool(e['observed']),velocity=e['observed_velocity'].tolist(),known=bool(e['velocity_known'])) for e in encoded['entities']])
                diagnostics.append(dict(ball=game.ball[0].tolist(),ball_velocity=game.ball_velocity[0].tolist(),paddle=float(game.body.position[0]),paddle_velocity=float(game.body.velocity[0]),paddle_activation=float(game.body.activation[0])))
                r=session.step(check_frozen=False);records.append(r);error=max(error,float(np.max(abs(x@theta.T-r['q']))))
        finally:del game.step
        session.verify_frozen();report=session.summary();terminal=fingerprint(session) if parity else None
    assert error<1e-10
    last=report['rallies'][-1] if report['rallies'] else dict(truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds)
    annotation=counter.summary(game,truncated=last['truncated'],terminated=last['terminated'],reward=last['reward'],seconds=last['seconds'])
    if report['complete']:
        assert annotation['own_returns']==report['hits']
        assert annotation['own_opportunities']==report['hits']+int(report['losses']>0)
    feature_file=f'features/train_{index:02d}.npz';np.savez(directory/feature_file,X=np.asarray(xs),actions=np.array([r['action'] for r in records]),q=np.array([r['q'] for r in records]),transition_indices=np.arange(len(records)))
    result=dict(seed=seed,id=f'train_{index:02d}',complete=report['complete'],config=asdict(config),first_hit_cutoff=False,weights_sha256=base.sha(directory/'carry.npy'),
        report=report,annotation=annotation,opportunity_events=events,feature_file=feature_file,records=records,belief_trace=beliefs,
        diagnostic_physics_only=diagnostics,maximum_q_reconstruction_error=error,transitions=session.transitions,initial_cameras=session.initial_cameras,
        learned_contents_unchanged=True,wall_seconds=time.monotonic()-began,parity_replay=None)
    if parity and report['complete']:
        reference=run_episode(weights,config,first_hit_cutoff=False,trace=True,terminal_digest=True,deadline_monotonic=deadline)
        assert reference['complete'] and reference['records']==records and reference['terminal_state_hash']==terminal
        result['parity_replay']=dict(passed=True,all_records_and_terminal_graph_exact=True,transition_cameras=reference['transitions'],initial_cameras=reference['initial_cameras'],wall_seconds=reference['wall_seconds'])
    base.atomic_json(directory/'episodes'/f'train_{index:02d}.json',result);return result


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');s=base.read(directory/'schedules.json')
    assert m['files']==pins() and m['source']==base.source_signature() and m['config']==CONFIG
    assert base.sha(directory/'carry.npy')==EXPECTED and not (directory/'work-clock.json').exists()
    began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));deadline=began+300;results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(s['train']));active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-3:
                try:i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(collect_one,str(directory),seed,i,deadline-2,i==0)]=(i,seed)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:active.pop(future);results.append(future.result())
            base.atomic_json(directory/'progress.json',dict(complete=sum(r['complete'] for r in results),episodes=len(results),policy_cameras=sum(r['transitions'] for r in results)))
            fill()
    complete=len(results)==64 and all(r['complete'] for r in results);annotations=[r['annotation'] for r in results];by_depth={}
    for depth in range(1,1+max((r['annotation']['own_opportunities'] for r in results),default=0)):
        events=[e for r in results for e in r['opportunity_events'] if e['kind']=='own' and e['ordinal']==depth]
        by_depth[str(depth)]=dict(opportunities=len(events),returns=sum(e['success'] for e in events),misses=sum(not e['success'] for e in events))
    parity=[r['parity_replay'] for r in results if r['parity_replay']]
    summary=dict(status='complete' if complete else 'bounded_partial',current_only_no_learning=True,train_serves=len(results),aggregate=aggregate(annotations),contact_depth=by_depth,
        per_game_return_tail={str(k):sum(r['report']['hits']>=k for r in results) for k in range(1,7)},
        own_losses=sum(r['report']['losses'] for r in results),opponent_miss_endings=sum(r['report']['wins'] for r in results),
        natural_opponent_misses=sum(r['report']['natural_serve_wins'] for r in results),timeouts=sum(r['report']['truncations'] for r in results),
        training_transition_cameras=sum(r['transitions'] for r in results),initial_cameras=sum(r['initial_cameras'] for r in results),
        parity_replay_transition_cameras=sum(r['transition_cameras'] for r in parity),parity_replay_initials=sum(r['initial_cameras'] for r in parity),
        all_actual_policy_transition_cameras=sum(r['transitions'] for r in results)+sum(r['transition_cameras'] for r in parity),
        wall_seconds=time.monotonic()-began,source_frozen=m['files']==pins() and m['source']==base.source_signature(),weights_unchanged=base.sha(directory/'carry.npy')==EXPECTED,
        notes='Native contact counts are diagnostic environment events, never observations/action labels. Successful and failed later contact exposure is policy-dependent; do not call conditional rates causal learning evidence.')
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['stage','run']);p.add_argument('directory');a=p.parse_args()
    if a.operation=='stage':stage(a.directory)
    else:run(a.directory)
