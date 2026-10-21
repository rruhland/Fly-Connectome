"""Posthoc physics/counting and paired fixed-policy snapshot effects only."""
import json,sqlite3
import numpy as np
from fly_connectome.control.environment import PongCamera
from return_metrics import ReturnCounter,aggregate


def annotate_job(directory,job):
    path=directory/(job['id']+'.db')
    if not path.exists():return dict(rallies=[],camera_rows=0,logical_visual_images=0)
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as c:
        rows=[json.loads(raw) for (raw,) in c.execute('SELECT record FROM replay_samples ORDER BY seq')]
    groups={};last=None
    for i,row in enumerate(rows):
        if row['transition']!=i:raise ValueError('stream transition sequence mismatch')
        episode=row['episode'];metadata=row['visual_learning']
        expected_input=1 if last is None else last['visual_learning']['next_observations']+int(last['episode']!=episode)
        if (metadata['enabled']!=job['learn_enabled'] or metadata['input']['observations']!=expected_input
                or metadata['next_observations']!=expected_input+1
                or metadata['input']['learning_observations']!=(expected_input if job['learn_enabled'] else 0)
                or metadata['next_learning_observations']!=(expected_input+1 if job['learn_enabled'] else 0)):
            raise ValueError('persistent logical visual counter mismatch')
        groups.setdefault(episode,[]).append(row);last=row
    if list(groups)!=list(range(len(groups))):raise ValueError('persistent episode schedule mismatch')
    rallies=[]
    for episode,records in groups.items():
        env=PongCamera(job['episode_seeds'][episode],max_episode_ticks=job['config']['max_episode_ticks'],
                       opponent_speed=job['config']['opponent_speed'])
        original=env.game.step;counter=ReturnCounter()
        def instrument(drive,**kwargs):return counter.step(env.game,original,drive,**kwargs)
        env.game.step=instrument
        for row in records:
            result=env.step(row['action'])
            for name in ('reward','hits','elapsed_seconds','physics_ticks','terminated','truncated'):
                if getattr(result,name)!=row[name]:raise ValueError(f'physical replay mismatch: {job["id"]}/{episode}/{name}')
        final=records[-1];partial=not (final['terminated'] or final['truncated'])
        if partial and episode!=next(reversed(groups)):raise ValueError('nonfinal episode is unfinished')
        rally=counter.summary(env.game,terminated=final['terminated'],truncated=final['truncated'],
            reward=sum(row['reward'] for row in records),seconds=sum(row['elapsed_seconds'] for row in records))
        p=env.game.config;x=float(env.game.ball[0,0]);vx=float(env.game.ball_velocity[0,0])
        rally.update(episode=episode,seed=job['episode_seeds'][episode],camera_rows=len(records),right_censored=partial,
            first_encounter_censored=bool(partial and counter.first_incoming is None))
        if partial:
            rally['unfinished_incoming_approach']=bool(vx<0 and x>=p.paddle_x+p.ball_radius)
            rally['unfinished_opponent_approach']=bool(vx>0 and x<=1-p.paddle_x-p.ball_radius)
        rallies.append(rally)
    return dict(rallies=rallies,camera_rows=len(rows),initial_images=len(groups),
        logical_visual_images=len(rows)+len(groups),learning_images=(len(rows)+len(groups)) if job['learn_enabled'] else 0,
        raw_hits=sum(row['hits'] for row in rows),completed_episode_hits=sum(r['own_returns'] for r in rallies if not r['right_censored']),
        partial_episode_hits=sum(r['own_returns'] for r in rallies if r['right_censored']),
        first_encounter_censored_tails=sum(r['first_encounter_censored'] for r in rallies),
        physics_reexecuted_cameras=len(rows),new_camera_observations_from_annotation=0,
        aggregate=aggregate(rallies))


def paired_snapshot_effects(groups,*,seed,draws=10000):
    """Resample shared serves, retaining both policies and both snapshots."""
    names=('seed1_before','seed1_after','seed2_before','seed2_after')
    if any(len(groups.get(name,[]))!=64 for name in names):return dict(complete=False)
    for i in range(64):
        signatures=[]
        for name in names:
            first=groups[name][i]['first_incoming']
            signatures.append(None if first is None else {k:v for k,v in first.items() if k!='success'})
        if any(s!=signatures[0] for s in signatures[1:]):raise ValueError('snapshot probe first geometry mismatch')
    index=np.random.default_rng(seed).integers(64,size=(draws,64));rates={};points={}
    for name in names:
        data=groups[name];hits=np.array([r['own_returns'] for r in data]);opps=np.array([r['own_opportunities'] for r in data])
        present=np.array([r['first_incoming'] is not None for r in data])
        first=np.array([bool(r['first_incoming'] and r['first_incoming']['success']) for r in data])
        if not present.sum() or not opps.sum():
            return dict(complete=True,analyzable=False,reason='A complete probe cohort has no incoming opportunities')
        with np.errstate(divide='ignore',invalid='ignore'):
            rates[name]=dict(first=first[index].sum(1)/present[index].sum(1),all=hits[index].sum(1)/opps[index].sum(1))
            points[name]=dict(first=float(first.sum()/present.sum()),all=float(hits.sum()/opps.sum()))
    deltas={label:{metric:rates[label+'_after'][metric]-rates[label+'_before'][metric] for metric in ('first','all')}
            for label in ('seed1','seed2')}
    report=dict(complete=True,serve_clusters=64,bootstrap_draws=draws,bootstrap_seed=seed,
        scope='Equal weighting of this fixed two-policy set; no per-episode IID or training-population inference',
        per_policy={},formal_noninferiority_gate=None)
    def describe(delta,point):
        valid=np.isfinite(delta)
        return dict(point=point,ci95=np.quantile(delta[valid],[.025,.975]).tolist() if valid.any() else None,
                    invalid_bootstrap_draws=int((~valid).sum()))
    for label,values in deltas.items():
        report['per_policy'][label]={metric:describe(delta,points[label+'_after'][metric]-points[label+'_before'][metric])
                                     for metric,delta in values.items()}
    report['equal_policy_average']={metric:describe(np.mean([deltas[label][metric] for label in deltas],axis=0),
        float(np.mean([points[label+'_after'][metric]-points[label+'_before'][metric] for label in deltas])))
        for metric in ('first','all')}
    return report
