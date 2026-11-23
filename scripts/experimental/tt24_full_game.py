"""Play complete alternating-service games with direct sensory-to-joint policies."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read,_write
from .tt8_service_arena import ServiceScore
from .tt19_view_full_game import digest_json
from .tt22_opponent import RallyArena,install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt22_rally_trial import CONFIG,alternating
from .tt23_coach_trial import PRACTICE,ReplyRanking
from .tt24_rally_bank import BROAD_SETTINGS,CACHE,DIMS
from .tt24_rally_policy import JointPolicy,ViewHistory
from .tt24_rally_trial import COACHED,STEP_SIZE,PublicBody

DIR=Path('artifacts/tt24')
SERVE_SOURCE=Path('artifacts/tt22/ordinary-game.json.gz')


def arena():
    return RallyArena(**dict(CONFIG,width=DIMS[0],height=DIMS[1]))


def serve_examples():
    env=arena()
    rows=[r for r in _read(SERVE_SOURCE)['branches']['online'] if r['role']==1 and r['outcome']['legal_serve']]
    features,postures,rates=[],[],[]
    try:
        for row in rows:
            env.reset(seed=row['seed'],server=1,render=False)
            h,b=ViewHistory(),PublicBody(env)
            for near,far in zip(row['actions'],row['rival_actions'],strict=True):
                near=np.array(near)
                x=h.observe(env.render_cameras(),env.joints,b.target)
                env._aim=env._command_to=lambda *args: np.array(far)
                _,_,_,_,_=env.step(near,render=False)
                h.issued(near)
                b.issued(near)
                features.append(x)
                postures.append(b.target.copy())
                rates.append(near)
            if digest_json(row['events'])!=digest_json(env.events):
                raise ValueError(f"archived service physics mismatch on {row['seed']}")
    finally:
        env.close()
    return tuple(map(np.array,(features,postures,rates))),[r['seed'] for r in rows]


def game(env,tape,policy,seed,first,mode):
    scoring=ServiceScore(first_server=first)
    rows=[]
    for point_seed in np.random.default_rng(seed).integers(1,2**31-1,100):
        if scoring.finished:
            return rows,list(scoring.score)
        role=scoring.server
        env.reset(seed=int(point_seed),server=role,render=False)
        tape.steps.clear()
        h,b=ViewHistory(),PublicBody(env)
        commands,history=[],[]
        digest=sha256()
        for tick in range(env.max_steps):
            images=env.render_cameras()
            for image in images.values():
                digest.update(image)
            x=h.observe(images,env.joints,b.target)
            history.append(x)
            if mode=='delay':
                x=x.copy()
                x[:14]=history[max(0,tick-6)][:14]
            u=policy.command(x,b.target)
            _,reward,end,cap,_=env.step(u,render=False)
            h.issued(u)
            b.issued(u)
            commands.append(u.tolist())
            if end or cap:
                break
        outcome=env.point_result()
        before=list(scoring.score)
        after=list(scoring.record(outcome))
        rows.append(dict(seed=int(point_seed),server=role,before=before,after=after,events=env.events.copy(),
                         outcome=outcome,reward=reward,longest=alternating(env.events),rgb=digest.hexdigest(),
                         near=commands,far=tape.tape(len(commands))))
    raise ValueError('100 points elapsed before game completion')


def verify(rows,first):
    env=arena()
    scoreboard=ServiceScore(first_server=first)
    try:
        for row in rows:
            if row['server']!=scoreboard.server or row['before']!=list(scoreboard.score):
                raise ValueError('service rotation differs')
            env.reset(seed=row['seed'],server=row['server'],render=False)
            for i,(near,far) in enumerate(zip(row['near'],row['far'],strict=True)):
                env._aim=env._command_to=lambda *args:np.array(far)
                _,reward,end,cap,_=env.step(near,render=False)
                if bool(end or cap)!=(i+1==len(row['near'])):
                    raise ValueError('terminal frame differs')
            if reward!=row['reward'] or digest_json(row['events'])!=digest_json(env.events) or env.point_result()!=row['outcome']:
                raise ValueError('referee replay differs')
            if list(scoreboard.record(env.point_result()))!=row['after']:
                raise ValueError('game score differs')
        return scoreboard.finished
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--games',type=int,default=2)
    args=p.parse_args()
    serve,seeds=serve_examples()
    with np.load(CACHE) as broad:
        original=[broad[k] for k in ('features','postures','rates')]
    combined=[np.concatenate((a,b)) for a,b in zip(original,serve)]
    policies={k:JointPolicy(*(original if k=='receiving' else combined),STEP_SIZE,**BROAD_SETTINGS)
              for k in ('receiving','combined','delay')}
    env=arena()
    rival=install_rally_opponent(env,target_profile=_read(COACHED)['metadata']['opening']['far_profile'])
    rival.confirm=ReplyRanking(rival,_read(PRACTICE)).confirm
    tape=FarTape(env)
    docs=[]
    try:
        for i in range(args.games):
            seed=6208939+i
            first=1 if i%2 else -1
            for mode,pupil in policies.items():
                rows,score=game(env,tape,pupil,seed,first,mode)
                exact=verify(rows,first)
                print(i,mode,score,sum(r['outcome']['legal_returns'] for r in rows),max(r['longest'] for r in rows),flush=True)
                docs.append(dict(seed=seed,first=first,mode=mode,rows=rows,score=score,verified=exact))
                _write(args.out,dict(serve_demonstrations=seeds,serve_frames=len(serve[0]),games=docs))
    finally:
        env.close()


if __name__=='__main__':
    main()
