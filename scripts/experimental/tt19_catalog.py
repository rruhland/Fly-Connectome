"""Plan RGB-conditioned shots from a catalogue of public CAD arm movements.

The dry model contains no incoming ball and receives no target measurements.
It supplies a factory-informed alternative to fitting the arm from RGB.
"""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from mujoco import rollout
from .tt4_arena import mujoco, TABLE_HEIGHT, SOLVER_SUBSTEPS
from .tt4_transfer_trial import _read, _write, _hash
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore
from .tt19_flight import FlightLaw, optical_start
from .tt19_practice_camera import PracticeCamera,BallEyes
from .tt19_shoot import HORIZON,_surface,_landings
from .tt19_so101 import SO101Arena,SOURCE_REVISION
from .tt19_stroke_trial import DONOR
from .tt19_view import ViewTransport
from .tt19_view_trial import SO101ViewArena


def programs(count, wide=False):
    rng=np.random.default_rng(188209611)
    offsets=rng.uniform(-1.,1.,(count,5))*rng.uniform(.1,2.5,(count,1))
    swings=rng.uniform(-1.,1.,(count,5))*rng.uniform(.6,5.5,(count,1))
    if wide:
        offsets[:,0]=rng.uniform(-6.2,6.2,count)
        offsets[:,1:4]*=1.35
    starts=rng.integers(1,16,count)
    periods=rng.integers(2,8,count)
    ramps=rng.integers(2,7,count)
    ticks=np.arange(HORIZON)[None,:]
    posture=np.minimum(1.,(ticks+1)/ramps[:,None])
    progress=np.clip((ticks-starts[:,None]+1)/periods[:,None],0.,1.)
    goals=posture[...,None]*offsets[:,None,:]+progress[...,None]*swings[:,None,:]
    tape=np.clip(np.diff(goals,axis=1,prepend=np.zeros((count,1,5))),-1.,1.)
    tape[0]=0.
    return tape


def dry_catalog(path,count,wide=False):
    start=perf_counter()
    arena=SO101Arena(venue='hall',detail='preview',strict_endline=True)
    try:
        arena.reset(seed=842,render=False)
        arena.ball[:]=[1.4,2.0,.15]
        arena.velocity[:]=0.
        for _ in range(8*arena.frameskip*SOLVER_SUBSTEPS):
            mujoco.mj_step(arena.model,arena.data)
        model=arena.model
        spec=mujoco.mjtState.mjSTATE_FULLPHYSICS
        initial=np.empty(mujoco.mj_stateSize(model,spec))
        mujoco.mj_getState(model,arena.data,initial,spec)
        # The catalogue uses the known simulator actuator increment and home
        # in addition to the published arm, gripper and paddle attachment.
        actuators=arena._arm_ids[1][2][:5]
        timestep=arena.frameskip*SOLVER_SUBSTEPS
        reference=arena.data.ctrl.copy()
        joint=1+int(model.joint('learner_paddle_free').qposadr[0])
        pose0=initial[joint:joint+7].copy()
        geom=model.geom('learner_racket')
        local_center=geom.pos.copy()
        local_normal=np.array([1.,0.,0.])
        # Read the fixed mounting rotation from the dry model.
        r=np.empty(9);mujoco.mju_quat2Mat(r,geom.quat)
        local_normal=r.reshape(3,3)[:,2].copy()

        def blade(pose):
            w=pose[...,3:4]; vec=pose[...,4:7]
            def rotate(axis):
                delta=2*np.cross(vec,axis)
                return axis+w*delta+np.cross(vec,delta)
            normal=rotate(local_normal)
            normal=np.where(normal[...,1:2]>=0,normal,-normal)
            xyz=pose[...,:3]+rotate(local_center)-[0.,0.,TABLE_HEIGHT]
            return np.concatenate((xyz,normal[...,[0,2]]),axis=-1)

        tapes=programs(count,wide)
        bank=[]
        for group in np.array_split(tapes,max(1,int(np.ceil(count/24)))):
            steps=np.repeat(group,timestep,axis=1)
            ctrl=np.broadcast_to(reference,(len(group),steps.shape[1],len(reference))).copy()
            q=reference[actuators]+np.cumsum(steps*arena._joint_increments[actuators],axis=1)
            ctrl[:,:,actuators]=np.clip(q,arena._joint_low[actuators],arena._joint_high[actuators])
            state,_=rollout.rollout(model,mujoco.MjData(model),initial[None],ctrl)
            poses=state[:,timestep-1::timestep,joint:joint+7]
            first=np.broadcast_to(pose0,(len(group),1,7))
            bank.append(blade(np.concatenate((first,poses),axis=1)))
        points=np.concatenate(bank)
        np.savez_compressed(path,tapes=tapes.astype('float32'), poses=points.astype('float32'))
        return dict(count=count,frames=HORIZON,source=SOURCE_REVISION,home=points[0,0].tolist(),
                    max_drift_m=float(np.max(np.linalg.norm(points[0,:,:3]-points[0,0,:3],axis=1))),
                    wall_seconds=perf_counter()-start)
    finally:
        arena.close()


def candidate_costs(poses,law,response,points,speeds,serving):
    hit,eta,ball,v,n,blade,gap,radial,approach=_surface(poses,points,speeds)
    cost,landing,net,outbound,first_y=_landings(ball,v,n,blade,law,response,serving)
    cost=cost+25*np.maximum(0.,gap)+3.*(~hit)+1.7*(~approach)
    cost+=3.8*np.maximum(0.,radial-.048)+.13*(radial/.075)**2
    cost+=6*((v[:,2]>-.014) if serving else (v[:,1]>-.015))
    return cost,eta,hit,landing,net,first_y,outbound,ball


def choose(poses,law,response,start,serving,robust):
    points,speeds=law.trajectory(*start,HORIZON,bounds=(.424,.80))
    parts=[candidate_costs(poses[i:i+256],law,response,points,speeds,serving)
           for i in range(0,len(poses),256)]
    cost,eta,hit,landing,net,first_y,outbound,ball=map(np.concatenate,zip(*parts))
    if robust:
        count=min(100,len(poses))
        selection=np.argpartition(cost,count-1)[:count]
        costs=[cost[selection]]
        # Estimate how sensitive the first physical contact is to optical error,
        # especially close to the edge of the racket or a predicted bounce.
        for offset in ((.012,0,.012),(-.012,0,-.012),(0,.018,-.013),(0,-.018,.013)):
            perturbed=points+np.array(offset)
            costs.append(candidate_costs(poses[selection],law,response,perturbed,speeds,serving)[0])
        stacked=np.sort(costs,axis=0)
        j=int(selection[np.argmin(stacked.mean(axis=0)+.45*stacked[-2])])
    else:
        j=int(np.argmin(cost))
    return (float(cost[j]),j,float(eta[j]),bool(hit[j]),landing[j].tolist(),float(net[j]),
            float(first_y[j]),outbound[j].tolist(),ball[j].tolist())


class Player:
    def __init__(self,catalog,law,response,method):
        self.method=method
        self.poses=catalog['poses']; self.tapes=catalog['tapes']
        self.law=law; self.response=response
        self.reset()

    def reset(self):
        self.history=[]
        self.plans=[]
        self.movement=np.zeros(5)
        self.ends=-1
        self.plan=None
        self.cleared=False

    def command(self,tick,xyz,server):
        self.history.append((tick,xyz))
        if self.plan is not None:
            start,index,until=self.plan
            frame=tick-start
            if frame<until:
                a=self.tapes[index,frame]
                self.movement+=a
                return np.r_[a,0.]
            self.plan=None
            self.ends=tick
        if np.max(abs(self.movement))>.002:
            a=-np.clip(self.movement,-1.,1.)
            self.movement+=a
            return np.r_[a,0.]
        if xyz is not None and xyz[1]>.10:
            self.cleared=True
        if self.plans and (tick-self.ends<5 or not self.cleared):
            return np.zeros(6)
        start=optical_start(self.history,self.law.gravity)
        if start is None or xyz is None:
            return np.zeros(6)
        p,v=start
        serving=server==1 and not self.plans
        if serving:
            ready=tick>=(8 if self.method=="early" else 13) and p[1]<-.6
        else:
            # Accept a reliable approach after the first far bounce, with
            # time left to prepare the arm and identify the near-side bounce.
            limit=.56 if self.method=='reply' else (.32 if self.method=='early' else -.24)
            ready=v[1]<-.031 and -.81<p[1]<limit and tick>=8
        if not ready:
            return np.zeros(6)
        fitted=choose(self.poses,self.law,self.response,start,serving,self.method=="robust")
        cost,index,eta,hit,*rest=fitted
        if not hit and (not serving or tick<12):
            return np.zeros(6)
        duration=min(HORIZON,max(2,int(np.ceil(eta))+2))
        self.plan=(tick,index,duration)
        self.plans.append(dict(tick=tick,index=index,duration=duration,cost=cost,eta=eta,hit=hit,
                               landing=rest[0],net=rest[1],first_y=rest[2],out=rest[3],collision=rest[4]))
        self.cleared=False
        a=self.tapes[index,0];self.movement+=a
        return np.r_[a,0.]


def eval_games(folder,cache,seeds,mode,native,stride=1,method="early"):
    catalog=dict(np.load(cache))
    donor=_read(DONOR)['series']
    law=FlightLaw.from_state(_read(folder/'flight-source.json.gz')['source_law']) if mode=='gantry' else FlightLaw()
    impulse=FlightMemory('impulse',donor['flight_history'],donor['flight_impulses']) if mode=='gantry' else FlightMemory('impulse')
    player=Player(catalog,law,impulse.response_matrix,method)
    rig='wrist+fixed' if native else 'fixed'
    arena=SO101ViewArena(venue='hall',strict_endline=True,width=240,height=180,max_steps=180,view_mode=rig)
    setup=arena.reset(seed=20017); eyes=BallEyes(setup); cam=PracticeCamera(arena)
    transport=ViewTransport(rig)
    results=[]
    try:
        for seed in seeds:
            rng=np.random.default_rng(seed)
            score=ServiceScore(first_server=-1);rows=[]
            while not score.finished and len(rows)<80:
                point=int(rng.integers(2**31));server=score.server
                arena.reset(seed=point,render=False,server=server);player.reset()
                tracker=transport.tracker()
                obs=[];actions=[]
                for t in range(arena.max_steps):
                    ball=None
                    if t%stride==0:
                        if native:
                            tracker.update(arena.render_cameras())
                            if tracker.ball is not None and tracker.ball.observed: ball=tracker.ball.position
                        else: ball=eyes.position(cam.capture(),True)
                    a=player.command(t,ball,server);actions.append(a.tolist());obs.append(None if ball is None else np.asarray(ball).tolist())
                    _,reward,done,timeout,_=arena.step(a,render=False)
                    if done or timeout:break
                result=arena.point_result(); score.record(result)
                rows.append(dict(seed=point,server=server,actions=actions,optical=obs,plans=player.plans,
                                 result=result,events=arena.events,rgb=tracker.digest.hexdigest()))
            g=dict(seed=seed,score=list(score.score),points=rows,finished=score.finished,
                   returns=sum(r['result']['legal_returns'] for r in rows),multiple=sum(r['result']['legal_returns']>=2 for r in rows),
                   serves=sum(r['server']==1 and r['result']['legal_serve'] for r in rows))
            results.append(g);print('held',g['seed'],g['score'],g['returns'],g['multiple'],g['serves'],flush=True)
            _write(folder/f'catalog-{method}-{mode}-{"native" if native else "fast"}.json.gz',dict(seeds=seeds,strict_endline=True,
                   cad_sha256=_hash(cache),native=native,games=results,stride=stride,source_world=law.state(),response=player.response.tolist()))
    finally: arena.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('cache',type=Path)
    p.add_argument('--build',action='store_true');p.add_argument('--count',type=int,default=2048)
    p.add_argument('--method',choices=('early','late','robust','reply'),default='early')
    p.add_argument('--native',action='store_true');p.add_argument('--mode',choices=['gantry','nominal'],default='gantry')
    p.add_argument('--stride',type=int,default=1);p.add_argument('--seeds',nargs='+',type=int,default=[7451032,99168517,6291160])
    a=p.parse_args()
    if a.build:
        _write(a.folder/'catalog-build.json.gz',dry_catalog(a.cache,a.count))
    eval_games(a.folder,a.cache,a.seeds,a.mode,a.native,a.stride,a.method)
