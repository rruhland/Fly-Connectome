"""Isolated presentation sessions over the unchanged frozen evaluator.

Watch mode uses native PongCamera/Pong directly. Human mode substitutes only the
right paddle's controller in a viewer-only Pong subclass. Camera capture, learned
left controller, vision and encoder remain FreshPongEvaluator's existing path.
There is no learning, journal, checkpoint save or background game clock.
"""
import math
import threading
import time
import uuid
import numpy as np
import torch
from ..pong import Pong,Outcome
from ..control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig


class HumanPong(Pong):
    """Native physics with human right-paddle drive instead of the bot's target.

    Deliberately isolated copy of Pong.step, with the desired-velocity line as the
    only controller difference. The ordinary benchmark never instantiates this.
    """
    human_axis=0.

    @torch.no_grad()
    def step(self,drive,*,auto_reset=True):
        if drive.shape!=(self.batch,) or not torch.isfinite(drive).all():
            raise ValueError('finite continuous motor drive required per environment')
        p=self.config;self.body.step(drive.detach())
        desired=torch.full_like(self.opponent,self.human_axis*p.opponent_speed)
        self.opponent_velocity.add_((desired-self.opponent_velocity).clamp(-p.opponent_acceleration*p.dt,p.opponent_acceleration*p.dt))
        self.opponent.add_(p.dt*self.opponent_velocity).clamp_(p.paddle_half,1-p.paddle_half)
        old_x=self.ball[:,0].clone();self.ball.add_(self.ball_velocity*p.dt)
        low=self.ball[:,1]<p.ball_radius;high=self.ball[:,1]>1-p.ball_radius
        self.ball[low,1]=2*p.ball_radius-self.ball[low,1]
        self.ball[high,1]=2*(1-p.ball_radius)-self.ball[high,1]
        self.ball_velocity[:,1]=torch.where(low|high,-self.ball_velocity[:,1],self.ball_velocity[:,1])
        left_plane,right_plane=p.paddle_x+p.ball_radius,1-p.paddle_x-p.ball_radius
        left=((old_x>=left_plane)&(self.ball[:,0]<=left_plane)&(self.ball_velocity[:,0]<0)&((self.ball[:,1]-self.body.position).abs()<=p.paddle_half+p.ball_radius))
        right=((old_x<=right_plane)&(self.ball[:,0]>=right_plane)&(self.ball_velocity[:,0]>0)&((self.ball[:,1]-self.opponent).abs()<=p.paddle_half+p.ball_radius))
        collision=left|right;center=torch.where(left,self.body.position,self.opponent)
        angle=((self.ball[:,1]-center)/p.paddle_half).clamp(-1,1)*.9
        self.ball_velocity[collision,0]=(torch.where(left,1.,-1.)*p.ball_speed*angle.cos())[collision]
        self.ball_velocity[collision,1]=(p.ball_speed*angle.sin())[collision]
        self.ball[left,0]=2*left_plane-self.ball[left,0];self.ball[right,0]=2*right_plane-self.ball[right,0]
        scores=(self.ball[:,0]>1).float()-(self.ball[:,0]<0).float();self.rally_steps.add_(1)
        result=Outcome(scores,left.float(),self.rally_steps.clone())
        if auto_reset:self.reset(scores!=0)
        return result


def vision_overlay(encoded):
    """Detached JSON derived only from the already encoded sensor belief.

    Never calls vision.step/encode and never reads environment ball/paddle state.
    Native positions are row,column pixels; transport velocities are normalized/s.
    """
    if encoded is None:return dict(entities=[],forecasts=[],sample=0)
    entities=[]
    for e in encoded['entities']:
        entities.append(dict(id=int(e['id']),position=e['position'].detach().tolist(),observed=bool(e['observed']),
            age_seconds=float(e['observation_age_seconds']),velocity=e['observed_velocity'].detach().tolist(),velocity_known=bool(e['velocity_known'])))
    forecasts=[]
    for f in encoded['forecasts']:
        weights=f.get('mixture_weights');centers=f.get('mixture_centers')
        components=[]
        if weights is not None and centers is not None:
            # Presentation-only top components; the model's complete mixture is untouched.
            for i in torch.argsort(weights,descending=True)[:4].tolist():
                components.append(dict(position=centers[i].detach().tolist(),weight=float(weights[i])))
        sigma=f.get('component_sigma_pixels')
        forecasts.append(dict(id=int(f['id']),horizon=int(f['horizon_samples']),position=f['position'].detach().tolist(),
            components=components,sigma=None if sigma is None else sigma.detach().tolist() if isinstance(sigma,torch.Tensor) else float(sigma),
            evidence_age_seconds=float(f.get('evidence_age_seconds',0))))
    return dict(entities=entities,forecasts=forecasts,sample=int(encoded['sample']))


class ViewerSession:
    """Request-driven memory only; no client requests means no simulated time."""
    def __init__(self,catalog,checkpoint='final',seed=123456789,mode='watch',clock=time.monotonic):
        self.catalog=catalog;self.clock=clock;self.lock=threading.RLock();self.last_seen=clock();self.revision=0;self.sequence=0
        self.playing=False;self.last_record=None;self.round_message='';self.round_until=0.
        self.reset(checkpoint,seed,mode)

    def reset(self,checkpoint,seed,mode):
        if checkpoint not in self.catalog:raise ValueError('unknown checkpoint')
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('seed must be an integer from 0 to 4294967295')
        if mode not in ('watch','human'):raise ValueError('unknown play mode')
        if hasattr(self,'evaluator'):self.evaluator.verify_frozen()
        c=self.catalog[checkpoint]
        config=PongEvaluationConfig(seed=seed,episodes=1000000,max_episode_ticks=2400,policy_kind='trained',action_hold_frames=1,opponent_speed=.65,policy_scaling=c.scaling)
        self.evaluator=FreshPongEvaluator(c.weights,config)
        if mode=='human':
            self.evaluator.env.game=HumanPong([seed],config=self.evaluator.env.game.config)
        self.evaluator._begin_episode()
        self.checkpoint=checkpoint;self.seed=seed;self.mode=mode;self.last_record=None;self.playing=False
        self.round_message='';self.round_until=0.;self.last_seen=self.clock();self.revision+=1;self.sequence+=1
        self.evaluator.verify_frozen()

    def pause(self):
        self.playing=False
        if self.mode=='human':self.evaluator.env.game.human_axis=0.

    def advance(self,frames=1,*,manual=False,human_axis=0.):
        if type(frames) is not int or not 1<=frames<=4:raise ValueError('advance requires 1 to 4 camera frames')
        if type(manual) is not bool:raise ValueError('manual must be boolean')
        if not isinstance(human_axis,(int,float)) or not math.isfinite(human_axis) or not -1<=human_axis<=1:raise ValueError('human axis must lie between -1 and 1')
        if self.mode=='human' and frames!=1:raise ValueError('human mode advances one camera at a time')
        now=self.clock()
        if now-self.last_seen>1.5:self.pause()
        self.last_seen=now
        if not self.playing and not manual:return []
        if manual:self.pause()
        if self.round_until>now and not manual:return []
        records=[]
        for _ in range(frames):
            if self.mode=='human':self.evaluator.env.game.human_axis=float(human_axis)
            r=self.evaluator.step(check_frozen=False);self.last_record=r;records.append(r);self.sequence+=1
            if r['terminated'] or r['truncated']:
                self.evaluator.verify_frozen()
                self.round_message=('Point · learned paddle scores' if r['reward']>0 else 'Miss · right paddle scores' if r['reward']<0 else '20-second limit · new serve')
                self.round_until=now+.65
                break
            self.round_message='';self.round_until=0.
        return records

    def packet(self):
        if self.clock()-self.last_seen>1.5:self.pause()
        session=self.evaluator;game=session.env.game
        # The raster is the exact last camera passed to vision. UI overlays are
        # returned separately and can never enter the observation path.
        frame=session.previous.detach().flatten().nonzero().flatten().tolist()
        rallies=session.rallies
        last=self.last_record
        return dict(checkpoint=self.checkpoint,seed=self.seed,mode=self.mode,playing=self.playing,revision=self.revision,sequence=self.sequence,
            camera=dict(width=64,height=64,on_pixels=frame),vision=vision_overlay(session.latest_encoded),
            action=None if last is None else int(last['action']),scores=None if last is None else last['q'],
            elapsed_seconds=float(session.physics_seconds),rally_seconds=float(session.episode_seconds),rally=int(session.episodes+1),
            hits=int(sum(r['hits'] for r in rallies)+ (0 if last and (last['terminated'] or last['truncated']) else session.episode_hits)),
            rally_hits=int(session.episode_hits),points=dict(learned=sum(r['reward']>0 for r in rallies),right=sum(r['reward']<0 for r in rallies)),
            timeouts=sum(r['truncated'] for r in rallies),round_message=self.round_message,
            observations=dict(initial=session.initial_cameras,transitions=session.transition_cameras),
            opponent=dict(kind='deterministic native bot' if self.mode=='watch' else 'human-controlled right paddle',speed=.65),
            # Only presentation geometry; the actual game is rendered from camera pixels.
            frozen=True)


class ViewerEngine:
    def __init__(self,catalog):
        self.catalog=catalog;self.sessions={};self.lock=threading.RLock()

    def create(self):
        with self.lock:
            now=time.monotonic()
            for key in list(self.sessions):
                if now-self.sessions[key].last_seen>600:del self.sessions[key]
            if len(self.sessions)>=8:
                raise ValueError('viewer has eight active tabs; close one and try again')
            identity=uuid.uuid4().hex;session=ViewerSession(self.catalog);self.sessions[identity]=session
            return identity,session.packet()

    def command(self,identity,command):
        with self.lock:
            if identity not in self.sessions:raise ValueError('viewer session expired; reload the page')
            state=self.sessions[identity]
            with state.lock:
                action=command.get('action')
                if action=='reset':state.reset(command.get('checkpoint',state.checkpoint),command.get('seed',state.seed),command.get('mode',state.mode))
                elif action=='pause':state.pause();state.revision+=1
                elif action=='play':state.playing=True;state.last_seen=state.clock();state.revision+=1
                elif action in ('advance','step'):
                    if command.get('revision')!=state.revision:return state.packet()
                    state.advance(command.get('frames',1),manual=action=='step',human_axis=command.get('human_axis',0.))
                elif action!='state':raise ValueError('unsupported viewer command')
                return state.packet()
