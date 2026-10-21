"""Physics-only annotations of saved legal actions; never policy observations."""
import numpy as np
from fly_connectome.control.environment import PongCamera

class ReturnCounter:
    def __init__(self):
        self.opportunities=self.returns=self.opponent_opportunities=self.opponent_returns=0
        self.post_own_opponent_opportunities=self.post_own_opponent_returns=0
        self.streak=self.maximum_streak=0
        self.first_incoming=None

    def step(self,game,original,drive,**kwargs):
        p=game.config;x=float(game.ball[0,0]);y=float(game.ball[0,1])
        vx=float(game.ball_velocity[0,0]);vy=float(game.ball_velocity[0,1])
        # Use Torch's actual float32 update, exactly matching collision detection.
        next_x=float((game.ball[0,0]+game.ball_velocity[0,0]*p.dt).item())
        left=vx<0 and x>=p.paddle_x+p.ball_radius and next_x<=p.paddle_x+p.ball_radius
        right=vx>0 and x<=1-p.paddle_x-p.ball_radius and next_x>=1-p.paddle_x-p.ball_radius
        tick=int(game.rally_steps[0])+1
        outcome=original(drive,**kwargs)
        hit=bool(outcome.hits[0])
        if hit and not left:raise ValueError('own hit without an incoming collision-plane opportunity')
        if left:
            self.opportunities+=1;self.returns+=int(hit)
            self.streak=self.streak+1 if hit else 0
            self.maximum_streak=max(self.maximum_streak,self.streak)
            if self.first_incoming is None:
                self.first_incoming=dict(success=hit,tick=tick,pre_ball=[x,y],pre_velocity=[vx,vy])
        if right:
            returned=float(game.ball_velocity[0,0])<0
            self.opponent_opportunities+=1;self.opponent_returns+=int(returned)
            if self.returns:
                self.post_own_opponent_opportunities+=1;self.post_own_opponent_returns+=int(returned)
        return outcome

    def summary(self,game,*,truncated,terminated,reward,seconds):
        p=game.config;x=float(game.ball[0,0]);vx=float(game.ball_velocity[0,0])
        return dict(own_opportunities=self.opportunities,own_returns=self.returns,
            failed_returns=self.opportunities-self.returns,first_incoming=self.first_incoming,
            maximum_consecutive_returns=self.maximum_streak,
            opponent_opportunities=self.opponent_opportunities,opponent_returns=self.opponent_returns,
            post_own_opponent_opportunities=self.post_own_opponent_opportunities,
            post_own_opponent_returns=self.post_own_opponent_returns,
            unfinished_incoming_approach=bool(truncated and vx<0 and x>=p.paddle_x+p.ball_radius),
            unfinished_opponent_approach=bool(truncated and vx>0 and x<=1-p.paddle_x-p.ball_radius),
            reward=reward,seconds=seconds,truncated=truncated,terminated=terminated,
            natural_serve_win=bool(reward>0 and not self.returns))

def replay_records(records,*,seed,opponent_speed,max_episode_ticks):
    """Match every stored physical field and annotate opportunities per rally."""
    env=PongCamera(seed,max_episode_ticks=max_episode_ticks,opponent_speed=opponent_speed)
    original=env.game.step;counter=ReturnCounter();rallies=[];seconds=reward=0.
    def instrument(drive,**kwargs):return counter.step(env.game,original,drive,**kwargs)
    env.game.step=instrument
    for i,row in enumerate(records):
        if env.needs_reset:env.reset();counter=ReturnCounter();seconds=reward=0.
        result=env.step(row['action'])
        for key in ('reward','hits','elapsed_seconds','physics_ticks','terminated','truncated'):
            if getattr(result,key)!=row[key]:raise ValueError(f'physical replay mismatch at{i}: {key}')
        seconds+=result.elapsed_seconds;reward+=result.reward
        if result.terminated or result.truncated:
            rallies.append(counter.summary(env.game,truncated=result.truncated,terminated=result.terminated,
                reward=reward,seconds=seconds))
    if records and not env.needs_reset:
        rallies.append(counter.summary(env.game,truncated=False,terminated=False,reward=reward,seconds=seconds))
    return rallies

def aggregate(rallies):
    opportunities=sum(r['own_opportunities'] for r in rallies);returns=sum(r['own_returns'] for r in rallies)
    first=[r['first_incoming']['success'] for r in rallies if r['first_incoming'] is not None]
    seconds=sum(r['seconds'] for r in rallies)
    opponent=sum(r['opponent_opportunities'] for r in rallies);opponent_returns=sum(r['opponent_returns'] for r in rallies)
    return dict(rallies=len(rallies),own_opportunities=opportunities,own_returns=returns,
        return_fraction=returns/opportunities if opportunities else None,
        first_incoming_opportunities=len(first),first_incoming_returns=sum(first),
        first_incoming_return_fraction=sum(first)/len(first) if first else None,
        rallies_with_at_least_two_returns=sum(r['maximum_consecutive_returns']>=2 for r in rallies),
        maximum_consecutive_returns=max((r['maximum_consecutive_returns'] for r in rallies),default=0),
        returns_per_physical_second=returns/seconds if seconds else None,physics_seconds=seconds,
        unfinished_incoming_approaches=sum(r['unfinished_incoming_approach'] for r in rallies),
        opponent_opportunities=opponent,opponent_returns=opponent_returns,
        opponent_return_fraction=opponent_returns/opponent if opponent else None,
        natural_serve_wins=sum(r['natural_serve_win'] for r in rallies),
        post_own_opponent_opportunities=sum(r['post_own_opponent_opportunities'] for r in rallies),
        post_own_opponent_returns=sum(r['post_own_opponent_returns'] for r in rallies))

def paired_evidence(by_policy,*,draws=10000,seed=193870291,
                    arms=('hold1','hold10'),comparison=None,expected_serves=24,
                    controls=('initial','up','noop','random1','random10')):
    """Paired serve resampling; contacts within a serve are never independent."""
    names=(*arms,*controls)
    if not controls or type(expected_serves) is not int or expected_serves<1:
        raise ValueError('positive serve count and nonempty matched controls required')
    if any(len(by_policy.get(n,[]))!=expected_serves for n in names):return dict(complete=False)
    for i in range(expected_serves):
        signatures=[]
        for name in names:
            first=by_policy[name][i]['first_incoming']
            signatures.append(None if first is None else {k:v for k,v in first.items() if k!='success'})
        if any(s!=signatures[0] for s in signatures[1:]):raise ValueError('first incoming trajectory differs before own contact')
    index=np.random.default_rng(seed).integers(expected_serves,size=(draws,expected_serves));rates={};initial={};points={}
    for name in names:
        data=by_policy[name];hits=np.array([r['own_returns'] for r in data]);opp=np.array([r['own_opportunities'] for r in data])
        present=np.array([r['first_incoming'] is not None for r in data]);first=np.array([bool(r['first_incoming'] and r['first_incoming']['success']) for r in data])
        with np.errstate(divide='ignore',invalid='ignore'):
            rates[name]=hits[index].sum(1)/opp[index].sum(1)
            initial[name]=first[index].sum(1)/present[index].sum(1)
        points[name]=dict(all=hits.sum()/opp.sum() if opp.sum() else None,
                          first=first.sum()/present.sum() if present.sum() else None)
    report=dict(complete=True,serve_clusters=expected_serves,bootstrap_draws=draws,bootstrap_seed=seed,
        first_incoming_geometry_identical=True,arms={})
    if any(v is None for p in points.values() for v in p.values()):
        report.update(analyzable=False,reason='a complete policy cohort has no incoming opportunities')
        return report
    for name in arms:
        delta=rates[name]-np.max(np.stack([rates[c] for c in controls]),axis=0)
        delta_first=initial[name]-np.max(np.stack([initial[c] for c in controls]),axis=0)
        valid=np.isfinite(delta)&np.isfinite(delta_first)
        point=points[name]['all']-max(points[c]['all'] for c in controls)
        first=points[name]['first']-max(points[c]['first'] for c in controls)
        ci=np.quantile(delta[valid],[.025,.975]).tolist() if valid.any() else None
        ci_first=np.quantile(delta_first[valid],[.025,.975]).tolist() if valid.any() else None
        report['arms'][name]=dict(all_opportunity_lift=point,first_incoming_lift=first,
            all_opportunity_lift_ci95=ci,first_incoming_lift_ci95=ci_first,
            invalid_bootstrap_draws=int((~valid).sum()),
            promising_development_signal=bool(valid.all() and point>=.2 and first>=.2 and ci[0]>0 and ci_first[0]>0))
    if comparison is not None:
        positive,negative=comparison
        delta=rates[positive]-rates[negative]
        delta_first=initial[positive]-initial[negative]
        valid=np.isfinite(delta)&np.isfinite(delta_first)
        report['paired_arm_comparison']=dict(positive=positive,negative=negative,
            all_opportunity_lift=points[positive]['all']-points[negative]['all'],
            first_incoming_lift=points[positive]['first']-points[negative]['first'],
            all_opportunity_lift_ci95=np.quantile(delta[valid],[.025,.975]).tolist() if valid.any() else None,
            first_incoming_lift_ci95=np.quantile(delta_first[valid],[.025,.975]).tolist() if valid.any() else None,
            invalid_bootstrap_draws=int((~valid).sum()),descriptive_not_multiplicity_adjusted=True)
    return report
