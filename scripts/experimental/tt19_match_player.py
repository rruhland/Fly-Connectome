"""Play complete strict games with a practiced server and RGB-conditioned receiver."""
import argparse
from pathlib import Path
import numpy as np
from .tt4_transfer_trial import _hash,_read,_write
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore
from .tt19_catalog import Player
from .tt19_consolidate import TrialBank
from .tt19_contextual import context
from .tt19_course import Actor
from .tt19_flight import FlightLaw
from .tt19_practice_camera import BallEyes,PracticeCamera
from .tt19_stroke_trial import DONOR
from .tt19_view import ViewTransport
from .tt19_view_trial import SO101ViewArena


class MatchPlayer:
    def __init__(self,bank,catalog,law,response,receiver):
        self.serves=TrialBank(**bank['1'])
        self.receives=TrialBank(**bank['-1'])
        self.player=Player(catalog,law,response,'late')
        self.receiver=receiver

    def reset(self,role):
        self.role=role
        self.history=[]
        self.actor=None
        self.phase='opening'
        self.player.reset()
        self.moved=np.zeros(5)

    def command(self,t,xyz):
        pos=None if xyz is None else np.asarray(xyz)
        if pos is not None:
            self.history.append((t,pos))
        if self.role<0 and self.receiver=='catalog' and self.phase=='opening':
            self.phase='receive'
        if self.phase=='opening':
            if t<4:
                return np.zeros(6)
            if self.actor is None:
                cue=context(self.history,self.role)
                bank=self.serves if self.role==1 else self.receives
                self.actor=Actor(bank.choose(cue,False),self.role,True)
                for tick,p in self.history: self.actor.observe(tick,p)
            elif pos is not None:
                self.actor.observe(t,pos)
            leaving=(len(self.history)>=2 and pos is not None and pos[1]>-.55 and
                     pos[1]-self.history[-2][1][1]>.018 and (self.role>0 or t>23))
            if leaving and t>16:
                self.phase='recover'
            else:
                command=self.actor.action(t)
                self.moved+=command[:5]
                return command
        if self.phase=='recover':
            cmd=-np.clip(self.moved,-1.,1.)
            self.moved+=cmd
            if np.max(abs(self.moved))<.0001:
                self.phase='receive'
                self.player.reset()
            return np.r_[cmd,0.]
        return self.player.command(t,pos,-1)


def load_match_player(checkpoint, cache):
    data = _read(checkpoint)
    if 'tt19_model_rollout' in data:
        raise ValueError('Use tt19_model_rollout_trial.load_checkpoint for this receiver')
    saved = data['tt19_match']
    if saved['catalog_sha256'] != _hash(cache):
        raise ValueError('The motion catalogue differs from the saved player')
    with np.load(cache) as archive:
        catalog = dict(archive)
    return MatchPlayer(saved['bank'], catalog, FlightLaw.from_state(saved['law']),
                       np.asarray(saved['response']), saved['receiver'])


def evaluate(folder,cache,tag,seeds,native,receiver,source,bank_source,*,engine=None,subset='full',endpoints=True,resume=None):
    if resume is not None:
        if engine is not None:
            raise ValueError('Cannot change the receiver while resuming a checkpoint')
        policy = load_match_player(resume, cache)
        donor = _read(resume)
        saved = donor['tt19_match']
        bank_source, source, receiver = saved['bank_source'], saved['source'], saved['receiver']
    else:
        path=folder/f'consolidated-{bank_source}-bank.json.gz'
        bank=_read(path)['banks']
        with np.load(cache) as archive:
            catalog = dict(archive)
        donor=_read(DONOR); old=donor['series'] if source else {}
        law=FlightLaw.from_state(_read(folder/'flight-source.json.gz')['source_law']) if source else FlightLaw()
        response=FlightMemory('impulse',old.get('flight_history',()),old.get('flight_impulses',())).response_matrix
        policy=MatchPlayer(bank,catalog,law,response,receiver)
        saved = dict(bank=bank,bank_source=bank_source,practice_sha256=_hash(path),source=source,
                     law=law.state(),response=response.tolist(),catalog_sha256=_hash(cache),receiver=receiver,
                     donor_sha256=_hash(DONOR))
    if engine: policy.player=engine(catalog,law,response)
    arena=SO101ViewArena(venue='hall',strict_endline=True,width=240,height=180,view_mode='wrist+fixed',max_steps=190)
    if subset!='full':
        from .tt19_skill_course import install_feeder
        install_feeder(arena,subset)
    cam=PracticeCamera(arena); eyes=BallEyes(arena.reset(seed=20017))
    transport=ViewTransport('wrist+fixed')
    result=dict(seeds=seeds,native=native,bank_source=bank_source,source=source,
                cache_sha256=_hash(cache),practice_sha256=saved['practice_sha256'],strict_endline=True,receiver=receiver,course=subset,games=[])
    try:
        for seed in seeds:
            rng=np.random.default_rng(seed); scoreboard=ServiceScore(first_server=-1); rows=[]
            while not scoreboard.finished and len(rows)<90:
                role=scoreboard.server; point=int(rng.integers(2**31))
                arena.reset(seed=point,server=role,render=False);policy.reset(role)
                tracker=transport.tracker(); actions=[];sightings=[];sources=[]
                for tick in range(arena.max_steps):
                    xyz=None
                    # The persisted serving strokes were learned from stereo every
                    # second frame. Preserve that timing after a change of sensor.
                    if tick%2==0 or policy.phase=='receive':
                        if native:
                            tracker.update(arena.render_cameras())
                            if tracker.ball is not None and tracker.ball.observed:
                                xyz=tracker.ball.position
                            sources.append((tick,tracker.sources.get('ball',())))
                        else: xyz=eyes.position(cam.capture(),True)
                    sights=None if xyz is None else np.asarray(xyz).tolist()
                    sightings.append(sights)
                    a=policy.command(tick,xyz);actions.append(a.tolist())
                    _,reward,done,timeout,_=arena.step(a,render=False)
                    if done or timeout:break
                outcome=arena.point_result();scoreboard.record(outcome)
                rows.append(dict(seed=point,server=role,actions=actions,optical=sightings,sources=sources,
                                 plans=policy.player.plans,result=outcome,events=arena.events,
                                 rgb=tracker.digest.hexdigest() if native else None))
            g=dict(seed=seed,score=list(scoreboard.score),points=rows,finished=scoreboard.finished,
                   returns=sum(r['result']['legal_returns'] for r in rows),multiple=sum(r['result']['legal_returns']>=2 for r in rows),
                   serves=sum(r['server']==1 and r['result']['legal_serve'] for r in rows))
            result['games'].append(g); print(tag,g['seed'],g['score'],g['returns'],g['multiple'],g['serves'],flush=True)
            _write(folder/f'match-{tag}.json.gz',result)
    finally: arena.close()
    if endpoints:
        donor['tt19_match'] = saved
        _write(folder/f'match-{tag}-player.json.gz',donor)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('cache',type=Path);p.add_argument('tag')
    p.add_argument('--nominal',action='store_true');p.add_argument('--native',action='store_true')
    p.add_argument('--bank',default='local');p.add_argument('--receiver',choices=('catalog','memory'),default='catalog')
    p.add_argument('--seeds',type=int,nargs='+',default=[20135460,4596208,36126412,337880,2125482,95337130])
    p.add_argument('--course',choices=('full','center','wide'),default='full')
    p.add_argument('--resume',type=Path,help='Play directly from a saved match checkpoint')
    a=p.parse_args();evaluate(a.folder,a.cache,a.tag,a.seeds,a.native,a.receiver,not a.nominal,a.bank,subset=a.course,resume=a.resume)
