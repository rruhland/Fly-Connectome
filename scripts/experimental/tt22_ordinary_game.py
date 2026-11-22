"""Two full games with ordinary serves, a physically steerable partner and online/frozen optics."""
import json

import numpy as np

from .tt4_transfer_trial import ROOT,_read,_write
from .tt8_service_arena import ServiceScore
from .tt19_flight import FlightLaw
from .tt20_natural import NaturalViewTransport
from .tt22_flight_inverse_trial import InversePlayer
from .tt22_match import CHECKPOINT,FACTORY,STANDARD_PICK,MatchStation,summarize
from .tt22_opponent import RallyArena,install_rally_opponent
from .tt22_rally import ReturnMemory
from .tt22_rally_trial import CONFIG

DIR=ROOT/'artifacts/tt22'
SEED=1668962128


class FrozenMemory(ReturnMemory):
    def observe(self,shot,reading):
        return


class StandardMatch(MatchStation):
    def __init__(self,feed):
        law=FlightLaw()
        self.arena=RallyArena(**CONFIG)
        self.rival=install_rally_opponent(self.arena,target_profile=feed)
        self.player=InversePlayer(law)
        lenses=self.arena.lens_intrinsics()
        self.optics=NaturalViewTransport('fixed',lens_fov=lenses,law=law,flight=False,associate=True)
        self.service_optics=NaturalViewTransport('left',lens_fov={'left':lenses['left']},flight=False)
        self.serve=np.array(_read(FACTORY)['programs'][STANDARD_PICK])


def verify(rows):
    arena=RallyArena(**CONFIG)
    scores=ServiceScore(first_server=-1)
    rng=np.random.default_rng(SEED)
    try:
        for row in rows:
            if (row['seed'],row['role'],row['score_before'])!=(int(rng.integers(2**31)),scores.server,list(scores.score)):
                raise ValueError('service schedule mismatch')
            arena.reset(seed=row['seed'],server=row['role'],render=False)
            saved_aim,saved_control=arena._aim,arena._command_to
            try:
                for i,(near,far) in enumerate(zip(row['actions'],row['rival_actions'],strict=True)):
                    arena._aim=arena._command_to=lambda *args: np.asarray(far)
                    _,r,done,cap,_=arena.step(near,render=False)
                    if bool(done or cap)!=(i+1==len(row['actions'])):
                        raise ValueError('point length mismatch')
            finally:
                arena._aim,arena._command_to=saved_aim,saved_control
            if r!=row['reward'] or json.dumps(arena.events)!=json.dumps(row['events']) or arena.point_result()!=row['outcome']:
                raise ValueError('physical outcome mismatch')
            if list(scores.record(arena.point_result()))!=row['score_after']:
                raise ValueError('score mismatch')
        if not scores.finished:
            raise ValueError('incomplete game')
    finally:
        arena.close()
    return len(rows)


def main():
    feed=_read(DIR/'tt22_opponent-learned.json.gz')['fit']['profile']
    practice=_read(CHECKPOINT)['evidence']
    branches=('online','frozen')
    memory=dict(online=ReturnMemory(practice),frozen=FrozenMemory(practice))
    station=StandardMatch(feed)
    scores={k:ServiceScore(first_server=-1) for k in branches}
    doc=dict(seed=SEED,spread=1.,pace=1.,feed=feed,initial_evidence=practice,branches={k:[] for k in branches})
    path=DIR/'ordinary-game.json.gz'
    try:
        for i,s in enumerate(np.random.default_rng(SEED).integers(2**31,size=90)):
            for k in branches[::1 if i%2==0 else -1]:
                score=scores[k]
                if score.finished:
                    continue
                before=list(score.score)
                row=station.play(int(s),score.server,memory[k])
                after=list(score.record(row['outcome']))
                doc['branches'][k].append(dict(row,score_before=before,score_after=after))
                _write(path,doc)
                print(k,i+1,after,row['longest'],flush=True)
            if all(s.finished for s in scores.values()):
                break
    finally:
        station.close()
    doc['verified']={k:verify(rows) for k,rows in doc['branches'].items()}
    doc['summary']={k:summarize(rows) for k,rows in doc['branches'].items()}
    doc['evidence_after']={k:v.evidence for k,v in memory.items()}
    _write(path,doc)


if __name__=='__main__':
    main()
