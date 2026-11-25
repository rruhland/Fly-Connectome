"""Final low/slow practice stage; identical actor and retained actual outcomes."""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import tt51_low_contact_practice as skill
from .tt4_transfer_trial import _read,_write,_hash

PLAN=skill.ROOT/'slow-stage-plan.json'


class SlowCoach(skill.ReachableCoach):
    def draw(self,card,leg,protocol,rng):
        r=np.random.default_rng([510256,card['seed'],card['reply_seed'],leg])
        # The coach's slower requested return is a declared practice scaffold.
        # It changes no ball dynamics, body, observation or score rule.
        if (len(self.calls)+1)%4==0:
            target,origin=skill.base.STANDARD_DRAW(card,leg,protocol,rng)
        else:
            target=(float(r.uniform(.09,.22)),float(r.uniform(-1.32,-1.24)),
                float(r.uniform(.955,.995)),float(r.uniform(2.25,2.70)))
            origin='tt51_slow_low_physical_practice'
        self.calls.append(dict(target=list(target),actor_unchanged=True,
            scaffold='slower far requested speed to make real low approaches less time-critical'))
        return target,origin


def prepare():
    assert not PLAN.exists();p=_read(skill.PLAN)
    p.update(practice=p['practice'][16:24],slow_stage_source_sha256=_hash(Path(__file__)),
        parent_contact_plan_sha256=_hash(skill.PLAN),
        stage_scope='Last eight originally preassigned practice cards. Same actor, all17 preceding actual outcomes carried. Lower-stage scores were all evaluator-ordinary; seek true low rewards with slower physical partner requests before any claim of low skill. No actor/parameter sweep.')
    _write(PLAN,p);print(_hash(PLAN),flush=True)


def run(a):
    if _read(PLAN)['slow_stage_source_sha256']!=_hash(Path(__file__)):raise ValueError('Slow coach changed')
    with patch.object(skill,'PLAN',PLAN),patch.object(skill,'ReachableCoach',SlowCoach):skill.run(a)
    d=_read(a.out);d['slow_stage_source_sha256']=_hash(Path(__file__));_write(a.out,d)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','practice'))
    p.add_argument('--out',type=Path,default=skill.ROOT/'slow-practice.json.gz')
    p.add_argument('--memory',type=Path,default=skill.ROOT/'lower-practice.json.gz')
    p.add_argument('--n',type=int,default=8);p.add_argument('--cap',type=int,default=4)
    p.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    a=p.parse_args();prepare() if a.phase=='prepare' else run(a)
