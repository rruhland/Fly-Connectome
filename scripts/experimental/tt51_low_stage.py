"""Continue the same reward learner on lower real physical practice deliveries."""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import tt51_low_contact_practice as skill
from .tt4_transfer_trial import _read,_write,_hash

PLAN=skill.ROOT/'low-stage-plan.json'


class LowCoach(skill.ReachableCoach):
    def draw(self,card,leg,protocol,rng):
        r=np.random.default_rng([510255,card['seed'],card['reply_seed'],leg])
        wins=sum(row['landed'] for row in self.memory.rows)
        if (len(self.calls)+1)%4==0:
            target,origin=skill.base.STANDARD_DRAW(card,leg,protocol,rng)
        else:
            spread=(.11,.21) if wins<6 else (.02,.26)
            height=(.965,1.005) if wins<6 else (.970,1.055)
            target=(float(r.uniform(*spread)),float(r.uniform(-1.32,-1.24)),
                float(r.uniform(*height)),float(r.uniform(3.20,3.65)))
            origin='tt51_lower_true_reward_practice'
        self.calls.append(dict(rewarded_attempts=wins,target=list(target),
            reason='lower requested band; actual receive-height telemetry remains evaluator-only'))
        return target,origin


def prepare():
    assert not PLAN.exists();p=_read(skill.PLAN)
    p.update(practice=p['practice'][8:16],lower_stage_source_sha256=_hash(Path(__file__)),
        parent_contact_plan_sha256=_hash(skill.PLAN),
        stage_scope='Same actor and learned motor-memory definition. Next eight preassigned practice cards. Nominal goals lowered because the two preceding actual scores came from ordinary evaluator heights; no low claim inferred from goal names.')
    _write(PLAN,p);print(_hash(PLAN),flush=True)


def run(a):
    if _read(PLAN)['lower_stage_source_sha256']!=_hash(Path(__file__)):raise ValueError('Lower coach changed')
    with patch.object(skill,'PLAN',PLAN),patch.object(skill,'ReachableCoach',LowCoach):skill.run(a)
    d=_read(a.out);d['lower_stage_source_sha256']=_hash(Path(__file__));_write(a.out,d)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','practice'))
    p.add_argument('--out',type=Path,default=skill.ROOT/'lower-practice.json.gz')
    p.add_argument('--memory',type=Path,default=skill.ROOT/'contact-practice.json.gz')
    p.add_argument('--n',type=int,default=8);p.add_argument('--cap',type=int,default=4)
    p.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    a=p.parse_args();prepare() if a.phase=='prepare' else run(a)
