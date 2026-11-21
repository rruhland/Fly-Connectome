"""Recover a dimensionless impact coefficient from a mini player's visual tracks."""
import argparse
import json
from pathlib import Path

import numpy as np

from .tt3_arena import heldout_cases, IMAGE_WIDTH, PIXELS_PER_METER
from .tt3_cycle import ConditionedNumeric
from .tt3_learning import play_point
from .tt3_recovery_learning import DelayMemory
from .tt3_table_checkpoint import make_player
from .tt3_visual import RoleModel
from .tt4_mini_pov import make_arena, read_source, use_stereo
from .tt4_transfer_trial import ROOT, _read, _hash


def local_fit(path, times, center):
    good = np.isfinite(path[times]).all(axis=1)
    if good.sum() < 3:
        return None
    x = times[good] - center
    return np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), path[times[good]], rcond=None)[0]


def fit_impacts(ball, blade):
    """Infer the impulse direction from a velocity change near the observed blade."""
    candidates = []
    for t in range(5, len(ball)-5):
        before = local_fit(ball, np.arange(t-5,t), t)
        after = local_fit(ball, np.arange(t+1,t+6), t)
        racket = local_fit(blade, np.arange(t-2,t+3), t)
        if before is None or after is None or racket is None:
            continue
        vin, vout = before[1], after[1]
        change = vout-vin
        impulse_speed = np.linalg.norm(change)
        if impulse_speed < .05 or vin @ vout > -.22*np.linalg.norm(vin)*np.linalg.norm(vout):
            continue
        gap = np.linalg.norm((before[0]+after[0])*.5 - racket[0])
        discontinuity = np.linalg.norm(before[0]-after[0])
        if gap > .23 or discontinuity > .10:
            continue
        normal = change / impulse_speed
        approach = -normal @ (vin-racket[1])
        if approach <= .025:
            continue
        coefficient = (vout-racket[1]) @ normal / approach
        if .05 < coefficient < 1.50:
            candidates.append((discontinuity + .08*gap, t, float(coefficient)))
    retained = []
    for _, t, value in sorted(candidates):
        if all(abs(t-s[0])>8 for s in retained):
            retained.append((t,value))
    return sorted(retained)


def estimate(directory, archive, view):
    values, source = read_source(directory)
    roles,numeric,clock = values
    course = next(c for c in _read(archive)['courses'] if c['view']==view)
    seed=course['seed']
    player=make_player(seed,RoleModel(**roles),ConditionedNumeric(numeric),DelayMemory(),clock)
    cases=heldout_cases(seed,len(course['points'])+1)
    samples=[]
    for i, (case,old) in enumerate(zip(cases[:-1],course['points'],strict=True)):
        arena=make_arena(case,cases[0].view_seed,view,160,120,.16)
        if i==0 and view=='first_person':
            use_stereo(player,arena.cameras)
        ordinary=player.binder.update
        balls,blades=[],[]
        def capture(*args,**kwargs):
            binding=ordinary(*args,**kwargs)
            for tr,history in ((binding.controlled,blades),(binding.relevant,balls)):
                history.append(tr.position if tr is not None and tr.observed else [np.nan]*3)
            return binding
        player.binder.update=capture
        try:
            row=play_point(arena,player)
        finally:
            player.binder.update=ordinary
        if row['action_sha256']!=old['action_sha256']:
            raise ValueError('mini source failed action replay')
        scaling=IMAGE_WIDTH / PIXELS_PER_METER
        for t,v in fit_impacts(np.asarray(balls)*scaling,np.asarray(blades)*scaling):
            samples.append(dict(point=i,tick=t,ratio=v))
    if len(samples)<3:
        raise ValueError(f'too few visual stroke samples: {samples}')
    return dict(view=view,seed=seed,points=len(course['points']),source=source,archive_sha256=_hash(archive),
                samples=samples,restitution=float(np.median([r['ratio'] for r in samples])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--archive',type=Path,default=ROOT/'artifacts/tt4/mini-first-person-course.json.gz')
    p.add_argument('--directory',type=Path,default=ROOT/'artifacts/tt3')
    args=p.parse_args()
    data={v:estimate(args.directory,args.archive,v) for v in ('external','first_person')}
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print([(v,len(d['samples']),d['restitution']) for v,d in data.items()])
