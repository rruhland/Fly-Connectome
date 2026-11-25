"""Observer-only full ordinary TT80 games, paired with the exact frozen parent."""
import argparse
from pathlib import Path
from time import monotonic, process_time
from unittest.mock import patch

import numpy as np

from . import tt80_prospective_course as tt80
from . import tt75_incoming_state as held
from .tt75_observer_adapter import ExposureSession, actor_class
from .tt4_transfer_trial import _read, _write, _hash

ROOT = held.ROOT
PLAN = ROOT / 'ordinary.protocol.json'
BASELINE = tt80.ROOT / 'frozen.json.gz'
PARENT = tt80.season.Player
ORDINARY = tt80.course.mpc.play


def freeze():
    baseline = _read(BASELINE)
    design = _read(tt80.PLAN)
    if len(baseline['rows']) != 32 or baseline['cards'] != design['practice'][:32]:
        raise ValueError('Need the complete fixed original first32 comparator')
    if baseline['carry_sha256'] != design['carry']:
        raise ValueError('Different parent checkpoint')
    files = [held.PLAN, held.ROOT/'held.json.gz', held.ROOT/'sensor-calibration.json',
             tt80.PLAN, BASELINE, tt80.season.CARRY,
             Path(__file__), Path(__file__).with_name('tt75_observer_adapter.py'),
             Path(__file__).with_name('tt75_retina.py'), Path(__file__).with_name('tt75_dense_fit.py'),
             Path(__file__).with_name('tt83_observer_adapter.py')]
    p = dict(schema='tt75_ordinary_observer_v1', cards=baseline['cards'], n=32,
             sources={str(f):_hash(f) for f in files},
             settings=_read(held.PLAN)['settings'],
             scope='All first32 fixed independent TT80 original-center games, original plain ball, fixed mounts, true continuous uniform partner. Same frozen CARRY, raw field, temporal readout, motor search, budget28 and cap240. Only flight observation changes. All failures and all assigned games count. No outcome-based endpoint, extra training or source selection. Causal standard tracker fallback; native state is never a controller input. Continuous120Hz actual cropped stereo plus standard30Hz global tracker. A disabled first-game physical parity check precedes the full enabled course.')
    if PLAN.exists() and _read(PLAN) != p:
        raise ValueError('Frozen ordinary protocol changed')
    _write(PLAN, p)


def run(args):
    plan = _read(PLAN)
    if any(_hash(Path(p)) != h for p,h in plan['sources'].items()):
        raise ValueError('Changed sealed ordinary inputs')
    if not 1 <= args.n <= plan['n']:
        raise ValueError('Outside fixed32 course')
    actor = actor_class(PARENT)
    coverage = tt80.season.fresh.shot.Coverage(_read(tt80.season.fresh.RISK))
    enabled = args.phase == 'dense'
    wall, cpu = monotonic(), process_time()

    def make(*a, **kw):
        return actor(*a, coverage=coverage, **kw)

    def play(*pos, **kw):
        arena, wide, rig = pos[:3]
        raw, model = pos[9:11]
        model.observation_enabled = enabled
        model.observer_home = np.asarray(pos[6]).copy()
        model.observer_raw = raw
        model.exposures = ExposureSession(arena,rig,wide,plan['settings']) if enabled else None
        try:
            row = ORDINARY(*pos, **kw)
            row['observer'] = (dict(model.exposures.stats(), counts=model.public.counts,
                                    decisions=model.observer_decisions) if enabled else dict(disabled=True))
            # Archived comparator covers this exact card/configuration.
            old = _read(BASELINE)
            index = old['cards'].index(pos[7])
            previous = old['rows'][index]
            same_actions = np.array_equal(np.asarray(row['actions']),np.asarray(previous['actions']))
            same_events = held.school.digest_json(row['events']) == held.school.digest_json(previous['events'])
            row['exact_frozen_parent'] = dict(index=index, actions_equal=same_actions,
                 events_equal=same_events, baseline_learner=previous['score']['learner'])
            if not enabled and not (same_actions and same_events):
                _write(ROOT/'disabled-mismatch.json.gz',dict(card=pos[7],row=row,
                    baseline_source_sha256=_hash(BASELINE),actual_control_ticks=len(row['actions'])))
                raise ValueError('Disabled observer changed ordinary controls or physics')
            print('observer',index,row['score']['learner'],'parent',previous['score']['learner'],
                  'coverage',None if not enabled else model.public.counts,flush=True)
            return row
        finally:
            if model.exposures is not None:
                model.exposures.close()

    with tt80.enabled(), patch.object(tt80.course,'PLAN',tt80.PLAN), \
         patch.object(tt80.course.mpc,'MPC',make), patch.object(tt80.course.mpc,'play',play):
        tt80.course.run(args)
    saved = _read(args.out)
    saved.update(tt75_protocol_sha256=_hash(PLAN),wall_seconds=monotonic()-wall,
                 cpu_seconds=process_time()-cpu,observer_enabled=enabled)
    _write(args.out,saved)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('plan','disabled','dense'))
    p.add_argument('--n',type=int,default=32)
    p.add_argument('--out',type=Path)
    p.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    p.add_argument('--resume',action='store_true')
    p.set_defaults(mode='frozen',deck='practice',cap=240,carry=tt80.season.CARRY)
    a=p.parse_args()
    if a.phase == 'plan':
        freeze()
    else:
        a.out=a.out or ROOT/('ordinary-'+a.phase+'.json.gz')
        run(a)
