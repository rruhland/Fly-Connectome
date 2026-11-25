"""Posthoc paid causal sensor intervention on a known ordinary failed offer.

This diagnoses the 5222 whole-game gain. It is not a new held-out evaluation
or a deployable source/time gate. The external harness switches the sensor;
the ordinary actor receives only its normal RGB-derived state and encoders.
"""
import argparse
from pathlib import Path
from time import monotonic, process_time
from unittest.mock import patch

import numpy as np

from . import tt75_ordinary_play as tt
from .tt75_observer_adapter import ExposureSession, actor_class
from .tt39_carry_joints import Carried
from .tt41_early_optics_carry import resume
from .tt4_transfer_trial import _read, _write, _hash

PLAN = tt.ROOT/'causal-return.protocol.json'
STOCK_OBSERVED = tt.tt80.course.mpc.observed


def freeze():
    old=_read(tt.BASELINE)
    index=5
    row=old['rows'][index]
    offer=row['placements']['delivered'][-1]
    if old['cards'][index]['id']!=5222 or offer['near_landed']:
        raise ValueError('Diagnostic source differs')
    _write(PLAN,dict(schema='tt75_posthoc_causal_return_v1',index=index,
        cut=int(offer['far_contact']['step'])+6,cap=4,
        sources={str(p):_hash(p) for p in (tt.BASELINE,tt.PLAN,tt.tt80.season.CARRY,
            Path(__file__),Path(__file__).with_name('tt75_observer_adapter.py'))},
        scope='Known diagnostic selected after 5222 whole-game21 versus1. Both real branches replay the ordinary stock actor with passive120HzRGB until six control ticks after the failed terminal offer far contact. The evaluation harness then activates the same dense observer in one branch; no source/time gate is added to the deployed actor. Baseline must reproduce all actual issued commands and physical events. Both branches must share bitwise commands and the exact physical event prefix up to the cut. Evaluator-only fork state is read after the chosen command. Score the next physically legal near return (cap4 totals) or natural failure; do not count a contact as a return. All repeated prefixes/cameras are paid. No training, parameter change, or claim of new independence.'))


def run(out):
    if out.exists():raise ValueError('Use a new diagnostic receipt')
    p=_read(PLAN)
    if any(_hash(Path(f))!=h for f,h in p['sources'].items()):
        raise ValueError('Changed sealed diagnostic inputs')
    old=_read(tt.BASELINE);index=p['index'];card=old['cards'][index];source=old['rows'][index]
    mpc=tt.tt80.course.mpc;course=tt.tt80.course
    observed_actor=actor_class(tt.PARENT)

    class PassivePrefix(observed_actor):
        def begin(self):
            self.observation_enabled=False
            super().begin()
            self.exposures.begin()

    rank=tt.tt80.season.fresh.shot.Coverage(_read(tt.tt80.season.fresh.RISK))
    doc=dict(schema='tt75_posthoc_causal_return_v1',protocol_sha256=_hash(PLAN),
             rows=[],complete=False,actual_control_ticks=0)
    wall,cpu=monotonic(),process_time()
    settings=_read(tt.PLAN)['settings']
    with tt.tt80.enabled(),mpc.centered(old['ready']):
      for mode in ('legacy','dense'):
        arena=mpc.arena();wide,rig,stride,low,high=mpc.resources(arena)
        model=None;session=None;record={};legacy_history=[]
        with np.load(mpc.BANK) as bank:raw=mpc.DeepField(bank,'local')
        try:
            with patch.object(mpc,'Fragments',lambda file,engine:Carried(file,engine,
                practice=[course.PRACTICE],held={card['id']})):
                model=PassivePrefix(mpc.BANK,raw,variant='elastic',budget=28,coverage=rank)
            carry=_read(tt.tt80.season.CARRY)
            resume(model,tt.tt80.season.CARRY.with_suffix('').with_suffix('.npz'),carry)
            model.serial+=index
            if model.priors.digest()!=source['memory']['before']:
                raise ValueError('Different earned source bank')
            session=ExposureSession(arena,rig,wide,settings)
            model.exposures=session;model.observer_home=np.asarray(old['ready']).copy();model.observer_raw=raw
            exchange=arena.exchange

            def observed(eye,tick,frames,q):
                # This is an explicitly declared experimental intervention,
                # outside the policy, at the identical paid physical prefix.
                model.observation_enabled=mode=='dense' and tick>=p['cut']
                x=STOCK_OBSERVED(eye,tick,frames,q)
                legacy_history.append(dict(tick=tick,x=None if x is None else x.tolist()))
                return x

            def issue(command,far):
                tick=arena.steps
                if tick<p['cut'] or mode=='legacy':
                    if tick>=len(source['actions']) or not np.array_equal(command,np.asarray(source['actions'][tick])):
                        raise ValueError('Stock source command diverged before the intervention')
                if tick==p['cut']:
                    # The actor has already chosen its command from images.
                    # Native p/v/spin below are evaluator-only receipt fields.
                    record.update(time=float(arena.data.time),encoders=arena.joints.tolist(),
                        evaluator_pv=np.r_[arena.ball,arena.velocity].tolist(),evaluator_spin=arena.spin.tolist(),
                        evaluator_qpos=arena.data.qpos.tolist(),evaluator_qvel=arena.data.qvel.tolist(),
                        evaluator_ctrl=arena.data.ctrl.tolist(),
                        evaluator_warmstart=arena.data.qacc_warmstart.tolist(),
                        event_prefix=tt.held.school.digest_json(arena.events))
                return exchange(command,far)

            with patch.object(mpc,'observed',observed),patch.object(arena,'exchange',issue):
                row=tt.ORDINARY(arena,wide,rig,stride,low,high,old['ready'],card,source['shot'],
                    raw,model,course.OPTIC,'frozen',p['cap'])
            prefix=[e for e in source['events'] if e['time']<=p['cut']/60+1.e-8]
            if record['event_prefix']!=tt.held.school.digest_json(prefix):
                raise ValueError('Physical source prefix differs')
            if mode=='legacy' and tt.held.school.digest_json(row['events'])!=tt.held.school.digest_json(source['events']):
                raise ValueError('Ordinary baseline continuation differs')
            row.update(mode=mode,fork=record,observer=session.stats(),legacy_history=legacy_history,
                dense_decisions=model.observer_decisions,
                next_offer_legally_returned=bool(row['score']['learner']>source['score']['learner']))
            doc['rows'].append(row);doc['actual_control_ticks']+=len(row['actions'])
            doc.update(wall_seconds=monotonic()-wall,cpu_seconds=process_time()-cpu)
            _write(out,doc)
            print(mode,row['score']['learner'],row['next_offer_legally_returned'],len(row['actions']),flush=True)
        except BaseException as error:
            doc.update(error=repr(error),charged_incomplete_branch_ticks=int(arena.steps),
                wall_seconds=monotonic()-wall,cpu_seconds=process_time()-cpu)
            _write(out,doc)
            raise
        finally:
            if session is not None:session.close()
            if model is not None:model.close()
            if wide.renderer._gl_context is not None:wide.renderer._gl_context.make_current()
            wide.close();arena.close()
    a,b=(r['fork'] for r in doc['rows'])
    if a!=b:raise ValueError('Evaluator fork states differ')
    doc.update(complete=True,matched_actual_fork=True)
    _write(out,doc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('plan','run'))
    p.add_argument('--out',type=Path,default=tt.ROOT/'causal-return.json.gz')
    a=p.parse_args();freeze() if a.mode=='plan' else run(a.out)
