"""Save a calibrated station learner between points and resume without motor probes."""
import argparse
from dataclasses import asdict
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .tt3_visual import RoleModel
from .tt4_arena import TT4Arena
from .tt4_camera import PerspectiveCamera
from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import Wiring, _counts, _point, _read, _write, load_sources, ROOT
from .tt5_hall_trial import prepare, probe_histories
from .tt5_player import RallyPlayer
from .tt5_player_body import FittedArm
from .tt5_vision import HallTracker, _Stereo


def save(path, player):
    if player.adjuster is not None:
        raise ValueError('checkpoint expects the standard rally player')
    tracker = player.tracker_factory()
    prior = player.prior
    state = dict(schema='tt5-between-points-v1',
                 optics=[asdict(c) for c in tracker.stereo.rig],
                 motion_supported_blade=tracker.motion_supported_blade,
                 apertures=[[[s.start,s.stop] for s in pair] for pair in tracker.apertures],
                 roles=asdict(prior.roles), clock=prior.clock, bounce=list(prior.bounce_values),
                 rng=prior.rng.bit_generator.state, fit_error=player.adapter.error,
                 fit_samples=player.adapter.samples, pitch_samples=player.adapter.pitch_samples,
                 gravity=player.gravity, stroke_speed=player.stroke_speed, transfer_strength=player.transfer_strength,
                 impact_restitution=player.impact_restitution)
    arrays = {name:getattr(player.adapter,name) for name in ('geometry','scripts','home','gains','lags')}
    coeff, scores, counts = prior.numeric
    delays, uses, continuations, public_scores = prior.delay
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temp = Path(stream.name)
    try:
        with stream:
            np.savez_compressed(stream, metadata=json.dumps(state,allow_nan=False),reference=tracker.reference,
                                coeff=coeff,scores=scores,counts=counts,delays=delays,uses=uses,
                                continuations=continuations,public_scores=public_scores,
                                role_rows=np.asarray(player.role_rows), **arrays)
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def load(path):
    with np.load(path,allow_pickle=False) as data:
        state=json.loads(data['metadata'].item())
        if state['schema']!='tt5-between-points-v1':
            raise ValueError('unrecognized station checkpoint')
        body=FittedArm(*(data[k] for k in ('geometry','scripts','home','gains','lags')),
                       state['fit_error'],state['fit_samples'],state['pitch_samples'])
        stereo=_Stereo(tuple(PerspectiveCamera(**c) for c in state['optics']))
        apertures=tuple(tuple(slice(a,b) for a,b in eye) for eye in state['apertures'])
        reference=data['reference']
        prior=TransferPrior('fresh',{},0)
        prior.numeric=tuple(data[k] for k in ('coeff','scores','counts'))
        prior.delay=tuple(data[k] for k in ('delays','uses','continuations','public_scores'))
        prior.clock=state['clock']
        prior.roles=RoleModel(**state['roles'])
        prior.bounce_values=state['bounce']
        def make_tracker():
            return HallTracker(stereo, reference, apertures,
                               motion_supported_blade=state.get('motion_supported_blade', False))
        player=RallyPlayer(body,prior,make_tracker,
                           stroke_speed=state['stroke_speed'],transfer_strength=state['transfer_strength'],
                           impact_restitution=state['impact_restitution'])
        player.role_rows=list(data['role_rows'])
        player.gravity=state['gravity']
        prior.rng.bit_generator.state=state['rng']
    return player


def setup(receipt,index):
    doc=_read(receipt)
    course=doc['courses']['first_person'][index]
    sched=course['schedule']
    geom=doc['geometry']
    if geom.get('fast_hall'):
        raise ValueError('expected the original native shading')
    arena=TT4Arena(sched['arena_seed'],view_seed=sched['view_seed'],view='first_person',**geom['arena'])
    return doc,course,Wiring(arena,sched['arena_seed'])


def build(args):
    doc,course,w=setup(args.receipt,args.installation)
    try:
        sched=course['schedule']
        pipeline, source = args.condition.split(':')
        if pipeline not in ('rally', 'fallback', 'impact') or args.condition not in course['runs']:
            raise ValueError('select a played rally, fallback, or impact condition from the receipt')
        hist,_,_=probe_histories(w,sched,doc['geometry'],args.cache)
        make_player, _ = prepare(pipeline, w.arena.cameras, hist)
        native,states,hashes=load_sources(args.first_person_mini)
        if any(hashes.get(key) != digest for key, digest in doc['inputs_sha256'].items()
               if key in hashes and (key in native or key == source)):
            raise ValueError('source checkpoint differs from the course receipt')
        prior=TransferPrior(source,native,sched['player_seed'],mini=states.get(source))
        live=make_player(prior)
        prior_points=course['runs'][args.condition]['points']
        if not 0 <= args.after < len(prior_points):
            raise ValueError('select a point boundary with a following scored point')
        for row in prior_points[:args.after]:
            if _point(w,live,row['seed'])['action_sha256'] != row['action_sha256']:
                raise ValueError('source course action tape diverged during checkpoint build')
        save(args.output,live)
        expected=prior_points[args.after]
        resumed=_point(w,load(args.output),expected['seed'])
        if resumed['action_sha256']!=expected['action_sha256'] or resumed['evaluator']!=expected['evaluator']:
            raise ValueError('resumed learner diverged')
        print(json.dumps({'seed':expected['seed'],'action_sha256':resumed['action_sha256'],
                          'outcome':resumed['evaluator']}))
    finally:
        w.arena.close()


def play(args):
    doc,course,w=setup(args.receipt,args.installation)
    try:
        player=load(args.checkpoint)
        if not all(np.allclose(c.basis,d.basis) and np.allclose(c.eye,d.eye) and np.isclose(c.fovy,d.fovy)
                   and (c.height,c.width)==(d.height,d.width) for c,d in zip(
                       w.arena.cameras,player.scene.tracker.stereo.rig,strict=True)):
            raise ValueError('camera calibration differs from checkpoint station')
        row=_point(w,player,args.seed)
        row.update(_counts(row))
        small=dict(schema='tt5-resumed-play-v1',geometry=doc['geometry'],
                   courses={'first_person':[dict(schedule=course['schedule'],view='first_person',
                          evaluator_wiring=w.state(),runs={'resumed':dict(points=[row])})]})
        _write(args.output,small)
        print(row['evaluator'])
    finally:
        w.arena.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for label in ('build','play'):
        group=sub.add_parser(label)
        group.add_argument('receipt',type=Path,help='station wiring and venue from a scored course')
        group.add_argument('output',type=Path)
        group.add_argument('--installation',type=int,default=0)
        if label=='build':
            group.add_argument('--cache',type=Path)
            group.add_argument('--condition', default='rally:legacy_mini')
            group.add_argument('--first-person-mini', type=Path,
                               default=ROOT/'artifacts/tt4/mini-first-person-player.json.gz')
            group.add_argument('--after',type=int,default=2)
        else:
            group.add_argument('--checkpoint',required=True,type=Path)
            group.add_argument('--seed',required=True,type=int)
    args=parser.parse_args()
    (build if args.action=='build' else play)(args)
