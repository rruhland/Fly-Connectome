"""Play through a simultaneous body, table and camera change on one scored stream."""
import argparse
import json
from pathlib import Path
import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_continuous_trial import prepare_point
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt10_series_trial import _restore, _save
from .tt17_motor_protocol import wiring_for
from .tt18_body_choice import BodyChoice, ChoicePlayer
from .tt18_online import BodyLearning, OnlineBodyPlayer, blank_body, carry_body
from .tt18_online_trial import metrics, play_point
from .tt18_plant import PLANTS, PlantSettings, TT18PlantArena
from .tt18_surface import SurfaceMemory, SurfacePlayer
from .tt18_viewplay import ViewTransport
from .tt18_viewplay_trial import DONOR, ViewPlayCapture, add_play_cameras, layout, specification


class WorldPlayer(OnlineBodyPlayer, SurfacePlayer):
    """Update both the camera-body response and the table's local bounce model."""


class CombinedArena(ViewPlayCapture, TT18PlantArena):
    def _model_document(self):
        document = super()._model_document()
        add_play_cameras(document)
        return document


def fresh_arena(spec, changed, target_body, *, render):
    plant = PLANTS['compact'] if changed else PlantSettings()
    settings = dict(spec['arena'], width=384, height=288)
    if changed:
        settings['body'] = target_body
    arena = CombinedArena(**settings, plant=plant)
    if render:
        cache_hall_stereo(arena)
    return arena


def load_controller(path, mode, seed, rule=None):
    saved = _read(path).get('tt18_combined', {})
    if saved and saved['mode'] != mode:
        raise ValueError('controller differs from saved player')
    adaptive = mode != 'stale_view'
    transport = ViewTransport(adaptive, saved.get('views'))
    state, library, player = _restore(path, tracker_factory=transport.tracker)
    player.__class__ = ChoicePlayer if mode == 'choice' else OnlineBodyPlayer
    if mode == 'world':
        player.__class__ = WorldPlayer
        player.surface_mode = 'transfer'
        if saved:
            player.surface = SurfaceMemory(**saved['surface'])
        elif rule is not None:
            player.surface = SurfaceMemory(rule['width'],rule['lifetime'])
        else:
            raise ValueError('a source-selected learning rule is required')
    if saved:
        body = (BodyChoice if mode == 'choice' else BodyLearning).restore(saved['body'])
    else:
        old = state['gain_memory']
        fitted = old['maps'][old['active']]
        model = library.models[fitted['model']]
        warm = carry_body(model, np.asarray(fitted['transform']))
        prior = blank_body(model.action_size) if mode == 'cold' else warm
        if mode == 'choice':
            body = BodyChoice(prior,seed=seed)
        else:
            body = BodyLearning(prior, enabled=mode != 'fixed_body', seed=seed, directed=mode != 'fixed_body')
    player.body_learning = body
    player.adapter = body.model
    if player.prior.span_memory is not None and not saved:
        player.prior.span_memory.active = None
    return state, library, player, transport


def persist(path, state, library, player, transport, mode):
    body = player.body_learning
    library.models.append(body.model)
    entry = {k:v for k,v in state['series'].items() if k not in
             ('completed','role_rows','flight_history','flight_impulses','optics','tt16_camera',
              'arena','calibration','probes','attempts')}
    entry['model'] = len(library.models)-1
    _save(path,state,library,player,entry,state['series']['completed']+1)
    saved = _read(path)
    saved['tt18_combined'] = dict(body=body.state(), views=transport.state(), mode=mode)
    if mode == 'world':
        saved['tt18_combined']['surface'] = player.surface.state()
    _write(path,saved)


def run(folder, seed, mode, *, resume=None, target_body='gantry'):
    spec = specification(seed)
    donor = DONOR if resume is None else resume
    source_rule = folder/'surface-source-rule.json.gz'
    rule = _read(source_rule)['choice'] if mode == 'world' and resume is None else None
    state, library, player, transport = load_controller(donor, mode, seed,rule)
    score = ServiceScore(first_server=-1)
    rows = []
    arena = None
    target = '' if target_body == 'gantry' else target_body+'-'
    prefix = folder/f'combined-{target}{seed}-{mode}'
    try:
        for i, (start, phase) in enumerate(zip(spec['starts'], spec['phases'], strict=True)):
            if i in (0,2):
                if arena is not None:
                    arena.close()
                arena = fresh_arena(spec, i == 2, target_body, render=True)
                wiring = wiring_for(arena,spec)
            stage, streams = layout('adaptive',i)
            arena.capture(streams)
            row = play_point(player,arena,wiring,seed,start,phase,score.server)
            track = player.scene.tracker
            row.update(stage=stage, changed=i>=2, streams=list(streams), rgb=track.digest.hexdigest(),
                       registration=track.registration, sightings=track.sightings)
            if mode == 'world':
                row['surface_events'] = player.surface_events
            rows.append(row)
            score.record(row['evaluator'])
            if score.finished:
                break
        if not score.finished:
            raise ValueError('unfinished game')
    finally:
        if arena is not None:
            arena.close()
    checkpoint = prefix.with_name(prefix.name+'-player.json.gz')
    persist(checkpoint,state,library,player,transport,mode)
    _,_,after,views = load_controller(checkpoint,mode,seed)
    if after.body_learning.state() != player.body_learning.state() or views.state() != transport.state():
        raise ValueError('checkpoint restore differs')
    donor_path = donor.resolve()
    donor_label = str(donor_path.relative_to(ROOT)) if donor_path.is_relative_to(ROOT) else donor.name
    game = dict(seed=seed,mode=mode,target_body=target_body,score=list(score.score),points=rows,metrics=metrics(rows),
                donor=donor_label, donor_sha256=_hash(donor), checkpoint_sha256=_hash(checkpoint))
    if mode == 'world':
        if rule is not None:
            game['source_rule_sha256'] = _hash(source_rule)
        if after.surface.state() != player.surface.state():
            raise ValueError('restored surface memory differs')
    verify(spec,game)
    _write(prefix.with_suffix('.json.gz'),game)
    print(prefix.name,score.score,game['metrics']['total_legal_returns'],flush=True)


def verify(spec, game):
    score = ServiceScore(first_server=-1)
    arena = None
    try:
        for i,row in enumerate(game['points']):
            if i in (0,2):
                if arena is not None:
                    arena.close()
                arena = fresh_arena(spec,i==2,game['target_body'],render=False)
                wiring = wiring_for(arena,spec)
            interface = PointInterface(wiring, prepare_point(arena,row['phase']),
                                       server=score.server,seed=game['seed'])
            interface.reset(seed=row['seed'],render=False)
            for action in row['actions']:
                interface.step(action,render=False)
            if (row['server'] != score.server or arena.point_result()!=row['evaluator']
                    or interface.applied!=row['applied'] or
                    json.dumps(arena.events,sort_keys=True)!=json.dumps(row['events'],sort_keys=True)):
                raise ValueError('combined fixture failed physical replay')
            score.record(row['evaluator'])
        if not score.finished or list(score.score)!=game['score']:
            raise ValueError('combined game score differs')
    finally:
        if arena is not None:
            arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('mode',choices=('carried','cold','fixed_body','stale_view','choice','world'))
    parser.add_argument('--seed',type=int,default=418336241)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--target-body',choices=('gantry','stock'), default='gantry')
    args=parser.parse_args()
    args.folder.mkdir(exist_ok=True,parents=True)
    run(args.folder,args.seed,args.mode,resume=args.resume,target_body=args.target_body)
