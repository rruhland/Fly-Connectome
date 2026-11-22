"""Compare duration-trained and label-permuted trajectory rankers in native rallies."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt20_natural import NaturalViewTransport
from .tt22_imaging_trial import verify_live
from .tt22_rally_trial import CONFIG, OUTPUT, RallyArena, alternating, opponent, rates
from .tt22_value_calibration import CALIBRATION, MODES as CALIBRATED, CalibratedValue
from .tt22_value_player import ValuePlayer
from .tt22_value_support import MODES as SUPPORT, SupportedValue, quality as support_quality


SOURCE = OUTPUT/'value-source.json.gz'
RECEIPT = OUTPUT/'value-live.json.gz'
CALIBRATED_RECEIPT = OUTPUT/'value-calibrated-live.json.gz'
SUPPORT_RECEIPT = OUTPUT/'value-supported-live.json.gz'
REPLICATION_RECEIPT = OUTPUT/'value-supported-replication.json.gz'
SEED = 1065840701
CALIBRATED_SEED = 1406715327
SUPPORT_SEED = 1224727723
REPLICATION_SEED = 137823193
MODES = ('duration','shuffled')


class Station:
    def __init__(self,models,gains,spread,pace,horizon,nominal_origin,*,rival='anchor',player_type=ValuePlayer):
        self.arena = RallyArena(spread,pace)
        self.rival = opponent(self.arena,rival)
        self.models = models
        self.nominal_origin = nominal_origin
        law = FlightLaw()
        self.player = player_type(law,next(iter(models.values())),gains,horizon=horizon)

    def close(self):
        self.player.close()
        self.arena.close()

    def point(self,seed,mode,*,audit=None):
        arena,player = self.arena,self.player
        arena.reset(seed=seed,server=-1,render=False)
        player.reset()
        player.value_model = self.models[mode]
        optics = NaturalViewTransport('fixed',lens_fov=arena.lens_intrinsics(),law=player.law,flight=False,associate=True)
        tracker = optics.tracker()
        actions,readings = [],[]
        for tick in range(arena.max_steps):
            tracker.update(arena.render_cameras(),tick=tick)
            mover,actor = tracker.ball,tracker.actor
            position = (np.asarray(mover.position) if mover is not None and mover.observed
                        and tracker.sources.get('ball_range')=='stereo' else None)
            player.actor_reading = (np.asarray(actor.position) if not self.nominal_origin and actor is not None and actor.observed
                                    and len(tracker.sources.get('blade',()))==2 else None)
            if position is not None:
                readings.append((tick,position.tolist()))
            action = player.command(tick,position)
            actions.append(action.tolist())
            _,reward,done,capped,_ = arena.step(action,render=False)
            if audit is not None:
                audit(tick+1,arena)
            if done or capped:
                break
        return dict(seed=seed,role=-1,reward=reward,actions=actions,optical=readings,rgb=tracker.digest.hexdigest(),
                    longest=alternating(arena.events),outcome=arena.point_result(),events=arena.events.copy(),
                    shots=player.shots,rival=self.rival.decisions.copy())


def report(rows):
    return dict(rates(rows),contact_points=sum(row['outcome']['contacts']>0 for row in rows),
                first_return_points=sum(row['outcome']['legal_returns']>0 for row in rows),
                later_return_points=sum(row['outcome']['legal_returns']>1 for row in rows),
                selected_portable=sum(pick['proposal']=='portable' for row in rows for pick in row['shots']))


def shared_first_choice(pair):
    first = [row['shots'][:1] for row in pair.values()]
    if any(first) and not all(first):
        raise ValueError('exactly one ranker received a first feasible set on the matched start')
    if all(first) and len({shot[0]['candidate_sha256'] for shot in first})!=1:
        raise ValueError('candidate sets differ at the common first decision')


def run(count,spread,pace,*,calibrated=False,supported=False,replication=False,
        rival='anchor',proposal='duration',output=None):
    model = _read(SOURCE)
    gains = model['candidates']['duration'][1:] if proposal=='duration' else [1.,1.]
    supported = supported or replication
    if supported:
        launch_seed = REPLICATION_SEED if replication else SUPPORT_SEED
    else:
        launch_seed = CALIBRATED_SEED if calibrated else SEED
    starts = np.random.default_rng(launch_seed).integers(2**31,size=count).tolist()
    if len(set(starts))!=count:
        raise ValueError('held rally launches overlap')
    models = model['models']
    modes,path,horizon = MODES,RECEIPT,model['frames']
    calibrated = calibrated or supported
    if calibrated:
        checkpoint = _read(CALIBRATION)
        if set(checkpoint['archived_seeds'])&set(starts):
            raise ValueError('archived training or validation used a held launch')
        modes,path,horizon = CALIBRATED,CALIBRATED_RECEIPT,checkpoint['horizon']
        models = {name:CalibratedValue(checkpoint,name) for name in modes}
        if supported:
            if set(_read(CALIBRATED_RECEIPT)['starts'])&set(starts):
                raise ValueError('support study reused a calibrated live launch')
            if replication and set(_read(SUPPORT_RECEIPT)['starts'])&set(starts):
                raise ValueError('support replication reused a pilot launch')
            modes,path = SUPPORT,REPLICATION_RECEIPT if replication else SUPPORT_RECEIPT
            models = {name:SupportedValue(checkpoint,name) for name in modes}
    model_path = CALIBRATION if calibrated else SOURCE
    data = dict(schema='tt22-value-rally-1',arena=CONFIG,source=model_path.name,source_sha256=_hash(model_path),
                spread=spread,pace=pace,rival=rival,starts=starts,portable_gains=gains,pairs=[])
    path = output or path
    if supported:
        data['archived_quality'] = support_quality(models,checkpoint)
    station = Station(models,gains,spread,pace,horizon,calibrated,rival=rival)
    began = perf_counter()
    try:
        for i,seed in enumerate(starts):
            order = modes[i%len(modes):]+modes[:i%len(modes)]
            pair = {name:station.point(seed,name) for name in order}
            shared_first_choice(pair)
            data['pairs'].append(pair)
            data['summary'] = {name:report([row[name] for row in data['pairs']]) for name in modes}
            _write(path,data)
            print('value',seed,{name:(row['longest'],row['outcome']['legal_returns'],
                                      [shot['choice'] for shot in row['shots']]) for name,row in pair.items()},flush=True)
    finally:
        station.close()
    data['verified'] = verify_live(data)
    data['wall_seconds'] = perf_counter()-began
    _write(path,data)
    print(data['summary'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=6)
    parser.add_argument('--spread',type=float,default=.30)
    parser.add_argument('--pace',type=float,default=.82)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--rival',choices=('anchor','varied'),default='anchor')
    parser.add_argument('--proposal',choices=('duration','neutral'),default='duration')
    parser.add_argument('--output',type=Path)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument('--calibrated',action='store_true')
    profile.add_argument('--supported',action='store_true')
    profile.add_argument('--replication',action='store_true')
    args = parser.parse_args()
    if args.verify:
        print('replayed',verify_live(_read(args.verify)))
    elif (args.rival!='anchor' or args.proposal!='duration') and args.output is None:
        parser.error('pass an independent output path for an altered bench')
    elif args.count>0:
        run(args.count,args.spread,args.pace,calibrated=args.calibrated,supported=args.supported,
            replication=args.replication,rival=args.rival,proposal=args.proposal,output=args.output)
    else:
        parser.error('positive paired count required')
