"""Use the native Pong lateral policy to prepare the blade before a late stroke."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import optical_start
from .tt19_model_rollout import ORIGIN
from .tt19_shoot import COURT, HORIZON
from .tt22_imaging_trial import verify_live
from .tt22_rally import face
from .tt22_rally_trial import CONFIG, OUTPUT
from .tt22_spatial import portable_contact
from .tt22_spatial_trial import SOURCE as NATIVE_PONG
from .tt22_value_calibration import CALIBRATION
from .tt22_value_support import SupportedValue
from .tt22_value_trial import SOURCE, Station, report
from .tt23_value_wide import RECEIPT as WIDE_RECEIPT, WidePlayer


DEVELOPMENT = OUTPUT/'value-ready-development.json.gz'
FRESH = OUTPUT/'value-ready-fresh.json.gz'
SEED = 1852297273
VALIDATION = (
    (2130715543,399331003,659743118,1520097002,949556383,2134479342),
    (1999740997,781303606,1321024455,358697341,131612783,464758626),
)
MIRRORED = (399331003,949556383,781303606,131612783)
READY_LIMIT = .29
READY_STEP = .128
MODES = ('pong','stationary')


class ReadyPlayer(WidePlayer):
    def __init__(self,law,model,gains,horizon):
        self.ready_coefficients = None
        self.ready_polarity = 1.
        self.ready_from = 0
        super().__init__(law,model,gains,horizon)
        self.initial_neutral = self.neutral.copy()
        self.ready_center = self.factory.pose(self.initial_neutral)[0]

    def reset(self):
        super().reset()
        self.ready_x = None
        self.preparation = []
        self.prepare_seconds = 0.
        if hasattr(self,'initial_neutral'):
            self.neutral = self.initial_neutral.copy()
            self.ready_x = self.ready_center[0]

    def command(self,tick,position,*,teacher=None):
        if self.ready_coefficients is not None:
            started = perf_counter()
            self.prepare(tick,position)
            self.prepare_seconds += perf_counter()-started
        return super().command(tick,position,teacher=teacher)

    def prepare(self,tick,position):
        center = self.ready_center
        goal_x = center[0]
        target = self.sim.twin.data.ctrl[self.sim.twin.axes[:5]]
        arriving = (tick>=self.ready_from and self.plan is None and self.outbound)
        if arriving and self.preparation and tick-self.preparation[-1]['tick']<=3:
            goal_x = self.ready_x
        estimate = optical_start([*self.history[-5:],(tick,position)],self.law.gravity) if arriving else None
        if estimate is not None:
            mover,speed = estimate
            if -.30<=mover[1]<=.62 and speed[1]<-.032:
                points,_ = self.law.trajectory(mover,speed,HORIZON,bounds=COURT)
                crossing = np.flatnonzero((points[:-1,1]>=center[1])&(points[1:,1]<center[1]))
                if len(crossing):
                    i = crossing[0]
                    mix = (center[1]-points[i,1])/(points[i+1,1]-points[i,1])
                    forecast_x = points[i,0]+mix*(points[i+1,0]-points[i,0])
                    actor_x = self.factory.pose(target)[0][0]
                    request = portable_contact(actor_x,mover[0],forecast_x,self.ready_coefficients)
                    goal_x = float(np.clip(self.ready_polarity*request,-READY_LIMIT,READY_LIMIT))
                    self.preparation.append(dict(tick=tick,mover_x=float(mover[0]),forecast_x=float(forecast_x),
                                                 actor_x=float(actor_x),request_x=float(request),goal_x=goal_x))
        next_x = float(self.ready_x+np.clip(goal_x-self.ready_x,-READY_STEP,READY_STEP))
        if abs(next_x-self.ready_x)<1.e-10:
            return
        aim = center.copy()
        aim[0] = next_x
        self.neutral = self.factory.solve(aim,face(.48,0.),target)
        self.ready_x = next_x


def point(station,seed,mode,coefficients,ready_from):
    started = perf_counter()
    player = station.player
    player.ready_coefficients = None if mode=='stationary' else coefficients
    player.ready_polarity = -1. if mode=='mirrored' else 1.
    player.ready_from = ready_from
    measurements = []

    def audit(step,arena):
        blade = arena.data.geom('learner_racket').xpos-ORIGIN
        ball = arena.ball-ORIGIN
        measurements.append(dict(step=step,blade=blade.tolist(),ball=ball.tolist()))

    row = station.point(seed,mode,audit=audit)
    row['preparation'] = player.preparation
    row['ready_from'] = ready_from
    row['evaluator_motion'] = measurements
    row['prepare_seconds'] = player.prepare_seconds
    row['point_seconds'] = perf_counter()-started
    last_shot = -1
    for shot in row['shots']:
        tick = shot['tick']
        first = next((note['tick'] for note in player.preparation if last_shot<note['tick']<tick),None)
        shot['preparation_lead_frames'] = None if first is None else tick-first
        last_shot = tick
    row['preparation_motor_effort'] = sum(float(np.abs(row['actions'][note['tick']][:5]).sum())
                                          for note in player.preparation)
    return row


def run(count,output,block=None):
    started = perf_counter()
    checkpoint = _read(CALIBRATION)
    gains = _read(SOURCE)['candidates']['duration'][1:]
    native = _read(NATIVE_PONG)
    index = int(np.argmax(np.asarray(native['totals'])/native['counts']))
    coefficients = native['coefficients'][index]
    development = _read(WIDE_RECEIPT)
    known = {pair['wide']['seed']:pair['wide'] for pair in development['pairs']}
    if block is not None:
        starts = list(VALIDATION[block-1])
        count = len(starts)
    else:
        starts = np.random.default_rng(SEED).integers(2**31,size=count).tolist() if count else list(known)
    archive = set(checkpoint['archived_seeds']) | set(known)
    if count:
        for name in ('value-live','value-calibrated-live','value-supported-live',
                     'value-supported-replication','value-online','value-inverse'):
            archive.update(_read(OUTPUT/(name+'.json.gz'))['starts'])
    if block is not None:
        pilot = _read(FRESH)
        if pilot['verified']!=2*len(pilot['starts']):
            raise ValueError('the fresh pilot must finish before validation')
        archive.update(pilot['starts'])
        archive.update(VALIDATION[2-block])
    if count and (len(set(starts))!=count or archive.intersection(starts)):
        raise ValueError('a fresh start overlaps earlier experience')
    ranker = SupportedValue(checkpoint,'kernel')
    modes = (*MODES,'mirrored') if block is not None else MODES
    data = dict(schema='tt23-value-ready-1',arena=CONFIG,source=CALIBRATION.name,
                source_sha256=_hash(CALIBRATION),pong_source=NATIVE_PONG.name,pong_sha256=_hash(NATIVE_PONG),
                pong_options=coefficients,portable_proposal_gains=gains,ready_step_m=READY_STEP,ready_limit_m=READY_LIMIT,
                spread=.30,pace=.82,rival='anchor',starts=starts,pairs=[],development=(count==0),
                validation_block=block,mirrored_starts=[seed for seed in starts if block is not None and seed in MIRRORED])
    station = Station(dict.fromkeys(modes,ranker),gains,data['spread'],data['pace'],checkpoint['horizon'],True,
                      player_type=ReadyPlayer)
    try:
        for index,seed in enumerate(starts):
            ready_from = 0
            if count:
                order = MODES if index%2==0 else MODES[::-1]
                if seed in data['mirrored_starts']:
                    # Put opposite movement before one pair and after the other in each block.
                    order = ('mirrored',*order) if index==1 else (*order,'mirrored')
                pair = {mode:point(station,seed,mode,coefficients,ready_from) for mode in order}
            else:
                stock = known[seed]
                ready_from = max(event['step'] for event in stock['events']
                                 if event['kind']=='contact' and event.get('side')==-1)
                pong = point(station,seed,'pong',coefficients,ready_from)
                if pong['actions'][:ready_from]!=stock['actions'][:ready_from]:
                    raise ValueError('preparation changed archived commands before the final opposing contact')
                pair = dict(pong=pong,stationary=stock)
            data['pairs'].append(pair)
            data['summary'] = {mode:report([rows[mode] for rows in data['pairs'] if mode in rows])
                               for mode in modes if any(mode in rows for rows in data['pairs'])}
            _write(output,data)
            print('ready blade',seed,{mode:(row['longest'],row['outcome']['contacts'],row['outcome']['legal_returns'])
                                     for mode,row in pair.items()},flush=True)
    finally:
        station.close()
    data['verified'] = verify_live(data)
    data['elapsed_seconds'] = perf_counter()-started
    _write(output,data)
    print(data['summary'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument('--fresh',type=int,default=0,help='Run this many fresh matched starts')
    cohort.add_argument('--block',type=int,choices=(1,2),help='Run the specified frozen validation block')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.fresh<0:
        parser.error('nonnegative fresh start count required')
    output = OUTPUT/f'value-ready-validation-{args.block}.json.gz' if args.block else (FRESH if args.fresh else DEVELOPMENT)
    run(args.fresh,args.output or output,args.block)
