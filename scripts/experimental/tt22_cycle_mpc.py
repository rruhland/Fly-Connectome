"""Use new causal stereo fits to guard and replace planned physical strokes."""

import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt21_coach_oracle import PHASE
from .tt22_cycle import ARENA, CyclePlanner, World
from .tt22_cycle_inverse import InverseReceiver, TRANSFER
from .tt22_cycle_optical import TransferredPlanner, rally_optical
from .tt22_cycle_optical_trial import OUTPUT, verify_optical


OLD = OUTPUT/'cycle-inverse.json.gz'
PHYSICAL = OUTPUT/'cycle-held.json.gz'
RESULT = OUTPUT/'cycle-mpc.json.gz'
CEILING_SEED = 530614602
PROSPECTIVE = 1706932580
LEAD = 4.


class FitPlanner(TransferredPlanner):
    """Use the common planner and widen only the event-triggered search."""

    def __init__(self, world, styles):
        super().__init__(world, styles)
        self.minimum = 0.

    def _options(self, side, frames, *, wide=False):
        if not self.minimum:
            return super()._options(side, frames, wide=wide)
        base, plain = CyclePlanner._options(self, side, frames, wide=False)
        extra, broad = CyclePlanner._options(self, side, frames, wide=True)
        choices = [(goal, note) for goal,note in zip(base+extra,plain+broad,strict=True)
                   if note['impact_frame'] >= self.minimum]
        return [goal for goal,_ in choices], [note for _,note in choices]

    def legal_suffix(self, rates):
        live = self.arena
        trial = self.check
        scratch = trial.arena
        mujoco.mj_copyData(scratch.data,scratch.model,live.data)
        for key in PHASE:
            if hasattr(live,key):
                setattr(scratch,key,deepcopy(getattr(live,key)))
        scratch.events = list(live.events)
        trial._rival_active()
        for action in rates:
            _, _, done, cap, _ = trial.step(np.r_[action,0.],np.zeros(4))
            if scratch.legal_returns>live.legal_returns:
                return True
            if done or cap:
                break
        return False

    def incoming_lead(self):
        path, _, bounce = self.flight.predict(1,self.dry.micro,32)
        if bounce is None:
            return None
        height = path[:,2]-TABLE_HEIGHT
        pick = np.flatnonzero((np.arange(len(path))>bounce) & (-1.10<path[:,1]) & (path[:,1]<-.758)
                              & (abs(path[:,0])<.425) & (.044<height) & (height<.48))
        if not len(pick):
            return None
        return float(pick[0]/self.dry.micro),float(pick[-1]/self.dry.micro)


class EventReceiver(InverseReceiver):
    """Check a pending tape once for each fitted incoming bounce phase."""

    def __init__(self,lenses,**kwargs):
        super().__init__(lenses,**kwargs)
        styles = self.planner.styles
        self.planner.close()
        self.planner = FitPlanner(self.world,styles)

    def reset(self):
        super().reset()
        self.event_phases = set()
        self.last_event = -4
        self.event_estimate = None
        self.mpc = []

    def _event(self,tick,estimate,remaining=None):
        phase = 'after_bounce' if self.seen_near_bounce else 'before_bounce'
        if self.inverse is None or phase in self.event_phases or tick-self.last_event<3:
            return False
        self._shadow(tick,estimate)
        # Use only the optical shadow to check whether later contact is possible.
        lead = self.planner.incoming_lead()
        record = dict(tick=tick,bounce_phase=phase,lead=lead,fit_error_mm=float(self.inverse[3]),
                      fit_observations=self.inverse[5],has_old_tape=remaining is not None,
                      eligible=bool(lead is not None and lead[1]>=LEAD),full_search=False,changed=False)
        self.mpc.append(record)
        if not record['eligible']:
            return False
        self.event_phases.add(phase)
        self.last_event = tick
        if remaining is not None:
            self.replanned = True
            record['old_is_legal'] = self.planner.legal_suffix(remaining)
            if record['old_is_legal']:
                return True
        record['full_search'] = True
        self.planner.minimum = LEAD
        try:
            replacement, note = self.planner.choose(1,wide=True)
        finally:
            self.planner.minimum = 0.
        record['search'] = note
        if replacement is not None:
            self.tape, self.start = replacement,tick
            if remaining is None:
                self.last_attempt = tick
            old = np.empty((0,5)) if remaining is None else remaining
            horizon = max(len(old),len(replacement))
            unchanged = np.array_equal(np.pad(old,((0,horizon-len(old)),(0,0))),
                                       np.pad(replacement,((0,horizon-len(replacement)),(0,0))))
            record['changed'] = not unchanged
            self.decisions.append(dict(**note,phase='mpc',donor_subset=False,
                                       optical=np.asarray(estimate[0]).tolist(),
                                       optical_velocity_m_s=(30*np.asarray(estimate[1])).tolist()))
        return True

    def _estimate(self,tick):
        estimate = super()._estimate(tick)
        self.event_estimate = estimate
        if estimate is None:
            return None
        xyz, speed = estimate
        if self.stage=='crossed' and speed[1]<-.035:
            self.event_phases.clear()
            self.last_event = tick-4
        due = self.attempts<4 and tick-self.last_attempt>=4 and xyz[1]<.77
        if (self.stage=='incoming' and self.tape is None and self.attempts>0
                and not due and speed[1]<-.039):
            self._event(tick,estimate)
        return estimate

    def _refinement_phase(self,tick,xyz):
        age = tick-self.start
        if self.inverse is not None and age>=2:
            remaining = self.tape[age:]
            if self._event(tick,self.event_estimate,remaining if len(remaining) else None):
                return None
        return super()._refinement_phase(tick,xyz)


class ExactBallDiagnostic(InverseReceiver):
    """Evaluator-only input swap downstream of the optical scheduler."""

    def __init__(self,lenses,live,**kwargs):
        super().__init__(lenses,**kwargs)
        self.diagnostic_live = live

    def reset(self):
        super().reset()
        self.substitutions = []

    def _shadow(self,tick,estimate):
        super()._shadow(tick,estimate)
        arena = self.world.arena
        live = self.diagnostic_live
        ball, velocity = arena._ball_joint,arena._ball_dof
        arena.data.qpos[ball:ball+3] = live.ball
        arena.data.qvel[velocity:velocity+3] = live.velocity
        arena.data.qvel[velocity+3:velocity+6] = live.spin
        # Leave the original optical bounce inference and any scheduling gates fixed.
        self.substitutions.append(dict(tick=tick,ball=live.ball.tolist(),velocity_m_s=live.velocity.tolist(),
                                       spin_rad_s=live.spin.tolist(),inferred_bounce=bool(arena._received_bounce)))
        mujoco.mj_forward(arena.model,arena.data)


def counts(rows):
    values = [r['legal_alternating_returns'] for r in rows]
    return dict(lengths=values,total=sum(values),ten_or_more=sum(n>=10 for n in values),
                twenty_or_more=sum(n>=20 for n in values),near=sum(r['result']['legal_returns'] for r in rows),
                far=sum(r['result']['rival_returns'] for r in rows),wins=sum(r['reward']>0 for r in rows))


def scores(pairs):
    full = [q for q in pairs.values() if 'stock' in q and 'mpc' in q]
    output = {mode:counts([p[mode] for p in full]) for mode in ('stock','mpc')}
    notes = [n for q in full for n in q['mpc']['mpc']]
    delta = [q['mpc']['legal_alternating_returns']-q['stock']['legal_alternating_returns'] for q in full]
    output['effects'] = dict(gains=sum(x>0 for x in delta),losses=sum(x<0 for x in delta),ties=sum(x==0 for x in delta),
                             accepted=sum(n['eligible'] for n in notes),old_tape_checks=sum('old_is_legal' in n for n in notes),
                             legal_old_preserved=sum(n.get('old_is_legal',False) for n in notes),
                             full_searches=sum(n['full_search'] for n in notes),changed=sum(n['changed'] for n in notes))
    return output


def run(path):
    original,learned = _read(OLD),_read(TRANSFER)
    physical = _read(PHYSICAL)
    privileged = next(r for r in physical['rows'] if r['seed']==CEILING_SEED)
    styles = [tuple(x) for x in learned['styles']]
    samples = learned['rows']['own-practice'][-1]['online_samples']
    fresh = np.random.default_rng(PROSPECTIVE).integers(2**31,size=4).tolist()
    saved = _read(path) if path.exists() else None
    if len(set(fresh))!=4 or (saved is None and set(fresh).intersection(previous_seeds(OUTPUT))):
        raise ValueError('prospective schedule overlaps prior points')
    world = World()
    far = CyclePlanner(world)
    lenses = world.arena.lens_intrinsics()
    common = dict(styles=styles,samples=samples)
    baseline = InverseReceiver(lenses,**common)
    mpc = EventReceiver(lenses,**common)
    injected = ExactBallDiagnostic(lenses,world.arena,**common)
    started = perf_counter()
    receipt = dict(schema='tt22-optical-mpc-1',arena=ARENA,source=OLD.name,source_sha256=_hash(OLD),
                   native_stock_verified=True,far='unchanged privileged physical cycle',
                   near_inputs=['stereo RGB through current frame','clock','own command history'],
                   minimum_lead_frames=LEAD,new_searches_per_bounce_phase=1,online_updates=0,
                   predeclared_fresh=fresh,old={},prospective={},crossover={},
                   implementation={name:_hash(Path(__file__).with_name(name)) for name in (
                       'tt22_cycle_mpc.py','tt22_cycle.py','tt22_cycle_inverse.py','tt22_flight_inverse.py')})

    previous_seconds = 0.
    if saved is not None:
        for key in ('schema', 'arena', 'source_sha256', 'predeclared_fresh'):
            if saved[key] != receipt[key]:
                raise ValueError('cannot resume a different optical comparison')
        receipt.update(old=saved['old'], prospective=saved['prospective'], crossover=saved['crossover'])
        previous_seconds = saved['seconds']

    def save():
        receipt['seconds'] = previous_seconds + perf_counter()-started
        receipt['old_summary'] = scores(receipt['old'])
        receipt['prospective_summary'] = scores(receipt['prospective'])
        _write(path,receipt)

    def play(player,seed,label):
        row = rally_optical(world,far,player,seed,verbose=False)
        row['fit_frames'] = len(player.fits)
        row['fit_planning_calls'] = player.shadow_fits
        if isinstance(player,EventReceiver):
            row['mpc'] = player.mpc
        if isinstance(player,ExactBallDiagnostic):
            row['evaluator_only_substitutions'] = player.substitutions
        row['verified'] = verify_optical([row])
        print('MPC',label,seed,row['legal_alternating_returns'], 'fits',len(player.fits),
              'eligible',sum(r['eligible'] for r in getattr(player,'mpc',[])),flush=True)
        return row

    try:
        for seed in original['seeds']:
            if str(seed) in receipt['old']:
                continue
            old = original['comparison'][str(seed)]['inverse']
            if old['verified']!=1:
                raise ValueError('expected a verified inverse-fit control')
            receipt['old'][str(seed)] = dict(stock=old,mpc=play(mpc,seed,'old MPC'))
            save()
        receipt['crossover']['privileged_reference_returns'] = privileged['legal_alternating_returns']
        receipt['crossover']['physical_source_sha256'] = _hash(PHYSICAL)
        if 'stock' not in receipt['crossover']:
            receipt['crossover']['stock'] = play(baseline,CEILING_SEED,'ceiling optical')
        save()
        if 'exact_ball_evaluator' not in receipt['crossover']:
            receipt['crossover']['exact_ball_evaluator'] = play(injected,CEILING_SEED,'ceiling exact ball')
        save()
        effects = receipt['old_summary']['effects']
        improved = (receipt['old_summary']['mpc']['total']>receipt['old_summary']['stock']['total']
                    and effects['gains']>effects['losses'])
        receipt['prospective_triggered'] = improved
        if improved:
            for seed in fresh:
                pair = receipt['prospective'].setdefault(str(seed), {})
                if 'stock' not in pair:
                    pair['stock'] = play(baseline,seed,'fresh optical')
                save()
                if 'mpc' not in pair:
                    pair['mpc'] = play(mpc,seed,'fresh MPC')
                save()
    finally:
        injected.close()
        mpc.close()
        baseline.close()
        far.close()
        world.close()
    save()
    print(receipt['old_summary'],receipt['prospective_summary'],flush=True)
    return receipt


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=RESULT)
    a=p.parse_args(); run(a.output)
