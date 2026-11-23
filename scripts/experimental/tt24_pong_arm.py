"""Measure what ordinary native Pong identifies about a physical SO-101 motor interface."""

import argparse
from pathlib import Path

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_motor_bank import I, STEP, decoder
from .tt24_ready import prepare_step
from .tt4_transfer_trial import _hash, _read, _write


SOURCE = ROOT / 'pong-arm-source.json.gz'
TARGET = ROOT / 'pong-arm-held.json.gz'
PRIOR = ROOT / 'held-donor-ordinary.json.gz'
NATIVE_CAP = 960
DIRECTION = I[0]
PARKED = np.array((0., -1.05, 2.7))
ACTUATION = .136
PADDLE_SCALE = 1.95
SOURCE_REFERENCE = 138440827
HEIGHT, WIDTH = 64, 96
ROWS = (np.arange(HEIGHT)+.5)/HEIGHT
COLS = (np.arange(WIDTH)+.5)/WIDTH


def wirings():
    restricted = I.copy()
    restricted[:, 0] = [.56,.06,-.06,.10,.48]
    dense = restricted.copy()
    # Four directions are remixed while the visibly exercised column is identical.
    for offset, angle in ((1, .97),(3, -.78)):
        c,s = np.cos(angle),np.sin(angle)
        dense[offset:offset+2,offset:offset+2] = [[c,-s],[s,c]]
    return restricted, dense


class PhysicalPaddle:
    """A native Pong paddle whose position comes from one physical encoder projection."""

    def __init__(self, arena, wiring, gamespec):
        self.arena, self.wiring = arena, wiring
        self.gamespec = gamespec
        self.position = torch.full((1,), .5)
        obs = arena.reset(seed=SOURCE_REFERENCE, split='practice', render=False)
        self.home = obs['joints'].astype(float)
        self.commands = []
        self.encoders = [self.home.tolist()]
        self.elapsed = 0

    def step(self, drive):
        if self.elapsed % 2 == 0:
            request = ACTUATION*float(drive[0])*DIRECTION
            command = np.clip(self.wiring@request,-1.,1.)
            prepare_step(self.arena,command,PARKED)
            joints = self.arena._observation(False)['joints'].astype(float)
            self.commands.append(request.tolist())
            self.encoders.append(joints.tolist())
            projected = .5 + PADDLE_SCALE*((joints-self.home)@DIRECTION)
            self.position[0] = np.clip(projected,self.gamespec.paddle_half,1.-self.gamespec.paddle_half)
        self.elapsed += 1


class ScreenPlayer:
    """Drive toward the intercept extrapolated from native Pong raster frames."""

    def __init__(self):
        self.samples = []
        self.target = .5

    def visual(self, image, step, p):
        # Native ball and paddle predicates are used only through the rendered pixels.
        first = int(np.argmin(abs(COLS-p.paddle_x)))
        paddle = image[:,first]
        position = float(paddle@ROWS/max(1,paddle.sum()))
        usable = image[:,8:-8]
        count = usable.sum()
        if count:
            point = np.array([usable.sum(axis=0)@COLS[8:-8], usable.sum(axis=1)@ROWS])/count
            self.samples.append((step,point))
            self.samples = self.samples[-5:]
            self.target = float(point[1])
            if len(self.samples) >= 3:
                earlier, past = self.samples[0]
                speed = 120.*(point-past)/(step-earlier)
                if speed[0]<-.08:
                    arrival = np.clip((point[0]-p.paddle_x-p.ball_radius)/-speed[0],0.,.75)
                    span = 1.-2.*p.ball_radius
                    u = (point[1]+arrival*speed[1]-p.ball_radius) % (2.*span)
                    self.target = float(p.ball_radius + min(u,2.*span-u))
        return float(np.clip(7.8*(self.target-position),-.98,.98))


@torch.no_grad()
def point(arena, seed, wiring, mode, *, replay=None):
    spec = Physics(opponent_speed=.35)
    game = Pong([seed],spec)
    paddle = PhysicalPaddle(arena,wiring,spec)
    game.body = paddle
    player = ScreenPlayer()
    drives, returns = [],0
    for tick in range(NATIVE_CAP):
        if tick%2==0:
            if replay is not None:
                request = replay[len(drives)]
            elif mode=='ordinary':
                request = player.visual(game.render(HEIGHT,WIDTH).numpy()[0],tick,spec)
            else:
                request = 0.
            drives.append(request)
        result = game.step(torch.tensor([request],dtype=torch.float32),auto_reset=False)
        returns += int(result.hits.item())
        score = int(result.scores.item())
        if score:
            break
    return dict(seed=seed, drives=drives, commands=paddle.commands, encoders=paddle.encoders,
                native_ticks=tick+1, score=score, returns=returns)


def fit_column(clips, *, shuffled=False):
    rng = np.random.default_rng(76372449)
    # Center each rally relative to its own encoder baseline.
    buffers = []
    for clip in clips:
        u = np.asarray(clip['commands'])@DIRECTION
        y = np.asarray(clip['encoders'])[1:]
        if shuffled:
            # Leave native action timing intact, but independently rotate matched encoder rows.
            y = np.roll(y,int(rng.integers(max(4,len(y)//5),max(5,4*len(y)//5))),axis=0)
        buffers.append((np.cumsum(STEP*u), y-y.mean(axis=0)))
    best = None
    # The filter is chosen by cross-rally prediction using only source command and encoder pairs.
    for alpha in (.08,.13,.21,.34,.52,.72,1.):
        designs = []
        for u,y in buffers:
            positions = np.empty(len(u))
            last = 0.
            for i,value in enumerate(u):
                last += alpha*(value-last)
                positions[i] = last
            designs.append((positions-positions.mean(),y))
        pair = lambda group: (np.concatenate([d[0] for d in group]),np.concatenate([d[1] for d in group]))
        fold = [designs[::2],designs[1::2]]
        errs = []
        for a,b in (fold,fold[::-1]):
            x,y = pair(a)
            estimate = (x@y+.02*DIRECTION)/(x@x+.02)
            x,y = pair(b)
            errs.append(float(np.mean((y-x[:,None]*estimate)**2)))
        x,y = pair(designs)
        vector = (x@y+.02*DIRECTION)/(x@x+.02)
        fit = dict(alpha=alpha, encoder_crossval_rms=float(np.sqrt(np.mean(errs))), column=vector.tolist())
        if best is None or fit['encoder_crossval_rms']<best['encoder_crossval_rms']:
            best = fit
    matrix = I.copy()
    matrix[:,0] = best['column']
    return matrix,best


def source(arena):
    limited,dense = wirings()
    rng = np.random.default_rng(1245692047)
    active,passive,verified = [],[],0
    for seed in rng.integers(1,2**31-1,18).tolist():
        actual = point(arena,seed,limited,'ordinary')
        stationary = point(arena,seed,limited,'still')
        raw = point(arena,seed,limited,'ordinary',replay=actual['drives'])
        if digest_json(actual)!=digest_json(raw):
            raise ValueError('native Pong and physical arm replay changed')
        # A second physical instalment with four changed columns receives the identical ordinary moves.
        alternate = point(arena,seed,dense,'ordinary',replay=actual['drives'])
        if digest_json(actual)!=digest_json(alternate):
            raise ValueError('the two actual command maps should agree on every source action')
        active.append(actual)
        passive.append(stationary)
        verified += 2
        print('native arm Pong',seed,(actual['score'],actual['returns']),
              (stationary['score'],stationary['returns']),flush=True)
    matched,fit = fit_column(active)
    shuffled,shuff = fit_column(active,shuffled=True)
    commands = np.concatenate([r['commands'] for r in active])
    spectrum = np.linalg.svd(commands,compute_uv=False)
    true_observed = limited@DIRECTION
    for result,matrix in ((fit,matched),(shuff,shuffled)):
        result['judge_observed_column_error'] = float(np.linalg.norm(matrix@DIRECTION-true_observed)/
                                                    np.linalg.norm(true_observed))
    assess = lambda rows: dict(wins=sum(r['score']>0 for r in rows), losses=sum(r['score']<0 for r in rows),
                               points_with_return=sum(r['returns']>0 for r in rows),
                               total_returns=sum(r['returns'] for r in rows),
                               physical_frames=sum(len(r['commands']) for r in rows))
    evidence = dict(ordinary=active,stationary=passive, command_singular_values=spectrum.tolist(),
                    source_physical_replays=verified, judge_restricted=limited.tolist(),judge_dense=dense.tolist(),
                    matched=fit, shuffled=shuff, configurations=dict(matched=matched.tolist(),shuffled=shuffled.tolist()),
                    game_summary=dict(ordinary=assess(active),stationary=assess(passive)),bundle_sha256=_hash(BUNDLE))
    _write(SOURCE,evidence)
    print(evidence['game_summary'],fit,shuff,spectrum.tolist(),flush=True)
    return evidence


def bounce(arena, policy, seed, wiring, estimate, *, tape=None):
    obs = arena.reset(seed=seed,split='held',render=tape is None)
    if tape is None:
        policy.begin_release(obs)
        inverse = decoder(estimate)
    actions = []
    for tick in range(arena.max_steps):
        if tape is None:
            raw = wiring@np.clip(inverse@policy.act(obs),-1.,1.)
            applied = np.clip(raw,-1.,1.)
        else:
            applied = np.asarray(tape[tick])
        actions.append(applied.tolist())
        obs,_,done = arena.step(applied,render=tape is None)
        if tape is not None and done != (tick+1==len(tape)):
            raise ValueError('wrist tape terminal tick changed')
        if done:
            break
    return dict(metrics=arena.judge_metrics(),actions=actions)


def summarize(rows):
    result = {}
    for label in rows[0]['conditions']:
        streak = np.asarray([item['conditions'][label]['metrics']['longest_consecutive_rebounds'] for item in rows])
        result[label] = dict(mean=float(streak.mean()), run10=int((streak>=10).sum()), maximum=int(streak.max()))
    return result


def historical_retention():
    """Replay the unchanged stock context on an earlier independent physical roster."""
    from .tt24_held_donor_eval import Installation, ObservedBundle, check_tape

    prior = _read(PRIOR)
    if prior['bundle_sha256'] != _hash(BUNDLE):
        raise ValueError('earlier stock reference was built with a different carried policy')
    head = CarriedWristPolicy().weights
    policy = ObservedBundle({'frozen':head})
    installation = Installation('stock',ordinary=True)
    dispenser = BounceArena(view='none')
    outcomes = []
    try:
        for row in prior['cases']:
            plan = row['plan']
            if plan['body'] != 'stock':
                continue
            dispenser.reset(seed=plan['launcher_seed'],split='held',regime=installation.regime,render=False)
            launch = tuple(x.copy() for x in
                           (dispenser._ball_position,dispenser._ball_velocity,dispenser._ball_spin))
            installation.velocity, installation.spin = launch[1:]
            returned = check_tape(installation,policy,plan,launch,'frozen')
            original = row['modes']['frozen']
            if any(digest_json(returned[key]) != digest_json(original[key]) for key in ('metrics','actions')):
                raise ValueError('return to previously unaltered stock did not reproduce its original tape')
            outcomes.append(dict(seed=plan['learner_seed'],returned=returned))
    finally:
        dispenser.close()
        installation.arena.close()
    return dict(context_matrix='original identity, held in separate memory',
                earlier_receipt=PRIOR.name, earlier_sha256=_hash(PRIOR), rows=outcomes)


def target(arena,evidence):
    playing, still = (evidence['game_summary'][key] for key in ('ordinary','stationary'))
    if playing['points_with_return'] <= still['points_with_return'] or playing['wins'] < 1:
        raise ValueError('the physical source did not demonstrate meaningful native Pong play')
    limited,dense = wirings()
    estimates = {key:np.array(v) for key,v in evidence['configurations'].items()}
    ordinary_seeds, changed_seeds = np.random.default_rng(217820758).integers(1,2**31-1,(2,10)).tolist()
    policy = CarriedWristPolicy()
    report = _read(TARGET) if TARGET.exists() else dict(source_sha256=_hash(SOURCE),changes={},physical_replays=0)
    if report['source_sha256']!=_hash(SOURCE) or evidence['bundle_sha256']!=_hash(BUNDLE):
        raise ValueError('source or carried game policy changed since the held protocol was set')
    for case,wiring,roster,raw in (
            ('restricted',limited,changed_seeds,dict(stale=I,**estimates)),
            ('dense',dense,changed_seeds,dict(stale=I,**estimates)),
            ('original_stock',I,ordinary_seeds,dict(frozen_original=I,retained_original=I,
                                                   wrong_context=estimates['matched']))):
        rows = report['changes'].setdefault(case,dict(cases=[]))['cases']
        for seed in roster[len(rows):]:
            outcomes = {}
            for label,estimate in raw.items():
                if label=='retained_original':
                    outcomes[label] = outcomes['frozen_original']
                    continue
                played = bounce(arena,policy,seed,wiring,estimate)
                replayed = bounce(arena,policy,seed,wiring,estimate,tape=played['actions'])
                if digest_json(played)!=digest_json(replayed):
                    raise ValueError(f'physical target replay changed on {seed}')
                report['physical_replays'] += 1
                outcomes[label] = played
            rows.append(dict(seed=seed,conditions=outcomes))
            _write(TARGET,report)
            print('wrist',case,seed,[(tag,item['metrics']['longest_consecutive_rebounds'])
                                    for tag,item in outcomes.items()],flush=True)
        report['changes'][case]['summary'] = summarize(rows)
        _write(TARGET,report)
    original = [row['conditions']['frozen_original']['actions']
                for row in report['changes']['original_stock']['cases']]
    report['bounce_action_singular_values'] = np.linalg.svd(np.concatenate(original),compute_uv=False).tolist()
    report['earlier_stock_retention'] = historical_retention()
    _write(TARGET,report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only',action='store_true')
    parser.add_argument('--held',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    arena = BounceArena(view='wrist',width=192,height=144,max_steps=300)
    try:
        evidence = _read(SOURCE) if args.held else source(arena)
        if not args.source_only:
            if args.held:
                last = evidence['ordinary'][-1]
                repeated = point(arena,last['seed'],wirings()[0],'ordinary',replay=last['drives'])
                if digest_json(last)!=digest_json(repeated):
                    raise ValueError('saved source play changed on the arm before wrist transfer')
            target(arena,evidence)
    finally:
        arena.close()


if __name__=='__main__':
    main()
