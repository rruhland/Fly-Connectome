"""Transfer only a reward-free update rate from game experience into changed Pong."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch

from arcade_hockey_pong import PongFlightActor
from arcade_transfer_session import CHECKPOINT
from fly_connectome.pong import Physics, Pong


RATES = (.08, .20, .45, .70, .90)
VARIANTS = ((.39, .79), (.79, .39), (1., 1.))
TICKS_PER_LIFE = 480
LIVES_PER_PHASE = 4


class RateLearner:
    def __init__(self, starting_coefficient, rate):
        self.coefficient = starting_coefficient
        self.rate = rate
        self.observations = []

    def observe_hint(self, ratio, uncertainty):
        # This comparison uses only fully observed pre/post bounce trajectories.
        pass

    def observe(self, value, uncertainty):
        # The tracker screens for intact wall flights before calling observe().
        self.coefficient += self.rate * (np.clip(value, 0., 1.) - self.coefficient)
        self.observations.append(float(self.coefficient))


def select_rate(blocks, anchor):
    # One-step prediction loss uses the next visible bounce. It consumes no
    # environment coefficients, source block labels, contact flags or target data.
    observed = [evt['coefficient'] for block in blocks for life in block['lives'] for evt in life['bounces']]
    if not observed:
        raise ValueError('cannot learn an adjustment rate without observed wall flights')
    losses = []
    for rate in RATES:
        learner = RateLearner(anchor, rate)
        errors = []
        for ratio in observed:
            errors.append((np.clip(ratio, 0, 1) - learner.coefficient) ** 2)
            learner.observe(ratio, None)
        losses.append(float(np.mean(errors)))
    return dict(rate=RATES[int(np.argmin(losses))], mse=losses, visible_bounces=len(observed))


def batched_life(models, timing, seeds, coefficient, ordinary, *, view=None, bank_side=None):
    # Physics runs the native Pong body and collision equations for every lane.
    # Independent stored learners keep only the wall estimate between points.
    config = replace(Physics(), opponent_speed=.55, wall_restitution=coefficient,
                     paddle_half=.10 if ordinary else .066, ball_speed=.55 if ordinary else 1.0)
    env = Pong(seeds, config)
    n = len(seeds)
    if not ordinary:
        for lane, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            angle = rng.uniform(.62, .99)
            x = rng.uniform(.72, .89)
            y = rng.uniform(.30, .43)
            sign = rng.choice((-1., 1.)) if bank_side is None else bank_side
            env.ball[lane] = torch.tensor([x, y if sign < 0 else 1-y])
            env.ball_velocity[lane] = torch.tensor([-math.cos(angle), sign * math.sin(angle)])
    actors = [PongFlightActor(timing, model) for model in models]
    def camera_frame():
        pixels = env.render(64, 64)
        return pixels if view is None else view(pixels)

    for actor, frame in zip(actors, camera_frame(), strict=True):
        actor.start(frame)
    active = np.ones(n, dtype=bool)
    seen = [len(model.observations) for model in models]
    rows = [dict(seed=int(s), commands=0, hits=0, opportunities=0, first_hit=False, score=0,
                 initial_estimate=float(m.coefficient)) for s,m in zip(seeds,models,strict=True)]
    ticks = frame_index = 0
    while ticks < TICKS_PER_LIFE and active.any():
        before_frame = active.copy()
        cmds = [actor.choose() if playing else 1 for actor,playing in zip(actors,active,strict=True)]
        drive = torch.tensor([(-1.,0.,1.)[c] for c in cmds])
        for lane in np.flatnonzero(active):
            rows[lane]['commands'] += 1
        cadence = (frame_index + 1) * 12 // 5 - frame_index * 12 // 5
        elapsed = np.zeros(n)
        for _ in range(min(cadence, TICKS_PER_LIFE - ticks)):
            old_x = env.ball[:,0].clone()
            vx = env.ball_velocity[:,0].clone()
            transition = env.step(drive, auto_reset=False)
            plane = config.paddle_x + config.ball_radius
            opportunity = ((old_x >= plane) & (old_x + vx*config.dt <= plane) & (vx < 0)).numpy()
            elapsed[active] += config.dt
            for lane in np.flatnonzero(active & opportunity):
                row = rows[lane]
                hit = int(transition.hits[lane])
                if not row['opportunities']:
                    row['first_hit'] = bool(hit)
                row['opportunities'] += 1
                row['hits'] += hit
            scores = transition.scores.numpy()
            for lane in np.flatnonzero(active & (scores != 0)):
                rows[lane]['score'] = int(scores[lane])
                active[lane] = False
            ticks += 1
        # An earlier substep can have ended this lane. Its terminal image is
        # omitted because it can contain post-goal motion in a batched rollout.
        frames = camera_frame()
        for lane in np.flatnonzero(active & before_frame):
            actors[lane].observe(frames[lane], elapsed[lane])
        frame_index += 1
    for lane,model in enumerate(models):
        rows[lane]['estimates_after_bounce'] = model.observations[seen[lane]:]
    return rows


def run(practice_path, seed, n):
    torch.set_num_threads(1)
    practice = json.loads(practice_path.read_bytes())
    wall = practice['common_pong_memory'] if 'common_pong_memory' in practice else practice['anchor']['wall']
    anchor = float(np.clip(wall['weighted_total']/wall['precision'],0,1)) if wall['precision'] else wall['default']
    sources = [label for label in ('hockey', 'pong', 'varied_pong') if label in practice]
    learned = {label:select_rate(practice[label],anchor) for label in sources}
    rates = {**{label:v['rate'] for label,v in learned.items()}, 'fixed_mid':.45, 'latest':1.0}
    timing = json.loads(CHECKPOINT.read_bytes())['timing']
    rng = np.random.default_rng(seed)
    courses = []
    for variant in VARIANTS:
        game_seeds = rng.integers(1,2**31-1,size=(LIVES_PER_PHASE*2,n))
        records = {}
        for alpha in sorted(set(rates.values())):
            models = [RateLearner(anchor,alpha) for _ in range(n)]
            lives = []
            for index,starts in enumerate(game_seeds):
                phase = index // LIVES_PER_PHASE
                # Include a natural native serve at the end of each phase.
                ordinary = index % LIVES_PER_PHASE == LIVES_PER_PHASE - 1
                lives.append(batched_life(models,timing,starts.tolist(),variant[phase],ordinary))
            records[alpha] = lives
        courses.append(dict(physics=variant, outcomes={k:records[a] for k,a in rates.items()}))
    return dict(schema='cross-game-adaptation-rate-v1', anchor=anchor, seed=seed,
                rates=rates, fitted=learned, source_frames={label:sum(b['frames'] for b in practice[label])
                    for label in sources}, courses=courses,
                practice_sha256=sha256(practice_path.read_bytes()).hexdigest(),
                source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__),Path(__file__).with_name('arcade_hockey_pong.py'),CHECKPOINT]})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('practice',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--seed',type=int,default=3911060)
    parser.add_argument('--streams',type=int,default=16)
    args=parser.parse_args()
    data=run(args.practice,args.seed,args.streams)
    with args.output.open('x') as f:
        json.dump(data,f,separators=(',',':'),allow_nan=False)
    print(data['rates'])
