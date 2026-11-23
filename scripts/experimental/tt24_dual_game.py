"""Learn to route two reward-trained physical motor heads from wrist observations."""

import argparse
import math
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import mujoco
from .tt24_donor_trial import DESIGNS, ROOT, DonorBounce, Native
from .tt24_overhead import OpposedArm, PrecisionArm
from .tt24_overhead_audit import replay as overhead_replay
from .tt24_public_body import sight
from .tt24_public_morph import OUTPUT as OBSERVER, load_morph


SOLO = ROOT / 'tt24_donor-refined.npz'
OVERHEAD = ROOT / 'overhead-source.npz'
PRECISION = ROOT / 'overhead-precision.npz'
SEED = 71833901
CAP = 220
CALIBRATION = 24
CHOICES = np.linspace(0., 1., 5)
BODY = DESIGNS[1]


class Overhead:
    """Step the original physical overhead receiver with a public wrist port."""

    def __init__(self, precise):
        kind = PrecisionArm if precise else OpposedArm
        self.arena = kind(BODY, max_steps=CAP, view='wrist', width=192, height=144)
        self.precise = precise
        self.top = self.arena._model.geom('opponent_racket').id

    def reset(self, seed, render):
        arena = self.arena
        arena.reset(seed=seed, split='held', regime='changed', render=False)
        self.hand = arena._data.mocap_pos[0]
        self.ceiling = self.hand[2]
        self.home = arena._data.geom_xpos[arena._blade].copy()
        self.speed_limit = .12 if self.precise else .50
        self.touching, self.ended = False, False
        self.previous = self.pairs = self.receives = self.ticks = 0
        return arena._observation(render)

    def step(self, action, render):
        arena = self.arena
        data, model = arena._data, arena._model
        dt = model.opt.timestep
        for _ in range(8):
            q, v = arena._ball_position, arena._ball_velocity
            data.ctrl[:5] = np.clip(data.ctrl[:5] + action*(5.2*dt), arena._control_low[:5], arena._control_high[:5])
            middle = np.clip(q[:2], self.home[:2]-.025, self.home[:2]+.025) if self.precise else q[:2]
            self.hand[:2] += np.clip(middle-self.hand[:2], -self.speed_limit*dt, self.speed_limit*dt)
            self.hand[2] = self.ceiling+.008*math.sin(17.*data.time)
            incoming = v[2]
            magnitude = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            arena._ball_force[:] = arena._drag_force_scale*magnitude*v
            mujoco.mj_step(model, data)
            hit = 0
            for contact in data.contact:
                if contact.efc_address < 0 or arena._ball_geom not in contact.geom:
                    continue
                first, second = contact.geom
                other = second if first == arena._ball_geom else first
                if other == self.top:
                    hit = 2
                elif other == arena._blade:
                    hit = 1
                else:
                    self.ended = True
            if hit and not self.touching:
                if hit == self.previous or (hit == 1 and incoming > -.20) or (
                        hit == 2 and (incoming < .20 or self.previous != 1)):
                    self.ended = True
                elif hit == 1:
                    self.receives += 1
                else:
                    self.pairs += 1
                self.previous = hit
            self.touching = bool(hit)
            if self.ended or q[2] < .51 or q[2] > 2.1 or abs(q[0]) > .66 or not -1.65 < q[1] < -.33:
                self.ended = True
                break
        self.ticks += 1
        done = self.ended or self.ticks >= CAP
        return arena._observation(render and not done), done

    def metrics(self):
        return dict(reward=self.pairs, run=self.pairs, ticks=self.ticks, near=self.receives)

    def verify(self, seed, tape, score):
        actual = overhead_replay(self.arena, seed, np.asarray(tape))
        expected = [score['reward'], score['run'], score['ticks'], score['near']]
        if actual != expected:
            raise ValueError(f'overhead tape replay diverged on seed {seed}')

    def close(self):
        self.arena.close()


class Solo:
    def __init__(self):
        self.arena = DonorBounce(BODY, max_steps=CAP, view='wrist', width=192, height=144)

    def reset(self, seed, render):
        return self.arena.reset(seed=seed, split='held', regime='changed', render=render)

    def step(self, action, render):
        state, _, done = self.arena.step(action, render=render)
        return state, done

    def metrics(self):
        data = self.arena.judge_metrics()
        return dict(reward=data['rebounds'], run=data['longest_consecutive_rebounds'],
                    ticks=data['steps'], near=data['paddle_contacts'], full=data)

    def verify(self, seed, tape, score):
        self.reset(seed, False)
        for command in tape:
            self.step(command, False)
        if self.metrics() != score:
            raise ValueError(f'solo tape replay diverged on seed {seed}')

    def close(self):
        self.arena.close()


def load_weights(name):
    with np.load(name, allow_pickle=False) as saved:
        return np.ascontiguousarray(saved['weights'], dtype=np.float64)


class OpenStart:
    """Use the first credible direct optical sighting to initialize a fresh episode."""

    def __init__(self, observer):
        self.keeper = load_morph(observer)
        self.frames = 0

    def observe(self, obs):
        keeper = self.keeper
        if self.frames == 0:
            box = sight(obs['rgb'])
            if np.isfinite(box).all():
                frame = keeper.decoder.body(obs['joints'])
                point = keeper.decoder.ball(box, frame)
                relative = point-frame[12:15]
                if np.all(np.abs(relative[:2]) < .18) and -.06 < relative[2] < .8:
                    keeper.previous = point
        value = keeper.observe(obs)
        if self.frames == 1 and len(keeper.history) == 2:
            speed = np.clip(60.*(keeper.history[-1]-keeper.history[-2]),
                            [-1.8,-1.8,-4.5], [1.8,1.8,4.5])
            keeper.velocity = .8*speed + .2*value[3:6]
            value[3:6] = keeper.velocity
        self.frames += 1
        return value


def rollout(runtime, native, seed, weights, *, observer=OBSERVER, open_start=False, stop=False, initial=None):
    rgb = not stop
    obs = runtime.reset(seed, rgb) if initial is None else initial
    keeper = (OpenStart(observer) if open_start else load_morph(observer)) if rgb else None
    action, issued = np.zeros(5), np.zeros(5)
    tape = []
    while True:
        if keeper is not None:
            value = np.ascontiguousarray(keeper.observe(obs), dtype=np.float64)
            native.command(value.ctypes.data, weights.ctypes.data, issued.ctypes.data, action.ctypes.data)
        tape.append(action.tolist())
        obs, done = runtime.step(action, rgb)
        if done:
            break
    outcome = runtime.metrics()
    runtime.verify(seed, tape, outcome)
    return dict(seed=seed, metrics=outcome, actions=tape)


def schedule(observer=OBSERVER, open_start=False):
    seed = SEED+71238 if open_start else SEED
    rng = np.random.default_rng(seed)
    values = rng.choice(2**31-2, 12+12+CALIBRATION+32, replace=False).tolist()
    ordered = np.repeat((0, 1), CALIBRATION//2)
    rng.shuffle(ordered)
    held = np.repeat((0, 1), 16)
    rng.shuffle(held)
    force = np.tile(np.array((0., .5, 1.)), 4)
    rng.shuffle(force)
    plan = dict(schema='tt24-dual-game-plan-1', seed=seed, horizon=CAP,
                gate=dict(wide=values[:12], precise=values[12:24]),
                practice=[[int(k), s] for k, s in zip(ordered, values[24:24+CALIBRATION], strict=True)],
                held=[[int(k), s] for k, s in zip(held, values[24+CALIBRATION:], strict=True)],
                randomized_initial_mixture=force.tolist(), reward_budget=CALIBRATION,
                observer_sha256=_hash(observer), body='center donor with public earlier alternate-shape calibration',
                heads={p.name:_hash(p) for p in (SOLO, OVERHEAD, PRECISION)},
                preregistered_gate='at least 6 of 12 overhead-head partner routes and mean one pair, plus mean one pair above solo; solo mean 4 and at least two rebounds ahead on its own game')
    if open_start:
        plan['perception_variant'] = 'accept first geometrically admissible sighting, use first difference on second frame'
    return plan


def summarize(rows):
    reward = np.array([r['metrics']['reward'] for r in rows])
    run = np.array([r['metrics']['run'] for r in rows])
    return dict(n=len(rows), mean_reward=float(reward.mean()), mean_run=float(run.mean()),
                at_least_one=int(np.count_nonzero(reward)), best_run=int(run.max()))


def gate(native, plan, output, observer, open_start):
    solo = Solo()
    anchor = load_weights(SOLO)
    receipt = dict(plan=plan, profiles={}, chosen=None)
    try:
        for label, precise, path in [('wide', False, OVERHEAD), ('precise', True, PRECISION)]:
            other = Overhead(precise)
            active = load_weights(path)
            try:
                events = {}
                for name, runtime in [('overhead', other), ('solo', solo)]:
                    events[name] = {key:[rollout(runtime, native, s, w, observer=observer, open_start=open_start,
                                                 stop=(key=='hold'))
                                         for s in plan['gate'][label]]
                                    for key, w in [('overhead_head', active), ('solo_head', anchor),
                                                   ('hold', np.zeros_like(anchor))]}
            finally:
                other.close()
            summary = {key:{actor:summarize(rows) for actor, rows in modes.items()} for key,modes in events.items()}
            overhead, bouncing = summary['overhead'], summary['solo']
            good, cross = overhead['overhead_head'], overhead['solo_head']
            other_task, mistaken = bouncing['solo_head'], bouncing['overhead_head']
            passed = (good['at_least_one'] >= 6 and good['mean_run'] >= 1.
                      and good['mean_run']-cross['mean_run'] >= 1.
                      and other_task['mean_run'] >= 4. and other_task['mean_run']-mistaken['mean_run'] >= 2.)
            receipt['profiles'][label] = dict(summary=summary, rows=events, passed=passed)
            if passed:
                receipt['chosen'] = label
            _write(output, receipt)
            print(label, summary, 'passed', passed, flush=True)
            if passed:
                break
    finally:
        solo.close()
    return receipt['chosen']


class Context:
    """The same generic wrist and encoder transform at every episode onset."""

    def __init__(self, observer):
        self.decoder = load_morph(observer).decoder

    def measure(self, obs):
        box = sight(obs['rgb'])
        body = self.decoder.body(obs['joints'])
        if np.isfinite(box).all():
            relative = self.decoder.ball(box, body) - body[12:15]
            detected = 1.
        else:
            relative = np.zeros(3)
            detected = 0.
        image = obs['rgb'].reshape(6, 24, 8, 24, 3).mean(axis=(1, 3)).ravel() / 255.
        return np.r_[relative/.080, detected, obs['joints']/.22, image/(.09*np.sqrt(len(image)))]


class Router:
    """Update a small contextual reward model after each scored experience."""

    def __init__(self, contextual=True):
        self.rows = []
        self.contextual = contextual

    @staticmethod
    def kernel(x, y, context):
        mix = np.exp(-.5*((x[:, None, -1]-y[None, :, -1])/.34)**2)
        if not context:
            return mix
        difference = x[:, None, :-1] - y[None, :, :-1]
        squared = np.einsum('ijk,ijk->ij', difference, difference)
        scene = .64*np.exp(-.5*squared)+.28*np.exp(-.125*squared)+.08
        return mix*scene

    def scores(self, context, *, explore=False):
        if not self.rows:
            return np.zeros(len(CHOICES))
        x, rewards = zip(*self.rows, strict=True)
        x = np.asarray(x)
        rewards = np.log1p(rewards)
        prior = rewards.mean()
        sample = np.c_[np.tile(context, (len(CHOICES),1)), CHOICES]
        precision = self.kernel(x, x, self.contextual)+.19*np.eye(len(x))
        at = self.kernel(sample, x, self.contextual)
        rating = prior + at @ np.linalg.solve(precision, rewards-prior)
        if explore:
            variance = 1.-np.sum(at*np.linalg.solve(precision,at.T).T,axis=1)
            rating += .43*np.sqrt(np.maximum(0.,variance))
        return rating

    def choose(self, context, *, explore=False):
        return float(CHOICES[np.argmax(self.scores(context, explore=explore))])

    def update(self, context, action, reward):
        self.rows.append((np.r_[context, action], float(reward)))


def run_router(native, plan, profile, observer, path, gate_path, open_start, exploratory):
    solo, partner = Solo(), Overhead(profile == 'precise')
    games = (solo, partner)
    other = PRECISION if profile == 'precise' else OVERHEAD
    first, second = load_weights(SOLO), load_weights(other)
    weights = {float(a):np.ascontiguousarray((1.-a)*first+a*second) for a in CHOICES}
    retina = Context(observer)
    router = Router()
    unconditional = Router(contextual=False)
    learning = []
    try:
        for t, (game, seed) in enumerate(plan['practice']):
            world = games[game]
            obs = world.reset(seed, True)
            context = retina.measure(obs)
            scores = router.scores(context, explore=True)
            invitations = plan['randomized_initial_mixture']
            mix = invitations[t] if t < len(invitations) else float(CHOICES[np.argmax(scores)])
            row = rollout(world, native, seed, weights[mix], observer=observer, open_start=open_start, initial=obs)
            # Only a scalar terminal task result enters the learning path.
            outcome = row['metrics']['reward']
            router.update(context, mix, outcome)
            unconditional.update(context, mix, outcome)
            learning.append(dict(trial=t, game=game, mixture=mix, context=context.tolist(), scores=scores.tolist(),
                                 tape=row))
            print('router learned from', t+1, 'mixture', mix, 'physical reward', outcome, flush=True)

        # Separately audit whether the calibration's contextual association
        # helps, while preserving the same earned rewards and mixture choices.
        shuffled = Router()
        rng = np.random.default_rng(SEED+924011)
        offsets = rng.permutation(len(learning))
        for index, row in enumerate(learning):
            shuffled.update(learning[int(offsets[index])]['context'], row['mixture'], row['tape']['metrics']['reward'])
        rows = []
        for game, seed in plan['held']:
            world = games[game]
            obs = world.reset(seed, True)
            context = retina.measure(obs)
            actions = dict(learned=router.choose(context), fixed_solo=0., fixed_overhead=1., half=.5,
                           fixed_fit=unconditional.choose(context), shuffled_context=shuffled.choose(context),
                           ideal=float(game))
            compared = {}
            for mix in sorted(set(actions.values())):
                compared[mix] = rollout(world, native, seed, weights[mix], observer=observer, open_start=open_start)
            tape = dict(game=game, seed=seed, context=context.tolist(), mixtures=actions,
                        predicted_scores=router.scores(context).tolist(), evaluations=compared)
            rows.append(tape)
            print('held', seed, game, {key:compared[mix]['metrics']['reward']
                                      for key,mix in actions.items()}, flush=True)
    finally:
        for world in games:
            world.close()
    results = {name:{game_name:summarize([r['evaluations'][r['mixtures'][name]] for r in rows if r['game']==i])
                     for i,game_name in enumerate(('solo','overhead'))} for name in rows[0]['mixtures']}
    _write(path, dict(schema='tt24-dual-game-routing-1', profile=profile, feasibility_sha256=_hash(gate_path),
                      exploratory_sparse_gate=exploratory,
                      observer_sha256=_hash(observer), fixed_head_digests={p.name:_hash(p) for p in (SOLO,other)},
                      calibration=learning, held=rows, summary=results,
                      online_scalar_reward_trials=len(learning), held_updates=0, hidden_state_labels=0))
    print('routing qualification', results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-only', action='store_true')
    parser.add_argument('--open-start', action='store_true')
    parser.add_argument('--exploratory-sparse', action='store_true',
                        help='Allow an exploratory comparison if both specialized heads have a physical advantage.')
    parser.add_argument('--observer', type=Path, default=OBSERVER)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    stem = '-open-start' if args.open_start else ''
    args.gate = args.gate or ROOT/f'dual-game{stem}-gate.json.gz'
    args.plan = args.plan or ROOT/f'dual-game{stem}-plan.json'
    args.report = args.report or ROOT/f'dual-game{stem}.json.gz'
    plan = schedule(args.observer, args.open_start)
    if args.plan.exists() and _read(args.plan) != plan:
        raise ValueError('registered timetable changed')
    if not args.plan.exists():
        _write(args.plan, plan)
    engine = Native()
    try:
        if args.gate.exists():
            qualified = _read(args.gate)
            if qualified['plan'] != plan:
                raise ValueError('feasibility and plan differ')
            selected = qualified['chosen']
        else:
            selected = gate(engine, plan, args.gate, args.observer, args.open_start)
            qualified = _read(args.gate)
        exploratory = False
        if selected is None and args.exploratory_sparse:
            for variant, proof in qualified['profiles'].items():
                overhead, solo = (proof['summary'][name] for name in ('overhead','solo'))
                if (overhead['overhead_head']['mean_run'] > overhead['solo_head']['mean_run']
                        and overhead['overhead_head']['mean_run'] > overhead['hold']['mean_run']
                        and solo['solo_head']['mean_run'] > solo['overhead_head']['mean_run']):
                    selected = variant
                    exploratory = True
                    break
        if selected is not None and not args.gate_only:
            run_router(engine, plan, selected, args.observer, args.report, args.gate, args.open_start, exploratory)
    finally:
        engine.working.cleanup()
