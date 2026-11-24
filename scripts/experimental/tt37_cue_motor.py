"""Use pre-contact RGB predictions solely to prepare direct joint posture in blind time."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt29_vision import VERTICAL
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, resources
from .tt36_reach import BODY_BANK, OUTPUT as TT36, Body, Reach, Relation
from .tt37_cue_fit import SCALE, predict, vector
from .tt37_cue_vision import CueGate, PaddleEye
from .tt37_opponent import arena, DIAGNOSTIC_CAP
from .tt37_opponent_benchmark import BASE, OUT, PLAN, mark


MOTOR_BANK = TT36 / 'reach-observations.npz'
REACH = TT36 / 'reach-checkpoint.json'
MOTOR_MODES = ('off', 'ball', 'cue', 'uniform')
BLIND_LIMIT_S = .48
AUTHORITY_RAD = .42


def to_plane(x, line=-.40):
    """Move a short, already-observed piece of flight to the common visual label plane."""
    p, v = x[:, :3], x[:, 3:6]
    damp = .133 * np.clip(np.linalg.norm(v, axis=1), 2.5, 12.)
    travel = (line - p[:, 1]) / v[:, 1]
    t = -np.log1p(-damp * travel) / damp
    move, fall = travel, (t - travel) / damp
    at = p + move[:, None] * v + fall[:, None] * VERTICAL
    speed = np.exp(-damp * t)[:, None] * v + move[:, None] * VERTICAL
    return np.c_[at[:, [0, 2]], speed]


class Preparation:
    """Retrieve encoder postures from successful physical demonstrations."""

    def __init__(self, obs):
        x, q = obs['optical'], obs['encoders']
        valid = (-.70 < x[:, 1]) & (x[:, 1] < -.25) & (x[:, 4] < -1.25) & (x[:, 5] < -.85)
        self.y = to_plane(x[valid]) / SCALE
        self.q = q[valid]
        self.tree = cKDTree(self.y)

    def goal(self, forecast, home):
        point = forecast / SCALE
        d, idx = self.tree.query(point, 12)
        dist = float(d[0])
        if dist > 1.85:
            return None, dist
        w = np.exp(-.5 * (d / max(.18, d[5])) ** 2)
        q = (w / w.sum()) @ self.q[idx]
        support = np.clip((1.85-dist)/1.10, 0., 1.)
        wanted = home.copy()
        wanted[:5] += support * np.clip(q - home[:5], -AUTHORITY_RAD, AUTHORITY_RAD)
        return wanted, dist


class EarlyJoint:
    """Combine a small one-shot learned posture with the unchanged incoming-ball pupil."""

    def __init__(self, model, mode, policy, rig, preparation, reach):
        self.model = model
        self.mode = model['chosen'] if mode == 'cue' else mode
        self.policy = policy
        self.preparation = preparation
        self.reach = reach
        self.gate = CueGate(model.get('line', 1.))
        self.paddle = PaddleEye({name: rig.fixed[name] for name in ('left', 'right')})
        self.pending = None
        self.started = -1.
        self.gates = []
        self.driven = 0
        self.handoffs = []

    def command(self, tick, now, vision, optical, q, frames):
        if frames is not None:
            blade = self.paddle.update(now, frames)
            ball = self.gate.update(vision, now)
            if ball is not None:
                context = dict(tick=tick, time_s=now, paddle_visible=blade is not None)
                self.gates.append(context)
                if self.mode != 'off' and blade is not None and optical is None:
                    state = dict(ball=ball, paddle=blade)
                    forecast = predict(self.model['models'][self.mode], vector(state, self.mode))
                    self.pending, distance = self.preparation.goal(forecast, self.policy.home)
                    self.started = now
                    context.update(forecast=forecast.tolist(), posture_support=distance,
                                   prepared=self.pending is not None)
        if optical is not None:
            if self.pending is not None:
                self.handoffs.append(dict(tick=tick, blind_s=now-self.started, x=float(optical[0]),
                                          y=float(optical[1])))
            self.pending = None
        if self.pending is not None and now - self.started > BLIND_LIMIT_S:
            self.pending = None
        if self.pending is not None:
            self.driven += 1
            return self.policy.drive(self.pending, q, self.policy.options['recovery'])
        offset = self.reach.bias(tick, optical, q, frames)
        return self.policy.command(optical, q, offset=offset)


def score_gates(a, gates):
    latest_near = float('-inf')
    contacts = [r for r in a.events if r['kind']=='contact']
    for i, g in enumerate(gates):
        now = g['time_s']
        prior_near = [r['time'] for r in contacts if r['side']==1 and r['time'] < now]
        lo = prior_near[-1] if prior_near else latest_near
        next_g = gates[i+1]['time_s'] if i+1 < len(gates) else float('inf')
        comp = [r['time'] for r in contacts if r['side']==-1 and lo < r['time'] < next_g]
        if comp:
            g['far_contact_s'] = comp[0]
            g['lead_s'] = comp[0] - now
        else:
            g['far_contact_s'] = g['lead_s'] = None


def play(a, renderer, rig, raw, stride, lo, hi, shot, card, mode, model, preparation, body, relation, authority, cap):
    a.launch(shot, card=card)
    a.partner.reset(seed=card['seed'])
    policy = JointPolicy(raw, stride, lo, hi)
    camera = Vision(rig, a.control_dt, raw=policy.raw)
    reach = Reach(body, relation, policy, rig, authority, 1)
    controller = EarlyJoint(model, mode, policy, rig, preparation, reach)
    actions = []
    for tick in range(a.max_steps):
        q = a.joints
        frames = renderer.capture() if tick % 2 == 0 else None
        optical = camera.update(tick, frames, q)
        action = controller.command(tick, tick * a.control_dt, camera.eye.latest, optical, q, frames)
        _, _, done, truncated, _ = a.exchange(action, FAR_COMMAND)
        actions.append(action.tolist())
        if done or truncated or a.legal_returns+a.rival_returns-1 >= cap:
            break
    score_gates(a, controller.gates)
    row = dict(shot=shot, opponent_seed=card['seed'], score=audit(a.events, opening_side=1),
               events=a.events, placements=a.point_placements(), actions=actions,
               motor_audit=a.motor_audit(), gates=controller.gates, handoffs=controller.handoffs,
               early_ticks=controller.driven)
    mark(a, row, cap)
    return row


def run(feeds, model_path, output, pool, modes, cap, limit, plan_path, resume=False):
    modes = list(modes)
    plan = _read(plan_path)
    if _hash(feeds) != plan['feed_sha256']:
        raise ValueError('Measured feeds changed')
    model = _read(model_path)
    cards = plan[pool][:limit]
    if pool != 'practice':
        for key, field in (('id', 'practice_releases'), ('seed', 'practice_seeds'),
                           ('reply_seed', 'practice_reply_seeds')):
            if {c[key] for c in cards} & set(model[field]):
                raise ValueError('Deployment overlaps the forecast practice card')
        if {c['id'] for c in cards} & set(_read(MOTOR_BANK.with_suffix('.json'))['training_releases']):
            raise ValueError('Deployment overlaps the preparation demonstrations')
    launches = measured_launches(feeds)
    raw = _read(BASE)
    with np.load(MOTOR_BANK) as obs:
        relation, preparation = Relation(obs), Preparation(obs)
    with np.load(BODY_BANK) as obs:
        body = Body(obs)
    authority = _read(REACH)['authority_rad']
    result = dict(schema='tt37-cue-motor-1', pool=pool, cap=cap, modes=modes, cards=cards, rows=[],
                  protocol='random', model_sha256=_hash(model_path), raw_sha256=_hash(BASE),
                  motor_bank_sha256=_hash(MOTOR_BANK), body_sha256=_hash(BODY_BANK),
                  reach_sha256=_hash(REACH), plan_sha256=_hash(plan_path),
                  blind_limit_s=BLIND_LIMIT_S, authority_rad=AUTHORITY_RAD)
    if resume:
        archived = _read(output)
        if any(archived[k] != v for k, v in result.items() if k != 'rows'):
            raise ValueError('Cannot resume with a changed course, policy or motor setup')
        if any(set(row) != set(modes) for row in archived['rows']) or len(archived['rows']) > len(cards):
            raise ValueError('Resume tape contains incomplete or unexpected point arms')
        if archived.get('complete'):
            return
        result['rows'] = archived['rows']
        result['recovered_points'] = len(result['rows'])
    first_index = len(result['rows'])
    started = process_time()
    with centered(raw['ready']):
        a = arena(protocol='random')
        # The card always resets the plant. Restore the count for the audit only.
        a.launches = first_index * len(modes)
        renderer, rig, stride, lo, hi = resources(a)
        try:
            for i, c in enumerate(cards[first_index:], first_index):
                arms = {}
                for mode in modes if i % 2 == 0 else reversed(modes):
                    data = play(a, renderer, rig, raw, stride, lo, hi, launches[c['id']], c,
                                mode, model, preparation, body, relation, authority, cap)
                    arms[mode] = data
                    print(pool, c['key'], mode, data['score']['learner'], data['early_ticks'], flush=True)
                # Same first outgoing ball on every arm, up to any near-side effect.
                first = [r['placements']['delivered'][0]['far_contact'] for r in arms.values()
                         if r['placements']['delivered']]
                near_requests = [r['placements']['requested'][0]['target'] for r in arms.values()
                                 if r['placements']['requested']]
                if len({digest_json(f) for f in first}) > 1 or len({digest_json(f) for f in near_requests}) > 1:
                    raise ValueError('Paired point changed the first far contact or draw')
                result['rows'].append(arms)
                _write(output, result)
        finally:
            renderer.close()
            a.close()
    result['cpu_s' if not resume else 'resume_cpu_s'] = process_time() - started
    result['complete'] = True
    _write(output, result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('feeds', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('pool', choices=('practice', 'held_openings', 'held_replies'))
    p.add_argument('output', type=Path)
    p.add_argument('--modes', choices=MOTOR_MODES, nargs='+', default=('off', 'ball', 'cue'))
    p.add_argument('--cap', type=int, default=DIAGNOSTIC_CAP)
    p.add_argument('--limit', type=int)
    p.add_argument('--plan', type=Path, default=PLAN)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    run(args.feeds, args.checkpoint, args.output, args.pool, args.modes, args.cap,
        args.limit, args.plan, args.resume)
