"""Observe optical far-paddle cues, then associate them with independently audited shots."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_cue_vision import CueGate, PaddleEye, plane_sample, quality
from .tt37_opponent_benchmark import BASE, OUT, PLAN
from .tt37_opponent_capacity import CapacityArena


PARAMETERS = dict(cards=(), strike_ready='lateral', width=384, height=288,
                  clearance_guard=False, fast_search=True, max_steps=1000,
                  rate_scale=.8, velocity_guard=True, motor_envelope=True,
                  guard_table=True, net_rescue=True)
LEADS = (.100, .167, .234)


def roster(plan, pool, repeats, seed_offset=0):
    if pool == 'practice':
        return [dict({k: v for k, v in card.items() if k not in ('opening',)},
                     seed=377201+53*r+977*i+100003*seed_offset,
                     reply_seed=388001+59*r+991*i+110003*seed_offset,
                     key=f'cue-practice-{i}-{r}')
                for i, card in enumerate(plan[pool]) for r in range(repeats)]
    return plan[pool]


def match(events, cues, observations, legs, lead=None):
    targets = []
    chosen = set()
    last_contact = float('-inf')
    for leg, far in enumerate(legs):
        now = far['hit']
        next_contact = next((e['time'] for e in events if e['kind'] == 'contact' and e['time'] > now + 1e-8),
                            float('inf'))
        arrivals = [p for p in observations if now + .10 <= p[0] < next_contact and now < p[2] < next_contact]
        prior = [(i, *p) for i, p in enumerate(cues) if last_contact < p[0] < now]
        if lead is not None:
            prior = [(i, *p) for i, *p in prior if now - lead - .035 < p[0] <= now - lead]
        last_contact = next_contact if next_contact < float('inf') else now
        if not prior or not arrivals:
            continue
        i, t, ball, paddle = prior[-1] if lead is not None else prior[0]
        chosen.add(i)
        label = min(arrivals, key=lambda p: (p[1], -p[0]))
        targets.append(dict(leg=leg, tick_s=t, contact_s=now, lead_s=now-t, ball=ball, paddle=paddle,
                            label_capture_s=label[0], label_plane_s=label[2], arriving=label[3]))
    return targets, len(chosen)


def capture_point(a, s, rig, shot, card, cap, line):
    a.launch(shot, card=card)
    a.partner.reset(seed=card['seed'])
    a.receiver.reset(seed=card['seed'])
    rgb = Vision(rig, a.control_dt, raw=True)
    paddle = PaddleEye({name: rig.fixed[name] for name in ('left', 'right')})
    gate = CueGate(line)
    cues, candidates, contacts, observations = [], [], [], []
    zero = np.zeros(a.action_size)
    for tick in range(a.max_steps):
        if tick % 2 == 0:
            now, frames = tick * a.control_dt, s.capture()
            rgb.update(tick, frames, a.joints)
            f = rgb.eye.latest
            incoming = plane_sample(f, now)
            if incoming is not None:
                distance, t, label = incoming
                observations.append((now, distance, t, label.tolist()))
            blade = paddle.update(now, frames)
            b = gate.update(f, now)
            if b is not None:
                cues.append((now, b.tolist(), None if blade is None else blade.tolist()))
            if quality(f, now) and f.velocity[1] > 1. and f.position[1] > .43:
                candidates.append((now, np.r_[f.position, f.velocity].tolist(),
                                   None if blade is None else blade.tolist()))
        _, _, ended, truncated, _ = a.exchange(zero, FAR_COMMAND)
        if ended or truncated or a.legal_returns+a.rival_returns-1 >= cap:
            break
    score = audit(a.events, opening_side=1)
    far = [r for r in score['shots'] if r['side'] == -1]
    contacts = [e['time'] for e in a.events if e['kind'] == 'contact' and e['side'] == -1]
    labelled, gates_used = match(a.events, cues, observations, far)
    return dict(card=card, own=score['learner'], far=len(far), far_contacts_s=contacts,
                physical_returns=score['returns'], gate_total=len(cues), gate_used=gates_used,
                gates=[p[0] for p in cues], causal=labelled,
                bins={f'{1000*lead:.0f}': match(a.events,candidates,observations,far,lead)[0]
                      for lead in LEADS},
                visually_labelled=sum(any(r['hit']+.10 <= p[0] and abs(p[2]-r['hit']) < 1.30
                                         for p in observations) for r in far))


def run(feeds, output, pool, repeats, cap, limit, seed_offset, plan, line):
    doc = _read(plan)
    if _hash(feeds) != doc['feed_sha256']:
        raise ValueError('Measured feed differs from the TT37 manifest')
    cards = roster(doc, pool, repeats, seed_offset)[:limit]
    launches = measured_launches(feeds)
    result = dict(schema='tt37-cue-capture-1', pool=pool, cap=cap,
                  protocol='random', mentor='finite-mass privileged capacity carriage, labels RGB only',
                  plan_sha256=_hash(plan), seed_offset=seed_offset, line=line, rows=[])
    with centered(_read(BASE)['ready']):
        a = CapacityArena(near_mode='wide', protocol='random', **PARAMETERS)
        s, rig, *_ = resources(a)
        try:
            for card in cards:
                point = capture_point(a, s, rig, launches[card['id']], card, cap, line)
                result['rows'].append(point)
                _write(output, result)
                print(pool, card['key'], point['far'], len(point['causal']),
                      sum(r['paddle'] is not None for r in point['causal']), flush=True)
        finally:
            s.close()
            a.close()
    result['complete'] = True
    _write(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('pool', choices=('practice', 'held_openings', 'held_replies'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cap', type=int, default=6)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--line', type=float, default=1.)
    parser.add_argument('--plan', type=Path, default=PLAN)
    args = parser.parse_args()
    run(args.feeds, args.output, args.pool, args.repeats, args.cap, args.limit,
        args.seed_offset, args.plan, args.line)
