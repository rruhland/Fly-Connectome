"""Locate rotary-arm failures in replayed physical contact and landing traces."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_arena import BALL_RADIUS, TABLE_HALF_WIDTH, mujoco
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_continuous_trial import point_condition
from .tt6_racket import set_racket_response
from .tt6_timing import DelayedWiring


class ContactAuditArena(TT6Arena):
    """Add evaluator-only diagnostic measurements without changing physics."""

    def reset(self, **kwargs):
        self.contact_audit = []
        self.closest = None
        return super().reset(**kwargs)

    def _contacts(self, incoming):
        if self._received_bounce and self._last_hitter == -1 and self.ball[1] < -.85 and self.velocity[1] < -.5:
            index = self._racket_ids[1]
            normal = self.data.geom_xmat[index].reshape(3, 3)[:, 2]
            displacement = self.ball - self.data.geom_xpos[index]
            axial = float(displacement @ normal)
            radial = float(np.linalg.norm(displacement - axial * normal))
            gap = np.hypot(max(0., abs(axial) - .006), max(0., radial - .076)) - BALL_RADIUS
            if self.closest is None or gap < self.closest['surface_gap']:
                self.closest = dict(surface_gap=float(gap), normal_gap=axial, lateral_gap=radial,
                                    displacement=displacement.tolist(), time=float(self.data.time))
        super()._contacts(incoming)

    def _racket_contact(self, side, location, incoming):
        if side == 1:
            index = self._racket_ids[1]
            normal = self.data.geom_xmat[index].reshape(3, 3)[:, 2].copy()
            if normal[1] < 0:
                normal *= -1
            center = self.data.geom_xpos[index]
            offset = self.ball - center
            radial = offset - (offset @ normal) * normal
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, index, velocity, 0)
            blade_velocity = velocity[3:] + np.cross(velocity[:3], np.asarray(location) - center)
            self.contact_audit.append(dict(time=float(self.data.time), normal=normal.tolist(),
                                           yaw=float(np.arctan2(normal[0], normal[1])),
                                           pitch=float(np.arctan2(normal[2], np.hypot(*normal[:2]))),
                                           incoming=list(incoming), position=self.ball.tolist(),
                                           radial_offset=radial.tolist(), blade_velocity=blade_velocity.tolist()))
        super()._racket_contact(side, location, incoming)


def replay(geometry, install, seed, points):
    result = []
    arena = ContactAuditArena(install['arena_seed'], body='rotary', view_seed=install['view_seed'], **geometry)
    wiring = TT6Wiring(arena, install['arena_seed'])
    try:
        for point in points:
            rubber, disturbance = point_condition(point['phase'])
            set_racket_response(arena, rubber)
            motor = DelayedWiring(wiring, disturbance, seed=seed)
            motor.reset(seed=point['seed'], render=False)
            for action in point['actions']:
                motor.step(action, render=False)
            if arena.point_result() != point['evaluator']:
                raise ValueError(f"point outcome diverged at seed {point['seed']}")
            separation = next((e for e in arena.events if e['kind'] == 'separation'
                               and e['collider'] == 'learner_racket'), None)
            contact = next(iter(arena.contact_audit), None)
            bounce = next((e for e in arena.events if e['kind'] == 'bounce' and contact is not None
                           and e['time'] > contact['time']), None)
            net_clip = contact is not None and any(
                e['kind'] == 'net' and contact['time'] < e['time'] and (bounce is None or e['time'] < bounce['time'])
                for e in arena.events)
            result.append(dict(seed=point['seed'], phase=point['phase'], reason=arena.reason,
                               returns=arena.legal_returns, rival_returns=arena.rival_returns,
                               closest=arena.closest, contact=contact, outgoing=separation,
                               first_landing=None if bounce is None else bounce['location'], net_clip=net_clip))
    finally:
        arena.close()
    return result


def summarize(points):
    contacts = [p for p in points if p['contact'] is not None]
    failures = [p for p in contacts if not p['returns']]
    launches = [p['outgoing']['velocity'] for p in failures if p['outgoing'] is not None]
    wide = [p for p in failures if p['first_landing'] is not None and abs(p['first_landing'][0]) > TABLE_HALF_WIDTH]
    return dict(points=len(points), with_contact=len(contacts), with_return=sum(bool(p['returns']) for p in points),
                contact_failures=dict(Counter(p['reason'] for p in failures)), wide_bounces=len(wide),
                failed_net_clips=sum(p['net_clip'] for p in failures),
                failed_abs_yaw=None if not failures else float(np.median([abs(p['contact']['yaw']) for p in failures])),
                failed_contact_blade_speed=None if not failures else np.median(
                    [p['contact']['blade_velocity'] for p in failures], axis=0).tolist(),
                failed_median_launch=None if not launches else np.median(launches, axis=0).tolist())


def audit(output, inputs):
    evidence = []
    hashes = {}
    for path in inputs:
        data = _read(path)
        hashes[path.name] = _hash(path)
        geometry = data['geometry']
        if 'visits' in data:
            for visit in data['visits']:
                if visit['body'] == 'rotary':
                    rows = replay(geometry, visit['installation'], data['seed'], visit['points'])
                    evidence.append(dict(input=path.name, condition='predictive', rows=rows))
        else:
            visit = data['held']
            for name, condition in visit['conditions'].items():
                rows = replay(geometry, visit['installation'], data['held_seed'], condition['points'])
                evidence.append(dict(input=path.name, condition=name, rows=rows))
    all_points = [point for condition in evidence for point in condition['rows']]
    summary = {phase: summarize([point for point in all_points if point['phase'] == phase])
               for phase in ('ordinary', 'soft', 'mixed', 'delay')}
    _write(output, dict(schema='tt6-rotary-contact-audit-v1', summary=summary, input_sha256=hashes, evidence=evidence))
    print(summary)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('inputs', nargs='*', type=Path, default=[ROOT / 'artifacts/tt6' / (name + '.json.gz') for name in
                                                            ('combined-rotary', 'rotary-adaptation',
                                                             'rotary-adaptation-confirm')])
    arguments = cli.parse_args()
    audit(arguments.output, arguments.inputs)
