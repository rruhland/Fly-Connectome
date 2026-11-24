"""Observe every ball contact during exact physical command replay."""

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

import numpy as np

from .tt20_fixture import ARCHIVED_FIXTURE
from .tt24_bounce import BounceArena, mujoco


DEFAULT_TAPES = Path('artifacts/tt24/gantry-source-arm-stock.json.gz')


def surface(name):
    if name == 'learner_racket':
        return 'blade'
    if name == 'learner_handle':
        return 'paddle_handle'
    if name.startswith(('tt20_clamp_', 'tt29_clamp_')):
        return 'table_clamp'
    if name.startswith(('tt20_rgb_', 'tt20_camera_', 'tt24_payload_', 'tt29_camera_')) or 'camera_mount' in name:
        return 'camera_mounting_hardware'
    if name.startswith(('learner_gripper_', 'learner_moving_jaw_')):
        return 'hand_or_jaw'
    if name.startswith('learner_'):
        return 'arm_link'
    return name


def interacting_geoms(model, ball):
    belongs = (model.geom_contype & model.geom_conaffinity[ball]) != 0
    belongs |= (model.geom_conaffinity & model.geom_contype[ball]) != 0
    belongs[ball] = False
    return np.flatnonzero(belongs)


class ClearanceAudit(BounceArena):
    """Read contact pairs and distances without giving them to the policy."""

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.collision_trace = []
        self.active_ball_contacts = set()
        self.last_blade_contact = None
        self.home_center = self._data.geom_xpos[self._blade].copy()
        live = interacting_geoms(self._model, self._ball_geom)
        self.guard_geoms = [int(i) for i in live if surface(self._model.geom(int(i)).name) not in ('blade', 'table', 'floor')]
        self.closest = {}
        self.substeps_measured = 0
        return observation

    def _judge_substep(self, incoming):
        active = set()
        data = self._data
        blade = data.geom_xpos[self._blade]
        normal = data.geom_xmat[self._blade].reshape(3, 3)[:, 2]
        normal = normal if normal[2] >= 0. else -normal
        ball_position = data.geom_xpos[self._ball_geom]
        delta = ball_position - blade
        gap = float(delta @ normal)
        radial = float(np.linalg.norm(delta - gap * normal))
        touched = []
        for contact in data.contact:
            if contact.efc_address < 0 or self._ball_geom not in contact.geom:
                continue
            other = int(contact.geom[1] if contact.geom[0] == self._ball_geom else contact.geom[0])
            if other in active:
                continue
            active.add(other)
            if other in self.active_ball_contacts:
                continue
            touched.append((other, contact))
        for other, contact in touched:
            name = self._model.geom(other).name
            category = surface(name)
            time_since_blade = None if self.last_blade_contact is None else data.time - self.last_blade_contact
            toward_ball_normal = contact.frame[:3] if contact.geom[1] == self._ball_geom else -contact.frame[:3]
            face = category == 'blade' and toward_ball_normal[2] > .55 and abs(toward_ball_normal @ normal) > .95
            if category == 'blade':
                self.last_blade_contact = data.time
            self.collision_trace.append(dict(t_s=float(data.time), surface=name, category=category,
                  blade_top_face=bool(face), after_prior_blade=time_since_blade is not None,
                  time_since_blade_s=time_since_blade, incoming_vz_m_s=incoming, outgoing_v=self._ball_velocity.tolist(),
                  ball=ball_position.tolist(), blade=blade.tolist(), contact_point=contact.pos.tolist(),
                  normal=toward_ball_normal.tolist(), blade_normal=normal.tolist(), radial_ball_offset_m=radial,
                  ball_height_above_blade_face_m=gap, blade_center_delta_m=(blade - self.home_center).tolist(),
                  blade_tilt_deg=math.degrees(math.acos(np.clip(normal[2], -1., 1.)))))
        self.active_ball_contacts = active
        self.substeps_measured += 1
        if self.substeps_measured % 8 == 1 and abs(delta[2]) < .18:
            phase = 'initial_drop' if self.last_blade_contact is None else 'after_blade'
            distance, name = self.closest.get(phase, (.18, None))
            for index in self.guard_geoms:
                measured = mujoco.mj_geomDistance(self._model, data, self._ball_geom, index, .18, None)
                if measured < distance:
                    distance, name = float(measured), self._model.geom(index).name
            self.closest[phase] = distance, name
        return super()._judge_substep(incoming)

    def contact_receipt(self):
        return dict(contacts=self.collision_trace, minimum_sampled_surface_gap=self.closest)


def _load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        return json.load(file)


def _write(path, data):
    encoded = json.dumps(data, separators=(',', ':'))
    with gzip.open(path, 'wt', encoding='utf-8') as file:
        file.write(encoded)


def _first_flights(rows):
    drifts, speeds, tilts = [], [], []
    for row in rows:
        events = row['audit']['contacts']
        index = next((idx for idx, event in enumerate(events) if event['category'] == 'blade'), None)
        if index is None or index + 1 == len(events):
            continue
        first, later = events[index:index + 2]
        elapsed = later['t_s'] - first['t_s']
        if elapsed < .065:
            continue
        horizontal = np.linalg.norm(np.subtract(later['ball'][:2], first['ball'][:2]))
        drifts.append(float(horizontal))
        speeds.append(float(horizontal / elapsed))
        tilts.append(first['blade_tilt_deg'])
    return dict(observed_next_contacts=len(drifts),
                median_world_xy_drift_m=float(np.median(drifts)) if drifts else None,
                median_world_xy_speed_m_s=float(np.median(speeds)) if speeds else None,
                median_blade_tilt_at_first_contact_deg=float(np.median(tilts)) if tilts else None)


def summarize(rows):
    history, ending = Counter(), Counter()
    situations = Counter()
    gap = {'initial_drop': [], 'after_blade': []}
    stopped_by_structure = []
    for row in rows:
        previous = row['judge']['other_collider']
        for contact in row['audit']['contacts']:
            history[contact['category']] += 1
            if contact['surface'] == previous:
                ending[contact['category']] += 1
                if contact['category'] not in ('table', 'floor', 'blade'):
                    phase = 'post_blade' if contact['after_prior_blade'] else 'initial_drop'
                    direction = 'ascending' if contact['incoming_vz_m_s'] > 0. else 'descending'
                    situations[f'{phase}_{direction}'] += 1
                    stopped_by_structure.append(contact)
        for phase, sample in row['audit']['minimum_sampled_surface_gap'].items():
            gap[phase].append(sample[0])
    def median(values):
        return float(np.median(values)) if values else None

    return dict(episodes=len(rows), contacts=dict(history), terminal_contacts=dict(ending),
                terminal_structure_frames=len(stopped_by_structure), terminal_structure_phases=dict(situations),
                terminal_structure_median_vz=median([row['incoming_vz_m_s'] for row in stopped_by_structure]),
                terminal_structure_median_radial_offset_m=median([row['radial_ball_offset_m'] for row in stopped_by_structure]),
                terminal_structure_median_blade_tilt_deg=median([row['blade_tilt_deg'] for row in stopped_by_structure]),
                median_minimum_sampled_surface_gap_m={phase: median(values) for phase, values in gap.items()},
                first_flight=_first_flights(rows))


def audit(source):
    d = _load(source)
    output = dict(schema='tt24-clearance-contact-replay-1', privileged_diagnostic=True,
                  source=str(source), arm=d['arm'], horizon=d['horizon'], outcomes={})
    arena = ClearanceAudit(view='none', arm=d['arm'], max_steps=d['horizon'], fixture=ARCHIVED_FIXTURE)
    try:
        for key, treatment in d['outcomes'].items():
            rows = []
            for expected in treatment['episodes']:
                arena.reset(seed=expected['seed'], split='held', regime=d['regime'])
                for command in expected['commands']:
                    _, _, done = arena.step(command, render=False)
                judge = arena.judge_metrics()
                assert done and json.loads(json.dumps(judge)) == {k: v for k, v in expected.items() if k != 'commands'}
                rows.append(dict(judge=judge, audit=arena.contact_receipt()))
            output['outcomes'][key] = dict(summary=summarize(rows), replay_exact=True, episodes=rows)
            print(key, output['outcomes'][key]['summary'], flush=True)
    finally:
        arena.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_TAPES)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24/clearance-contacts.json.gz'))
    args = parser.parse_args()
    _write(args.out, audit(args.source))


if __name__ == '__main__':
    main()
