"""Optional five-axis physical far-court practice robot for SO-101 rallies."""

from copy import copy
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import qmc

from .tt4_arena import AIR_DRAG, BALL_RADIUS, GRAVITY, RACKET_BOUNCE, TABLE_FORWARD, TABLE_HEIGHT, mujoco
from .tt20_fixture import TT20Arena
from .tt20_opponent import RivalSearch
from .tt22_opponent import ANCHOR, Incoming, RallyOpponent, _rank


AXES = ('cross', 'depth', 'height', 'swivel', 'loft')
TRAVEL = ((-.50, .50), (-.30, .36), (-.16, .47), (-.68, .68), (-.45, .98))
RATE = np.array((2.65, 3.4, 2.45, 9., 9.))
PARK = np.array((0., .265, -.035, 0., .25))
MASS = (1.6, .84, .52, .17, .15)
GAINS = ((2600, 105, 120), (2600, 80, 132), (2900, 67, 96), (57, 2.2, 7), (57, 2.2, 7))
JOINT_AXES = ('1 0 0', '0 1 0', '0 0 1', '0 0 1', '-1 0 0')
COURSE = qmc.Sobol(4, scramble=True, seed=61713721).random_base2(5) - .5
FRAMES = 57


def _value(values):
    return ' '.join(str(v) for v in values)


def _structure(parent, name, center, size, color, *, physical=False):
    attrs = dict(name=f'tt22_precision_{name}', type='box', pos=_value(center), size=_value(size), rgba=color)
    if physical:
        attrs.update(contype='2', conaffinity='5')
    return ET.SubElement(parent, 'geom', **attrs)


def _add_robot(document, origin):
    world = document.find('worldbody')
    base = world.find("body[@name='rival_base']")
    for child in list(base):
        base.remove(child)
    base.set('pos', _value(origin))
    base.attrib.pop('quat', None)
    steel, joint = '.28 .35 .40 1', '.46 .55 .60 1'
    for side, label in ((-1, 'left'), (1, 'right')):
        _structure(base, label + '_post', (side*.57, .24, -.165), (.034, .052, .80), steel, physical=True)
        _structure(base, label + '_foot', (side*.57, .24, -.933), (.12, .16, .035), steel, physical=True)
    _structure(base, 'crossbeam', (0., .24, .625), (.606, .045, .039), steel, physical=True)
    names = ('cross_carriage', 'depth_carriage', 'elbow', 'swivel_stage', 'loft_stage')
    drives = document.find('actuator')
    for drive in list(drives):
        if drive.get('name', '').startswith('rival_'):
            drives.remove(drive)
    for section in ('tendon', 'equality'):
        for item in list(document.find(section)):
            if item.get('name', item.get('tendon1', '')).startswith('rival_'):
                document.find(section).remove(item)
    stage = base
    stages = []
    for index, (axis_name, body_name, vector, limits, gains, mass) in enumerate(
        zip(AXES, names, JOINT_AXES, TRAVEL, GAINS, MASS, strict=True)
    ):
        stage = ET.SubElement(stage, 'body', name='rival_' + body_name)
        stages.append(stage)
        linear = index < 3
        ET.SubElement(stage, 'joint', name='rival_' + axis_name, type='slide' if linear else 'hinge',
                      axis=vector, range=_value(limits), damping='3' if linear else '.035',
                      armature='.021' if linear else '.003')
        ET.SubElement(stage, 'inertial', pos='0 0 0', mass=str(mass),
                      diaginertia='.016 .016 .016' if linear else '.001 .001 .001')
        kp, kv, limit = gains
        ET.SubElement(drives, 'position', name='rival_' + axis_name, joint='rival_' + axis_name,
                      kp=str(kp), kv=str(kv), ctrlrange=_value(limits), forcerange=_value((-limit, limit)))
    _structure(stages[0], 'cross_carriage', (0., .24, .615), (.064, .069, .062), joint)
    _structure(stages[0], 'depth_rail', (0., .025, .610), (.029, .345, .025), steel)
    _structure(stages[1], 'depth_block', (0., 0., .603), (.053, .063, .042), joint)
    _structure(stages[1], 'vertical_rail', (0., .044, .23), (.018, .021, .399), steel)
    _structure(stages[2], 'vertical_block', (0., .035, .055), (.039, .042, .038), joint)
    ET.SubElement(stages[3], 'geom', name='tt22_precision_gimbal', type='sphere', pos='0 .046 0',
                  size='.030', rgba=joint)
    ET.SubElement(stage, 'geom', name='rival_racket', **{'class': 'racket_collision'},
                  type='cylinder', size='.076 .006', quat='.7071067811865476 .7071067811865476 0 0',
                  rgba='.24 .67 .95 1')


def reference_impact(ball, incoming, anchor):
    """Approximate the desired impulse before scoring real contact solutions."""
    x, y, z = ball
    target_x, target_y, target_z, target_speed = anchor
    total = y - target_y
    speed = target_speed / TABLE_FORWARD * np.exp(AIR_DRAG * total)
    landing = np.clip(target_y + .30 - .55 * (target_z - .19), -.721, -.47)
    elapsed = np.expm1(AIR_DRAG * (y - landing)) / (AIR_DRAG * speed)
    drag = AIR_DRAG * speed
    fall = TABLE_HEIGHT + BALL_RADIUS - z
    vertical = (fall + GRAVITY * elapsed / drag) * drag / -np.expm1(-drag * elapsed) - GRAVITY / drag
    requested = np.array(((target_x - x) * speed / total, -speed, vertical))
    delta = requested - incoming
    normal = delta / np.linalg.norm(delta)
    speed_normal = incoming @ normal + np.linalg.norm(delta) / (1. + RACKET_BOUNCE)
    return np.arctan2(normal[2], np.linalg.norm(normal[:2])), np.arctan2(normal[0], -normal[1]), speed_normal


class PrecisionOpponent(RallyOpponent):
    """Use private exact planning for the far Cartesian robot only."""

    def __init__(self, arena):
        self.arena = arena
        self.origin = np.asarray(arena.model.body('rival_base').pos).copy()
        RivalSearch.__init__(self, arena, None)
        self.incoming = Incoming(arena)
        self.check = copy(arena)
        self.check.data = mujoco.MjData(arena.model)

    def target(self):
        return self.arena.precision_anchor

    def command(self, bias, target_depth, scale):
        arena = self.arena
        current = arena.data.ctrl[self.axes]
        home = np.clip((PARK - current) / (self.micro * self.increment), -.9, .9)
        if arena._last_hitter == -1 or arena.velocity[1] <= .20:
            return home
        age = arena.steps - self.start
        if self.tape is not None and 0 <= age < len(self.tape):
            return self.tape[age]
        y = arena.ball[1]
        stage = int(y > -.48) + int(y > .34) + int(arena._received_bounce)
        separated = arena._racket_ids[1] not in arena._previous_touches
        if stage > self.stage and stage > 0 and y < 1.06 and separated:
            self.stage = stage
            self.tape, record = self.plan()
            self.start = arena.steps
            self.decisions.append(dict(tick=arena.steps, phase=stage, ball=arena.ball.tolist(), **record))
            if self.tape is not None:
                return self.tape[0]
        return home

    def goals(self, windows, frames):
        timeline = np.arange(frames) * self.arena.control_dt
        goals, parameters = [], []
        for when, point, incoming in windows:
            pitch, yaw, pace = reference_impact(point, incoming, self.arena.precision_anchor)
            for lift, side, impulse, timing in COURSE:
                loft = float(np.clip(pitch + .49 * lift, -.41, .92))
                swivel = float(np.clip(yaw + .29 * side, -.63, .63))
                normal = np.array((np.sin(swivel)*np.cos(loft), -np.cos(swivel)*np.cos(loft), np.sin(loft)))
                speed = float(np.clip(pace + 1.50 * impulse, -.91, 2.5))
                lead = float(.051 + .078 * timing)
                center = point - (BALL_RADIUS + .006) * normal - self.origin
                if np.any(center < np.asarray(TRAVEL)[:3, 0]) or np.any(center > np.asarray(TRAVEL)[:3, 1]):
                    continue
                shift = np.clip(speed * (timeline - when + lead), -.16, .17)[:, None] * normal
                targets = np.column_stack((center + shift, np.full(frames, swivel), np.full(frames, loft)))
                goals.append(targets)
                parameters.append(dict(eta=float(when), pitch=loft, yaw=swivel, paddle_speed=speed, lead=lead,
                                       ball_center=point.tolist()))
        return np.asarray(goals), parameters

    def plan(self):
        began = perf_counter()
        arena = self.arena
        frames = min(FRAMES, arena.max_steps - arena.steps)
        windows = self.incoming.windows(frames)
        goals, parameters = self.goals(windows, frames)
        record = dict(windows=len(windows), candidates=len(goals), predicted_legal=0, exact_trials=0,
                      confirmed=False, target=list(arena.precision_anchor))
        if len(goals):
            tapes, paths = self.predict(goals)
            ranked = _rank(arena, paths[..., self.position:self.position+3],
                           paths[..., self.velocity:self.velocity+3], arena.precision_anchor)
            record['predicted_legal'] = len(ranked)
            chosen = self.confirm(ranked, tapes, record)
            if chosen is not None:
                index = chosen['index']
                return tapes[index], dict(record, **chosen, selected=parameters[index], wall_seconds=perf_counter()-began)
        return None, dict(record, wall_seconds=perf_counter()-began)


class PrecisionArena(TT20Arena):
    """Keep the native learner API, venue, ball and referee with a different far robot."""

    def __init__(self, *args, precision_anchor=ANCHOR, **kwargs):
        self.precision = None
        self.precision_anchor = tuple(map(float, precision_anchor))
        super().__init__(*args, **kwargs)
        self.precision = PrecisionOpponent(self)
        self.precision_command = np.zeros(len(AXES))

    def _model_document(self):
        document = super()._model_document()
        self.rival_base_depth = self.plant.table_half_length + .10
        _add_robot(document, (0., self.rival_base_depth, .97))
        return document

    def _arm_axes(self, side):
        return (AXES, (), RATE) if side == -1 else super()._arm_axes(side)

    def _arm_home(self, side):
        return (PARK, ()) if side == -1 else super()._arm_home(side)

    def _choose_rival_response(self):
        super()._choose_rival_response()
        self._rival_attempt = True
        if self.precision is not None:
            self.precision.new_shot()

    def reset(self, *args, **kwargs):
        if self.precision is not None:
            self.precision.reset()
        self.precision_actions = []
        result = super().reset(*args, **kwargs)
        self._rival_attempt = True
        return result

    def _aim(self, side, *args):
        if side != -1:
            return super()._aim(side, *args)
        self.precision_command = self.precision.command(*args)
        self.precision_actions.append(self.precision_command.tolist())
        return self.precision_command
