"""Catch a table-tennis ball after it banks off a plate and crosses a side-view rail."""

from dataclasses import dataclass
import math
from xml.etree import ElementTree

import numpy as np

from .tt4_arena import (AIR_DRAG, BALL_MASS, MAGNUS_COEFFICIENT, MODEL_PATH,
                        SPIN_DECAY, BALL_INERTIA)
from .tt7_surface_trial import DAMPING

import mujoco


WIDTH, HEIGHT = 288, 112
SCALE, FLOOR_ROW = 78., 96
CENTER = (WIDTH - 1) / 2
DT = 1. / 60.
PLATE = .020
RAIL = .115
LIMIT = 1.66
RATE = 3.0
MOUTH = .065
GATE_ROW = round(FLOOR_ROW - RAIL * SCALE)


@dataclass
class Opening:
    positions: np.ndarray
    crossing_time: float
    cart_start: float

    @property
    def endpoint(self):
        return float(self.positions[-1, 0])


class ReboundCourt:
    """Use the original ball and contact material in a court with just one plate."""

    def __init__(self, material):
        original = ElementTree.parse(MODEL_PATH).getroot()
        scene = ElementTree.Element('mujoco')
        for field in ('compiler', 'option', 'default'):
            scene.append(original.find(field))
        world = ElementTree.SubElement(scene, 'worldbody')
        plate = original.find("./worldbody/geom[@name='table']")
        plate.set('pos', '0 0 -0.025')
        plate.set('size', '2.4 1 0.025')
        plate.set('solref', f'-100000 {-DAMPING[material]}')
        world.append(plate)
        world.append(original.find("./worldbody/body[@name='ball']"))
        self.model = mujoco.MjModel.from_xml_string(ElementTree.tostring(scene, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.backdrop = np.full((HEIGHT, WIDTH, 3), (20, 32, 49), dtype=np.uint8)
        self.backdrop[FLOOR_ROW:FLOOR_ROW + 3] = (89, 109, 131)
        self.backdrop[FLOOR_ROW + 3:] = (35, 54, 69)
        self.backdrop[9:FLOOR_ROW:8, 8:-8:9] = (32, 45, 65)
        self.backdrop[GATE_ROW:GATE_ROW + 1] = (50, 70, 87)
        self.shaded = self.backdrop.copy()
        self.shaded[3:FLOOR_ROW, 5:-5] = (48, 63, 78)

    def opening(self, seed):
        """Roll out the ball independently of player motion until the scoring gate."""
        rng = np.random.default_rng(seed)
        direction = rng.choice((-1., 1.))
        x = -direction * rng.uniform(.54, .79)
        z = rng.uniform(.63, .89)
        vx = direction * rng.uniform(1.85, 2.90)
        vz = -rng.uniform(1.75, 2.55)
        spin = direction * rng.uniform(65., 115.)
        cart = rng.uniform(-.60, .60)
        data = self.data
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = x, 0., z, 1., 0., 0., 0.
        data.qvel[:] = vx, 0., vz, 0., spin, 0.
        mujoco.mj_forward(self.model, data)
        positions = [(x, z)]
        ascending = False
        steps = round(DT / self.model.opt.timestep)
        for step in range(1, 800):
            vx, _, vz, _, spin, _ = data.qvel
            force = data.xfrc_applied[1]
            drag = -BALL_MASS * AIR_DRAG * math.hypot(vx, vz)
            magnus = BALL_MASS * MAGNUS_COEFFICIENT * spin
            force[:] = (drag * vx + magnus * vz, 0., drag * vz - magnus * vx,
                        0., -BALL_INERTIA * SPIN_DECAY * spin, 0.)
            prev_x, prev_z = data.qpos[[0, 2]]
            mujoco.mj_step(self.model, data)
            x, z = data.qpos[[0, 2]]
            ascending |= z >= RAIL and data.qvel[2] > 0.
            if ascending and z < RAIL:
                fraction = (prev_z - RAIL) / (prev_z - z)
                end = prev_x + fraction * (x - prev_x)
                time = float((step - 1 + fraction) * self.model.opt.timestep)
                return Opening(np.array([*positions, (end, RAIL)]), time, float(cart))
            if step % steps == 0:
                positions.append((x, z))
            if not ascending and z < RAIL and data.qvel[2] < 0. and len(positions) > 40:
                break
        raise ValueError(f'opening {seed} did not bank above the rail')

    def frame(self, ball, cart):
        """Render an orthographic RGB image. The closed screen removes the ball."""
        rgb = (self.shaded if ball is None else self.backdrop).copy()
        center = round(CENTER + SCALE * cart)
        rgb[GATE_ROW + 3:GATE_ROW + 5, center - 5:center + 6] = (68, 208, 243)
        rgb[GATE_ROW:GATE_ROW + 3, [center - 5, center + 5]] = (68, 208, 243)
        if ball is not None:
            x, z = ball
            col, row = CENTER + SCALE * x, FLOOR_ROW - SCALE * z
            x, y = round(col), round(row)
            left, right = max(0, x - 2), min(WIDTH, x + 3)
            top, bottom = max(0, y - 2), min(HEIGHT, y + 3)
            if right <= left or bottom <= top:
                return rgb
            cols, rows = np.arange(left, right), np.arange(top, bottom)
            kernel = np.maximum(0., 2.25 - np.hypot(cols - col, (rows - row)[:, None]))
            strength = np.minimum(1., kernel)
            brush = rgb[top:bottom, left:right]
            visible = strength > 0.
            brush[visible] = np.column_stack((160. + 95. * strength[visible],
                                               100. + 80. * strength[visible],
                                               np.full(np.count_nonzero(visible), 62))).astype(np.uint8)
        return rgb


def steer(cart, command, dt):
    return min(LIMIT, max(-LIMIT, cart + command * RATE * dt))


def pixels(rgb):
    """Convert only the colored pixels and the published camera scale to metres."""
    red = rgb[:, :, 0]
    rows, cols = np.nonzero(red > 160)
    ball = None
    if len(rows):
        weights = red[rows, cols].astype(float) - 160.
        col = weights @ cols / weights.sum()
        row = weights @ rows / weights.sum()
        ball = np.array([(col - CENTER) / SCALE, (FLOOR_ROW - row) / SCALE])
    rack = np.flatnonzero(red[GATE_ROW + 4] == 68)
    cart = (rack.mean() - CENTER) / SCALE
    return ball, cart
