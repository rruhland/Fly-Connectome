"""Render the corrected SO-101 fixture and its native wrist feeds in MuJoCo."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .tt20_fixture import TABLE_EDGE
from .tt24_bounce import BounceArena
from .tt25_reference_receiver import ReferenceReceiver


SIZE = (640, 480)
SEED = 42


def _save(path, rgb):
    Image.fromarray(rgb).save(path)


def _outside(model, data, output, views):
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    with mujoco.Renderer(model, width=SIZE[0], height=SIZE[1]) as renderer:
        for name, center, distance, azimuth, elevation in views:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = center
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            _save(output / f'{name}.png', renderer.render())


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    dimensions = dict(width=SIZE[0], height=SIZE[1], detail='cad', fixture=TABLE_EDGE)
    regulation = ReferenceReceiver(view_mode='wrist+fixed', **dimensions)
    try:
        regulation.reset(seed=SEED, server=-1, render=False)
        for _ in range(25):
            regulation.step(np.zeros(regulation.action_size), render=False)
        _save(output / 'fixture-wrist-rgb.png', regulation.render_wrist())
        base = regulation.data.body('learner_base').xpos
        board = regulation.data.geom('tt29_camera_board').xpos
        _outside(regulation.model, regulation.data, output, (
            ('fixture-regulation', base + (0., .14, .15), 1.21, 145, -16),
            ('fixture-clamp', base + (0., -.018, .075), .56, 142, -7),
            ('fixture-wrist-board', board + (0., .037, -.024), .365, -50, -23),
        ))
    finally:
        regulation.close()

    side_on = ReferenceReceiver(paddle_grip='side-on', **dimensions)
    try:
        side_on.reset(seed=SEED, render=False)
        side_on._color_roles()
        grip = side_on.data.body('learner_gripper').xpos
        blade = side_on.data.geom('learner_racket').xpos
        _outside(side_on.model, side_on.data, output, (
            ('fixture-side-on', .45 * grip + .55 * blade, .57, 60, -30),
        ))
    finally:
        side_on.close()

    solo = BounceArena(**dimensions)
    try:
        solo.reset(seed=SEED, render=False)
        for _ in range(4):
            solo.step(np.zeros(solo.action_size), render=False)
        _save(output / 'fixture-solo-wrist-rgb.png', solo.render_wrist())
        _save(output / 'fixture-solo.png', solo.render_external())
    finally:
        solo.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tt29'))
    render(parser.parse_args().output)
