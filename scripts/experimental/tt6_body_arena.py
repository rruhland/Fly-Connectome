"""Alternate physical receiver arms on the same regulation MuJoCo table."""

import xml.etree.ElementTree as ET

import numpy as np

from .tt4_arena import MODEL_PATH, TT4Arena, mujoco
from .tt4_hall import hall_model
from .tt4_transfer_trial import Wiring


BODY_AXES = {
    'gantry': (('lateral', 'depth', 'lift', 'wrist'), (), (2.85, 2.15, 2.55, 5.8)),
    'offset': (('carriage', 'lift', 'sweep', 'depth', 'wrist'), ('yaw',), (2.05, 2.55, 7.1, 2.0, 5.8)),
    'rotary': (('yaw', 'shoulder', 'elbow', 'wrist', 'wrist_turn', 'wrist_roll'),
               ('pitch', 'yaw'), (4.5, 7.0, 8.5, 7.5, 2.7, 3.0)),
}
BODIES = ('stock', *BODY_AXES)


def body_xml(body):
    """Replace only the receiver's mechanism, preserving the ball, table and rival."""
    model = ET.parse(MODEL_PATH).getroot()
    mechanism = ET.parse(MODEL_PATH.with_name(f'tt6_body_{body}.xml')).getroot()
    world = model.find('worldbody')
    old_body = world.find("body[@name='learner_base']")
    index = list(world).index(old_body)
    world.remove(old_body)
    world.insert(index, mechanism.find('body'))
    for section in ('tendon', 'equality', 'actuator'):
        output = model.find(section)
        for item in list(output):
            if item.get('name', item.get('tendon1', '')).startswith('learner_'):
                output.remove(item)
        output[:0] = list(mechanism.findall(f'{section}/*'))
    return model


class TT6Arena(TT4Arena):
    """Public RGB, normalized motor rates and reward for multiple physical bodies."""

    def __init__(self, *args, body='stock', render_mode='mujoco', **kwargs):
        if body not in BODIES:
            raise ValueError(f'unknown receiver body {body!r}')
        self.body = body
        super().__init__(*args, render_mode=render_mode, **kwargs)

    def _load_model(self):
        if self.body == 'stock':
            return super()._load_model()
        model = body_xml(self.body)
        if self.venue == 'hall':
            return hall_model(model)
        return mujoco.MjModel.from_xml_string(ET.tostring(model, encoding='unicode'))

    def _arm_axes(self, side):
        if side == 1 and self.body in BODY_AXES:
            axes, passive, rates = BODY_AXES[self.body]
            return axes, passive, np.array(rates)
        return super()._arm_axes(side)

    def _arm_home(self, side):
        if side == 1 and self.body == 'rotary':
            home = (0., -.61, 1.41, .28, 0., 0.)
            return home, (-home[1] - home[2], -home[0])
        if side == 1 and self.body in BODY_AXES:
            axes, passive, _ = BODY_AXES[self.body]
            return (*np.zeros(len(axes) - 1), .27), np.zeros(len(passive))
        return super()._arm_home(side)


class TT6Wiring(Wiring):
    """Hide the sign and permutation of however many commands the receiver accepts."""

    def __init__(self, arena, seed):
        self.arena = arena
        rng = np.random.default_rng(np.random.SeedSequence([seed, 41708]))
        self.order = rng.permutation(arena.action_size)
        self.signs = rng.choice((-1, 1), arena.action_size)

    def step(self, action, *, render=True):
        physical = np.empty(self.arena.action_size)
        physical[self.order] = self.signs * action
        return self.arena.step(physical, render=render)
