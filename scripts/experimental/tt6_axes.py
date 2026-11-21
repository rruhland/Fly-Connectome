"""Compile offset-arm fixtures with different physical joint axes."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_arena import mujoco
from .tt4_hall import hall_model
from .tt6_body_arena import TT6Arena, TT6Wiring, body_xml


AXIS_LAYOUTS = {
    'standard': {},
    'held-a': {'carriage': ('y', 16), 'lift': ('x', -13), 'sweep': ('x', 14), 'depth': ('z', -18)},
    'held-b': {'carriage': ('y', -14), 'lift': ('x', 19), 'sweep': ('y', -17), 'depth': ('z', 15)},
    'practice-a': {'carriage': ('y', 12), 'lift': ('x', 11), 'sweep': ('x', -12), 'depth': ('z', -12)},
    'practice-b': {'carriage': ('y', -12), 'lift': ('x', -10), 'sweep': ('y', 12), 'depth': ('z', 11)},
}


class TiltedAxisArena(TT6Arena):
    """Use the five-channel offset receiver with altered local joint directions."""

    def __init__(self, *args, layout='standard', **kwargs):
        if layout not in AXIS_LAYOUTS:
            raise ValueError(f'unknown axis layout {layout!r}')
        self.layout = layout
        super().__init__(*args, body='offset', **kwargs)

    def _load_model(self):
        xml = body_xml('offset')
        for name, (direction, degrees) in AXIS_LAYOUTS[self.layout].items():
            joint = xml.find(f".//joint[@name='learner_{name}']")
            original = np.fromstring(joint.attrib['axis'], sep=' ')
            altered = Rotation.from_euler(direction, degrees, degrees=True).apply(original)
            axis = ' '.join(map(str, altered))
            joint.set('axis', axis)
            if name == 'sweep':
                # Both hinges share the same local zero reference for the passive coupling.
                xml.find(".//joint[@name='learner_yaw_link']").set('axis', axis)
        if self.venue == 'hall':
            return hall_model(xml)
        return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))


def pulse_receipt(layout, seed):
    """Measure divergent tip motion from identical public inputs in both models."""
    arenas = [TiltedAxisArena(seed, layout=name) for name in ('standard', layout)]
    try:
        baseline, altered = arenas
        physical = ('carriage', 'lift', 'sweep', 'depth')
        wiring = [TT6Wiring(arena, seed) for arena in arenas]
        rows = []
        for index, name in enumerate(physical):
            axes = [arena.model.joint(f'learner_{name}').axis.copy() for arena in arenas]
            angle = np.rad2deg(np.arccos(np.clip(axes[0] @ axes[1], -1., 1.)))
            command = np.zeros(baseline.action_size)
            command[np.flatnonzero(wiring[0].order == index)[0]] = .80
            displacement = []
            for interface in wiring:
                interface.reset(seed=seed, render=False)
                home = interface.arena.paddle.copy()
                for _ in range(5):
                    interface.step(command, render=False)
                displacement.append(interface.arena.paddle - home)
            rows.append(dict(axis=name, rotation_degrees=float(angle), public_command=command.tolist(),
                             model_axes=[axis.tolist() for axis in axes], steps=5,
                             standard_tip_delta=displacement[0].tolist(), tilted_tip_delta=displacement[1].tolist(),
                             tip_difference_meters=float(np.linalg.norm(displacement[0] - displacement[1]))))
        return dict(seed=seed, layout=layout, action_size=altered.action_size, axes=rows)
    finally:
        for arena in arenas:
            arena.close()
