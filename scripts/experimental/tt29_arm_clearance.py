"""Check the SO-101 and its held paddle for nonadjacent CAD intersections."""

from itertools import combinations

import mujoco
import numpy as np

from .tt19_so101 import BLADE, CAMERA_BODY, GRIP_BODY, PADDLE_BODY


_LINKS = (
    ('learner_base',),
    ('learner_shoulder',),
    ('learner_upper_arm',),
    ('learner_elbow',),
    ('learner_wrist',),
    (GRIP_BODY, 'learner_moving_jaw_so101_v1', 'learner_wrist_camera_mount', CAMERA_BODY),
)


class ArmClearance:
    """Screen simulated states using convex MuJoCo distances on the actual scene."""

    def __init__(self, model, *, tolerance=.003):
        self.model = model
        self.tolerance = tolerance
        names = {model.body(i).name: i for i in range(model.nbody)}
        shell = {}
        for i, body in enumerate(model.geom_bodyid):
            # Solo bouncing duplicates link shells for ball-only contact.
            if model.geom_group[i] != 3:
                shell.setdefault(body, []).append(i)
        links = [tuple(g for name in group for g in shell.get(names[name], ())) for group in _LINKS]
        paddle = shell[names[PADDLE_BODY]]
        blade = model.geom(BLADE).id
        surfaces = [i for i in shell[0] if model.geom(i).name in ('table', 'table_right', 'net')]
        if 'tt20_table_end_clamp' in names:
            surfaces.extend(shell[names['tt20_table_end_clamp']])
        pairs = []
        for left, right in combinations(range(len(links)), 2):
            if right - left > 1:
                pairs.extend((a, b) for a in links[left] for b in links[right])
        pairs.extend((a, b) for group in links[:-2] for a in group for b in paddle)
        pairs.extend((a, blade) for a in links[-2])
        pairs.extend((a, b) for group in (*links[1:], paddle) for a in group for b in surfaces)
        self.first, self.second = np.asarray(pairs, dtype=int).T
        self.radii = model.geom_rbound[self.first] + model.geom_rbound[self.second]
        self.labels = [self._label(i) for i in range(model.ngeom)]

    def _label(self, index):
        geom = self.model.geom(index)
        body = self.model.body(geom.bodyid[0]).name
        if geom.name:
            return geom.name
        if geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH:
            return f'{body}/{self.model.mesh(geom.dataid[0]).name}'
        return body

    def distances(self, data, *, horizon=.01, update=True):
        """Yield named checked pairs within the distance horizon in metres."""
        if update:
            mujoco.mj_kinematics(self.model, data)
        delta = data.geom_xpos[self.first] - data.geom_xpos[self.second]
        near = np.einsum('ij,ij->i', delta, delta) <= (self.radii + horizon) ** 2
        for first, second in zip(self.first[near], self.second[near], strict=True):
            distance = mujoco.mj_geomDistance(self.model, data, first, second, horizon, None)
            if distance < horizon:
                yield self.labels[first], self.labels[second], distance

    def violations(self, data, *, update=True):
        """Return sampled penetrations larger than the CAD comparison tolerance."""
        return [pair for pair in self.distances(data, update=update) if pair[2] < -self.tolerance]
