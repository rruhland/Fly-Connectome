"""Opt-in single-threaded native kernels for the existing physical practice partner."""

import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from mujoco import rollout
import numpy as np

from .tt4_arena import BALL_INERTIA, BALL_MASS, BALL_RADIUS, SPIN_DECAY, TABLE_HEIGHT, mujoco
from .tt29_partner import IsolatedImpact, PartnerArena, PracticePartner
from .tt29_rally_physics import BallFlight, DRAG, LIFT, TABLE_TOP


@lru_cache(maxsize=1)
def kernels():
    package = Path(mujoco.__file__).parent
    library = next(package.glob("libmujoco.so.*"))
    workspace = TemporaryDirectory(prefix="tt29-partner-native-")
    target = Path(workspace.name) / "flight.so"
    source = Path(__file__).with_suffix(".cpp")
    subprocess.run(["c++", "-std=c++17", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                    "-I", str(package / "include"), str(source), str(library),
                    f"-Wl,-rpath,{package}", "-o", str(target)], check=True)
    dll = ctypes.CDLL(str(target))
    pointer, integer, number = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    dll.normal_quaternions.argtypes, dll.normal_quaternions.restype = (pointer, pointer, integer), None
    dll.departure_spins.argtypes, dll.departure_spins.restype = (pointer, pointer, integer, integer), None
    dll.flight_window.argtypes = (pointer, pointer, pointer, pointer, pointer, pointer, integer, integer,
        integer, integer, ctypes.c_bool, ctypes.c_bool, number, number, number, number, number, number,
        number, number, pointer)
    dll.flight_window.restype = integer
    return dll, workspace


class NativeBallFlight(BallFlight):
    """Use the same isolated table and time step with compiled state collection."""

    def __init__(self, arena):
        super().__init__(arena)
        self.tables = np.array(sorted(self.surface), dtype=np.intc)
        self.net = self.model.geom("net").id
        self.kernel = kernels()[0].flight_window

    def _window(self, position, velocity, spin, side, bounced, horizon, *, min_y=None):
        p, v, w = (np.ascontiguousarray(values, dtype=float) for values in (position, velocity, spin))
        steps = round(horizon / self.dt)
        outgoing = min_y is not None
        samples = np.empty((steps, 7 if outgoing else 10))
        count = self.kernel(self.model._address, self.data._address, p.ctypes.data, v.ctypes.data, w.ctypes.data,
                            self.tables.ctypes.data, len(self.tables), self.net, steps, side, bounced, outgoing,
                            min_y if outgoing else 0., TABLE_HEIGHT, TABLE_TOP, BALL_MASS, BALL_INERTIA,
                            SPIN_DECAY, DRAG if outgoing else self.drag, LIFT if outgoing else self.lift,
                            samples.ctypes.data)
        return samples[:count]

    def path(self, position, velocity, spin, side, bounced=False, horizon=.75):
        return self._window(position, velocity, spin, side, bounced, horizon)

    def outgoing(self, position, velocity, spin, min_y):
        return self._window(position, velocity, spin, -1, False, .9, min_y=min_y)


class NativeImpact(IsolatedImpact):
    """Batch conversions while retaining native sphere-cylinder collisions."""

    def __init__(self, arena):
        super().__init__(arena)
        self.kernel = kernels()[0]

    def solve(self, incoming, normal, velocity):
        count = len(incoming)
        initial = np.zeros((count, 1 + self.model.nq + self.model.nv))
        initial[:, 1:4] = incoming[:, 1:4]
        initial[:, 4] = 1.
        approach = np.sum((velocity - incoming[:, 4:7]) * normal, axis=1)
        delta = np.maximum(.001, approach * self.dt) + .001
        centers = incoming[:, 1:4] - (BALL_RADIUS + self.half_thickness + delta[:, None]) * normal
        normals = np.ascontiguousarray(normal)
        orientations = np.empty((count, 4))
        self.kernel.normal_quaternions(normals.ctypes.data, orientations.ctypes.data, count)
        initial[:, 8:11] = centers
        initial[:, 11:15] = orientations
        initial[:, 15:21] = incoming[:, 4:10]
        initial[:, 21:24] = velocity
        with rollout.Rollout() as engine:
            states, _ = engine.rollout(self.model, self.data, initial, nstep=9)
        leave = np.ascontiguousarray(states[:, -1])
        spin = np.empty((count, 3))
        self.kernel.departure_spins(leave.ctypes.data, spin.ctypes.data, count, leave.shape[1])
        return leave[:, 1:4], leave[:, 15:18], spin, centers, orientations


class FastPracticePartner(PracticePartner):
    def __init__(self, arena, **settings):
        super().__init__(arena, **settings)
        self.flight = NativeBallFlight(arena)
        self.impact = NativeImpact(arena)

    def _outgoing(self, position, velocity, spin, min_y):
        return self.flight.outgoing(position, velocity, spin, min_y)


class FastPartnerArena(PartnerArena):
    """Explicit alternative to the unchanged default partner arena."""

    def __init__(self, **settings):
        super().__init__(**settings)
        default = self.partner
        self.partner = FastPracticePartner(self, targets=default.targets, guard_table=default.guard_table,
                                           verify_approach=default.verify_approach)
