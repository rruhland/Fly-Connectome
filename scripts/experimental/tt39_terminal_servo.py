"""Finite-difference trajectory shooting for achieved optical paddle targets."""

import argparse
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import process_time
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import DRAG, NOMINAL_NO_LOAD

OUT = Path("artifacts/tt39-terminal")


@lru_cache(maxsize=1)
def body_library():
    folder = TemporaryDirectory(prefix="tt39-body-")
    package = Path(mujoco.__file__).parent
    target = Path(folder.name) / "body.so"
    lib = next(package.glob("libmujoco.so.*"))
    subprocess.run([
        "c++", "-std=c++17", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
        "-I", str(package / "include"), str(Path(__file__).with_name("tt39_terminal_body.cpp")),
        str(lib), f"-Wl,-rpath,{package}", "-o", str(target),
    ], check=True)
    function = ctypes.CDLL(str(target)).body_terminals
    p, i, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    function.argtypes = (p, p, p, i, i, i, i, p, p, p, p, p, p, d, d, p)
    function.restype = None
    return function, folder


class GroundedBody:
    """Only a separate body initialized from measured encoders and issued references."""

    def __init__(self, engine):
        assert engine.body_model is not None
        self.engine = engine
        self.model = engine.body_model
        self.data = mujoco.MjData(self.model)
        self.racket = engine.twin._racket_ids[1]
        self.micro = engine.predict.micro
        self.motors = np.ascontiguousarray(engine.actuators, dtype=np.intc)
        self.dofs = np.ascontiguousarray(engine.dofs, dtype=np.intc)
        self.low = engine.twin._joint_low
        self.high = engine.twin._joint_high
        self.step = engine.twin._joint_increments
        self.limits = np.ascontiguousarray(engine.limits)
        self.call, _ = body_library()

    def snapshot(self):
        mujoco.mj_copyData(self.data, self.model, self.engine.body_state)
        twin = self.engine.twin
        i, j = twin._ball_joint, twin._ball_dof
        # The ball's collision groups are disabled in this model. Park the irrelevant free body.
        self.data.qpos[i:i+7] = [0., 0., -2., 1., 0., 0., 0.]
        self.data.qvel[j:j+6] = 0.
        mujoco.mj_forward(self.model, self.data)

    def endpoints(self, references):
        x = np.ascontiguousarray(references, dtype=float)
        if x.ndim != 3 or x.shape[2] != 6 or not 1 <= x.shape[1] <= mpc.HORIZON:
            raise ValueError("Body readout expects a batch of bounded short reference sequences")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite reference")
        n, steps, _ = x.shape
        out = np.empty((n, 9))
        self.call(self.model._address, self.data._address, x.ctypes.data, n, steps, self.micro, self.racket,
                  self.motors.ctypes.data, self.dofs.ctypes.data, self.step.ctypes.data,
                  self.low.ctypes.data, self.high.ctypes.data, self.limits.ctypes.data,
                  NOMINAL_NO_LOAD, .8, out.ctypes.data)
        return out


def desired_pose(p, incoming, speed):
    """A nominal centered outgoing impulse proposes a physical endpoint to test."""
    pace = float(speed)
    dx, dy, terminal_z = -p[0], .71 - p[1], .765
    duration = np.expm1(DRAG * dy) / (DRAG * pace)
    decay = DRAG * pace
    travel = -np.expm1(-decay * duration) / decay
    vertical = (terminal_z - p[2] + (9.81 / decay) * (duration - travel)) / travel
    outgoing = np.array([dx / travel, pace, vertical])
    n = outgoing - incoming
    n /= np.linalg.norm(n)
    normal_speed = np.clip(((outgoing + .76 * incoming) @ n) / 1.76, .12, 1.85)
    return np.r_[p - .026 * n, n, normal_speed]


def endpoint_vector(end, wanted):
    """Position, signed face orientation, and velocity through that face."""
    return np.column_stack((end[:, :3] - wanted[:3], .106 * (end[:, 3:6] - wanted[3:6]),
                            .037 * (end[:, 6:9] @ wanted[3:6] - wanted[6])))


def shot_basis(n):
    """Two smooth temporal basis functions per joint, not learned stroke fragments."""
    t = np.arange(1, n+1) / n
    shift = np.minimum(1., t * 1.55)
    swing = (t**3 - .34*t) * 1.50
    return np.c_[shift, swing]


def edits(anchor, params, basis):
    values = np.asarray(params).reshape(-1, 2, 5)
    addition = np.einsum('da,naj->ndj', basis, values)
    return anchor + np.pad(addition, ((0, 0), (0, 0), (0, 1)))


def solve_terminal(body, anchor, goal, *, rounds=3):
    """Bounded damped Gauss-Newton with forward finite-difference motor rollouts."""
    basis = shot_basis(len(anchor))
    theta = np.zeros(10)
    eps = .032
    jac_trials = np.eye(10) * eps
    n = 0
    initial = None
    current = None
    for _ in range(rounds):
        pts = np.vstack((theta, theta + jac_trials))
        ends = body.endpoints(edits(anchor, pts, basis))
        n += len(pts)
        if not np.isfinite(ends[0]).all():
            return None
        current = ends[0]
        residual = endpoint_vector(ends, goal)
        if initial is None:
            initial = current.copy()
        error = np.linalg.norm(residual[0])
        J = ((residual[1:] - residual[0]) / eps).T
        good = np.isfinite(J).all(axis=0)
        if not good.any():
            break
        J[:, ~good] = 0.
        delta = -np.linalg.solve(J.T @ J + .016**2 * np.eye(10), J.T @ residual[0])
        tries = np.clip(theta + np.array([.65, 1.15])[:, None] * np.clip(delta, -.30, .30), -.62, .62)
        follow = body.endpoints(edits(anchor, tries, basis))
        n += len(tries)
        quality = np.linalg.norm(endpoint_vector(follow, goal), axis=1)
        quality[~np.isfinite(quality)] = np.inf
        win = int(np.argmin(quality))
        if quality[win] > .97*error:
            break
        theta = tries[win]
        current = follow[win]
        if quality[win] < .012:
            break
    if current is None or np.linalg.norm(endpoint_vector(current[None], goal)) > .10:
        return None
    return edits(anchor, theta[None], basis)[0], dict(
        before=initial.tolist(), after=current.tolist(), goal=goal.tolist(), trajectories=n,
        weighted_before=float(np.linalg.norm(endpoint_vector(initial[None], goal))),
        weighted_after=float(np.linalg.norm(endpoint_vector(current[None], goal))),
    )


class TerminalServo(mpc.MPC):
    """Rescue trajectories on low central replies after the stock planner fails."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.body = GroundedBody(self.engine)

    def begin(self):
        super().begin()
        self.last_shoot = -100

    def endpoints_for(self, q):
        twin = self.engine.twin
        path = self.engine.nominal.flight.path(twin.ball, twin.velocity, twin.spin, 1,
                                                bounced=self.engine.bounced, horizon=.36)
        if not len(path):
            return []
        eta = path[:, 0]
        p = path[:, 1:4]
        use = (eta >= .083) & (eta < .34) & (p[:, 1] < -.97) & (p[:, 1] > -1.74)
        use &= (p[:, 2] > .785) & (p[:, 2] < 1.055) & (abs(p[:, 0]) < .145)
        indices = np.flatnonzero(use)
        if not len(indices):
            return []
        out = []
        for k in np.unique(np.round(np.linspace(indices[0], indices[-1], 5)).astype(int)):
            frames = int(round(path[k, 0] / mpc.DT))
            if frames < 5 or frames > mpc.HORIZON - 1 or not use[k]:
                continue
            j = int(np.argmin(abs(eta - frames * mpc.DT)))
            for speed in (3.50, 4.05):
                goal = desired_pose(path[j, 1:4], path[j, 4:7], speed)
                angles, nominal = self.geometry.fit(goal[:3], goal[3:6], q[:5])
                center_error = np.linalg.norm(nominal[0] - goal[:3])
                tilt_error = np.linalg.norm(nominal[1] - goal[3:6])
                if center_error < .090 and tilt_error < .68:
                    out.append((center_error + .08*tilt_error, frames, goal, angles, nominal))
        # Preserve candidate time diversity without a budget proportional to searched angles.
        selected = []
        for target in sorted(out, key=lambda r: r[0]):
            if not any(target[1] == other[1] for other in selected):
                selected.append(target)
            if len(selected) >= 3:
                break
        return selected

    def schedule(self, q, reference, frames, goal, angles, nominal):
        _, _, jac, angle = nominal
        j = np.vstack((jac, .095 * angle))
        v = np.r_[goal[3:6] * goal[6], np.zeros(3)]
        drive = j.T @ np.linalg.solve(j @ j.T + .018**2 * np.eye(6), v)
        drive = np.clip(drive, -6., 6.)
        t = (1 + np.arange(frames)) * mpc.DT
        u = t / t[-1]
        aim = (2*u**3-3*u**2+1)[:,None]*q[:5] + (-2*u**3+3*u**2)[:,None]*angles
        aim += (u**3-u**2)[:,None] * t[-1] * drive
        # A modest unoptimized nominal actuator lookahead seeds the realized-body solve.
        aim += .034 * np.gradient(aim, mpc.DT, axis=0)
        return np.c_[aim, np.full(frames, reference[5])], drive

    def search(self, tick, x, q, reference, base):
        super().search(tick, x, q, reference, base)
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        note = self.notes[-1]
        if note.get("chosen") or tick-self.last_shoot < 5 or not (-1.50 < x[1] < -.53) or abs(x[0]) > .23:
            return
        self.last_shoot = tick
        start = process_time()
        self.body.snapshot()
        goals, proof = [], []
        for _, frames, goal, angles, nominal in self.endpoints_for(q):
            anchor, drive = self.schedule(q, reference, frames, goal, angles, nominal)
            result = solve_terminal(self.body, anchor, goal)
            if result is None:
                continue
            prefix, record = result
            extra = np.arange(1, mpc.HORIZON - frames + 1) * mpc.DT
            follow = np.c_[prefix[-1, :5] + np.minimum(extra, .078)[:, None] * drive[None] * .60,
                           np.full(len(extra), reference[5])]
            goals.append(np.vstack((prefix, follow)))
            proof.append(dict(frames=frames, **record))
        self.diag["shoot_solves"] += len(proof)
        note["shooting"] = proof
        if goals:
            loss, legal, _, native, _ = self.trial(goals, long=True)
            ids = np.flatnonzero(legal)
            self.diag["shoot_legal"] += len(ids)
            note["shoot_legal"] = int(len(ids))
            if len(ids):
                agree, difficulty = self.uncertainty([goals[i] for i in ids], loss[ids])
                fitted = np.flatnonzero(agree >= 1)
                if len(fitted):
                    k = int(fitted[np.argmin(difficulty[fitted]+3.1*(2-agree[fitted]))])
                    pick = int(ids[k])
                    self.best = goals[pick]
                    self.tick = tick
                    self.lease = mpc.LEASE
                    self.diag["shoot_issued"] += 1
                    note.update(chosen="terminal_body_shooting", robust=int(agree[k]),
                                terminal_index=pick, terminal_contact_s=float(native[pick, 1]),
                                terminal_center_radius_m=float(native[pick, 12]))
        self.diag["shoot_cpu_s"] += process_time() - start


def run(args):
    base = mpc.MPC
    use = TerminalServo if args.mode == "terminal" else base
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with patch.object(mpc, "MPC", use):
        mpc.run(args.feeds, args.bank, args.plan, args.deck, args.learning, args.output,
                optic=args.optics, cap=args.cap, limit=args.limit, variant="elastic", budget=args.budget,
                progress=args.progress)
    d = _read(args.output)
    d["terminal_servo"] = args.mode
    d["terminal_code_sha256"] = _hash(Path(__file__))
    d["terminal_native_sha256"] = _hash(Path(__file__).with_name("tt39_terminal_body.cpp"))
    _write(args.output, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("terminal", "stock"))
    p.add_argument("deck", choices=("development", "held_openings", "held_replies", "guarded_challenge"))
    p.add_argument("output", type=Path)
    p.add_argument("--bank", type=Path, default=mpc.BANK)
    p.add_argument("--plan", type=Path, default=mpc.PLAN)
    p.add_argument("--optics", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--learning", choices=("live", "frozen"), default="live")
    p.add_argument("--limit", type=int)
    p.add_argument("--cap", type=int, default=200)
    p.add_argument("--budget", type=int, default=28)
    p.add_argument("--progress", type=int, default=0)
    run(p.parse_args())
