"""Shoot toward a physically reachable full paddle pose on the observed flight."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt39_terminal_fork as forks
from . import tt39_terminal_servo as servo
from .tt4_transfer_trial import _hash, _read, _write


def parameterization(length):
    t = np.arange(1, length + 1)/length
    # Early displacement can wind up a move without fixing its terminal command.
    return np.column_stack((np.minimum(t * 1.50, 1.), 1.35*(t**3 - .25*t), 4.*t*(1.-t)))


def references(anchor, basis, parameters):
    out = np.einsum("ta,kaj->ktj", basis, np.asarray(parameters).reshape(-1, 3, 5))
    return anchor + np.pad(out, ((0, 0), (0, 0), (0, 1)))


def shoot(body, anchor, goal):
    basis = parameterization(len(anchor))
    theta = np.zeros(15)
    eps = .049
    perturbed = np.eye(15)*eps
    initial = body.endpoints(anchor[None])[0]
    if not np.isfinite(initial).all():
        return None
    present = initial
    score = np.linalg.norm(servo.endpoint_vector(present[None], goal))
    calls = 1
    for _ in range(4):
        plus = body.endpoints(references(anchor, basis, np.vstack((theta+perturbed, theta-perturbed))))
        calls += len(plus)
        res = servo.endpoint_vector(plus, goal)
        jacobian = ((res[:15] - res[15:]) / (2*eps)).T
        usable = np.isfinite(jacobian).all(axis=0)
        if not usable.any():
            break
        jacobian[:, ~usable] = 0.
        err = servo.endpoint_vector(present[None], goal)[0]
        j = jacobian
        step = np.clip(-j.T @ np.linalg.solve(j @ j.T + .011**2*np.eye(7), err), -.31, .31)
        attempts = np.clip(theta + np.array([.60, 1., 1.55])[:, None]*step, -.75, .75)
        end = body.endpoints(references(anchor, basis, attempts))
        costs = np.linalg.norm(servo.endpoint_vector(end, goal), axis=1)
        costs[~np.isfinite(costs)] = np.inf
        calls += len(end)
        k = int(np.argmin(costs))
        if costs[k] >= score-.0003:
            break
        theta, present, score = attempts[k], end[k], costs[k]
        if score < .010:
            break
    if score > .075:
        return None
    return references(anchor, basis, theta[None])[0], dict(
        before=initial.tolist(), after=present.tolist(), goal=goal.tolist(), trajectories=calls,
        weighted_before=float(np.linalg.norm(servo.endpoint_vector(initial[None], goal))),
        weighted_after=float(score),
    )


class ReachableServo(servo.TerminalServo):
    def endpoints_for(self, q):
        twin = self.engine.twin
        path = self.engine.nominal.flight.path(twin.ball, twin.velocity, twin.spin, 1,
                                                bounced=self.engine.bounced, horizon=.37)
        if not len(path):
            return []
        times, p = path[:, 0], path[:, 1:4]
        use = (times >= .083) & (times < .335) & (p[:,1] > -1.74) & (p[:,1] < -1.03)
        use &= (p[:,2] > .785) & (p[:,2] < 1.075) & (abs(p[:,0]) < .15)
        candidates = np.flatnonzero(use)
        if not len(candidates):
            return []
        rows = []
        for k in np.unique(np.rint(np.linspace(candidates[0], candidates[-1], 6)).astype(int)):
            frames = int(round(times[k] / mpc.DT))
            if not use[k] or not 5 <= frames < mpc.HORIZON:
                continue
            j = int(np.argmin(abs(times - frames * mpc.DT)))
            ball, incoming = path[j,1:4], path[j,4:7]
            for pace in (3.45, 4.45):
                goal = servo.desired_pose(ball, incoming, pace)
                n = goal[3:6]
                tangent = np.array([0., 0., 1.]) - n*n[2]
                tangent /= max(.01,np.linalg.norm(tangent))
                for shift in (0., .035):
                    center = goal[:3] + shift*tangent
                    seed = q[:5].copy()
                    if shift:
                        seed[2:4] += np.array([.28,-.39])
                    angles, fitted = self.geometry.fit(center, n, seed)
                    p_fit, n_fit = fitted[0], fitted[1]
                    delta = ball-p_fit
                    signed = float(delta @ n_fit)
                    radius = np.linalg.norm(delta-signed*n_fit)
                    angle = np.linalg.norm(n_fit-n)
                    # Accept a full reachable pose only if the planned ball lies on its face neighborhood.
                    if not (.001 < signed < .055 and radius < .060 and angle < .39):
                        continue
                    terminal = np.r_[p_fit, n_fit, np.clip(goal[6]-.20,.20,1.42)]
                    rank = 1.2*radius + .11*angle + 1.5*abs(signed-.025)
                    rows.append((rank, frames, terminal, angles, fitted))
        out = []
        for row in sorted(rows,key=lambda r:r[0]):
            if not any(abs(row[1]-old[1]) < 2 for old in out):
                out.append(row)
            if len(out) >= 4:
                break
        return out

    def schedule(self, q, reference, frames, goal, angles, nominal):
        initial = self.engine.body_state.qvel[self.engine.dofs][:5]
        _, _, jac, angle = nominal
        j = np.vstack((jac, .12*angle))
        v = np.r_[goal[3:6]*goal[6], np.zeros(3)]
        drive = np.clip(j.T @ np.linalg.solve(j @ j.T + .016**2*np.eye(6),v),-6.5,6.5)
        t = (1+np.arange(frames))*mpc.DT
        u = t/t[-1]
        aim = (2*u**3-3*u**2+1)[:,None]*q[:5] + (-2*u**3+3*u**2)[:,None]*angles
        aim += t[-1] * ((u**3-2*u*u+u)[:,None]*initial + (u**3-u*u)[:,None]*drive)
        aim += .048 * np.gradient(aim,mpc.DT,axis=0)
        return np.c_[aim, np.full(frames,reference[5])], drive


def run(args):
    with patch.object(servo,"solve_terminal",shoot),patch.object(forks,"TerminalServo",ReachableServo):
        forks.course(args.bank,args.optics,args.output,args.limit)
    d = _read(args.output)
    d["reachable_terminal_sha256"] = _hash(Path(__file__))
    _write(args.output,d)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--limit",type=int,default=len(forks.SOURCES))
    p.add_argument("--bank",type=Path,default=mpc.BANK)
    p.add_argument("--optics",type=Path,default=Path("artifacts/tt37/rebound.npz"))
    run(p.parse_args())
