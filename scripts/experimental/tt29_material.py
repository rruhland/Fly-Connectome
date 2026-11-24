"""Test optical material identification and uncertain-contact ranking in native rallies."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_action_memory import Experience, RecallPlanner
from .tt29_conditions import CONDITIONS, apply
from .tt29_endurance import audit
from .tt29_partner import FOREHAND_LANES, PartnerArena, PracticePartner
from .tt29_rally_physics import BallFlight, OUTPUT, RallyArena, TABLE_TOP, measured_launches, mujoco
from .tt29_vision import FlightPractice, OpticalFlight, measurement_blocks
from .tt29_vision_twin import OpticalPlanner, Rig


DAMPING = np.linspace(16., 42., 14)
IMPACT_SPEED = np.linspace(.9, 4.5, 13)
PADDLE_HYPOTHESES = ((18., .61), (34., .46))
STARTS = (7271, 7345, 4527, 8207)


def restitution_grid(twin):
    """Calibrate known candidate parameters on an independent native ball and table."""
    flight = BallFlight(twin)
    m, d = flight.model, flight.data
    response = np.zeros((len(DAMPING), len(IMPACT_SPEED)))
    for i, damping in enumerate(DAMPING):
        m.geom("table").solref[1] = -damping
        for j, speed in enumerate(IMPACT_SPEED):
            mujoco.mj_resetData(m, d)
            d.qpos[:3] = (0., .5, TABLE_TOP + .0001)
            d.qvel[:3] = (0., 3., -speed)
            touching = False
            for _ in range(35):
                mujoco.mj_step(m, d)
                if touching and not d.ncon:
                    response[i, j] = d.qvel[2] / speed
                    break
                touching |= bool(d.ncon)
    return response


class MaterialEvidence:
    """Infer an effective table damping from visual vertical rebound alone."""

    def __init__(self, reference):
        self.reference = reference
        self.samples = deque(maxlen=16)
        self.received = 0
        self.value = 24.
        self.history = []

    def update(self, t, speed, ratio, error):
        self.samples.append((speed, ratio, error))
        self.received += 1
        if len(self.samples) < 3:
            return
        observation = np.array(self.samples)
        expected = np.stack([np.interp(observation[:, 0], IMPACT_SPEED, row) for row in self.reference])
        residual = np.abs(expected - observation[:, 1])
        weight = 1. / np.maximum(.5, observation[:, 2])
        loss = (np.minimum(residual, .10) * weight).sum(axis=1)
        loss += .018 * ((DAMPING - 24.) / 12.)**2
        prior = np.exp(-(loss - loss.min()) / .026)
        self.value = float(np.average(DAMPING, weights=prior))
        self.history.append((t, self.value))


class MaterialVision(OpticalFlight):
    """Reuse the stereo bounce fit, waiting for one additional post-impact frame."""

    def __init__(self, rig, practice, evidence):
        super().__init__(rig, practice, learn=True)
        self.evidence = evidence
        self.recorded = None
        self.delayed = None

    def _split_bounce(self, times, unique, projection, target, damping, smooth_error):
        fit = super()._split_bounce(times, unique, projection, target, damping, smooth_error)
        if fit is not None:
            self.delayed = (fit[4], smooth_error)
        return fit

    def _fit(self, capture):
        super()._fit(capture)
        if self.delayed is None:
            return
        impact, error = self.delayed
        if capture - impact < .035 or self.recorded == impact:
            return
        self.recorded = impact
        if self.latest is None or self.bounce_time != impact or len(self.last_frame_views) < 2:
            return

        samples = list(self.rays)
        times = np.array([s.time for s in samples])
        unique = np.unique(times)
        if np.count_nonzero(unique < impact) < 3 or np.count_nonzero(unique > impact) < 3:
            return
        projection, target = measurement_blocks(samples)
        result = super()._split_bounce(times, unique, projection, target, self.latest.damping, error + .25)
        if result is None:
            return
        _, _, rms, sigma, refined, before, after = result
        if abs(refined - impact) <= .025 and before[2] < -1.25 and rms < 1.2 and sigma < .07:
            ratio = -after[2] / before[2]
            if .74 < ratio < 1.16:
                self.evidence.update(capture, -before[2], ratio, rms)


class MaterialRollout:
    """Re-rank a handful of nominal stroke candidates across paddle materials."""

    def __init__(self, predictor):
        self.base = predictor
        self.ensemble_calls = 0

    def trajectories(self, goals, target, cutoff):
        rates, base = self.base.trajectories(goals, target, cutoff)
        candidates = np.flatnonzero(np.isfinite(base[:, 0]))
        if len(candidates) < 2:
            return rates, base
        candidates = candidates[np.argsort(base[candidates, 0])[:8]]
        blade = self.base.model.geom("learner_racket")
        original = blade.solref[1], blade.friction[0]
        losses = []
        self.ensemble_calls += 1
        try:
            for damping, friction in PADDLE_HYPOTHESES:
                blade.solref[1], blade.friction[0] = -damping, friction
                _, scores = self.base.trajectories(goals[candidates], target, cutoff)
                losses.append(np.where(np.isfinite(scores[:, 0]), scores[:, 0], 7.))
        finally:
            blade.solref[1], blade.friction[0] = original
        output = base.copy()
        output[:, 0] = np.nan
        output[candidates, 0] = .64 * base[candidates, 0] + .36 * np.mean(losses, axis=0)
        return rates, output


class MaterialPlanner(RecallPlanner):
    def __init__(self, twin, memory, evidence, seed):
        super().__init__(twin, memory, seed=seed)
        self.evidence = evidence
        self.predict = MaterialRollout(self.predict)

    def plan(self):
        damping = self.evidence.value
        for name in self.arena.table_panels:
            self.flight.model.geom(name).solref[1] = -damping
            self.predict.base.model.geom(name).solref[1] = -damping
        return super().plan()


def memory_from(checkpoint):
    memory = Experience()
    for row in checkpoint["checkpoint"]:
        memory.add(np.array(row["flight"]), np.array(row["targets"]), row["time_to_contact"])
    return memory


def evaluate(arena, twin, rig, observer, practice, checkpoint, shot, *, adapt, cap):
    memory = memory_from(checkpoint)
    arena.launch(shot)
    twin.reset(seed=shot["id"], server=-1, render=False)
    rig.begin()
    eye = MaterialVision(rig, practice, observer)
    policy = OpticalPlanner(twin, seed=290110)
    planner = MaterialPlanner(twin, memory, observer, 290110) if adapt else RecallPlanner(twin, memory, seed=290110)
    policy.nominal = planner
    actual = arena.joints.copy()
    command = actual.copy()
    low, high = twin._joint_low[policy.actuators], twin._joint_high[policy.actuators]
    actions, before = [], observer.received
    for step in range(arena.max_steps):
        q = arena.joints.copy()
        now = step * arena.control_dt
        eye.observe(now, arena.render_cameras(), q)
        action = policy.command(step, eye.latest, q, (q - actual) / arena.control_dt, command)
        actual = q
        command = np.clip(command + action * planner.rates * arena.control_dt, low, high)
        _, _, done, truncated, _ = arena.exchange(action, np.zeros(5))
        actions.append(action.tolist())
        if done or truncated or arena.legal_returns + arena.rival_returns >= cap + 1:
            break
    result = audit(arena.events)
    return dict(shot=shot, result=arena.point_result(), score=result, actions=actions,
                events=arena.events.copy(), observations=observer.received-before,
                inferred_damping=observer.value, ensemble_calls=planner.predict.ensemble_calls if adapt else 0)


def run(feeds, output, checkpoint, ids, limit):
    output.parent.mkdir(parents=True, exist_ok=True)
    launches = measured_launches(feeds)
    source = _read(checkpoint)
    config = dict(strike_ready="lateral", width=384, height=288, fast_search=True, clearance_guard=True, max_steps=625)
    summary = dict(checkpoint_sha256=_hash(checkpoint), launches_sha256=_hash(feeds),
                   limit=limit, locked_examples=len(source["checkpoint"]), starts=ids, branches={})
    for material in ("materials", "stock"):
        for mode in ("frozen", "adaptive"):
            arena = PartnerArena(**config, targets=FOREHAND_LANES, guard_table=True)
            twin = RallyArena(**config)
            rig = Rig.from_arena(arena)
            if material != "stock":
                apply(arena, CONDITIONS[material])
                arena.partner = PracticePartner(arena, targets=FOREHAND_LANES, guard_table=True)
            arena.streams = ("left", "right")
            practice = FlightPractice()
            observer = MaterialEvidence(restitution_grid(twin))
            name = f"{material}_{mode}"
            rows = summary["branches"][name] = []
            try:
                for sid in ids:
                    row = evaluate(arena, twin, rig, observer, practice, source, launches[sid],
                                   adapt=mode == "adaptive", cap=limit)
                    rows.append(row)
                    print(name, sid, row["score"]["returns"], row["score"]["learner"],
                          row["observations"], round(row["inferred_damping"], 2), flush=True)
                    _write(output, summary)
            finally:
                arena.close()
                twin.close()
            summary.setdefault("optical_evidence", {})[name] = dict(samples=list(observer.samples), history=observer.history)
            _write(output, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT / "camera-action-memory.json.gz")
    parser.add_argument("--output", type=Path, default=OUTPUT / "material-adaptation.json.gz")
    parser.add_argument("--ids", type=int, nargs="+", default=STARTS)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    run(args.feeds, args.output, args.checkpoint, args.ids, args.limit)
