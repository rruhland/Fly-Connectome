"""Retry the stock stereo planner after a bounded approach toward an incoming ball."""

import argparse
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt29_vision_twin
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_arm_clearance import ArmClearance
from .tt29_rally_physics import NOMINAL_NO_LOAD, SOLVER_SUBSTEPS, mujoco
from .tt29_vision_replay import clearance_final
from .tt31_curriculum import ERRORS, MOVEMENTS, play_card
from .tt31_curriculum_opponent import replay_arena


CARD_INDICES = (("fixed", 0), ("periodic", 0), ("periodic", 1))
JOG_TICKS = 5
MAX_JOGS = 4
RETRY_MARGIN_S = .145


class ApproachCheck:
    """Bound the partial command through the same nominal torque-speed envelope."""

    def __init__(self, twin):
        self.twin = twin
        self.model = copy(twin.model)
        self.data = mujoco.MjData(self.model)
        self.guard = ArmClearance(self.model, tolerance=0.)
        self.axes = twin._arm_ids[1][2]
        self.rated = self.model.actuator_forcerange[self.axes].copy()
        self.ball = twin._geom_ids["ball"]

    def clear(self, action):
        a, m, d = self.twin, self.model, self.data
        mujoco.mj_copyData(d, m, a.data)
        increment = action * a._joint_increments[self.axes]
        horizon = .01
        nearest = horizon
        for _ in range(JOG_TICKS * a.frameskip * SOLVER_SUBSTEPS):
            d.ctrl[self.axes] = np.clip(d.ctrl[self.axes] + increment,
                                       a._joint_low[self.axes], a._joint_high[self.axes])
            normalized = d.qvel[a._near_dofs] / NOMINAL_NO_LOAD
            m.actuator_forcerange[self.axes] = self.rated * np.column_stack((
                np.clip(1 + normalized, 0., 1.), np.clip(1 - normalized, 0., 1.),
            ))
            mujoco.mj_step(m, d)
            if np.max(np.abs(d.qvel[a._near_dofs])) > NOMINAL_NO_LOAD:
                return None
            if any(c.efc_address >= 0 and self.ball in c.geom and
                   a._racket_ids[1] in c.geom for c in d.contact):
                return None
            distance = min((gap for _, _, gap in self.guard.distances(d, horizon=horizon)), default=horizon)
            nearest = min(nearest, distance)
            if nearest < .001:
                return None
        return nearest


class ApproachPlanner(tt29_vision_twin.OpticalPlanner):
    """Let a valid stock tape interrupt the approach at every new plan attempt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remaining = None
        self.count = 0
        self.check = None
        self.samples = []

    def approach(self, step, command):
        a = self.twin
        path = self.nominal.flight.path(a.ball, a.velocity, a.spin, 1, a._received_bounce)
        candidates = path[(path[:, 2] > -1.43) & (path[:, 2] < -1.17) &
                          (path[:, 3] > .88) & (path[:, 3] < 1.31)]
        if not len(candidates):
            return None
        hit = candidates[np.argmin(abs(candidates[:, 2] + 1.29))]
        if hit[0] < RETRY_MARGIN_S:
            return None
        self.count += 1
        ik = self.nominal.ik
        start = command[:5]
        base, normal = ik.pose(start)
        desired = hit[1:4] - .026 * normal
        goal, (endpoint, _, *_) = ik.fit(desired, normal, start)
        delta = goal - start
        per_frame = a.control_dt * self.nominal.rates[:5]
        rate = delta / (JOG_TICKS * per_frame)
        rate *= min(1., a.rate_scale / max(np.max(np.abs(rate)), 1e-9))
        action = np.r_[rate, 0.]
        next_center, _ = ik.pose(start + rate * JOG_TICKS * per_frame)
        before = np.linalg.norm(base - desired)
        after = np.linalg.norm(next_center - desired)
        if after > before - .005:
            return None
        if self.check is None:
            self.check = ApproachCheck(a)
        gap = self.check.clear(action)
        self.samples.append(dict(step=step, eta_s=float(hit[0]), predicted=hit[1:4].tolist(),
                                 initial_error_m=float(before), jog_error_m=float(after),
                                 reachable_pose_error_m=float(np.linalg.norm(endpoint - desired)),
                                 nominal_clearance_m=gap, action=action.tolist()))
        return None if gap is None else action

    def command(self, step, flight, angles, rate, target):
        was_incoming = self.incoming
        last_try = self.previous_try
        stock = super().command(step, flight, angles, rate, target)
        if not self.incoming or not was_incoming:
            self.count = 0
            self.remaining = None
        if self.tape is not None or np.any(stock):
            self.remaining = None
            return stock
        if self.incoming and last_try != self.previous_try and self.count < MAX_JOGS:
            action = self.approach(step, target)
            self.remaining = None if action is None else (step + JOG_TICKS, action)
        if self.remaining is not None and self.incoming and step < self.remaining[0]:
            return self.remaining[1]
        return stock


def verify(row, entry):
    arena = replay_arena([entry])
    cad = replay_arena([entry], detail="cad")
    guard = ArmClearance(cad.model, tolerance=0.)
    shadow = mujoco.MjData(cad.model)
    try:
        arena.launch(row["shot"])
        arena.partner.reset(seed=entry["seed"])
        arena.sample_clearance(True)
        arena._cad_reader = guard, shadow
        for action in row["actions"]:
            arena.exchange(action, np.zeros(5))
        event_hash = digest_json(arena.events)
        if event_hash != digest_json(row["events"]) or arena.point_placements() != row["placements"]:
            raise ValueError("Action replay changed the physical course")
        result = clearance_final(arena, guard, shadow)
        if result["samples"] != round(arena.data.time / arena.model.opt.timestep) + 1:
            raise ValueError("CAD replay omitted a physics state")
        return dict(events_sha256=event_hash, clearance=result, motor=arena.motor_audit())
    finally:
        arena.close()
        cad.close()


def evaluate(feeds, plan, output):
    deck = _read(plan)
    if _hash(feeds) != deck["feed_sha256"]:
        raise ValueError("Physics cards and the measured launch file differ")
    cards = {(c["mode"], c["index"]): c for c in deck["staged"]}
    chosen = [cards[key] for key in CARD_INDICES]
    result = dict(schema="tt31-approach-v1", feeds_sha256=_hash(feeds), plan_sha256=_hash(plan),
                  movement_sha256=_hash(MOVEMENTS), optical_sha256=_hash(ERRORS), opening_side=1,
                  cards=chosen, return_target=4, rows=[])
    policies = []

    class RecordApproach(ApproachPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            policies.append(self)

    with TemporaryDirectory(prefix="tt31-approach-") as scratch:
        temp = Path(scratch) / "point.json.gz"
        for card in chosen:
            baseline, _ = play_card(feeds, card, MOVEMENTS, ERRORS, temp, practice=False, cap=4)
            with patch.object(tt29_vision_twin, "OpticalPlanner", RecordApproach):
                approach, _ = play_card(feeds, card, MOVEMENTS, ERRORS, temp, practice=False, cap=4)
            approach["jogs"] = policies.pop().samples
            for row in (baseline, approach):
                row["replay"] = verify(row, card)
            result["rows"].append(dict(card=card, original=baseline, approach=approach))
            _write(output, result)
            print(card["key"],
                  [(row["score"]["returns"], row["score"]["learner"],
                    [(r["observed_lane"], r["near_landed"]) for r in row["placements"]["delivered"]])
                   for row in (baseline, approach)],
                  "jogs", len(approach["jogs"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.feeds, args.plan, args.output)
