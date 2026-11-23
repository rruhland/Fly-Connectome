"""Search second-return strokes after live replies and hold out opening variants."""

import json

import mujoco
import numpy as np
from scipy.stats import qmc

from .tt4_camera import PerspectiveCamera, triangulate
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt19_view_trial import CAMERAS
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_policy import orange_center
from .tt24_rally_trial import COACHED, PublicBody, STEP_SIZE
from .tt25_pose_demonstrations import incoming, poses, suggest
from .tt25_second_ball import feed
from .tt25_through_contact import center_jacobian
from .tt26_grip_course import Grip, OUT, PRIOR, ReferenceReceiver

SOURCE = OUT / "live-points.json.gz"
OUTPUT = OUT / "match-continuation.json.gz"
DRAW = qmc.Sobol(6, scramble=True, seed=996137).random_base2(5)
SCALE = np.array((.22, .30, .15, .65, .85, .5))


class ReplyVision:
    def __init__(self, width, height):
        self.rig = tuple(PerspectiveCamera(eye, goal, width, height, fovy)
                         for _, eye, goal, fovy in (CAMERAS[name] for name in ("left", "right")))
        self.previous = [None, None]
        self.points = []
        self.reached_near = False
        self.reached_far = False

    def observe(self, frames, tick):
        pixels = [orange_center(frames[k], prev) for k, prev in zip(("left", "right"), self.previous, strict=True)]
        if any(p is None for p in pixels):
            return None
        self.previous = pixels
        pixel_scale = self.rig[0].width / 480
        ray = [(p + .5) * pixel_scale - .5 for p in pixels]
        point = triangulate(self.rig, *ray)
        if point is None:
            return None
        p = np.array(point)
        if not (-2.0 < p[1] < 2.0 and abs(p[0]) < 1.15 and -.04 < p[2] < 1.0):
            return None
        self.reached_far |= self.reached_near and p[1] > 1.05
        self.reached_near |= p[1] < -.65
        self.points.append((tick, p))
        if len(self.points) < 3 or not self.reached_far or not .30 < p[1] < 1.55:
            return None
        (a, first), (b, middle), (_, last) = self.points[-3:]
        if tick - a > 5 or min(first[1] - middle[1], middle[1] - last[1]) < .035:
            return None
        v = 30 * (last - first) / (tick - a)
        return np.r_[p, v]


class Snapshot:
    """Branch physical search from a common reply and preserve referee history."""

    def __init__(self, arena):
        self.state = vars(arena).copy()
        self.state["events"] = arena.events.copy()
        self.data = mujoco.MjData(arena.model)
        mujoco.mj_copyData(self.data, arena.model, arena.data)

    def restore(self, arena):
        vars(arena).update(self.state)
        arena.events = self.state["events"].copy()
        mujoco.mj_copyData(arena.data, arena.model, self.data)


def archived_step(arena, tape, tick, action):
    far = tape[tick] if tick < len(tape) else np.zeros(len(arena._arm_ids[-1][2]))
    arena._aim = arena._command_to = lambda *args: np.asarray(far)
    return arena.step(action, render=False)


def command(action_size, issued, action, tick, switch):
    goal, impact, velocity, lead = action
    phase = np.clip((tick - switch - impact + lead) / 30, -.14, .18)
    target = np.asarray(goal) + phase * np.asarray(velocity)
    u = np.zeros(action_size)
    u[:5] = np.clip((target - issued[:5]) / STEP_SIZE[:5], -1, 1)
    return u


def candidate(arena, state, row, action):
    state.restore(arena)
    switch = arena.steps
    axes = arena._arm_ids[1][2]
    for tick in range(switch, arena.max_steps):
        near = command(arena.action_size, arena.data.ctrl[axes], action, tick, switch)
        _, _, done, cap, _ = archived_step(arena, row["rival_actions"], tick, near)
        if done or cap or arena.legal_returns >= 2:
            break
    impact = next((e for e in arena.events[len(state.state["events"]):]
                   if e["kind"] == "contact" and e["side"] == 1), None)
    out = next((e for e in arena.events[len(state.state["events"]):]
                if e["kind"] == "separation" and e.get("collider") == "learner_racket"), impact)
    rank = -100.
    if out is not None:
        vx, vy, vz = out["velocity"]
        rank = vy + .45 * vz - .45 * abs(vx)
    return dict(second=arena.legal_returns >= 2, contact=impact is not None,
                impact=impact["step"] - switch if impact else None, rank=rank,
                reason=arena.reason, events=digest_json(arena.events))


def practice(arena, case, bank, coeff):
    row = case["results"]["bounce"]
    arena.launch(case["shot"])
    vision = ReplyVision(arena.width, arena.height)
    for tick, near in enumerate(row["actions"]):
        key = vision.observe(arena.render_cameras(), tick)
        if key is not None:
            break
        archived_step(arena, row["rival_actions"], tick, near)
    if key is None:
        return dict(trigger=None, trials=[], selected=None)
    recorded = [e for e in row["events"] if e["step"] <= tick]
    if digest_json(recorded) != digest_json(arena.events):
        raise ValueError("Continuation prefix diverged from the archived point")
    if arena._last_hitter != -1 or arena.legal_returns != 1 or arena.velocity[1] >= 0:
        return dict(trigger=tick, selected=None, trials=[], reason="early optical trigger")
    start = arena.data.ctrl[arena._arm_ids[1][2]][:5].copy()
    if arena._received_bounce:
        return dict(trigger=tick, selected=None, trials=[], reason="late optical trigger")
    ball = feed(dict(ball=arena.ball, velocity=arena.velocity, spin=arena.spin))
    goals = suggest(bank, incoming(ball, coeff), start_angles=start, observation_delay=0.)
    state = Snapshot(arena)
    trials = []
    def test(action):
        stat = candidate(arena, state, row, action)
        trials.append(dict(action=action, **stat))
        return stat
    for goal, _ in goals:
        action = (goal.tolist(), 0, [0.] * 5, 0)
        if test(action)["second"]:
            break
    if not any(r["second"] for r in trials):
        anchors = sorted((r for r in trials if r["contact"]), key=lambda r: -r["rank"])[:4]
        for anchor in anchors:
            goal = np.asarray(anchor["action"][0])
            jac = center_jacobian(arena, goal)
            inverse = jac.T @ np.linalg.inv(jac @ jac.T + .0004 * np.eye(3))
            for u in DRAW:
                cart = [1.4 * (u[0] - .5), .30 + u[1] * 1.6, -.15 + u[2] * 1.1]
                velocity = inverse @ cart
                velocity *= min(1., 5.2 / np.max(np.abs(velocity)))
                target = goal + .10 * np.array((u[3] - .5, u[4] - .5, 0., u[5] - .5, u[4] - .5))
                action = (target.tolist(), anchor["impact"], velocity.tolist(), 3 * u[3])
                if test(action)["second"]:
                    break
            if trials[-1]["second"]:
                break
    selected = next((r for r in trials if r["second"]), None)
    if selected and candidate(arena, state, row, selected["action"])["events"] != selected["events"]:
        raise ValueError("Development stroke diverged on replay")
    return dict(trigger=tick, trials=trials, selected=selected, key=key.tolist())


def play(arena, tape, case, memory):
    row = case["results"]["bounce"]
    arena.launch(case["shot"])
    tape.steps.clear()
    vision = ReplyVision(arena.width, arena.height)
    body = PublicBody(arena)
    action = trigger = donor = None
    actions = []
    for tick in range(arena.max_steps):
        if trigger is None:
            key = vision.observe(arena.render_cameras(), tick)
            if key is not None:
                trigger = tick
                if memory:
                    example = min(memory, key=lambda r: np.linalg.norm((key - np.asarray(r["key"])) / SCALE))
                    action = example["selected"]["action"]
                    donor = example["idx"]
        if action is not None:
            near = command(arena.action_size, body.target, action, tick, trigger)
        else:
            near = row["actions"][tick] if tick < len(row["actions"]) else np.zeros(arena.action_size)
        actions.append(np.asarray(near).tolist())
        _, _, done, cap, _ = arena.step(near, render=False)
        body.issued(np.asarray(near))
        if done or cap:
            break
    return dict(near=arena.legal_returns, far=arena.rival_returns - 1, contacts=arena.contacts,
                trigger=trigger, donor=donor, actions=actions, rival_actions=tape.tape(len(actions)),
                events=arena.events.copy(), outcome=arena.point_result())


def main():
    cases = [c for c in _read(SOURCE)["cases"] if c["results"]["bounce"]["far"] > 0]
    dev = [c for c in cases if c["idx"] < 12]
    held = [c for c in cases if c["idx"] >= 12]
    coeff = json.loads((PRIOR / "recorded-flight.json").read_text())["held_scores"]["match_fit"]["coefficients"]
    arena = ReferenceReceiver()
    live = ReferenceReceiver(stationary_opponent=False)
    arena.max_steps = live.max_steps = 160
    grip, live_grip = Grip(arena.model), Grip(live.model)
    install_rally_opponent(live, target_profile=_read(COACHED)["metadata"]["opening"]["far_profile"])
    tape = FarTape(live)
    training, results = [], []
    try:
        banks = {}
        for angle in sorted({r["angle"] for r in dev}):
            grip.set(angle)
            banks[angle] = poses(arena)
        for case in dev:
            grip.set(case["angle"])
            result = practice(arena, case, banks[case["angle"]], coeff)
            training.append(dict(angle=case["angle"], idx=case["idx"], id=case["shot"]["id"], **result))
            print("Practice", case["angle"], case["idx"], result["trigger"], len(result["trials"]),
                  sum(r["contact"] for r in result["trials"]), bool(result["selected"]), flush=True)
            _write(OUTPUT, dict(source_sha256=_hash(SOURCE), training=training, held=results))
        for case in held:
            grip.set(case["angle"])
            live_grip.set(case["angle"])
            memory = [t for t in training if t["selected"] and t["angle"] == case["angle"]]
            result = play(live, tape, case, memory)
            arena.launch(case["shot"])
            for tick, near in enumerate(result["actions"]):
                archived_step(arena, result["rival_actions"], tick, near)
            if digest_json(arena.events) != digest_json(result["events"]):
                raise ValueError("Held continuation did not physically replay")
            original = case["results"]["bounce"]
            if result["trigger"] is not None:
                cut = result["trigger"]
                prefix = [e for e in result["events"] if e["step"] <= cut]
                archived = [e for e in original["events"] if e["step"] <= cut]
                if digest_json(prefix) != digest_json(archived):
                    raise ValueError("Held optical trigger did not follow the same physical prefix")
            results.append(dict(angle=case["angle"], idx=case["idx"], id=case["shot"]["id"],
                                baseline_near=original["near"], result=result))
            print("Held", case["angle"], case["idx"], result["trigger"], result["near"], flush=True)
        _write(OUTPUT, dict(source_sha256=_hash(SOURCE), training=training, held=results))
    finally:
        arena.close()
        live.close()


if __name__ == "__main__":
    main()
