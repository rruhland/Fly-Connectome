"""Evaluate stereo feedback for the second return on donor-disjoint full-size points."""

import argparse
import json
from collections import deque
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .tt4_arena import TABLE_HEIGHT
from .tt4_camera import triangulate
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_so101 import BLADE, GRIP_BODY, _GRASP_POS
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_policy import orange_center
from .tt24_rally_trial import COACHED, STEP_SIZE, PublicBody
from .tt25_recorded_exchange import step_ball, table_model
from .tt26_grip_course import PRIOR, Grip, ReferenceReceiver
from .tt26_match_continuation import OUTPUT as DEMONSTRATIONS
from .tt26_match_continuation import SCALE, SOURCE, ReplyVision, archived_step, command

OUT = ROOT / "artifacts" / "tt27"
TABLE_ORIGIN = np.array([0., 0., TABLE_HEIGHT])


class StereoReply(ReplyVision):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.seen = deque(maxlen=16)
        self.reply = []
        self.incoming = None
        self.outgoing = False

    def read(self, frames, tick):
        pixels = [orange_center(frames[k], prev) for k, prev in zip(("left", "right"), self.previous, strict=True)]
        if any(point is None for point in pixels):
            return None
        self.previous = pixels
        scale = self.rig[0].width / 480
        point = triangulate(self.rig, *((pixel + .5) * scale - .5 for pixel in pixels))
        if point is None or not (abs(point[0]) < 1.15 and -2.25 < point[1] < 2 and -.04 < point[2] < 1):
            return None
        point = np.asarray(point)
        self.reached_far |= self.reached_near and point[1] > 1.05
        self.reached_near |= point[1] < -.65
        self.follow(tick, point)
        return point

    def follow(self, tick, point):
        self.seen.append((tick, point))
        if self.incoming is None and len(self.seen) >= 3 and self.reached_far and .30 < point[1] < 1.55:
            (a, first), (_, middle), (_, last) = list(self.seen)[-3:]
            if tick - a <= 5 and min(first[1] - middle[1], middle[1] - last[1]) >= .035:
                self.incoming = (tick, np.r_[last, 30 * (last - first) / (tick - a)])
                self.reply = list(self.seen)[-2:]
        elif self.incoming is not None and len(self.seen) >= 3:
            (_, first), (_, middle), (_, last) = list(self.seen)[-3:]
            self.outgoing |= last[1] < -.65 and min(last[1] - middle[1], middle[1] - first[1]) > .035
        if self.incoming is not None:
            self.reply.append((tick, point))


def schedule():
    return [case for case in _read(SOURCE)["cases"] if case["results"]["bounce"]["far"] > 0]


def collect(path):
    arena = ReferenceReceiver()
    arena.max_steps = 160
    grip = Grip(arena.model)
    demos = {(row["angle"], row["idx"]): row for row in _read(DEMONSTRATIONS)["training"]}
    traces = []
    try:
        for case in schedule():
            angle, index = case["angle"], case["idx"]
            if index >= 12:
                continue
            row = case["results"]["bounce"]
            demo = demos[angle, index]
            grip.set(angle)
            arena.launch(case["shot"])
            eye = StereoReply(arena.width, arena.height)
            axes = arena._arm_ids[1][2]
            track = []
            for tick in range(arena.max_steps):
                point = eye.read(arena.render_cameras(), tick)
                if eye.incoming is not None:
                    track.append(dict(
                        tick=tick, image=None if point is None else point.tolist(),
                        ball=arena.ball.tolist(), velocity=arena.velocity.tolist(), spin=arena.spin.tolist(),
                        joints=arena.joints.tolist(), target=arena.data.ctrl[axes].tolist(),
                    ))
                if eye.incoming is not None and demo["selected"]:
                    near = command(arena.action_size, arena.data.ctrl[axes], demo["selected"]["action"], tick,
                                   eye.incoming[0])
                elif tick < len(row["actions"]):
                    near = row["actions"][tick]
                else:
                    near = np.zeros(arena.action_size)
                _, _, done, cap, _ = archived_step(arena, row["rival_actions"], tick, near)
                if done or cap or arena.legal_returns >= 2:
                    break
            if demo["selected"] and digest_json(arena.events) != demo["selected"]["events"]:
                raise ValueError("Demonstration did not replay")
            if eye.incoming is None or eye.incoming[0] != demo["trigger"]:
                raise ValueError("Inherited visual handoff did not replay")
            traces.append(dict(angle=angle, idx=index, id=case["shot"]["id"], trace=track,
                               key=eye.incoming[1].tolist(), selected=demo["selected"], events=arena.events.copy()))
            print("trace", angle, index, len(track), bool(demo["selected"]), flush=True)
    finally:
        arena.close()
    _write(path, dict(openings_sha256=_hash(SOURCE), demonstrations_sha256=_hash(DEMONSTRATIONS), traces=traces))


class Flight:
    """Predict interception from the most recent continuous RGB flight segment."""

    def __init__(self):
        self.model = table_model()
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body("ball").id
        self.table = self.model.geom("table").id
        self.coeff = json.loads((PRIOR / "recorded-flight.json").read_text())["held_scores"]["match_fit"]["coefficients"]
        self.gravity = self.model.opt.gravity.copy()
        self.spin = np.array([35., 0., 0.])
        self.cross = np.empty(3)

    def estimate(self, readings):
        ticks, points = zip(*readings, strict=True)
        points = np.asarray(points)
        near_bounce = np.flatnonzero((points[:-1, 1] < -.3) & (points[:-1, 2] < .11)
                                    & (np.diff(points[:, 2]) > .018))
        bounced = len(near_bounce) > 0
        if bounced:
            first = near_bounce[0]
            while first > 0 and points[first - 1, 2] < points[first, 2]:
                first -= 1
            first += 1
            if len(points) - first < 3:
                return None
        else:
            first = 0
        first = max(first, len(points) - 7)
        points = points[first:]
        if len(points) < 3:
            return None
        offsets = (np.asarray(ticks[first:]) - ticks[-1]) / 30
        last = points[-1]
        velocity = (offsets @ (points - last)) / (offsets @ offsets)
        for _ in range(3):
            a = self.acceleration(velocity)
            change = self.acceleration(velocity + .01 * a) - a
            displaced = np.outer(.5 * offsets**2, a) + np.outer(offsets**3 / .06, change)
            velocity = (offsets @ (points - last - displaced)) / (offsets @ offsets)
        if velocity[1] > -.5:
            return None
        return ticks[-1], last, velocity, bounced

    def acceleration(self, velocity):
        drag = self.coeff[0] * np.linalg.norm(velocity) * velocity
        magnus = self.coeff[1] * np.cross(self.spin, velocity)
        return self.gravity - drag + magnus

    def predict(self, estimate):
        tick, point, velocity, bounced = estimate
        return self._predict(tick, tuple(point), tuple(velocity), bounced)

    @lru_cache(maxsize=1024)
    def _predict(self, tick, point, velocity, bounced):
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = point + TABLE_ORIGIN
        d.qvel[:3] = velocity
        d.qvel[3:6] = self.spin
        mujoco.mj_forward(m, d)
        path = []
        terrain = False
        for substep in range(1080):
            step_ball(m, d, self.coeff, self.cross, self.body)
            touching = d.ncon and any(self.table in contact.geom and contact.efc_address >= 0 for contact in d.contact)
            if touching and not terrain:
                if bounced or d.qpos[1] > 0:
                    break
                bounced = True
            terrain = touching
            p = d.qpos[:3] - TABLE_ORIGIN
            if p[1] < -2.28 or p[2] < -.02:
                break
            if bounced and p[1] < -1.55 and substep % 4 == 0:
                path.append(np.r_[tick + 30 * d.time, p, d.qvel[:3]])
        return np.asarray(path)


class PoseMap:
    """Solve small spatial corrections with the arm's calibrated chain and face."""

    def __init__(self, arena):
        self.model = arena.model
        self.data = mujoco.MjData(self.model)
        self.grip = self.model.body(GRIP_BODY).id
        self.blade = self.model.geom(BLADE).id
        self.joints = arena._arm_ids[1][0][:5]
        actuators = arena._arm_ids[1][2][:5]
        self.low, self.high = arena._joint_low[actuators], arena._joint_high[actuators]
        self.local = _GRASP_POS + self.model.geom_pos[self.blade]

    def face(self, joints):
        self.data.qpos[self.joints] = joints
        mujoco.mj_kinematics(self.model, self.data)
        orientation = self.data.xmat[self.grip].reshape(3, 3)
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.model.geom_quat[self.blade])
        return self.data.xpos[self.grip] + orientation @ self.local, orientation @ rot.reshape(3, 3)[:, 2]

    def offset(self, q, position, tilt):
        return self._offset(tuple(q), tuple(position), tuple(tilt), tuple(self.model.geom_quat[self.blade]))

    @lru_cache(maxsize=4096)
    def _offset(self, q, position, tilt, _blade_orientation):
        q = np.asarray(q)
        center, normal = self.face(q)
        desired = center + position
        target = normal + tilt
        target /= np.linalg.norm(target)

        def error(candidate):
            p, n = self.face(candidate)
            return np.r_[(p - desired) / .013, (n - target) / .04, (candidate - q) / .85]

        fit = least_squares(error, np.clip(q, self.low + 1e-6, self.high - 1e-6),
                            bounds=(self.low, self.high), max_nfev=12)
        return fit.x - q, np.linalg.norm(error(fit.x)[:3])


def examples(traces):
    memory = []
    for row in traces:
        if row["selected"] is None:
            continue
        trigger = row["trace"][0]["tick"]
        hit = next(event for event in row["events"] if event["kind"] == "contact" and event["side"] == 1
                   and event["step"] > trigger)
        body = next(sample for sample in row["trace"] if sample["tick"] == hit["step"] - 1)
        q = np.array(body["joints"][:5])
        swing = row["selected"]["action"]
        memory.append(dict(angle=row["angle"], idx=row["idx"], id=row["id"], key=row["key"],
                           action=swing, hit=np.asarray(hit["ball"]) - TABLE_ORIGIN,
                           incoming=np.asarray(hit["incoming"]), contact=hit["time"] * 30 - trigger, q=q))
    return memory


class Recall:
    def __init__(self, eye, body, poses, flight, memory, mode):
        self.eye, self.body, self.poses, self.flight = eye, body, poses, flight
        self.memory = memory
        self.mode = mode
        self.example = None
        self.shift = None
        self.time = None
        self.last_update = None
        self.bounced = False
        self.updates = []
        self.late = False

    def choose(self, tick):
        trigger, key = self.eye.incoming
        if self.example is None:
            self.example = min(self.memory, key=lambda e: self.image_cost(e, key))
        if self.mode == "recall" or self.eye.outgoing or self.last_update == tick:
            return
        reading = self.flight.estimate(self.eye.seen)
        if reading is None or reading[0] != tick or (self.bounced and not reading[3]):
            return
        self.last_update = tick
        self.bounced |= reading[3]
        if self.mode == "once" and self.updates:
            return
        path = self.flight.predict(reading)
        if not len(path):
            return
        options = [self.example]
        if self.mode in ("feedback", "once", "tilt", "early") and not self.late:
            options = self.memory
        choices = []
        for e in options:
            i = np.argmin(np.abs(path[:, 2] - e["hit"][1]))
            prediction = path[i]
            if abs(prediction[2] - e["hit"][1]) > .08:
                continue
            pos = prediction[1:4] - e["hit"]
            if self.mode == "early" and not self.bounced:
                pos[2] = 0.
            if np.linalg.norm(pos) > .63:
                continue
            tilt = np.zeros(3)
            if self.mode == "tilt":
                tilt = .17 * ((prediction[4:] - e["incoming"]) / np.linalg.norm(e["incoming"]))
                if self.poses.face(e["q"])[1][1] < 0:
                    tilt *= -1
            shift, error = self.poses.offset(e["q"], pos, tilt)
            delta = prediction[0] - trigger - e["contact"]
            dt = prediction[0] - tick
            required = np.max(abs(np.asarray(e["action"][0]) + shift - self.body.target[:5]) / STEP_SIZE[:5])
            rank = (error + .9 * np.linalg.norm(shift) + .6 * np.linalg.norm(pos)
                    + .1 * self.image_cost(e, key) + .22 * np.linalg.norm(prediction[4:] - e["incoming"])
                    + .18 * max(required - dt + 3, 0))
            if e is self.example:
                rank -= .20
            choices.append((rank, e, shift, delta, dt))
        if not choices:
            return
        _, selected, shift, delta, dt = min(choices, key=lambda r: r[0])
        self.late |= dt <= 13
        if dt < 1:
            return
        self.example = selected
        self.shift = shift if self.mode != "timing" else np.zeros(5)
        self.time = delta if self.mode != "spatial" else 0.
        self.updates.append(dict(tick=tick, expert=selected["idx"], id=selected["id"], shift=self.shift.tolist(),
                                 delta=self.time, intercept_tick=tick + dt, bounced=self.bounced))

    @staticmethod
    def image_cost(e, key):
        return float(np.linalg.norm((key - e["key"]) / SCALE))

    def act(self, tick):
        if self.eye.incoming is None or not self.memory:
            return None
        self.choose(tick)
        goal, impact, velocity, lead = self.example["action"]
        if self.shift is not None:
            goal = np.asarray(goal) + self.shift
            impact = impact + self.time
        return command(len(self.body.target), self.body.target, (goal, impact, velocity, lead), tick,
                       self.eye.incoming[0])




MODES = ("recall", "once", "timing", "spatial", "track", "feedback", "tilt", "early", "shadow")
DEV_DONORS = (2999, 11953, 13009)
HELD_DONORS = (4104, 10542)


def play(arena, tape, case, memory, poses, flight, mode, *, trace=None, source=None, reuse_reply=False):
    original = case["results"]["bounce"]
    arena.launch(case["shot"])
    if tape is not None:
        tape.steps.clear()
    live_commands = arena._aim, arena._command_to
    body = PublicBody(arena)
    eye = StereoReply(arena.width, arena.height)
    if mode == "shadow":
        from .tt27_reply_shadow import ShadowPlanner, ShadowRecall

        planner = ShadowPlanner()
        pilot = ShadowRecall(eye, body, poses, flight, memory, case["angle"], planner)
    else:
        pilot = Recall(eye, body, poses, flight, memory, mode)
    actions = []
    images = {p["tick"]: p["image"] for p in trace} if trace else None
    prefix = []
    second_touch = False
    for tick in range(arena.max_steps):
        if not second_touch:
            if images is None:
                if not eye.outgoing and (mode != "recall" or eye.incoming is None):
                    eye.read(arena.render_cameras(), tick)
            elif images.get(tick) is not None:
                eye.follow(tick, np.asarray(images[tick]))
                if eye.incoming is None:
                    eye.incoming = tick, np.asarray(source)
                    eye.reply = [(tick, np.asarray(images[tick]))]
        if mode == "shadow":
            pilot.observe_joints(arena.joints)
        near = pilot.act(tick)
        if near is None:
            near = original["actions"][tick] if tick < len(original["actions"]) else np.zeros(arena.action_size)
        if eye.incoming is not None and not prefix:
            prefix = arena.events.copy()
            expected = [event for event in original["events"] if event["step"] <= tick]
            if digest_json(prefix) != digest_json(expected):
                raise ValueError("Reply handoff changed the physical opening")
        actions.append(np.asarray(near).tolist())
        if tape is None:
            _, _, done, cap, _ = archived_step(arena, original["rival_actions"], tick, near)
            second_touch |= arena.contacts >= 2
        elif reuse_reply and arena.contacts < 2 and tick < len(original["rival_actions"]):
            tape.steps[tick] = original["rival_actions"][tick]
            try:
                _, _, done, cap, _ = archived_step(arena, original["rival_actions"], tick, near)
            finally:
                arena._aim, arena._command_to = live_commands
        else:
            _, _, done, cap, _ = arena.step(near, render=False)
        body.issued(np.asarray(near))
        if done or cap or (tape is None and arena.legal_returns >= 2):
            break
    if mode == "shadow":
        planner.close()
    return dict(near=arena.legal_returns, far=arena.rival_returns - 1, contacts=arena.contacts,
                trigger=eye.incoming[0] if eye.incoming is not None else None,
                donor=pilot.example["id"] if pilot.example is not None else None,
                updates=pilot.updates, actions=actions, rival_actions=tape.tape(len(actions)) if tape else None,
                events=arena.events.copy(), outcome=arena.point_result())


def development(path, destination, modes):
    archive = _read(path)
    memory = examples(archive["traces"])
    cases = {(c["angle"], c["idx"]): c for c in schedule()}
    arena = ReferenceReceiver()
    arena.max_steps = 160
    grip, poses, flight = Grip(arena.model), PoseMap(arena), Flight()
    results = []
    try:
        for r in archive["traces"]:
            if r["id"] not in DEV_DONORS:
                continue
            grip.set(r["angle"])
            bank = [e for e in memory if e["angle"] == r["angle"] and e["id"] in DEV_DONORS and e["id"] != r["id"]]
            runs = {mode: play(arena, None, cases[r["angle"], r["idx"]], bank, poses, flight, mode,
                               trace=r["trace"], source=r["key"]) for mode in modes}
            results.append(dict(angle=r["angle"], idx=r["idx"], id=r["id"], results=runs))
            print("development", r["angle"], r["idx"],
                  {k: (v["near"], v["contacts"], len(v["updates"])) for k, v in runs.items()}, flush=True)
    finally:
        arena.close()
    _write(destination, dict(traces_sha256=_hash(path), roster=DEV_DONORS, cases=results))
    print("development totals", {m: [sum(r['results'][m][k] >= 2 for r in results) for k in ("near", "contacts")]
                                 for m in modes}, flush=True)


def evaluate(path, destination, modes, *, cases=None, transport=None, upstream=SOURCE, reuse_reply=False):
    archive = _read(path)
    arena, check = ReferenceReceiver(stationary_opponent=False), ReferenceReceiver()
    arena.max_steps = check.max_steps = 160
    grip, check_grip = Grip(arena.model), Grip(check.model)
    poses, flight = PoseMap(arena), Flight()
    profile = _read(COACHED)["metadata"]["opening"]["far_profile"]
    install_rally_opponent(arena, target_profile=profile)
    tape = FarTape(arena)
    memory = [e for e in examples(archive["traces"]) if e["id"] in DEV_DONORS]
    results = []
    if cases is None:
        cases = [case for case in schedule() if case["shot"]["id"] in HELD_DONORS]
    donors = sorted({case["shot"]["id"] for case in cases})
    if set(donors) & {example["id"] for example in memory}:
        raise ValueError("The evaluation includes a demonstration donor")
    try:
        for case in cases:
            angle = case["angle"]
            grip.set(angle)
            check_grip.set(angle)
            bank = [e for e in memory if e["angle"] == angle] if transport is None else transport(memory, grip, poses, angle)
            runs = {}
            for mode in modes:
                result = play(arena, tape, case, bank, poses, flight, mode, reuse_reply=reuse_reply)
                check.launch(case["shot"])
                for tick, near in enumerate(result["actions"]):
                    archived_step(check, result["rival_actions"], tick, near)
                if digest_json(result["events"]) != digest_json(check.events):
                    raise ValueError("Held physical point diverged on replay")
                runs[mode] = result
            results.append(dict(angle=case["angle"], idx=case["idx"], id=case["shot"]["id"],
                                baseline=case["results"]["bounce"]["near"], results=runs))
            print("held", case["angle"], case["idx"],
                  {k: (v["near"], v["contacts"], len(v["updates"])) for k, v in runs.items()}, flush=True)
            _write(destination, dict(openings_sha256=_hash(upstream), traces_sha256=_hash(path), development=DEV_DONORS,
                                     held=donors, modes=modes, cases=results, replayed=len(results) * len(modes),
                                     archived_reply=bool(reuse_reply)))
    finally:
        arena.close()
        check.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "development", "evaluate"))
    parser.add_argument("--traces", type=Path, default=OUT / "reply-traces.json.gz")
    parser.add_argument("--modes", nargs="+", choices=MODES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-reply", action="store_true")
    args = parser.parse_args()
    if args.stage == "collect":
        collect(args.traces)
    elif args.stage == "development":
        development(args.traces, args.output or OUT / "reply-development.json.gz", args.modes or MODES[:-1])
    else:
        evaluate(args.traces, args.output or OUT / "reply-held.json.gz", args.modes or ("recall", "track", "once", "shadow"),
                 reuse_reply=args.reuse_reply)
