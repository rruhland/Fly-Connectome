"""Use measured paddle-bounce response to rank regulation receiving proposals."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np

from .tt4_arena import BALL_RADIUS, TABLE_HEIGHT
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_so101 import BLADE, SO101_HOME
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil
from .tt25_pose_demonstrations import OBSERVE, episode, poses
from .tt25_recorded_exchange import step_ball, table_model
from .tt25_recorded_flight import OUTPUT as TT25
from .tt25_reference_receiver import ReferenceReceiver

OUTPUT = ROOT / "artifacts" / "tt26"
SOURCE = ROOT / "artifacts" / "tt24" / "bounce-ident.json.gz"
SOURCE_MODE = "wrist_fitted_motion"
WRIST_CHECKPOINT = SOURCE.with_name("bounce-ident-pupil.npz")
SOFT_DAMPING = 85.0
SOFT_SEEDS = tuple(range(261_101, 261_125))
SEED = 1460519
BUDGETS = (4, 12, 24)


@dataclass(frozen=True)
class ImpactResponse:
    restitution: float
    slip: float

    def rebound(self, velocity, spin, normals):
        normal_speed = normals @ velocity
        # Reversing the surface normal leaves the normal impulse unchanged.
        # The ball's contact patch lies on the inward side of the selected face.
        face = np.where(normal_speed[:, None] < 0, normals, -normals)
        tangential_velocity = velocity - normal_speed[:, None] * normals
        slip = tangential_velocity - BALL_RADIUS * np.cross(spin, face)
        return velocity - (1 + self.restitution) * normal_speed[:, None] * normals - self.slip * slip


class ContactRecorder(BounceArena):
    """Instrument existing stock-arm rollouts without changing the simulator."""

    def __init__(self, *, damping=None, **kwargs):
        super().__init__(**kwargs)
        self.damping = damping

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self.damping is not None:
            self._model.geom_solref[self._blade] = (-100000, -self.damping)
        self.impacts = []
        self._entry = None
        self._before = self._sample()
        return result

    def _sample(self):
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self._model, self._data, mujoco.mjtObj.mjOBJ_GEOM, self._blade, velocity, 0
        )
        return (
            self._ball_velocity.copy(),
            self._data.xmat[self._ball_body].reshape(3, 3) @ self._ball_spin,
            self._data.geom_xpos[self._blade].copy(),
            velocity,
        )

    def _judge_substep(self, incoming):
        touching = self._touching
        reward = super()._judge_substep(incoming)
        if not touching and self._contact is not None:
            blade = self._data.geom_xmat[self._blade].reshape(3, 3).copy()
            if blade[2, 2] < 0:
                blade[:, 1:] *= -1
            contact = next(
                c for c in self._data.contact
                if c.efc_address >= 0 and self._blade in c.geom and self._ball_geom in c.geom
            )
            v, w, center, paddle = self._before
            paddle_speed = paddle[3:] + np.cross(paddle[:3], contact.pos - center)
            relative = blade.T @ (v - paddle_speed)
            surface_slip = relative - BALL_RADIUS * (blade.T @ np.cross(w, blade[:, 2]))
            self._entry = (self._contact.impact_index, blade, paddle_speed, relative, surface_slip[:2])
        if touching and not self._touching and self._flight is not None:
            index, blade, paddle_speed, relative, surface_slip = self._entry
            departed = blade.T @ (self._ball_velocity - paddle_speed)
            self.impacts.append(
                dict(index=index, relative_in_mps=relative.tolist(), incoming_slip_mps=surface_slip.tolist(),
                     relative_out_mps=departed.tolist())
            )
        self._before = self._sample()
        return reward


def fit_response(rows, *, shuffle=False):
    velocity = np.array([r["relative_in_mps"] for r in rows])
    surface_slip = np.array([r["incoming_slip_mps"] for r in rows])
    outgoing = np.array([r["relative_out_mps"] for r in rows])
    changes = np.column_stack((velocity[:, :2] - outgoing[:, :2], outgoing[:, 2]))
    if shuffle:
        changes = changes[np.random.default_rng(SEED).permutation(len(changes))]
    normal = -velocity[:, 2]
    restitution = np.clip(np.dot(normal, changes[:, 2]) / np.dot(normal, normal), 0, 1)
    slip = np.clip(np.sum(surface_slip * changes[:, :2]) / np.sum(surface_slip**2), 0, 1)
    return ImpactResponse(float(restitution), float(slip))


def soft_source():
    """Run fresh soft-paddle drops with the existing public wrist pupil."""
    arena = ContactRecorder(damping=SOFT_DAMPING, width=192, height=144)
    try:
        for seed in SOFT_SEEDS:
            policy = WristPupil.load(WRIST_CHECKPOINT)
            obs = arena.reset(seed=seed, split="held")
            commands = []
            for _ in range(arena.max_steps):
                command = policy.command(obs)
                commands.append(command.tolist())
                obs, _, done = arena.step(command)
                if done:
                    break
            yield dict(commands=commands, **arena.judge_metrics())
    finally:
        arena.close()


def source_response(soft):
    source = soft_source() if soft else _read(SOURCE)["cohorts"][SOURCE_MODE]["episodes"]
    arena = ContactRecorder(view="none", damping=SOFT_DAMPING if soft else None)
    rows, samples = [], []
    try:
        for row in source:
            arena.reset(seed=row["seed"], split=row["split"], regime=row["regime"], render=False)
            for command in row["commands"]:
                arena.step(command, render=False)
            metrics = arena.judge_metrics()
            # JSON stores recorded dataclass tuples as arrays.
            expected = {k: v for k, v in row.items() if k != "commands"}
            if json.loads(json.dumps(metrics)) != json.loads(json.dumps(expected)):
                raise ValueError(f"Could not reproduce bounce source {row['seed']}")
            verified = {r["impact_index"] for r in metrics["rebound_events"]}
            impacts = [dict(seed=row["seed"], **r) for r in arena.impacts if r["index"] in verified]
            if {r["index"] for r in impacts} != verified:
                raise ValueError(f"Missed an impact in bounce source {row['seed']}")
            samples.extend(impacts)
            rows.append(row if soft else dict(seed=row["seed"], steps=metrics["steps"],
                                              verified_rebounds=len(verified)))
    finally:
        arena.close()
    models = dict(bounce=fit_response(samples), cold=ImpactResponse(.85, 0),
                  shuffled=fit_response(samples, shuffle=True))
    parent = WRIST_CHECKPOINT if soft else SOURCE
    path = OUTPUT / ("match-soft-rebound-source.json.gz" if soft else "match-rebound-source.json.gz")
    _write(path, dict(
        source=parent.name, source_sha256=_hash(parent), source_mode=SOURCE_MODE, episodes=rows,
        fitted_observations="simulator state immediately before and after verified bounce contacts",
        paddle_damping_override=SOFT_DAMPING if soft else None,
        verified_impacts=samples, shuffle_seed=SEED, fits={k: asdict(v) for k, v in models.items()},
    ))
    print("Source", len(rows), len(samples), models, flush=True)
    return models, path


def incoming(model, shot, coefficients):
    """Report the incoming leg after the first near-side physical table bounce."""
    data = mujoco.MjData(model)
    data.qpos[:3] = [shot["pos_x"], shot["pos_y"], TABLE_HEIGHT + shot["pos_z"]]
    data.qvel[:3] = [shot[f"vel_{a}"] for a in "xyz"]
    data.qvel[3:] = [shot[f"w_vel_{a}"] for a in "xyz"]
    body, table = model.body("ball").id, model.geom("table").id
    spin = np.zeros(3)
    track = []
    bounced, last = False, False
    for tick in range(800):
        step_ball(model, data, coefficients, spin, body)
        touched = any(table in c.geom and c.efc_address >= 0 for c in data.contact)
        if touched and not last:
            if bounced or data.qpos[1] > 0:
                break
            bounced = True
        if data.qpos[2] < TABLE_HEIGHT - 0.12:
            break
        if bounced and tick % 2 == 0:
            track.append((data.time, data.qpos[:3].copy(), data.qvel[:3].copy(), spin.copy()))
        last = touched
    return track


def rank(poses_bank, track, models):
    """Keep the TT25 search objective, varying only its predicted paddle rebound."""
    tree, goals, centers, normals = poses_bank
    best = {name: np.full(len(goals), np.inf) for name in models}
    for time, position, velocity, spin in track:
        ids = np.array(tree.query_ball_point(position, 0.093), dtype=int)
        if not len(ids):
            continue
        faces = normals[ids]
        displacement = centers[ids] - position
        plane = np.abs(np.sum(faces * displacement, axis=1))
        tangent = np.sqrt(np.maximum(0, np.sum(displacement**2, axis=1) - plane**2))
        arrival = np.max(np.abs(goals[ids] - SO101_HOME[:5]), axis=1) / 5.2 + OBSERVE / 30
        geometry = 8 * np.abs(plane - .022) + 3 * tangent + 4 * np.maximum(0, arrival - time)
        for name, model in models.items():
            out = model.rebound(velocity, spin, faces)
            air = (out[:, 2] + np.sqrt(np.maximum(
                0, out[:, 2] ** 2 + 2 * 9.81 * (position[2] - TABLE_HEIGHT - BALL_RADIUS)
            ))) / 9.81
            landing = position[:2] + .84 * out[:, :2] * air[:, None]
            crossing = -position[1] / np.maximum(.01, out[:, 1])
            net_z = position[2] + out[:, 2] * crossing - .5 * 9.81 * crossing**2
            loss = geometry + np.abs(landing[:, 1] - .56) + .7 * np.abs(landing[:, 0])
            loss += 4 * np.maximum(0, .94 - net_z)
            best[name][ids] = np.minimum(best[name][ids], loss)
    rankings = {}
    for name, losses in best.items():
        selected = []
        for i in np.argsort(losses):
            if not np.isfinite(losses[i]):
                break
            if not selected or min(np.linalg.norm(goals[i] - goals[j]) for j in selected) > .22:
                selected.append(int(i))
                if len(selected) == BUDGETS[-1]:
                    break
        rankings[name] = selected
    return rankings


def practice(arena, shots, train, models):
    bank = poses(arena)
    flight = json.loads((TT25 / "recorded-flight.json").read_text())
    coefficients = flight["held_scores"]["match_fit"]["coefficients"]
    physics = table_model()
    memories = {name: {budget: [] for budget in BUDGETS} for name in models}
    cases = []
    for sid in train:
        shot = shots[sid]
        ranks = rank(bank, incoming(physics, shot, coefficients), models)
        outcomes, observations = {}, {}
        if len({len(r) for r in ranks.values()}) != 1:
            raise ValueError("The rankings did not receive identical probe counts")
        trials = {}
        for name, ranked in ranks.items():
            trials[name] = []
            for index in ranked:
                goal = bank[1][index]
                if index not in outcomes:
                    outcomes[index], _ = episode(arena, shot, goal)
                result = outcomes[index]
                trials[name].append(dict(pose_index=index, goal=goal.tolist(), **result))
            for budget in BUDGETS:
                good = next((r for r in trials[name][:budget] if r["legal"]), None)
                if good is None:
                    continue
                index = good["pose_index"]
                if index not in observations:
                    repeated, key = episode(arena, shot, bank[1][index], collect=True)
                    if repeated != outcomes[index]:
                        raise ValueError("Training action did not reproduce under the camera")
                    observations[index] = key
                memories[name][budget].append((sid, observations[index], bank[1][index]))
        cases.append(dict(id=sid, trials=trials, unique_physical_probes=len(outcomes)))
        print("Practice", sid, {name: next((i for i, r in enumerate(v, 1) if r["legal"]), None)
                                for name, v in trials.items()}, flush=True)
    return cases, memories


def evaluate(arena, shots, held, memories):
    with np.load(TT25 / "recorded-pose-imitation.npz", allow_pickle=False) as file:
        scale = file["scale"]
    # A camera observation is common to all options because all wait for the
    # same four images with their motors held at the normal home pose.
    cases = []
    for cohort, ids in held.items():
        for sid in ids:
            shot = shots[sid]
            held_result, key = episode(arena, shot, collect=True)
            outcomes = {None: held_result}
            scores = {}
            for name, checkpoints in memories.items():
                scores[name] = {}
                for budget, examples in checkpoints.items():
                    chosen = min(examples, key=lambda e: np.linalg.norm((e[1] - key) / scale), default=None)
                    donor, _, goal = chosen if chosen is not None else (None, None, None)
                    if donor not in outcomes or (donor is not None and not np.array_equal(outcomes[donor][0], goal)):
                        result, _ = episode(arena, shot, goal)
                        outcomes[donor] = (goal, result)
                    result = outcomes[donor][1] if donor is not None else outcomes[None]
                    if result["legal"]:
                        memory = (np.stack([e[1] for e in examples]), np.stack([e[2] for e in examples]), scale)
                        verified, _ = episode(arena, shot, memory=memory)
                        if verified != result:
                            raise ValueError("Held camera policy did not reproduce chosen physical action")
                    scores[name][budget] = dict(donor=donor, **result)
            cases.append(dict(id=sid, cohort=cohort, key=key.tolist(), scores=scores))
        print("Held", cohort, {name: {b: sum(c["scores"][name][b]["legal"] for c in cases if c["cohort"] == cohort)
                                      for b in BUDGETS} for name in memories}, flush=True)
    return cases


def run(path, soft=False):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != json.loads((TT25 / "recorded-secondary-reference.json").read_text())["sha256"]:
        raise ValueError("Unexpected recorded table-tennis dataset")
    shots = {r["id"]: r for r in json.loads(raw)}
    lesson = _read(TT25 / "recorded-pose-imitation.json.gz")
    nearby = _read(TT25 / "recorded-block-confirmation.json.gz")["cohorts"]
    held = dict(nearby_1=[r["id"] for r in nearby[0]], nearby_2=[r["id"] for r in nearby[1]],
                unfiltered=lesson["held"])
    train = lesson["train"]
    if set(train) & set(sum(held.values(), [])):
        raise ValueError("The practice and held rosters overlap")
    models, source_path = source_response(soft)
    arena = ReferenceReceiver()
    if soft:
        arena.model.geom(BLADE).solref[:] = (-100000, -SOFT_DAMPING)
    try:
        training, memories = practice(arena, shots, train, models)
        assessment = evaluate(arena, shots, held, memories)
    finally:
        arena.close()
    checkpoint = {
        name: {b: [dict(id=sid, key=key.tolist(), goal=q.tolist()) for sid, key, q in examples]
               for b, examples in budgets.items()}
        for name, budgets in memories.items()
    }
    filename = "match-soft-rebound.json.gz" if soft else "match-rebound.json.gz"
    _write(OUTPUT / filename, dict(
        source_sha256=_hash(source_path), paddle_damping_override=SOFT_DAMPING if soft else None,
        recorded_shot_sha256=digest, budgets_per_shot=BUDGETS,
        training=training, held=assessment, checkpoint=checkpoint,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deepmind", type=Path)
    parser.add_argument("--soft", action="store_true", help="Use the same softened blade for bouncing and receiving.")
    args = parser.parse_args()
    run(args.deepmind, soft=args.soft)
