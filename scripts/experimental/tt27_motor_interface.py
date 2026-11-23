"""Identify remapped motor requests from the first commands of ordinary points."""

import argparse
from dataclasses import asdict, dataclass
from itertools import permutations, product
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import AXES, SO101_HOME
from .tt19_view_full_game import digest_json
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import PublicBody, STEP_SIZE
from .tt25_pose_demonstrations import OBSERVE
from .tt25_reference_receiver import ReferenceReceiver
from .tt26_bounce_effects import CHECKPOINT, CONTROL_INTERVAL, JointEffects
from .tt26_receiving_speed import draw
from .tt27_world_camera import CameraMounts, FEEDS, POSES, watch


OUT = Path("artifacts/tt27/motor-interface.json.gz")
SEED = 178611937
FRESH_SEED = 50279911
AXIS_COUNT = 5
BUDGETS = (1, 2, 3, 4, 6, 8)
HOLD_BUDGETS = (2, 4)
NATIVE = (0, 1, 2, 3, 4)
SHARED = (2, 0, 4, 1, 3)
ALL_POSITIVE = (1, 1, 1, 1, 1)


@dataclass(frozen=True)
class Interface:
    permutation: tuple
    signs: tuple
    delay: int

    def apply(self, request):
        return np.asarray(self.signs) * request[list(self.permutation)]

    def invert(self, wanted):
        request = np.empty(AXIS_COUNT)
        request[list(self.permutation)] = np.asarray(self.signs) * wanted
        return request


IDENTITY = Interface(NATIVE, ALL_POSITIVE, 0)
SOURCE = (
    Interface(SHARED, (1, -1, 1, -1, 1), 0),
    Interface(SHARED, (-1, 1, 1, -1, -1), 1),
)
HELD = (
    (Interface(SHARED, (-1, -1, 1, -1, 1), 1), .024, .96),
    (Interface(SHARED, (1, 1, -1, 1, -1), 0), -.028, 1.04),
    (Interface((2, 4, 0, 1, 3), (-1, 1, 1, -1, 1), 1), .032, .98),
    (Interface((2, 4, 0, 1, 3), (1, -1, 1, 1, -1), 0), -.024, 1.03),
)


class Candidates:
    """The learner is given the signed-permutation family and two possible delays."""

    def __init__(self, model, lower, upper, dt):
        orders = tuple(permutations(range(AXIS_COUNT)))
        signs = tuple(product((1, -1), repeat=AXIS_COUNT))
        self.maps = [Interface(order, polarity, delay)
                     for order, polarity, delay in product(orders, signs, (0, 1))]
        self.permutations = np.asarray([c.permutation for c in self.maps])
        self.group_size = 2 * len(signs)
        self.signs = np.asarray([c.signs for c in self.maps])
        self.delays = np.asarray([c.delay for c in self.maps])
        self.model, self.lower, self.upper = model, lower, upper
        self.steps = round(dt / CONTROL_INTERVAL)
        if not np.isclose(self.steps * CONTROL_INTERVAL, dt):
            raise ValueError("A match tick must be a whole number of source model ticks")
        self.correction = None

    @staticmethod
    def features(q, velocity, target, command):
        q, velocity, target, command = np.broadcast_arrays(q, velocity, target, command)
        return np.concatenate((np.ones_like(q[..., :1]), (q - SO101_HOME[:AXIS_COUNT]) / .2,
                               velocity / 5., (target - q) / .2, command), axis=-1)

    def learn_clock_correction(self, traces):
        features, outcomes = [], []
        for trace in traces:
            target = np.asarray(SO101_HOME[:AXIS_COUNT])
            for state, command in zip(np.asarray(trace["encoders"]), trace["requests"], strict=True):
                q, v, q_next, v_next = state
                command = np.asarray(command)
                q_hat, v_hat = self.model.advance(q, v, target, command, steps=self.steps)
                features.append(self.features(q, v, target, command))
                outcomes.append(np.r_[q_next - q_hat, .02 * (v_next - v_hat)])
                target = np.clip(target + STEP_SIZE[:AXIS_COUNT] * command, self.lower, self.upper)
        x, y = np.stack(features), np.stack(outcomes)
        self.correction = np.linalg.solve(x.T @ x + .12 * np.eye(x.shape[1]), x.T @ y)
        residual = y - x @ self.correction
        self.scales = np.maximum(np.sqrt(np.mean(residual**2, axis=0)), .003)
        return dict(frames=len(x), bounce_error=np.sqrt(np.mean(y**2, axis=0)).tolist(),
                    calibrated_error=np.sqrt(np.mean(residual**2, axis=0)).tolist())

    def loss(self, trace, *, model="corrected"):
        angles = np.broadcast_to(np.asarray(SO101_HOME[:AXIS_COUNT]), (len(self.maps), AXIS_COUNT)).copy()
        requests = np.vstack((np.zeros(AXIS_COUNT), trace["requests"]))
        evidence = np.zeros(len(self.maps))
        rows = []
        for tick, (q, velocity, q_next, v_next) in enumerate(np.asarray(trace["encoders"])):
            u = self.signs * requests[tick + 1 - self.delays[:, None], self.permutations]
            target = np.clip(angles + STEP_SIZE[:AXIS_COUNT] * u, self.lower, self.upper)
            if model != "generic":
                # Both target and normalized command histories are candidate-specific.
                q_hat, v_hat = self.model.advance(q, velocity, angles, u, steps=self.steps)
                if model == "corrected" and self.correction is not None:
                    prediction = np.concatenate((q_hat, .02 * v_hat), axis=1)
                    prediction += self.features(q, velocity, angles, u) @ self.correction
                    gap = (prediction - np.r_[q_next, .02 * v_next]) / self.scales
                    residual = gap**2
                else:
                    residual = ((q_hat - q_next) / .013)**2 + .35 * ((v_hat - v_next) / .85)**2
            else:
                # A generic diagonal servo approximation given the same encoder history.
                q_hat = q + .19 * (angles + target - 2 * q) + .009 * velocity
                residual = ((q_hat - q_next) / .013)**2
            evidence += np.minimum(residual, 36.).sum(axis=1)
            angles = target
            rows.append(evidence.copy())
        return np.asarray(rows)

    def best(self, evidence, previous=None):
        score = evidence.copy()
        if previous is not None:
            # Soft prior with 0.5 probability on a repeated routing. Signs and
            # delay are independent and have the same prior as the reset learner.
            preferred = np.all(self.permutations == previous, axis=1)
            score[preferred] -= 2 * np.log(119.)
        return self.maps[int(np.argmin(score))]

    def remember(self, evidence):
        # Average over signs and delay before retaining an order. One uncertain
        # sign cannot dominate the evidence for a permutation.
        groups = evidence.reshape(-1, self.group_size)
        pair = int(np.argmax(logsumexp(-groups * .5, axis=1)))
        return self.maps[pair * self.group_size].permutation

    def pool_order(self, evidence):
        groups = evidence.reshape(-1, self.group_size)
        return np.repeat(-2 * logsumexp(-.5 * groups, axis=1), self.group_size)

    def continue_with(self, evidence, previous):
        prior = np.all(self.permutations == previous, axis=1)
        return self.remember(evidence - 2 * np.log(119.) * prior)

    def equivalent_command_counts(self, trace, actual):
        """Evaluator count of candidate interfaces that execute the same prefix."""
        requests = np.vstack((np.zeros(AXIS_COUNT), trace["requests"]))
        same = np.ones(len(self.maps), dtype=bool)
        counts = {}
        for tick in range(len(trace["requests"])):
            issued = self.signs * requests[tick + 1 - self.delays[:, None], self.permutations]
            expected = actual.apply(requests[tick + 1 - actual.delay])
            same &= np.all(np.isclose(issued, expected, rtol=0., atol=1e-9), axis=1)
            budget = tick + 1 - OBSERVE
            if budget in BUDGETS:
                counts[budget] = int(same.sum())
        return counts


def select(arena, mounts, shot, keys, scale):
    frames = watch(arena, shot, mounts, ((0, 0, 0),) * 2)
    history = ViewHistory()
    for rgb, joints in frames:
        cue = history.observe(rgb, joints, joints)[:14]
    return int(np.argmin(np.linalg.norm((keys - cue) / scale, axis=1)))


def play(arena, shot, goal, actual, inferred, dofs, *, budget=None, tape=None):
    """The physical channel sees requests; the policy sees only its inferred routing."""
    arena.launch(shot)
    body = PublicBody(arena)
    requests, executed, measurements = [], [], []
    zero = np.zeros(AXIS_COUNT)
    samples = lambda: (arena.joints[:AXIS_COUNT], arena.data.qvel[dofs].copy())
    for tick in range(arena.max_steps):
        prior = np.asarray(requests[-1]) if requests else zero
        if tape is None:
            wanted = zero
            if tick >= OBSERVE:
                # The pending request still has to reach the body when delay is one.
                pending = inferred.apply(prior) * inferred.delay
                target = np.clip(body.target[:AXIS_COUNT] + STEP_SIZE[:AXIS_COUNT] * pending,
                                 body.lower[:AXIS_COUNT], body.upper[:AXIS_COUNT])
                wanted = np.clip((goal - target) / STEP_SIZE[:AXIS_COUNT], -1., 1.)
            request = inferred.invert(wanted)
        else:
            request = np.asarray(tape[tick])
        action = actual.apply(prior if actual.delay else request)
        before = samples() if budget is not None else None
        _, _, done, cap, _ = arena.step(np.r_[action, 0.], render=False)
        body.issued(np.r_[inferred.apply(prior if inferred.delay else request), 0.])
        requests.append(request.tolist())
        executed.append(action.tolist())
        if before is not None:
            measurements.append((*before, *samples()))
        stop = budget is not None and tick + 1 >= OBSERVE + budget
        if done or cap or arena.legal_returns or stop:
            break
    return dict(requests=requests, executed=executed, encoders=np.asarray(measurements).tolist(),
                contact=arena.contacts, legal=arena.legal_returns,
                incoming=arena.rival_returns, events=digest_json(arena.events))


def course(rng, donors, count, shift=0., speed=1.):
    feeds = [draw(donors, sorted(donors), rng, i) for i in range(count)]
    for shot in feeds:
        shot["pos_x"] += shift
        shot["vel_y"] *= speed
    return feeds


def describe_fit(value, actual):
    sign_errors = sum(x != y for x, y in zip(value.signs, actual.signs, strict=True))
    route_errors = sum(x != y for x, y in zip(value.permutation, actual.permutation, strict=True))
    return dict(interface=asdict(value), exact=value == actual, signs_wrong=sign_errors,
                routes_wrong=route_errors, delay_wrong=value.delay != actual.delay)


def first_fit(fits, actual):
    correct = [budget for budget in BUDGETS if budget in fits and fits[budget] == actual]
    return next((budget for budget in correct if all(
        fits[b] == actual for b in fits if b >= budget)), None)


def summary(report):
    blocks = report["held"]
    rows = [row for block in blocks for row in block["rows"]]
    totals = {name: {metric: sum(row["results"][name][metric] for row in rows)
                     for metric in ("legal", "contact")} for name in rows[0]["results"]}
    for budget in report["scoring_budgets"]:
        carry = f"carry_{budget}"
        for other in (f"reset_{budget}", "frozen"):
            difference = [row["results"][carry]["legal"] - row["results"][other]["legal"] for row in rows]
            totals[carry][f"wins_losses_vs_{other}"] = [
                sum(value > 0 for value in difference), sum(value < 0 for value in difference)
            ]
    return totals


def run(path, per_setting, bounce_only, continuation):
    with np.load(POSES, allow_pickle=False) as data:
        keys, goals, scale = (data[k] for k in ("keys", "goals", "scale"))
    archive = [r for r in _read(FEEDS)["cases"] if r["angle"] < 0]
    donors = {r["shot"]["id"]: r["shot"] for r in archive}
    rng = np.random.default_rng(SEED)
    fresh_rng = np.random.default_rng(FRESH_SEED) if continuation else rng
    scoring_budgets = (*HOLD_BUDGETS, 8) if continuation else HOLD_BUDGETS
    arena = ReferenceReceiver()
    mounts = CameraMounts(arena)
    body = PublicBody(arena)
    dofs = [arena.model.joint("learner_" + axis).dofadr[0] for axis in AXES[:AXIS_COUNT]]
    candidates = Candidates(JointEffects.load(), body.lower[:AXIS_COUNT], body.upper[:AXIS_COUNT], arena.control_dt)
    report = dict(schema="tt27-motor-interface-1", model_sha256=_hash(CHECKPOINT), shots_sha256=_hash(FEEDS),
                  pose_sha256=_hash(POSES), seed=FRESH_SEED if continuation else SEED, source_seed=SEED,
                  source=[], held=[],
                  provided_structure="five driven axes, signed permutation, delay of zero or one control tick",
                  observations="ordinary stereo RGB, own issued requests, joint angles and velocities",
                  fit_budgets=BUDGETS, scoring_budgets=scoring_budgets,
                  frozen="last source setting's inferred interface", continued_history=continuation)
    path.parent.mkdir(parents=True, exist_ok=True)

    def trace(shot, actual):
        chosen = select(arena, mounts, shot, keys, scale)
        row = play(arena, shot, goals[chosen], actual, IDENTITY, dofs, budget=max(BUDGETS))
        return dict(goal=chosen, shot=shot, **row)

    try:
        if not bounce_only:
            clips = []
            for old in archive[:16]:
                tape = np.asarray(old["results"]["bounce"]["actions"])[:, :AXIS_COUNT]
                measured = play(arena, old["shot"], None, IDENTITY, IDENTITY, dofs,
                                budget=min(max(BUDGETS), len(tape) - OBSERVE), tape=tape)
                clips.append(dict(id=old["shot"]["id"], **measured))
            report["previous_game_feedback"] = candidates.learn_clock_correction(clips)
            report["previous_game_feedback"]["roster"] = [r["id"] for r in clips]
        previous = None
        previous_group_loss = np.zeros(len(candidates.maps))
        for actual in SOURCE:
            traces = [trace(shot, actual) for shot in course(rng, donors, 4)]
            source_evidence = sum(candidates.loss(row)[-1] for row in traces)
            previous_group_loss += candidates.pool_order(source_evidence)
            previous = candidates.remember(source_evidence if bounce_only else previous_group_loss)
            last_source = candidates.best(source_evidence)
            report["source"].append(dict(actual=asdict(actual), inferred=describe_fit(last_source, actual),
                                         retained_permutation=previous, traces=traces))
            print("source", describe_fit(last_source, actual), previous, flush=True)
        for actual, shift, speed in HELD:
            adaptation, *fresh = course(fresh_rng, donors, per_setting + 1, shift, speed)
            history = trace(adaptation, actual)
            loss = candidates.loss(history)
            generic = candidates.loss(history, model="generic")
            fits = {name: {} for name in ("carry", "reset", "generic")}
            for budget in BUDGETS:
                index = OBSERVE + budget - 1
                if index < len(loss):
                    fits["carry"][budget] = candidates.best(loss[index], previous)
                    fits["reset"][budget] = candidates.best(loss[index])
                    fits["generic"][budget] = candidates.best(generic[index])
            modes = dict(frozen=last_source, oracle=actual, factory=IDENTITY)
            for budget in scoring_budgets:
                for name in fits:
                    modes[f"{name}_{budget}"] = fits[name].get(budget, IDENTITY)
            rows = []
            for shot in fresh:
                chosen = select(arena, mounts, shot, keys, scale)
                results, cached = {}, {}
                for name, inferred in modes.items():
                    if inferred not in cached:
                        result = play(arena, shot, goals[chosen], actual, inferred, dofs)
                        if play(arena, shot, goals[chosen], actual, inferred, dofs, tape=result["requests"]) != result:
                            raise ValueError("The physical outcome changed on request replay")
                        cached[inferred] = result
                    results[name] = cached[inferred]
                rows.append(dict(shot=shot, goal=chosen, results=results, verified_commands=len(cached)))
            scores = {name: dict(legal=sum(row["results"][name]["legal"] for row in rows),
                                 contacts=sum(row["results"][name]["contact"] for row in rows))
                      for name in modes}
            detail = {name: {b: describe_fit(m, actual) for b, m in selections.items()}
                      for name, selections in fits.items()}
            earliest = {name: first_fit(selections, actual) for name, selections in fits.items()}
            before = previous
            later = []
            if continuation:
                later = [trace(shot, actual) for shot in course(fresh_rng, donors, 3, shift, speed)]
                more = sum(candidates.loss(row)[-1] for row in later)
                previous = candidates.continue_with(loss[-1] + more, before)
            else:
                previous = candidates.remember(loss[-1])
            report["held"].append(dict(actual=asdict(actual), shift_m=shift, forward_speed_factor=speed,
                                       before_retained_permutation=before, next_retained_permutation=previous,
                                       earliest_sustained_exact=earliest, fits=detail, adaptation=history,
                                       equivalent_executed_prefixes=candidates.equivalent_command_counts(history, actual),
                                       ordinary_history_before_next_switch=later,
                                       scored_modes={name: describe_fit(m, actual) for name, m in modes.items()},
                                       scores=scores, rows=rows))
            print("held", len(report["held"]), "permutations", before, previous, "first exact", earliest,
                  "legal", {name: r["legal"] for name, r in scores.items()}, flush=True)
            report["summary"] = summary(report)
            _write(path, report)
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--per-setting", type=int, default=12)
    parser.add_argument("--bounce-only", action="store_true", help="Skip the earlier 30 Hz match correction")
    parser.add_argument("--continuation", action="store_true", help="Use fresh feeds and continue learning between settings")
    args = parser.parse_args()
    if args.per_setting <= 0:
        parser.error("--per-setting must be positive")
    run(args.output, args.per_setting, args.bounce_only, args.continuation)
