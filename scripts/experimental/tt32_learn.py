"""Learn how far to recover from actual outcomes on the following ball."""

import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory as motor
from . import tt29_optical_outcomes as optics
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import OUTPUT as TT31, card, neutral_delivery, play_card
from .tt31_curriculum_opponent import replay_arena
from .tt31_curriculum_summary import compare_delivery, first_deliveries, held_difference
from .tt31_ready import centered


OUTPUT = TT31.parent / "tt32"
START = TT31 / "ready-staged-checkpoint.json.gz"
STRENGTH = (1.0, 0.65, 0.35)
SIDES = ("left", "right")
HELD = {"periodic": TT31 / "recovery-periodic.json.gz", "full": TT31 / "recovery-full.json.gz"}
TRAIN_CONTROL = TT31 / "confirm-recovery.json.gz"
CAPS = {"periodic": 24, "full": 16}

class RecoveryDistance(motor.RallyObjective):
    """Keep the fixed rule's hit verification and adapt only the recovery distance."""

    def __init__(self, solver, minimum_pace, *, choices):
        super().__init__(solver, minimum_pace)
        self.choices = choices
        self.batches = []
        self.verified = self.attempted = 0

    def trajectories(self, goals, target, cutoff):
        rates, score = super().trajectories(goals, target, cutoff)
        valid = np.flatnonzero(np.isfinite(score[:, 0]))
        if not len(valid):
            return rates, score
        i = valid[np.argmin(score[valid, 0])]
        arena = self.solver.arena
        start = int(np.ceil((score[i, 1] + 0.065) / arena.control_dt))
        if start < 1 or start >= goals.shape[1] - 2:
            return rates, score
        x = float(arena.ball[0] + arena.velocity[0] * score[i, 1])
        side = "left" if x < 0 else "right"
        strength = STRENGTH[self.choices[side]]
        pose = np.asarray(arena._arm_home(1)[0])
        new = goals[i : i + 1].copy()
        q = new[0, start - 1]
        n = len(new[0]) - start
        duration = max(0.26, float(np.max(abs(pose - q))) / 3.0)
        u = np.clip(np.arange(1, n + 1) * arena.control_dt / duration, 0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        # Preserve the fixed arithmetic when using its original full distance.
        delta = pose - q
        if strength != 1.0:
            delta = strength * delta
        new[0, start:] = q + smooth[:, None] * delta
        self.attempted += 1
        changed, verified = super().trajectories(new, target, cutoff)
        if np.isfinite(verified[0, 0]) and np.allclose(score[i, 1:6], verified[0, 1:6], atol=0.002, rtol=0.):
            self.verified += 1
            edited = np.flatnonzero(np.any(rates[i] != changed[0], axis=1))
            rates[i] = changed[0]
            score[i] = verified[0]
            self.batches.append((rates, int(i), dict(context=side, optical_hit_x_m=x,
                strength=strength, predicted_hit_s=float(arena.data.time + score[i, 1]),
                edited_from=int(edited[0]) if len(edited) else None)))
        return rates, score


class IssuedRecovery(optics.OpticalOutcomes):
    """Link checked commands to the executed tape without reading live ball coordinates."""

    def __init__(self, *args, **settings):
        super().__init__(*args, **settings)
        self.adopted = []

    def command(self, tape):
        if tape is not None and tape is not self.last_tape:
            recovery = None
            for batch, index, candidate in reversed(self.solver.batches):
                if np.shares_memory(batch[index], tape):
                    recovery = candidate
                    break
            self.adopted.append(dict(tick=self.step, recovery=recovery, tape=tape))
        self.solver.batches.clear()
        super().command(tape)


def next_rewards(row, records, dt):
    """Attribute the next legal return after an applied recovery and a real offer."""
    replies = row["placements"]["delivered"]
    finished = any(e["kind"] in ("point", "timeout", "let") for e in row["events"])
    completed = []
    for i, reply in enumerate(replies[:-1]):
        if not reply["near_landed"]:
            continue
        strike = reply["near_contact"]
        received = replies[i + 1]
        if i == len(replies) - 2 and not finished and not received["near_landed"]:
            continue
        eligible = [c for c in records if c["tick"] < strike["step"]]
        if not eligible or eligible[-1]["recovery"] is None:
            continue
        active = eligible[-1]
        choice = active["recovery"]
        index = choice["edited_from"]
        if index is None or abs(strike["time"] - choice["predicted_hit_s"]) > .080:
            continue
        tick = active["tick"] + index
        separated = next(e for e in row["events"] if e["time"] > strike["time"] and e["kind"] == "separation"
                         and e.get("collider") == "learner_racket")
        if tick >= len(row["actions"]) or tick * dt < separated["time"]:
            continue
        if not np.array_equal(row["actions"][tick], active["tape"][index]):
            continue
        completed.append(choice | dict(after_leg=i, command_tick=tick, observed_current_lane=reply["observed_lane"],
            observed_next_lane=received["observed_lane"], next_contact=received["near_contact"] is not None,
            next_legal=received["near_landed"], separated_s=separated["time"]))
    return completed


def play(feeds, entry, pose, choices, temporary, cap):
    objectives = []
    observers = []

    def make_objective(solver, minimum_pace):
        objective = RecoveryDistance(solver, minimum_pace, choices=choices)
        objectives.append(objective)
        return objective

    def make_observer(*args, **settings):
        observer = IssuedRecovery(*args, **settings)
        observers.append(observer)
        return observer

    with ExitStack() as stack:
        stack.enter_context(centered(pose))
        stack.enter_context(patch.object(motor, "RallyObjective", make_objective))
        stack.enter_context(patch.object(optics, "OpticalOutcomes", make_observer))
        row, _ = play_card(feeds, entry, START, START, temporary, practice=False, cap=cap)
    objective, = objectives
    observer, = observers
    row["recovery_candidates"] = objective.attempted
    row["verified_recovery_candidates"] = objective.verified
    row["next_return_rewards"] = next_rewards(row, observer.adopted, objective.solver.arena.control_dt)
    row["adopted_recovery_tapes"] = sum(c["recovery"] is not None for c in observer.adopted)
    return row


def prepare(feeds, output):
    reference = _read(TRAIN_CONTROL)
    ready = reference["ready"]
    cards = reference["cards"]
    if len(cards) != 4 or len(reference["rows"]) != 4:
        raise ValueError("Incomplete periodic practice roster")
    held = {k: _read(path)["cards"] for k, path in HELD.items()}
    training_ids = {c["id"] for c in cards}
    if training_ids.intersection(c["id"] for group in held.values() for c in group):
        raise ValueError("A held release also appears in practice")
    full = []
    shots = measured_launches(feeds)
    with centered(ready):
        arena = replay_arena([])
        try:
            for i, entry in enumerate(cards):
                for j in range(12):
                    candidate = card(entry["id"], "full", i, 320331 + 127 * i + j)
                    receipt = neutral_delivery(arena, shots, candidate)
                    if receipt is not None:
                        full.append(dict(card=candidate, first_delivery=receipt))
                        print("training offer", candidate["id"], receipt["observed_lane"], flush=True)
                        break
                else:
                    raise ValueError(f"No safe first offer on {entry['id']}")
        finally:
            arena.close()
    _write(output / "learn-plan.json.gz", dict(feed_sha256=_hash(feeds), initial_checkpoint_sha256=_hash(START),
           steady_ready=ready, practice=dict(periodic=cards, full=[r["card"] for r in full]), held=held,
           training_first_delivery=full, caps=CAPS, strengths=STRENGTH))


def run(feeds, output, phase, arm):
    path = output / "learn-plan.json.gz"
    deck = _read(path)
    if _hash(START) != deck["initial_checkpoint_sha256"] or _hash(feeds) != deck["feed_sha256"]:
        raise ValueError("Practice and evaluation have changed common inputs")
    if phase == "practice":
        choices = dict.fromkeys(SIDES, arm)
        points = [(mode, entry) for mode, cards in deck["practice"].items() for entry in cards]
        name = f"learn-practice-{arm}.json.gz"
    else:
        checkpoint = _read(output / "learn-checkpoint.json")
        if checkpoint["plan_sha256"] != _hash(path) or checkpoint["strengths"] != deck["strengths"]:
            raise ValueError("The learned recovery checkpoint changed training plans")
        choices = checkpoint["chosen"]
        points = [(phase, entry) for entry in deck["held"][phase]]
        name = f"learn-held-{phase}.json.gz"
    report = dict(phase=phase, ready=deck["steady_ready"], plan_sha256=_hash(path), opening_side=1,
                  cards=[entry for _, entry in points], choices=choices, rows=[], caps=CAPS)
    with TemporaryDirectory(prefix="tt32-learn-") as d:
        for mode, entry in points:
            row = play(feeds, entry, report["ready"], choices, Path(d) / "point.json.gz", CAPS[mode])
            row["mode"] = mode
            report["rows"].append(row)
            _write(output / name, report)
            n = row["next_return_rewards"]
            print(name, entry["id"], mode, row["score"]["learner"], row["score"]["returns"],
                  [(r["context"], r["next_legal"]) for r in n], flush=True)


def practiced(output, plan):
    result = []
    cards = [entry for group in plan["practice"].values() for entry in group]
    base = _read(TRAIN_CONTROL)
    for arm in range(len(STRENGTH)):
        path = output / f"learn-practice-{arm}.json.gz"
        doc = _read(path)
        if doc["plan_sha256"] != _hash(output / "learn-plan.json.gz") or doc["cards"] != cards or len(doc["rows"]) != len(cards):
            raise ValueError(f"Incomplete practice trial for recovery arm {arm}")
        if arm == 0:
            for row, expected in zip(doc["rows"][:4], base["rows"], strict=True):
                for field in ("actions", "events", "placements"):
                    if digest_json(row[field]) != digest_json(expected[field]):
                        raise ValueError(f"Fixed recovery failed equivalence for {field}")
        result.append((path, doc["rows"]))
    return result


def fit(output):
    path = output / "learn-plan.json.gz"
    plan = _read(path)
    results = practiced(output, plan)
    trials = {}
    chosen = {}
    for side in SIDES:
        counts = []
        for _, rows in results:
            outcomes = [r["next_legal"] for row in rows for r in row["next_return_rewards"] if r["context"] == side]
            counts.append(dict(trials=len(outcomes), legal=sum(outcomes), mean=(1 + sum(outcomes)) / (2 + len(outcomes))))
        # Uniform beta prior. An alternate with fewer than two physical trials
        # is too weakly sampled to replace the fixed recovery.
        allowed = [0] + [i for i, c in enumerate(counts[1:], 1) if c["trials"] >= 2]
        chosen[side] = max(allowed, key=lambda i: counts[i]["mean"])
        trials[side] = counts
    _write(output / "learn-checkpoint.json", dict(plan_sha256=_hash(path), strengths=STRENGTH,
           practice_sha256=[_hash(p) for p, _ in results], stats=trials, chosen=chosen,
           base_movement_and_optical_checkpoint_sha256=_hash(START)))
    print(chosen, trials, flush=True)


def stats(rows):
    arrivals = Counter(r["observed_lane"] or "unknown" for row in rows for r in row["placements"]["delivered"][1:])
    received = Counter(r["observed_lane"] or "unknown" for row in rows for r in row["placements"]["delivered"][1:]
                       if r["near_landed"])
    return dict(arm=sum(r["score"]["learner"] for r in rows), both=sum(r["score"]["returns"] for r in rows),
                cpu_s=sum(r["process_cpu_s"] for r in rows), later_arrivals=dict(arrivals), later_landed=dict(received),
                points=[dict(id=r["shot"]["id"], seed=r["opponent_seed"], arm=r["score"]["learner"], both=r["score"]["returns"])
                        for r in rows])


def summary(output):
    path = output / "learn-plan.json.gz"
    plan = _read(path)
    fit_path = output / "learn-checkpoint.json"
    trained = _read(fit_path)
    checkpoint = trained["base_movement_and_optical_checkpoint_sha256"]
    comparison = {}
    for phase, base_path in HELD.items():
        learned_path = output / f"learn-held-{phase}.json.gz"
        learned = _read(learned_path)
        base = _read(base_path)
        if (learned["plan_sha256"] != _hash(path) or learned["choices"] != trained["chosen"]
                or learned["cards"] != base["cards"] or len(learned["rows"]) != len(base["rows"])
                or learned["caps"] != plan["caps"] or learned["ready"] != base["ready"]
                or base["checkpoint_sha256"] != checkpoint):
            raise ValueError(f"Incomplete or changed held assessment for {phase}")
        rows, prior = learned["rows"], base["rows"]
        if any(r[field] != checkpoint for r in rows + prior
               for field in ("initial_checkpoint_sha256", "initial_optical_checkpoint_sha256")):
            raise ValueError(f"Movement or optical checkpoint mismatch for {phase}")
        same = {field: sum(digest_json(a[field]) == digest_json(b[field]) for a, b in zip(rows, prior, strict=True))
                for field in ("actions", "events", "placements")}
        comparison[phase] = dict(fixed=stats(prior), learned=stats(rows), fixed_sha256=_hash(base_path),
                                 learned_sha256=_hash(learned_path), paired=held_difference(rows, prior),
                                 exact_repetitions=same,
                                 first_offer=compare_delivery(first_deliveries(rows), first_deliveries(prior)))
        print(phase, comparison[phase]["paired"], flush=True)
    practice = [rows for _, rows in practiced(output, plan)]
    matches = [compare_delivery(first_deliveries(practice[0]), first_deliveries(rows)) for rows in practice[1:]]
    _write(output / "learn-summary.json", dict(checkpoint_sha256=_hash(fit_path), reference_equivalent_points=4,
           comparison=comparison, practice=[stats(rows) for rows in practice], practice_first_offers=matches))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("phase", choices=("prepare", "practice", "fit", "periodic", "full", "summary"))
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--arm", type=int, choices=range(len(STRENGTH)), default=0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase == "prepare":
        prepare(a.feeds, a.output)
    elif a.phase == "fit":
        fit(a.output)
    elif a.phase == "summary":
        summary(a.output)
    else:
        run(a.feeds, a.output, a.phase, a.arm)


if __name__ == "__main__":
    main()
