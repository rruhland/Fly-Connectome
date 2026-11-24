"""Learn a mild visual drift correction in continued wrist-only bouncing."""

from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_public_body import stock
from .tt29_bounce import COURSE, OUTPUT, WarmWrist, load_checkpoint, run_drop
from .tt29_bounce_continued import RESULT as EARLIER, compare, configure, make_deck
from .tt29_bounce_promotion import PROMOTION, RETAINED

RESULT = OUTPUT / "tt29_bounce_drift.json.gz"
DEPLOY = OUTPUT / "tt29_bounce_drift_deploy.json"
SEEDS = (295193243, 297441883, 296300445)
COUNTS = (12, 6, 8)
GAINS = np.array([0., .40, .85])


class DriftWrist(WarmWrist):
    """Aim the next paddle normal from lateral motion seen after a rebound."""

    def begin_release(self, obs):
        super().begin_release(obs)
        self.correction = np.zeros(5)
        self.correction_events = []
        self.fall = -100
        self.rise = -100
        self.pending = []

    def motion_adjustment(self, state, joints, ellipse):
        gain = self.checkpoint["drift_gain"]
        if not gain or ellipse is None:
            return self.adjustment + self.correction
        height = state[2] - state[8]
        if state[5] < -.60 and height < .145:
            self.fall = self.tick
        if (0 <= self.tick - self.fall <= 5 and self.tick - self.rise > 8 and state[5] > .30
                and -.015 < height < .175):
            self.rise = self.tick
            self.pending.clear()
        if 0 <= self.tick - self.rise <= 4 and state[5] > .20:
            self.pending.append(state[:5].copy())
            if len(self.pending) == 3:
                x, y, _, vx, vy = np.median(self.pending, axis=0)
                aim = -gain * np.array([vx + .95*x, vy + .95*y]) / 3.8
                aim *= min(1., .055 / max(1e-6, np.linalg.norm(aim)))
                reference = self.rig.body(joints)[12:17]
                jacobian = np.column_stack([(self.rig.body(joints + dq)[12:17] - reference) / .004
                                            for dq in np.eye(5) * .004])
                units = np.array([.020, .020, .020, .060, .060])
                design = jacobian / units[:, None]
                desired = np.r_[np.zeros(3), aim] / units
                shift = np.linalg.solve(design.T @ design + .4 * np.eye(5), design.T @ desired)
                self.correction = np.clip(shift, -.085, .085)
                self.correction_events.append(dict(tick=self.tick, optical_drift_m_s=[vx, vy],
                    lateral_offset_m=[x, y], normal_adjustment=aim.tolist(), joint_adjustment=self.correction.tolist()))
        return self.adjustment + self.correction


def play(arena, drop, snapshot):
    seed, regime, difficult = drop
    configure(arena, difficult)
    contenders = []

    def capture(checkpoint, bounds):
        actor = DriftWrist(checkpoint, bounds) if "drift_gain" in checkpoint else WarmWrist(checkpoint, bounds)
        contenders.append(actor)
        return actor

    row = run_drop(arena, seed, regime, "held", snapshot, policy=capture)
    if run_drop(arena, seed, regime, "held", tape=row["actions"]) != row:
        raise ValueError(f"The drift release did not replay at seed {seed}")
    return row | dict(corrections=getattr(contenders[0], "correction_events", []))


def match(arena, deck, snapshots):
    rows = []
    for drop in deck:
        seed, regime, difficult = drop
        outcomes = {key: play(arena, drop, checkpoint) for key, checkpoint in snapshots.items()}
        rows.append(dict(seed=seed, regime=regime, difficult=difficult, results=outcomes))
    return rows


def schedule():
    practice, gate, held = [make_deck(s, n) for s, n in zip(SEEDS, COUNTS, strict=True)]
    # Give each gain one ordinary and two difficult exploratory attempts.
    practice = [(seed, regime, index >= len(GAINS)) for index, (seed, regime, _) in enumerate(practice)]
    previous = _read(COURSE)
    promotion = _read(PROMOTION)
    continuation = _read(EARLIER)
    used = previous["practice"] + previous["held"] + promotion["held"]
    used.extend(row for gate_round in promotion["rounds"] for row in gate_round["rows"])
    used.extend(row for split in ("practice", "validation", "held") for row in continuation[split])
    seeds = [seed for seed, _, _ in practice + gate + held]
    if len(set(seeds)) != len(seeds) or set(seeds) & {row["seed"] for row in used}:
        raise ValueError("The drift decks overlap earlier practice or assessment")
    return practice, gate, held


def run():
    started = perf_counter()
    incumbent = load_checkpoint(RETAINED)
    practice, gate, held = schedule()
    arena = stock(fixture="table-edge")
    arena.max_steps = 140
    trials = np.zeros(len(GAINS), dtype=int)
    rewards = np.zeros(len(GAINS))
    rows = []
    report = dict(schema="tt29-bounce-visual-drift-1", incumbent_sha256=_hash(RETAINED), seeds=SEEDS,
                  practice=rows, gains=GAINS.tolist(), caps=dict(practice=140, gate=240, held=300),
                  learner_inputs="wrist RGB, own encoders and completed drop reward")

    def prefer(indices):
        score = (rewards + .3) / (trials + 1.)
        return int(indices[np.argmax(score[indices])])

    try:
        for index, drop in enumerate(practice):
            option = index % len(GAINS) if index < 9 else prefer(np.arange(len(GAINS)))
            proposal = incumbent | dict(drift_gain=float(GAINS[option]))
            row = play(arena, drop, proposal)
            if drop[2]:
                trials[option] += 1
                rewards[option] += np.sqrt(max(0, row["reward"] - 1.))
            rows.append(dict(seed=drop[0], regime=drop[1], difficult=drop[2], choice=option, **row))
            _write(RESULT, report)
            print("drift practice", index + 1, "gain", GAINS[option], "rebound", row["rebounds"],
                  "visual corrections", len(row["corrections"]), flush=True)
        best = prefer(np.arange(1, len(GAINS)))
        report["training"] = dict(difficult_attempts=trials.tolist(), difficult_bonus=rewards.tolist(), candidate=best)
        challenger = incumbent | dict(drift_gain=float(GAINS[best]))
        snapshots = {"0": incumbent, "challenger": challenger}
        report["candidate"] = challenger
        arena.max_steps = 240
        report["validation"] = match(arena, gate, snapshots)
        result = compare(report["validation"])
        promoted = result["all"]["promote"] and result["difficult"]["qualified_rebound_change"] > 0
        report.update(validation_summary=result, promoted=promoted)
        _write(DEPLOY, challenger if promoted else incumbent)
        report["deployment_sha256"] = _hash(DEPLOY)
        _write(RESULT, report)
        print("drift gate", result, "promoted", promoted, flush=True)
        arena.max_steps = 300
        report["held"] = match(arena, held, snapshots)
        report["held_summary"] = compare(report["held"])
        report["verified_tapes"] = len(rows) + 2 * (len(gate) + len(held))
        report["wall_s"] = perf_counter() - started
        _write(RESULT, report)
        print("drift held", report["held_summary"], flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    run()
