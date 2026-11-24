"""Continue wrist bouncing after table practice and gate a new motion checkpoint."""

from dataclasses import replace
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_metric_stereo import triangulate
from .tt24_public_body import motion_fit, stationary, stock
from .tt28_dose import mkplan, solo_summary
from .tt29_bounce import COURSE, OUTPUT, Decoder, load_checkpoint, run_drop
from .tt29_bounce_promotion import PROMOTION, RETAINED, accept

RESULT = OUTPUT / "tt29_bounce_continued.json.gz"
DEPLOY = OUTPUT / "tt29_bounce_continued_deploy.json"
TABLE = OUTPUT / "tt29_bounce_retained_receive.json.gz"
SOURCE_SEED, GATE_SEED, HELD_SEED = 29338272, 29409973, 29286456
COUNT, GATE, HELD = 15, 8, 10
PRACTICE_CAP, VALIDATION_CAP = 160, 240
BIASES = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, .045], [0, 0, 0, 0, -.045],
                   [0, 0, 0, .045, 0], [0, 0, 0, -.045, 0]])


def make_deck(seed, count):
    return [(s, "changed" if i % 4 in (1, 2) else "standard", i % 2 == 1)
            for i, (s, _) in enumerate(mkplan(seed, count))]


def configure(arena, hard):
    def samples(seed, split, regime):
        sample = BounceArena.sample_drop(seed, split, regime)
        if not hard:
            return sample
        x, y, z = sample.offset
        vx, vy, vz = sample.velocity
        return replace(sample, offset=(1.23*x, 1.23*y, z + .034), velocity=(1.60*vx, 1.60*vy, vz),
                       spin=tuple(1.35*value for value in sample.spin))

    arena.sample_drop = samples


def checked_drop(arena, snapshot, seed, regime, hard, *, collect=False):
    configure(arena, hard)
    row = run_drop(arena, seed, regime, "held", snapshot, collect=collect, optics_ticks=60)
    again = run_drop(arena, seed, regime, "held", tape=row["actions"])
    if again != {k: v for k, v in row.items() if k != "optics"}:
        raise ValueError(f"Continued bounce replay changed for {seed}")
    return row


def motion(rows, eyes, checkpoint):
    clips, objects = [], []
    for row in rows:
        optics = row["optics"]
        clips.append(dict(enc=np.asarray(optics["enc"]), wrist=np.array(optics["wrist"], dtype=float)))
        points, _, _ = triangulate(eyes, np.array(optics["stereo"], dtype=float), tolerance=.018)
        objects.append(points)
    rig = Decoder(checkpoint, np.array([0., 0., .23]))
    fitted, samples = motion_fit(rig, clips, np.concatenate(objects), observer=2)
    return dict(mean=fitted.mean.tolist(), scale=fitted.scale.tolist(), weights=fitted.weights.tolist(),
                samples=samples, blend=.35)


def assess(arena, snapshots, deck):
    rows = []
    for seed, regime, hard in deck:
        results = {label: checked_drop(arena, snapshot, seed, regime, hard) for label, snapshot in snapshots.items()}
        rows.append(dict(seed=seed, regime=regime, difficult=hard, results=results))
    return rows


def compare(rows):
    def info(deck):
        result = accept(deck, "0", "challenger")
        result["by_checkpoint"] = solo_summary(deck)
        return result
    return dict(all=info(rows), ordinary=info([r for r in rows if not r["difficult"]]),
                difficult=info([r for r in rows if r["difficult"]]))


def run():
    started = perf_counter()
    incumbent = load_checkpoint(RETAINED)
    post_table = _read(TABLE)
    if post_table["solo_checkpoint_sha256"] != _hash(RETAINED) or post_table["post_table_matches"] != 8:
        raise ValueError("The retained checkpoint has no matching post-table retention receipt")
    practice, gate, held = [make_deck(seed, count) for seed, count in ((SOURCE_SEED, COUNT), (GATE_SEED, GATE), (HELD_SEED, HELD))]
    earlier = _read(COURSE)
    promotion = _read(PROMOTION)
    used = [r["seed"] for r in earlier["practice"] + earlier["held"] + promotion["held"]]
    used.extend(r["seed"] for test in promotion["rounds"] for r in test["rows"])
    planned = [seed for seed, _, _ in practice + gate + held]
    if set(planned) & set(used) or len(set(planned)) != len(planned):
        raise ValueError("A release seed overlaps an earlier course")
    arena = stock(fixture="table-edge")
    arena.max_steps = PRACTICE_CAP
    learned = None
    rewards = np.zeros(len(BIASES))
    trials = np.zeros(len(BIASES), dtype=int)
    rows = []
    report = dict(schema="tt29-post-table-solo-continuation-1", incumbent_sha256=_hash(RETAINED),
                  preceding_table_sha256=_hash(TABLE), seeds=dict(practice=SOURCE_SEED, gate=GATE_SEED, held=HELD_SEED),
                  caps=dict(practice=PRACTICE_CAP, gate=VALIDATION_CAP, held=300),
                  practice=rows, choices_rad=BIASES.tolist())

    def best():
        return int(np.argmax((rewards + np.array([1., .6, .6, .6, .6])) / (trials + 1)))

    def snapshot(action):
        candidate = incumbent | dict(reference_bias_rad=BIASES[action].tolist())
        return candidate if learned is None else candidate | dict(continued_motion=learned)

    try:
        arena.reset(seed=practice[0][0], render=False)
        eyes = stationary(arena)
        for index, (seed, regime, hard) in enumerate(practice):
            action = index % len(BIASES) if index < 10 else best()
            row = checked_drop(arena, snapshot(action), seed, regime, hard, collect=True)
            trials[action] += 1
            rewards[action] += np.sqrt(row["reward"])
            rows.append(dict(seed=seed, regime=regime, difficult=hard, action=action, **row))
            if index + 1 in (5, 10, COUNT):
                learned = motion(rows, eyes, incumbent)
                report["candidate"] = snapshot(best())
                report["training"] = dict(attempts=trials.tolist(), sqrt_rewards=rewards.tolist(), chosen=best())
                _write(RESULT, report)
                print("continued drops", index + 1, "samples", learned["samples"], "candidate", best(), flush=True)
        arena.max_steps = VALIDATION_CAP
        candidates = {"0": incumbent, "challenger": report["candidate"]}
        report["validation"] = assess(arena, candidates, gate)
        report["validation_summary"] = compare(report["validation"])
        decision = report["validation_summary"]
        promoted = decision["all"]["promote"] and decision["difficult"]["qualified_rebound_change"] >= 0
        report["promoted"] = promoted
        chosen = candidates["challenger"] if promoted else candidates["0"]
        _write(DEPLOY, chosen)
        report["deployment_sha256"] = _hash(DEPLOY)
        _write(RESULT, report)
        print("gate", decision, "promoted", promoted, flush=True)
        arena.max_steps = 300
        report["held"] = assess(arena, candidates, held)
        report["held_summary"] = compare(report["held"])
        report["verified"] = len(rows) + 2 * (len(report["validation"]) + len(report["held"]))
        report["wall_s"] = perf_counter() - started
        _write(RESULT, report)
        print("held", report["held_summary"], flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    run()
