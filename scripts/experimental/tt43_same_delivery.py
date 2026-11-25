"""On a fixed sample of the record rally's deliveries, remove only the experience earned in this point."""

import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt40_within_rally import replay_memory
from .tt41_early_optics_carry import PRACTICE, resume, save

SOURCE = course.ROOT / "learning-24.json.gz"
BOUNDARY = course.ROOT / "learning-12.json.gz"
INITIAL = course.ROOT / "before-game-20.npz"
IDENT = 14000
# Early, middle, late and the terminal ball. Report all sampled deliveries.
ORDINALS = (2, 4, 8, 16, 24, 32, 38, 39)


class Ablated(Carried):
    def keep(self, archive, leg):
        count = super().keep(archive, leg)
        if leg[-1] == self.before_reward:
            self.online.clear()
            self.online.extend(self.original)
            self.swap_tick = archive[-1][0]
        return count


def run(out, feeds):
    source, boundary = _read(SOURCE), _read(BOUNDARY)
    i = next(j for j, c in enumerate(source["cards"]) if c["id"] == IDENT)
    row, card = source["rows"][i], source["cards"][i]
    ready = source["ready"]
    if row["score"]["learner"] != 38 or (source["plan_sha256"] != _hash(course.PLAN)):
        raise ValueError("Witness mismatch")
    with np.load(mpc.BANK) as f:
        raw = mpc.DeepField(f, "local")
    roster = {c["id"] for c in source["cards"]}
    with mpc.centered(ready):
        a = mpc.arena()
        tools = mpc.resources(a)
        model = mpc.MPC(mpc.BANK, raw, variant="elastic", budget=28)
        try:
            model.priors = Ablated(
                mpc.BANK, model.engine, practice=[PRACTICE], held=roster
            )
            model.priors.before_reward = -1
            if not INITIAL.exists():
                resume(model, BOUNDARY.with_suffix("").with_suffix(".npz"), boundary)
                for j in range(len(boundary["rows"]), i):
                    model.serial += 1
                    replay_memory(
                        a,
                        tools,
                        model.priors,
                        source["cards"][j],
                        source["rows"][j],
                        ready,
                        course.OPTIC,
                        j + 1,
                    )
                if model.priors.digest() != row["memory"]["before"]:
                    raise ValueError("Point entry memory mismatch")
                save(model, INITIAL)
            receipt = (
                _read(out)
                if out.exists()
                else {
                    "witness": _hash(SOURCE),
                    "entry_checkpoint": _hash(INITIAL),
                    "ordinals": list(ORDINALS),
                    "rows": [],
                }
            )
            if (
                receipt["witness"] != _hash(SOURCE)
                or receipt["entry_checkpoint"] != _hash(INITIAL)
                or receipt["ordinals"] != list(ORDINALS)
            ):
                raise ValueError("Incompatible partial paired intervention")
            before = {
                **source,
                "rows": source["rows"][:i],
                "checkpoint_sha256": _hash(INITIAL),
            }
            shots = mpc.measured_launches(feeds)
            for k in ORDINALS[len(receipt["rows"]) :]:
                resume(model, INITIAL, before)
                if model.priors.digest() != row["memory"]["before"]:
                    raise ValueError("Initial memory mismatch")
                model.priors.original = list(model.priors.online)
                model.priors.swap_tick = None
                model.priors.before_reward = k - 1
                changed = mpc.play(
                    a,
                    *tools,
                    ready,
                    card,
                    shots[IDENT],
                    raw,
                    model,
                    course.OPTIC,
                    "live",
                    2 * k,
                )
                tick = model.priors.swap_tick
                if (
                    tick is None
                    or changed["actions"][: tick + 1] != row["actions"][: tick + 1]
                ):
                    raise ValueError(
                        "Changed motion before the intended reward boundary"
                    )
                baseline = row["placements"]["delivered"][k - 1]
                deliveries = changed["placements"]["delivered"]
                measured = deliveries[k - 1] if len(deliveries) >= k else None
                if measured is None or digest_json(
                    measured["far_contact"]
                ) != digest_json(baseline["far_contact"]):
                    raise ValueError("Different delivery after the intervention")
                # First command divergence can be later than this incoming ball.
                compare = zip(changed["actions"], row["actions"])
                first = next((j for j, (x, y) in enumerate(compare) if x != y), None)
                receipt["rows"].append(
                    {
                        "ordinal": k,
                        "removed_at_tick": tick,
                        "first_altered_command": first,
                        "identical_far_contact": True,
                        "same_near_bounce": digest_json(measured["near_bounce"])
                        == digest_json(baseline["near_bounce"]),
                        "live_returned": baseline["near_landed"],
                        "restored_entry_returned": measured["near_landed"],
                        "live_landing": row["score"]["shots"][2 * k - 1]["landing_xyz"]
                        if baseline["near_landed"]
                        else None,
                        "ablated_score": changed["score"]["learner"],
                        "post_actions": changed["actions"][tick + 1 :],
                        "post_events": [
                            e
                            for e in changed["events"]
                            if e["time"] > (tick + 1) * mpc.DT + 1e-9
                        ],
                    }
                )
                _write(out, receipt)
                print(
                    k,
                    measured["near_landed"],
                    baseline["near_landed"],
                    first,
                    flush=True,
                )
        finally:
            model.close()
            tools[0].close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    args = p.parse_args()
    run(args.out, args.feeds)
