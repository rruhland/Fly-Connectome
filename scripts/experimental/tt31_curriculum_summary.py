"""Verify matched practice exposures and compare held live game cards."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_curriculum import ERRORS, MOVEMENTS, OUTPUT, split_cards
from .tt31_curriculum_opponent import lane


def by_side(replies):
    groups = {}
    for side in ("left", "middle", "right", None):
        observed = [r for r in replies if r["observed_lane"] == side]
        groups[side or "unknown"] = dict(delivered=len(observed),
                                         contacts=sum(r["near_contact"] is not None for r in observed),
                                         legal=sum(r["near_landed"] for r in observed))
    return groups


def placement_summary(rows):
    replies = [r for row in rows for r in row["placements"]["delivered"]]
    observed = [r for r in replies if r["observed_lane"] is not None]
    measured = [r for r in observed if r["receive_line"] is not None]
    x_positions = [r["receive_line"]["xyz"][0] for r in measured]
    errors = [r["receive_line"]["xyz"][0] - r["target"][0] for r in measured]
    targets = [tuple(request["target"]) for row in rows for request in row["placements"]["requested"]]
    return dict(points=len(rows), near_legal=sum(r["score"]["learner"] for r in rows),
                alternating_legal=sum(r["score"]["returns"] for r in rows),
                requested=len(targets), distinct_requested=len(set(targets)),
                by_actual_side=by_side(replies), actual_side_matches_goal=sum(
                    lane(r["target"][0]) == r["observed_lane"] for r in observed),
                actual_side_measured=len(observed), reference_line_samples=len(measured),
                receive_x_range_m=[min(x_positions), max(x_positions)] if x_positions else None,
                receive_x_vs_placement_x_mean_absolute_m=float(np.mean(np.abs(errors))) if errors else None,
                cpu_s=sum(r["process_cpu_s"] for r in rows))


def read_run(directory, deck, deck_path, phase, order):
    path = directory / f"course-{phase}-{order}.json.gz"
    receipt = _read(path)
    expected = split_cards(deck, order, phase)
    if receipt["plan_sha256"] != _hash(deck_path) or receipt["cards"] != expected:
        raise ValueError(f"Course roster changed in {path.name}")
    if len(receipt["rows"]) != len(expected):
        raise ValueError(f"Incomplete course in {path.name}")
    for row, entry in zip(receipt["rows"], expected, strict=True):
        if row["opponent_seed"] != entry["seed"] or row["shot"]["id"] != entry["id"]:
            raise ValueError(f"Unexpected launch or placement seed in {path.name}")
        if row["placements"]["card"] != entry:
            raise ValueError(f"Unexpected placement card in {path.name}")
        score = audit(row["events"], opening_side=1)
        if {key: value for key, value in score.items() if key != "shots"} != row["score"]:
            raise ValueError(f"Changed referee score in {path.name}")
        if len(row["placements"]["delivered"]) != score["opponent"]:
            raise ValueError(f"An opposing legal shot went uncounted in {path.name}")
        if sum(r["near_landed"] for r in row["placements"]["delivered"]) != score["learner"]:
            raise ValueError(f"A learner legal shot went uncounted in {path.name}")
        if phase == "held" and row["initial_examples"] != row["final_examples"]:
            raise ValueError(f"Evaluation updated the saved movement bank in {path.name}")
    return receipt, _hash(path)


def first_deliveries(rows):
    replies = {}
    for row in rows:
        card = row["placements"]["card"]
        replies[card["key"]] = next(iter(row["placements"]["delivered"]), None)
    return replies


def compare_delivery(left, right):
    if set(left) != set(right):
        raise ValueError("The arm policies did not receive the same card inventory")
    fields = ("target", "far_contact", "near_bounce", "receive_line")
    matches = Counter({key: 0 for key in ("both_delivered", *fields)})
    for key, a in left.items():
        b = right[key]
        if a is not None and b is not None:
            matches["both_delivered"] += 1
            for field in fields:
                matches[field] += int(digest_json(a[field]) == digest_json(b[field]))
    return dict(cards=len(left), **matches)


def held_difference(left, right):
    result = {}
    for score in ("learner", "returns"):
        delta = [a["score"][score] - b["score"][score] for a, b in zip(left, right, strict=True)]
        result[score] = dict(total=sum(delta), longer=sum(d > 0 for d in delta),
                             tied=sum(d == 0 for d in delta), shorter=sum(d < 0 for d in delta))
    return result


def run(directory):
    deck_path = directory / "course-plan.json.gz"
    deck = _read(deck_path)
    inputs = {"plan": _hash(deck_path), "original_movement": _hash(MOVEMENTS), "original_error": _hash(ERRORS)}
    practice = {}
    held = {}
    for order in ("staged", "mixed", "original"):
        if order != "original":
            practice[order], inputs[f"practice_{order}"] = read_run(directory, deck, deck_path, "practice", order)
            inputs[f"checkpoint_{order}"] = _hash(directory / f"course-{order}-checkpoint.json.gz")
            if inputs[f"checkpoint_{order}"] != practice[order]["checkpoint_sha256"]:
                raise ValueError(f"Incomplete final checkpoint for {order}")
        held[order], inputs[f"held_{order}"] = read_run(directory, deck, deck_path, "held", order)
        initial = inputs["original_movement"] if order == "original" else inputs[f"checkpoint_{order}"]
        error = inputs["original_error"] if order == "original" else initial
        if held[order]["initial_movement_sha256"] != initial or held[order]["initial_error_sha256"] != error:
            raise ValueError(f"Assessment loaded a different checkpoint for {order}")
    practice_rows = {order: doc["rows"] for order, doc in practice.items()}
    practice_match = compare_delivery(*(first_deliveries(rows) for rows in practice_rows.values()))
    if practice_match["both_delivered"] != practice_match["cards"]:
        raise ValueError("Practice differed in actual near opportunities")
    held_rows = {order: doc["rows"] for order, doc in held.items()}
    opening_replies = first_deliveries(held_rows["original"])
    held_openings = [dict(id=item["id"], seed=item["seed"],
                          actual_first_side=(opening_replies[item["key"]] or {}).get("observed_lane"),
                          score={order: dict(arm=rows[i]["score"]["learner"], both=rows[i]["score"]["returns"])
                                 for order, rows in held_rows.items()})
                     for i, item in enumerate(deck["held"])]
    pairs = (("staged", "original"), ("mixed", "original"), ("staged", "mixed"))
    _write(directory / "course-summary.json", dict(
        schema="tt31-course-summary-1", input_sha256=inputs,
        practice_first_opportunity_match=practice_match,
        practice={order: dict(total=placement_summary(rows),
                              mode={mode: placement_summary([r for r in rows if r["placements"]["card"]["mode"] == mode])
                                    for mode in deck["practice_counts"]},
                              checkpoint_size=dict(initial=rows[0]["initial_examples"],
                                                   final=rows[-1]["final_examples"],
                                                   optical=rows[-1]["final_optical_samples"],
                                                   censored=rows[-1]["final_censored_samples"]))
                  for order, rows in practice_rows.items()},
        held={order: placement_summary(rows) for order, rows in held_rows.items()},
        held_first_opportunity_matches={f"{a}_vs_{b}": compare_delivery(first_deliveries(held_rows[a]),
                                                                      first_deliveries(held_rows[b]))
                                        for a, b in pairs},
        held_paired={f"{a}_vs_{b}": held_difference(held_rows[a], held_rows[b]) for a, b in pairs},
        held_openings=held_openings,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
