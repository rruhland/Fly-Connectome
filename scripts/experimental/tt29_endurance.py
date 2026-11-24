"""Score one continuous point from solver contacts, separations and legal landings."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read


def summarize(values):
    if not values:
        return None
    return dict(
        zip(
            ("p10", "median", "p90"),
            np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
            strict=True,
        )
    )


def audit(events):
    """Exclude the launch. Pair each real racket collision with its subsequent legal bounce."""
    rackets = {"learner_racket": 1, "rival_racket": -1}
    shots = []
    pending = None
    last_time = -1.0
    finished = False
    previous_contact = None
    opening_counted = False
    illegal_touch = False
    for event in events:
        t = float(event["time"])
        if finished and event["kind"] == "separation" and abs(t - last_time) < 1e-9:
            continue
        if t < last_time - 1e-9 or finished:
            raise ValueError("Time reversal or an event following point termination")
        last_time = t
        kind = event["kind"]
        if kind == "contact":
            if pending is not None:
                # A second touch without a legal landing cannot extend the rally.
                illegal_touch = True
            side = event["side"]
            pending = dict(
                side=side,
                hit=t,
                incoming_mps=float(np.linalg.norm(event["incoming"])),
                contact_xyz=event["location"],
                interval_s=None if previous_contact is None else t - previous_contact,
            )
            previous_contact = t
        elif (
            kind == "separation"
            and event.get("collider") in rackets
            and pending is not None
        ):
            if pending["side"] != rackets[event["collider"]]:
                raise ValueError("Racket identity changed across contact")
            pending["outgoing_mps"] = float(np.linalg.norm(event["velocity"]))
            pending["spin_rad_s"] = float(np.linalg.norm(event["spin"]))
        elif kind == "legal_return":
            if (
                pending is None
                and not shots
                and not opening_counted
                and event["side"] == -1
            ):
                opening_counted = True
                continue  # ReferenceReceiver counts the physical landing of the recorded opening.
            if (
                illegal_touch
                or pending is None
                or pending["side"] != event["side"]
                or "outgoing_mps" not in pending
            ):
                raise ValueError(
                    "Legal landing lacks a matching physical racket separation"
                )
            if shots and shots[-1]["side"] == pending["side"]:
                raise ValueError("Nonalternating returns")
            shots.append(dict(pending, landing=t, landing_xyz=event["location"]))
            pending = None
        elif kind in ("point", "timeout", "let"):
            finished = True
    return dict(
        returns=len(shots),
        learner=sum(r["side"] == 1 for r in shots),
        opponent=sum(r["side"] == -1 for r in shots),
        outgoing_mps=summarize([r["outgoing_mps"] for r in shots]),
        learner_outgoing_mps=summarize(
            [r["outgoing_mps"] for r in shots if r["side"] == 1]
        ),
        opponent_outgoing_mps=summarize(
            [r["outgoing_mps"] for r in shots if r["side"] == -1]
        ),
        interval_s=summarize(
            [r["interval_s"] for r in shots if r["interval_s"] is not None]
        ),
        shots=shots,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    doc = _read(args.file)
    for row in doc if isinstance(doc, list) else doc["rows"]:
        report = audit(
            row.get("result", row)["events"] if "events" not in row else row["events"]
        )
        print({k: v for k, v in report.items() if k != "shots"})
