"""Measure which physical outcomes were observable to the outgoing-arc learner."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write


def coverage(source):
    pairs = []
    missed_ball = []
    for row in _read(source)["rows"]:
        events = row["events"]
        hits = [e for e in events if e["kind"] == "contact" and e["side"] == 1]
        observations = row["optical_outcomes"]["attempts"]
        for i, event in enumerate(hits):
            end = hits[i + 1]["time"] if i + 1 < len(hits) else float("inf")
            landed = any(e["kind"] == "legal_return" and e["side"] == 1 and event["time"] <= e["time"] < end
                         for e in events)
            # The native event clock is used only here, after the actor has finished.
            nearby = [o for o in observations if abs(o["predicted_strike_s"] - event["time"]) < .12]
            measured = any(o["measured"] is not None for o in nearby)
            pairs.append(dict(id=row["shot"]["id"], step=event["step"], landed=landed, observed=measured))
        last_incoming = max((e["time"] for e in events if e["kind"] == "legal_return" and e["side"] == -1), default=0.)
        if events[-1]["kind"] == "point" and events[-1]["winner"] == -1 and not any(h["time"] >= last_incoming for h in hits):
            missed_ball.append(row["shot"]["id"])
    return dict(source_sha256=_hash(source), contacts=pairs, no_contact_losses=missed_ball)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    _write(args.output, coverage(args.source))
