"""Summarize an outgoing ranker comparison and replay its longest changed point."""
import argparse
from pathlib import Path

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance, guarded, verify_clearance
from .tt39_opportunity_map import projected_height
from .tt43_outgoing_coverage import ROOT


def stats(rows):
    projected = [
        projected_height(p["receive_line"])
        for r in rows for p in r["placements"]["delivered"][1:]
        if p["near_bounce"] and p["receive_line"]
    ]
    scores = [r["score"]["learner"] for r in rows]
    return dict(own=sum(scores), max=max(scores, default=0),
                opening=sum(bool(r["placements"]["delivered"] and
                                 r["placements"]["delivered"][0]["near_landed"]) for r in rows),
                low=sum(h < .99 for h in projected), tracked_later=len(projected))


def verify(doc, index):
    card, row = doc["cards"][index], doc["rows"][index]
    with mpc.centered(doc["ready"]):
        arena, cad = mpc.arena(), mpc.arena(detail="cad")
        try:
            clearance = CourtClearance(cad.model, articulated=True)
            arena.launch(row["shot"], card=card)
            arena.partner.reset(seed=card["seed"])
            guarded(arena, clearance)
            for action in row["actions"]:
                arena.exchange(action, mpc.FAR_COMMAND)
            verify_clearance(arena)
            if any(digest_json(a) != digest_json(b) for a, b in [
                (arena.events, row["events"]),
                (arena.point_placements(), row["placements"]),
                (mpc.audit(arena.events, opening_side=1), row["score"]),
            ]):
                raise RuntimeError("Independent action-only replay differs")
            return dict(id=card["id"], real_legal=row["score"]["learner"],
                        opponent_legal=row["score"]["opponent"], launches=arena.launches,
                        actions=len(row["actions"]), simulation_s=float(arena.data.time),
                        event_sha256=digest_json(arena.events), clearance=arena.clearance_result(),
                        motor=arena.motor_audit(), placements=len(row["placements"]["requested"]))
        finally:
            cad.close()
            arena.close()


def report(args):
    doc, control = _read(args.live), _read(args.control)
    if _hash(args.control) != doc["baseline_sha256"] or doc["cards"] != control["cards"][:len(doc["cards"])]:
        raise ValueError("Incorrect archived stock branch")
    rows = doc["rows"]
    stock = control["rows"][:len(rows)]
    changed = [i for i, r in enumerate(rows) if r["first_changed_action"] is not None]
    r = dict(complete=doc["complete"], points=len(rows), assigned=len(doc["cards"]),
             baseline_sha256=_hash(args.control), candidate_sha256=_hash(args.live),
             stock=stats(stock), candidate=stats(rows), changed=len(changed),
             changed_assignments=[
                 dict(index=i, id=doc["cards"][i]["id"],
                      first_changed_action=rows[i]["first_changed_action"],
                      stock=stock[i]["score"]["learner"], new=rows[i]["score"]["learner"],
                      reranks=len(rows[i]["reranks"]))
                 for i in changed
             ])
    if args.verify and changed:
        longest = max(changed, key=lambda i: rows[i]["score"]["learner"])
        r["longest_changed_cad_replay"] = verify(doc, longest)
    _write(args.output, r)
    print(r, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, default=ROOT / "outgoing-coverage-live.json.gz")
    parser.add_argument("--control", type=Path, default=course.ROOT / "no-more-practice.json.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "outgoing-coverage-report.json")
    parser.add_argument("--verify", action="store_true")
    report(parser.parse_args())
