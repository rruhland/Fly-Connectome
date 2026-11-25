"""Offline integrity checks shared by the TT42 feedback comparisons."""

from collections import deque
from types import SimpleNamespace

from .tt4_transfer_trial import _hash
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_opponent import draw
from .tt38_joint_mpc import DT, Fragments
from .tt41_early_optics_carry import resume


def verify_checkpoint(tape, rows, sha256, *, capacity=768):
    """Use the original restorer without constructing a live actor or renderer."""
    priors = SimpleNamespace(online=deque(maxlen=capacity))
    priors.digest = lambda: Fragments.digest(priors)
    model = SimpleNamespace(priors=priors)
    resume(model, tape, dict(rows=rows, checkpoint_sha256=sha256))
    return dict(sha256=_hash(tape), serial=model.serial, **model.priors.digest())


def verify_row(card, row):
    """Recompute the physical score and the independent native-opponent draws."""
    if (row["placements"]["card"] != card or row["opponent_seed"] != card["seed"]
            or audit(row["events"], opening_side=1) != row["score"]):
        raise ValueError(f"Physical assignment or referee score changed for {card['id']}")
    for leg, actual in enumerate(row["placements"]["requested"]):
        expected, _ = draw(card, leg, "random", None)
        if list(expected) != actual["target"]:
            raise ValueError(f"Requested opponent target changed for {card['id']} leg {leg}")


def action_prefix(before, after):
    a, b = before["actions"], after["actions"]
    fork = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if fork is None and len(a) != len(b):
        fork = min(len(a), len(b))
    end = len(a) if fork is None else fork
    prefix = [[e for e in row["events"] if e["step"] <= end] for row in (before, after)]
    if digest_json(prefix[0]) != digest_json(prefix[1]):
        raise ValueError("Physical events changed before the first different issued command")
    return fork, prefix[0]


def _verdict(row, time):
    previous = None
    for offer in row["placements"]["delivered"]:
        far = offer["far_contact"]
        if far is None:
            continue
        if far["time"] > time + 1e-7:
            return offer
        previous = offer
        near = offer["near_contact"]
        if near is None or near["time"] > time:
            return offer
    return previous


def offer_contrast(before, after, fork):
    """Identify native offer equality and local return outcome near the command fork."""
    if fork is None:
        return None
    a, b = (_verdict(r, fork * DT) for r in (before, after))
    if a is None or b is None:
        return None
    matched = all(digest_json(a[k]) == digest_json(b[k]) for k in
                  ("far_contact", "near_bounce", "target", "receive_line"))
    r = dict(same_incoming=matched, leg=[p["leg"] for p in (a, b)],
             touches=[p["near_contact"] is not None for p in (a, b)],
             landed=[p["near_landed"] for p in (a, b)])
    if matched:
        r["native_receive_line"] = a["receive_line"]
        r["native_near_shot"] = []
        for row, offer in ((before, a), (after, b)):
            near = offer["near_contact"]
            r["native_near_shot"].append(next((s for s in row["score"]["shots"]
                                                if near and s["side"] == 1 and s["hit"] == near["time"]), None))
    return r
