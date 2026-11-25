"""Check whether nearby command timing and magnitude distinguish a scored joint movement from a miss."""

from pathlib import Path
from unittest.mock import patch
import numpy as np

from . import tt38_joint_mpc as mpc
from .tt44_paired_credit import prefix, warm
from .tt52_matched_movements import ROOT
from .tt4_transfer_trial import _read, _write


def variants(goal, reference):
    # Small timing and amplitude errors in five actuated joints. Each sequence
    # is a bounded ref path in the same coordinate system as the native search.
    goal = np.asarray(goal)
    t = np.arange(len(goal))
    augmented = np.vstack((reference, goal))
    for lag in (-1.0, -0.5, 0.5, 1.0):
        at = np.clip(t + 1 + lag, 0, len(goal))
        yield np.column_stack(
            [np.interp(at, np.arange(len(augmented)), u) for u in augmented.T]
        )
    for s in (-0.065, 0.065):
        yield reference + (1.0 + s) * (goal - reference)


def check():
    data = _read(ROOT / "pairs.json.gz")
    cache = {}
    receipts = []
    ready = None
    for c in data["pairs"]:
        scored, failed = max(
            ((w, l) for w in c["rows"] for l in c["rows"] if w["reward"] > l["reward"]),
            key=lambda pair: np.linalg.norm(
                np.asarray(pair[0]["first_actions"]) - pair[1]["first_actions"]
            ),
        )
        for m in (scored, failed):
            cache.setdefault(m["source"], None)
            if cache[m["source"]] is None:
                cache[m["source"]] = _read(Path(m["source"]))
        dw = cache[scored["source"]]
        r = dw["rows"][scored["row"]]
        if ready != dw["ready"]:
            if ready is not None:
                actor.close()
                im.close()
                arena.close()
                ctx.__exit__(None, None, None)
            ready = dw["ready"]
            ctx = mpc.centered(ready)
            ctx.__enter__()
            arena = mpc.arena()
            im, rig, stride, lo, hi = mpc.resources(arena)
            with patch.object(mpc, "Fragments", lambda *a: None):
                actor = mpc.MPC(mpc.BANK, None, variant="elastic")
        t = c["tick"]
        card = dw["cards"][scored["row"]]
        history = prefix(arena, card, r, t, im, arena._arm_ids[1][2], window=40)
        eye = warm(arena, rig, actor, history)
        ref = arena.data.ctrl[arena._arm_ids[1][2]].copy()
        q = arena.joints.copy()
        mpc.observed(eye, t, im.capture() if t % 2 == 0 else None, q)
        actor.engine.observe(t, eye.eye, q, ref)
        if not actor.engine.sync():
            continue
        pair = []
        for m in (scored, failed):
            row = cache[m["source"]]["rows"][m["row"]]
            start = ref.copy()
            goals = []
            for u in row["actions"][t : t + mpc.HORIZON]:
                start = np.clip(start + stride * np.asarray(u), lo, hi)
                goals.append(start)
            original = mpc.tail(np.asarray(goals), mpc.HORIZON)
            options = [original, *variants(original, ref)]
            loss, ok, *_ = actor.trial(options, long=True)
            state_votes, quality = actor.uncertainty(options, loss)
            pair.append(
                dict(
                    reward=m["reward"],
                    nominal_legal=bool(ok[0]),
                    nominal_quality=float(quality[0]),
                    nominal_state_votes=int(state_votes[0]),
                    nearby_nominal=int(ok[1:].sum()),
                    nearby_state_votes=int(state_votes[1:].sum()),
                    nearby_mean_cost=float(np.mean(quality[1:])),
                    phase_legal=ok[1:].astype(int).tolist(),
                )
            )
        receipts.append(dict(id=c["id"], tick=t, rows=pair))
        _write(ROOT / "camera-margin.json", dict(cases=receipts))
    if ready is not None:
        actor.close()
        im.close()
        arena.close()
        ctx.__exit__(None, None, None)
    print(len(receipts), flush=True)


if __name__ == "__main__":
    check()
