"""Measure actual score stability along the motor change that separated matched hit and miss."""

from pathlib import Path
import numpy as np
from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt44_paired_credit import State, prefix
from .tt52_matched_movements import ROOT
from .tt4_transfer_trial import _hash, _read, _write


def motions(r, tick, n=110):
    u = np.asarray(r["actions"][tick : tick + n])
    return np.pad(u, ((0, n - len(u)), (0, 0)))


def run():
    table = ROOT / "pairs.json.gz"
    plan = _read(table)
    output = ROOT / "lines.json.gz"
    receipt = {"source": _hash(table), "cases": []}
    cache = {}
    ready = None
    for case in plan["pairs"]:
        coords = case["rows"]
        # Select before seeing interpolated outcomes. Both endpoints have an
        # identical physical prefix, but their issued joint continuations differ.
        w, l = max(
            ((w, l) for w in coords for l in coords if w["reward"] > l["reward"]),
            key=lambda pair: np.linalg.norm(
                np.asarray(pair[0]["first_actions"]) - pair[1]["first_actions"]
            ),
        )
        for entry in (w, l):
            p = Path(entry["source"])
            if p not in cache:
                if _hash(p) != plan["sources"][str(p)]:
                    raise ValueError("Changed source")
                cache[p] = _read(p)
        dw, dl = cache[Path(w["source"])], cache[Path(l["source"])]
        r, loser = dw["rows"][w["row"]], dl["rows"][l["row"]]
        card = dw["cards"][w["row"]]
        t = case["tick"]
        if r["actions"][:t] != loser["actions"][:t] or dw["ready"] != dl["ready"]:
            raise ValueError("Different causal prefix")
        if ready != dw["ready"]:
            if ready is not None:
                im.close()
                a.close()
                ctx.__exit__(None, None, None)
            ready = dw["ready"]
            ctx = mpc.centered(ready)
            ctx.__enter__()
            a = mpc.arena()
            im, rig, *_ = mpc.resources(a)
        readings = prefix(a, card, r, t, im, a._arm_ids[1][2], window=26)
        snap = State(a)
        eye = mpc.Vision(rig, a.control_dt, raw=True)
        eye.eye = mpc.LearnedOptics(eye.eye, course.OPTIC)
        xx = None
        for k, frame, q, ref in readings[-26:]:
            xx = mpc.observed(eye, k, frame, q)
        q = a.joints.copy()
        ref = a.data.ctrl[a._arm_ids[1][2]].copy()
        xx = mpc.observed(eye, t, im.capture() if not t % 2 else None, q)
        win, lose = motions(r, t), motions(loser, t)
        probes = []
        for mix in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
            snap.restore(a)
            events = []
            cmd = (
                lose + mix * (win - lose)
                if mix not in (0.0, 1.0)
                else (win if mix else lose)
            )
            success = False
            for u in cmd:
                _, _, done, cap, info = a.exchange(u, mpc.FAR_COMMAND)
                events.extend(info["events"])
                success = any(
                    e["kind"] == "legal_return" and e["side"] == 1
                    for e in info["events"]
                )
                if success or done or cap:
                    break
            if mix in (0.0, 1.0) and success != bool(mix):
                raise ValueError("Could not reproduce the physical endpoint")
            hit = next(
                (e for e in events if e["kind"] == "contact" and e.get("side") == 1),
                None,
            )
            probes.append(
                dict(
                    mix=mix,
                    reward=int(success),
                    hit=hit is not None,
                    contact_tick=None if hit is None else hit["time"] * 60,
                )
            )
        receipt["cases"].append(
            dict(
                id=case["id"],
                prefix=case["prefix"],
                tick=t,
                observed=None if xx is None else xx.tolist(),
                q=q.tolist(),
                ref=ref.tolist(),
                win=win[:21].tolist(),
                loss=lose[:21].tolist(),
                probes=probes,
            )
        )
        _write(output, receipt)
        print(case["id"], t, [p["reward"] for p in probes], flush=True)
    if ready is not None:
        im.close()
        a.close()
        ctx.__exit__(None, None, None)
    receipt["complete"] = True
    _write(output, receipt)


if __name__ == "__main__":
    run()
