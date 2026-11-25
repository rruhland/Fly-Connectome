"""Audit one fixed ready pose against the clocks and joint motion of native strike witnesses."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_contacts import record
from .tt31_ready import centered
from .tt37_opponent import arena

ROOT = Path("artifacts")
OUT = ROOT / "tt43_stance"
RECORD = ROOT / "tt42/random-38"
LOW_FILES = ("long-low.json.gz", "long-low-flipped.json.gz", "long-low-alternate-strict.json.gz")
RATE = 4.16 # actual reference slew limit with .8 action clipping
DELAY = (0., 1/30, 2/30, .1, .15, .2)


def packet(events, c, **fields):
    """Use the last native outgoing, far strike, and table separation before this near strike."""
    prior = [e for e in events if e["time"] < c["time"]]
    bounce = next(e for e in reversed(prior)
                  if e["kind"] == "legal_return" and e.get("side") == -1)
    far = next(e for e in reversed(prior) if e.get("collider") == "rival_racket")
    table = next(e for e in prior if e["time"] > bounce["time"]
                 and e.get("collider") in ("table", "table_right"))
    clocks = dict(far=c["time"] - far["time"], rebound=c["time"] - table["time"])
    return dict(c, **fields, clocks=clocks, far=far, table=table)


def witnesses():
    d = _read(RECORD.with_suffix(".json"))
    tape = _read(RECORD.with_name(RECORD.name + "-tape.json.gz"))["rows"][0]
    if digest_json(tape["events"]) != d["event_sha256"]:
        raise ValueError("Record tape and contact verification differ")
    regular = [packet(tape["events"], c, id=d["card"]["id"], leg=i, group="regular")
               for i, c in enumerate(d["contacts"]) if c["landed"]]
    low = []
    verified = {}
    for name in ("long-low-diverse-audit.json", "long-low-alternate-strict-audit.json"):
        aud = _read(ROOT / "tt40" / name)
        verified.update({(r["origin"], r["id"]):r["events_sha256"] for r in aud["rows"]
                         if r["clearance"]["penetration_samples"] == 0})
    for filename in LOW_FILES:
        d = _read(ROOT / "tt40" / filename)
        with centered(d["ready"]):
            a = arena()
            try:
                for r in d["rows"]:
                    for win in r["wins"]:
                        key = (filename, r["id"])
                        if not win["answered"] or digest_json(win["events"]) != verified.get(key):
                            continue
                        a.launch(win["shot"], card=r["card"])
                        a.partner.reset(seed=r["card"]["seed"])
                        scored = [e for e in record(a, win, opening_side=1) if e["landed"]]
                        contact = scored[r["leg"]]
                        low.append(packet(win["events"], contact, id=r["id"], leg=r["leg"],
                                          group="low", source=filename))
            finally:
                a.close()
    if len(low) != 4 or len(regular) != 38:
        raise ValueError(f"Unexpected witness census {len(low)} low, {len(regular)} regular")
    return regular, low


def interval(q, times, latency, rate=RATE):
    """Necessary (not sufficient) all-witness box for a stationary pose at the cue."""
    t = np.asarray(times) - latency
    if np.min(t) < 0:
        return None
    lo = np.max(q - rate * t[:, None], axis=0)
    hi = np.min(q + rate * t[:, None], axis=0)
    return lo, hi


def minimax(q, times, rate=RATE):
    """Most slack in seconds before beginning any archived endpoint movement."""
    q = np.asarray(q)
    times = np.asarray(times)
    each = (times[:, None, None] + times[None, :, None]
            - np.abs(q[:, None, :] - q[None, :, :]) / rate) / 2.
    found = np.unravel_index(np.argmin(each), each.shape)
    slack = float(each[found])
    # The boxes remain algebraically defined when the time slack is negative.
    lo, hi = interval(q, times, slack, rate)
    return dict(slack_s=slack, obstructing_indices=list(map(int, found)),
                ready=((lo + hi) / 2.).tolist(), allowed_low=lo.tolist(), allowed_high=hi.tolist())


def clocks_summary(rows, home):
    q = np.array([r["joints_rad"] for r in rows])
    out = {}
    for cue in ("far", "rebound"):
        times = np.array([r["clocks"][cue] for r in rows])
        out[cue] = dict(minimax(q, times), actual_lead_range_s=[float(times.min()), float(times.max())],
                        from_stock_geometric=[int(np.sum(np.max(np.abs(q-home),axis=1) / RATE <= times-d+1e-10))
                                              for d in DELAY])
    return out


def analyze(output):
    regular, low = witnesses()
    boot = np.asarray(_read(RECORD.with_name(RECORD.name + "-tape.json.gz"))["ready"])
    negative = [r for r in low if r["face"] == "negative"]
    positive14843 = next(r for r in low if r["face"] == "positive")
    alternate = [r for r in negative if r["id"] != positive14843["id"]] + [positive14843]
    summaries = {k:clocks_summary(rows, boot[:5]) for k,rows in
                 (("regular",regular), ("low_negative",negative),
                  ("regular_plus_low_negative",regular+negative),
                  ("regular_plus_low_mixed",regular+alternate))}
    _write(output,dict(schema="tt43-native-witness-envelope-1", motor_radians_per_s=RATE,
                       delays_s=DELAY, boot=boot.tolist(), regular=regular, low=low,
                       summaries=summaries,
                       scope="Necessary max-joint reference travel for replayed strike endpoints, not a motor simulation or exhaustive IK bound."))
    for k,v in summaries.items():
        print(k, {cue:(round(t['slack_s'],3), t['obstructing_indices'], np.round(t['ready'],2).tolist(),
                            t['from_stock_geometric']) for cue,t in v.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, default=OUT / "witness-envelope.json.gz", nargs="?")
    analyze(p.parse_args().out)
