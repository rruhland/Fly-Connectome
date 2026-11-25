"""Predict the next observed flight from the opponent's pre-contact stereo appearance."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt44_paired_credit import State, prefix
from .tt4_transfer_trial import _read, _write

ROOT = Path("artifacts/tt50_opponent_vision")
SRC = course.ROOT / "learning-48.json.gz"
# Generous fixed-image windows surrounding the end of the table in the two
# calibrated cameras. No arena pose, segmentation labels or depth buffer.
WINDOWS = {"left": (108, 204), "right": (171, 267)}
V0, V1 = 64, 124


def crop(images):
    return np.stack([images[n][V0:V1, l:r] for n, (l, r) in WINDOWS.items()])


def roster(source):
    entries = []
    seen = set()
    for i, (card, row) in enumerate(zip(source["cards"], source["rows"])):
        offers = row["placements"]["delivered"]
        if card["id"] in seen or len(offers) < 2 or offers[1]["far_contact"] is None:
            continue
        seen.add(card["id"])
        entries.append(i)
    rng = np.random.default_rng(505037)
    rng.shuffle(entries)
    return entries[:24]


def collect(indices, special=None):
    d = _read(SRC)
    root = ROOT / ("observations" if special is None else "difficult")
    sources = {str(SRC): d}
    root.mkdir(parents=True, exist_ok=True)
    roster_indices = roster(d)
    with mpc.centered(d["ready"]):
        a = mpc.arena()
        image, rig, *_ = mpc.resources(a)
        try:
            for j in indices:
                if j >= (len(roster_indices) if special is None else len(special)):
                    continue
                out = root / f"case-{j:02d}.npz"
                if out.exists():
                    continue
                if special is None:
                    i, leg, source = roster_indices[j], 1, str(SRC)
                else:
                    c = special[j]
                    i, leg, source = c["index"], c["leg"], c["source"]
                if source not in sources:
                    sources[source] = _read(Path(source))
                doc = sources[source]
                if doc["ready"] != d["ready"]:
                    raise ValueError("Different native arm initialization")
                card, row = doc["cards"][i], doc["rows"][i]
                request = row["placements"]["requested"][leg]
                anchor = int(np.floor(request["time_s"] / mpc.DT + 1e-7))
                warm = prefix(a, card, row, anchor, image, a._arm_ids[1][2], window=20)
                state = State(a)
                if len(a.partner.requests) != leg:
                    raise ValueError("Fork must precede the fresh reply")
                observer = mpc.Vision(rig, a.control_dt, raw=True)
                observer.eye = mpc.LearnedOptics(observer.eye, course.OPTIC)
                for t, pixels, q, _ in warm[-20:]:
                    mpc.observed(observer, t, pixels, q)
                seeds = np.random.default_rng([5050, j]).integers(1, 2**30, size=6)
                if special is not None:
                    seeds = np.r_[
                        card["reply_seed"],
                        np.random.default_rng([51137, j]).integers(1, 2**30, size=5),
                    ]
                ticks, streams, pixels, balls, contacts, labels = [], [], [], [], [], []
                for b, seed in enumerate(seeds):
                    state.restore(a)
                    a.card = dict(card, reply_seed=int(seed))
                    eye = deepcopy(observer)
                    hit = None
                    label = None
                    old = None
                    for t in range(anchor, anchor + 175):
                        frames = image.capture() if t % 2 == 0 else None
                        x = mpc.observed(eye, t, frames, a.joints)
                        if hit is None and frames is not None:
                            ticks.append(t)
                            streams.append(b)
                            pixels.append(crop(frames))
                            balls.append(x if x is not None else np.full(6, np.nan))
                        if hit is not None and x is not None and x[4] < -0.9:
                            if old is not None and old[1] > -1.10 >= x[1]:
                                label = old + (x - old) * (
                                    (-1.10 - old[1]) / (x[1] - old[1])
                                )
                                break
                            old = x
                        u = (
                            row["actions"][t]
                            if t < len(row["actions"])
                            else np.zeros(6)
                        )
                        _, _, done, cap, info = a.exchange(u, mpc.FAR_COMMAND)
                        for e in info["events"]:
                            if (
                                e["kind"] == "contact"
                                and e.get("side") == -1
                                and hit is None
                            ):
                                hit = e["time"] / mpc.DT
                        if done or cap:
                            break
                    contacts.append(np.nan if hit is None else hit)
                    labels.append(np.full(6, np.nan) if label is None else label)
                tmp = out.with_suffix(".tmp.npz")
                np.savez_compressed(
                    tmp,
                    id=card["id"],
                    index=i,
                    leg=leg,
                    source=source,
                    seeds=seeds,
                    anchor=anchor,
                    tick=ticks,
                    stream=streams,
                    rgb=np.asarray(pixels),
                    ball=balls,
                    contact=contacts,
                    label=labels,
                )
                tmp.replace(out)
                print(
                    j,
                    card["id"],
                    np.isfinite(np.asarray(labels)[:, 0]).sum(),
                    flush=True,
                )
        finally:
            image.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("start", type=int)
    p.add_argument("stride", type=int)
    args = p.parse_args()
    collect(range(args.start, 24, args.stride))
