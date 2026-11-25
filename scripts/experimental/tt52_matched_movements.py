"""Find direct joint continuations that physically diverged from a bit-identical action prefix."""

from collections import defaultdict
from hashlib import blake2b
from pathlib import Path

import numpy as np

from . import tt49_sustain_value as value
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _read, _write, _hash

ROOT = Path("artifacts/tt52_matched_movements")


def collect():
    reserved = {
        c["id"] for k in ("practice", "assessment") for c in _read(season.PLAN)[k]
    }
    groups = defaultdict(dict)
    seen = set()
    sources = {}
    for path in value.sources():
        if not path.exists():
            continue
        doc = _read(path)
        if not isinstance(doc, dict) or "cards" not in doc or not doc.get("rows"):
            continue
        used = False
        for idx, (c, r) in enumerate(zip(doc["cards"], doc["rows"])):
            if c["id"] in reserved or not r.get("searches") or not r.get("actions"):
                continue
            offers = r.get("placements", {}).get("delivered", [])
            if not offers or any(not o.get("far_contact") for o in offers):
                continue
            actions = np.ascontiguousarray(r["actions"], dtype="<f8")
            course_key = (
                c["id"],
                c["seed"],
                c["reply_seed"],
                blake2b(actions, digest_size=16).digest(),
            )
            if course_key in seen:
                continue
            seen.add(course_key)
            ticks = [(n["tick"], n) for n in r["searches"] if "p" in n]
            start = 0
            hist = blake2b(digest_size=16)
            far = np.array([o["far_contact"]["time"] * 60 for o in offers])
            for t, n in ticks:
                hist.update(actions[start:t].tobytes())
                start = t
                leg = int(np.searchsorted(far, t) - 1)
                if leg < 0:
                    continue
                o = offers[leg]
                end = o.get("near_contact")
                arr = np.array(n["p"])
                if (
                    not -1.60 < arr[1] < -0.49
                    or (end and t >= end["time"] * 60)
                    or t + 5 > len(actions)
                ):
                    continue
                key = (
                    c["id"],
                    c["seed"],
                    0 if leg == 0 else c["reply_seed"],
                    tuple(doc["ready"]),
                    t,
                    hist.digest(),
                )
                sub = actions[t : min(t + 21, len(actions))]
                ak = blake2b(sub, digest_size=16).digest()
                reward = int(bool(o.get("near_landed")))
                touch = int(bool(end))
                future = int(
                    leg + 1 < len(offers) and bool(offers[leg + 1].get("near_landed"))
                )
                if ak not in groups[key]:
                    groups[key][ak] = dict(
                        source=str(path),
                        row=idx,
                        leg=leg,
                        tick=t,
                        id=c["id"],
                        position=n["p"],
                        reward=reward,
                        touch=touch,
                        next_return=future,
                        until_touch=None if not end else end["time"] * 60 - t,
                        first_actions=actions[t : t + 5, :5].tolist(),
                    )
                used = True
        if used:
            sources[str(path)] = _hash(path)
    paired = []
    different = 0
    for key, actions in groups.items():
        rows = list(actions.values())
        if len(rows) < 2:
            continue
        different += 1
        if max(r["reward"] for r in rows) == min(r["reward"] for r in rows):
            continue
        if len(set(str(r["first_actions"]) for r in rows)) < 2:
            continue
        paired.append(dict(id=key[0], tick=key[-2], prefix=key[-1].hex(), rows=rows))
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        ROOT / "pairs.json.gz",
        dict(
            sources=sources,
            unique_games=len(seen),
            decisions=len(groups),
            different_continuations=different,
            different_legal=len(paired),
            releases=len(set(r["id"] for r in paired)),
            pairs=paired,
        ),
    )
    print(
        len(seen),
        len(groups),
        different,
        len(paired),
        len(set(r["id"] for r in paired)),
        flush=True,
    )


if __name__ == "__main__":
    collect()
