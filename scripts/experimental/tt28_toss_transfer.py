"""Use bounce-derived motor directions to explore in eight real attempts per recorded trajectory family."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT as PREVIOUS
from .tt27_outcome_receiver import independent
from .tt28_toss_prior import OUT, DOSES, motor_basis


def directions(staged=False):
    rng = np.random.default_rng(2703204)
    random_axes, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    basis = dict(frozen=np.zeros((3, 5)), isotropic=random_axes[:3], wrist=np.eye(5)[[3, 4, 0]])
    for kind in ("ordinary", "oblique"):
        for dose in DOSES:
            with np.load(OUT / f"toss-{kind}-{dose}.npz") as f:
                basis[f"{kind}-{dose}"] = f["directions"]
    shuffled = _read(OUT / "toss-oblique.json.gz")["rows"]
    for row, offset in zip(shuffled, rng.permutation([r["offset"] for r in shuffled]), strict=True):
        row["offset"] = offset
    basis["shuffled"], _ = motor_basis(shuffled)
    if staged:
        with np.load(OUT / "toss-staged-96.npz") as f:
            basis = {"staged-96": f["directions"]}
    # Make the sign convention independent of the SVD implementation.
    result = {}
    for k, dirs in basis.items():
        signed = dirs.copy()
        for v in signed:
            if v[np.argmax(abs(v))] < 0:
                v *= -1
        result[k] = np.vstack([np.zeros(5), *[v for d in signed for v in (0.20 * d, -0.20 * d)]])
    return result


def cases(shots, confirmation):
    source = _read(PREVIOUS / "action-coverage.json.gz")["rows"]
    excluded = {r["id"] for r in source}
    excluded.update(
        r["id"] for b in _read(PREVIOUS / "lateral-confirmation.json.gz")["blocks"] for r in b["rows"]
    )
    seed = 178522
    dev = independent(shots, seed=seed, excluded=excluded, near_count=1, unrestricted_count=19)
    if confirmation:
        excluded.update(s["id"] for _, s in dev)
        return [
            (g, s["id"], None)
            for g, s in independent(
                shots, seed=184605, excluded=excluded, near_count=1, unrestricted_count=19
            )
        ]
    rows = [(g, s["id"], None) for g, s in dev]
    for r in source:
        # Historical contact with no successful goal is an explicit practice scaffold.
        if not any(t["legal"] for t in r["scores"]):
            prior = next((i for i, t in enumerate(r["scores"]) if t["contact"]), None)
            if prior is not None:
                rows.append(("coached_contact", r["id"], prior))
    return rows


def run(path, confirmation, staged=False):
    lookup, _ = load_match(path)
    banks = directions(staged)
    with np.load(PREVIOUS / "lateral-memory.npz") as f:
        keys, goals, scale = (f[n] for n in ("keys", "goals", "scale"))
    arena = ReferenceReceiver()
    results = []
    suffix = ("-staged" if staged else "") + ("-confirm" if confirmation else "")
    report = OUT / f"toss-target{suffix}.json.gz"
    try:
        for group, sid, coached in cases(lookup, confirmation):
            rng = np.random.default_rng(sid ^ 296149)
            winner = {n: None for n in banks}
            rows = []
            for t in range(12):
                shot = lookup[sid] if t == 0 else draw(lookup, [sid], rng, t)
                blank, k = episode(arena, shot, collect=True)
                base = (
                    goals[
                        coached
                        if coached is not None
                        else np.argmin(np.linalg.norm((keys - k) / scale, axis=1))
                    ]
                    if k is not None
                    else None
                )
                scores = {}
                motions = {}
                for name, dirs in banks.items():
                    use = winner[name] if winner[name] is not None else (t if t < 7 else 0)
                    score = blank
                    if base is not None:
                        goal = base + dirs[use]
                        tag = tuple(goal)
                        if tag not in motions:
                            score, _ = episode(arena, shot, goal=goal)
                            if score["legal"]:
                                verify, _ = episode(
                                    arena,
                                    shot,
                                    choose_goal=lambda image: (
                                        goals[
                                            coached
                                            if coached is not None
                                            else np.argmin(np.linalg.norm((keys - image) / scale, axis=1))
                                        ]
                                        + dirs[use]
                                    ),
                                )
                                if verify != score:
                                    raise ValueError("Successful goal did not replay from RGB")
                            motions[tag] = score
                        score = motions[tag]
                    scores[name] = dict(index=use, legal=score["legal"], contact=score["contact"])
                    if t < 8 and score["legal"] and winner[name] is None:
                        winner[name] = use
                rows.append(scores)
            results.append(dict(group=group, id=sid, coached=coached, rows=rows))
            if len(results) % 5 == 0:
                print("completed", len(results), flush=True)
        _write(
            report,
            dict(
                seed=184605 if confirmation else 178522,
                directions={k: v.tolist() for k, v in banks.items()},
                results=results,
            ),
        )
    finally:
        arena.close()
    for group in sorted({r["group"] for r in results}):
        subset = [r for r in results if r["group"] == group]
        print(
            group,
            {
                n: (
                    sum(any(t[n]["legal"] for t in r["rows"][:8]) for r in subset),
                    sum(t[n]["legal"] for r in subset for t in r["rows"][8:]),
                )
                for n in banks
            },
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("--confirmation", action="store_true")
    p.add_argument("--staged", action="store_true")
    args = p.parse_args()
    run(args.feeds, args.confirmation, args.staged)
