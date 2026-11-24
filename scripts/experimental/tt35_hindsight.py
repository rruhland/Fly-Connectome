"""Re-use the raw pupil's own missed motions at the ball locations where they passed the paddle."""

import argparse
from pathlib import Path

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt34_latency import OUT, PUPIL
from .tt35_error import Observations, passage

DIRECTORY = OUT.parent / "tt35"
PRACTICE = OUT / "visual-practice-0.json.gz"


def collect(feeds, output):
    policy = _read(PUPIL)
    cards = _read(PRACTICE)["cards"]
    shots = measured_launches(feeds)
    samples = []
    games = []
    with centered(policy["ready"]):
        arena = replay_arena([])
        stereo, rig, stride, low, high = joint.resources(arena)
        view = Observations(stereo, rig)
        try:
            for card in cards:
                arena.launch(shots[card["id"]], card=card)
                arena.partner.reset(seed=card["seed"])
                view.begin()
                actor = joint.JointPolicy(policy, stride, low, high)
                eye = joint.Vision(rig, arena.control_dt, raw=True)
                history = []
                for tick in range(3600):
                    q = arena.joints.copy()
                    image = view.capture() if tick % 2 == 0 else None
                    optical = eye.update(tick, image, q)
                    action = actor.command(optical, q)
                    if optical is not None:
                        history.append(
                            (tick, optical, q, actor.reference.copy(), action)
                        )
                    _, _, done, truncated, _ = arena.exchange(action, joint.FAR_COMMAND)
                    if (
                        done
                        or truncated
                        or arena.legal_returns + arena.rival_returns - 1 >= 16
                    ):
                        break
                score = audit(arena.events, opening_side=1)["learner"]
                crossing = passage(view.rows, history[0][0]) if history else None
                count = 0
                if score == 0 and crossing is not None:
                    gap = np.array(crossing["gap_xz_m"])
                    # The relabelling assumes local translation, so exclude offsets larger than a blade span.
                    if 0.03 <= np.linalg.norm(gap) <= 0.22:
                        for step, optical, q, target, action in history:
                            age = (crossing["tick"] - step) * arena.control_dt
                            if 0 <= age <= 0.38:
                                samples.append(
                                    (optical, q, target, action, [step, 0, age], gap)
                                )
                                count += 1
                games.append(
                    {
                        "card": card,
                        "own_returns": score,
                        "passage": crossing,
                        "used_ticks": count,
                    }
                )
        finally:
            stereo.close()
            arena.close()
    if not samples:
        raise ValueError("No observed near misses in the practice course")
    fields = ("optical", "joints", "targets", "actions", "ticks", "gap")
    np.savez_compressed(
        output,
        **dict(zip(fields, map(np.asarray, zip(*samples, strict=True)), strict=True)),
    )
    _write(
        output.with_suffix(".json"),
        {
            "feed_sha256": _hash(feeds),
            "pupil_sha256": _hash(PUPIL),
            "practice_sha256": _hash(PRACTICE),
            "used_ticks": len(samples),
            "games": games,
        },
    )
    print(len(samples), [g["used_ticks"] for g in games], flush=True)


def fit(raw, source, mode, output):
    from tempfile import TemporaryDirectory

    with np.load(source) as archive:
        data = dict(archive)
    receipt = _read(source.with_suffix(".json"))
    sign = {
        "aligned": -1.0,
        "as-observed": 0.0,
        "static": 0.0,
        "familiar": 0.0,
        "reverse": 1.0,
    }[mode]
    data["optical"][:, [0, 2]] += sign * data.pop("gap")
    if mode == "familiar":
        # Match the time weights and sample count while using only old demonstration states.
        rng = np.random.default_rng(355104)
        with np.load(raw) as original:
            ages = original["ticks"][:, 2]
            ix = [
                rng.choice(np.flatnonzero(abs(ages - age) <= 1 / 30.0))
                for age in data["ticks"][:, 2]
            ]
            data["optical"] = original["optical"][ix]
            data["joints"] = original["joints"][ix]
    if mode in ("static", "familiar"):
        policy = _read(PUPIL)
        home = np.asarray(policy["ready"])
        coords = joint.features(
            data["optical"], policy["options"]["features"], data["joints"], home
        )
        basis = joint.activation(
            coords, np.asarray(policy["centers"]), policy["options"]["width"]
        )
        data["targets"] = home + basis @ np.asarray(policy["weights"])
    # Velocities are unused by this raw-position/velocity + encoder policy fit.
    data["velocities"] = np.zeros_like(data["joints"])
    with TemporaryDirectory() as folder:
        dataset = Path(folder) / "scoreless.npz"
        np.savez_compressed(dataset, **data)
        _write(
            dataset.with_suffix(".json"),
            {
                "sources": {source.name: _hash(source)},
                "receipts": [
                    {"card": g["card"], "strikes": 0}
                    for g in receipt["games"]
                    if g["used_ticks"]
                ],
            },
        )
        joint.fit(raw, output, _read(PUPIL)["options"] | {"additional": [dataset]})
    model = _read(output)
    _write(output, model | {"relabel": mode, "practice_samples": receipt["used_ticks"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("collect", "aligned", "as-observed", "reverse", "static", "familiar"),
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--feeds", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument(
        "--source", type=Path, default=DIRECTORY / "hindsight-practice.npz"
    )
    args = parser.parse_args()
    if args.mode == "collect":
        collect(args.feeds, args.output)
    else:
        fit(args.raw, args.source, args.mode, args.output)
