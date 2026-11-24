"""Change only the opponent's later placements after the same physically delivered opening."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt31_curriculum_opponent as opponent
from .tt4_transfer_trial import _hash, _read, _write
from .tt31_ready import centered
from .tt31_replay import replay
from .tt33_joint import point, resources
from .tt34_latency import PUPIL
from .tt35_hindsight import DIRECTORY


def varied_opponent(opening, reply_seed):
    original = opponent.placement

    def requests(mode, index, rng):
        normal = original(mode, index, rng)
        if index == opening:
            return normal
        return original(mode, index, np.random.default_rng([reply_seed, index]))

    return patch.object(opponent, "placement", requests)


def film():
    source = DIRECTORY / "replies-aligned.json.gz"
    doc = _read(source)
    i = doc["reply_seeds"].index(355008)
    card = doc["cards"][i]
    target = DIRECTORY / "varied-5530.json"
    with TemporaryDirectory() as folder, varied_opponent(card["index"], 355008):
        path = Path(folder) / "point.json.gz"
        _write(
            path,
            doc | {"cards": [card], "rows": [doc["rows"][i]], "reply_seeds": [355008]},
        )
        replay(path, card["id"], target, DIRECTORY / "varied-5530.mp4", cad=True)
    _write(
        target,
        _read(target) | {"full_stream_sha256": _hash(source), "reply_seed": 355008},
    )


def run(mode):
    reference = _read(DIRECTORY / "confirm-nominal.json.gz")
    index = next(i for i, c in enumerate(reference["cards"]) if c["id"] == 5530)
    card, launch = reference["cards"][index], reference["rows"][index]["shot"]
    path = PUPIL if mode == "nominal" else DIRECTORY / f"hindsight-{mode}.json.gz"
    model = _read(path)
    result = {
        "checkpoint_sha256": _hash(path),
        "selected_release": 5530,
        "selection": "Four arm returns with the aligned readout on the earlier confirmation course",
        "ready": model["ready"],
        "opening_side": 1,
        "cards": [],
        "rows": [],
        "reply_seeds": [],
    }
    with centered(model["ready"]):
        arena = opponent.replay_arena([])
        stereo, rig, stride, lo, hi = resources(arena)
        try:
            for seed in range(355001, 355009):
                with varied_opponent(card["index"], seed):
                    arena.cards = iter([card])
                    row = point(
                        arena, stereo, rig, model, stride, lo, hi, launch, card, cap=16
                    )
                result["cards"].append(card)
                result["rows"].append(row)
                result["reply_seeds"].append(seed)
            _write(DIRECTORY / f"replies-{mode}.json.gz", result)
        finally:
            stereo.close()
            arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("nominal", "aligned", "static", "film"))
    args = parser.parse_args()
    if args.mode == "film":
        film()
    else:
        run(args.mode)
