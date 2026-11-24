"""Render one entire archived TT37 point and verify every referee event against the command tape."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from . import tt31_replay
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_opponent import arena


def film(path, movie, title):
    data = _read(path)
    i = max(range(len(data["rows"])), key=lambda i: data["rows"][i]["score"]["returns"])
    doc = {
        "ready": data["ready"],
        "cards": [data["cards"][i]],
        "rows": [data["rows"][i]],
        "opening_side": 1,
    }
    draw = tt31_replay.draw_frame

    def labeled(a, renderer, observer, unused, total):
        return draw(a, renderer, observer, title, total)

    def create(source, idx, **kwargs):
        return arena([source["cards"][idx]], **kwargs)

    with TemporaryDirectory() as folder:
        small = Path(folder) / "point.json.gz"
        info = Path(folder) / "audit.json"
        _write(small, doc)
        with (
            patch.object(tt31_replay, "make_arena", create),
            patch.object(tt31_replay, "draw_frame", labeled),
        ):
            tt31_replay.replay(small, doc["cards"][0]["id"], info, movie, cad=True)
        receipt = _read(info)
        receipt.update(
            full_source=path.name,
            full_sha256=_hash(path),
            card=doc["cards"][0],
            own=doc["rows"][0]["score"]["learner"],
        )
        _write(movie.with_suffix(".json"), receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("movie", type=Path)
    p.add_argument("title")
    a = p.parse_args()
    film(a.source, a.movie, a.title)
