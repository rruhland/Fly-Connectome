"""Replay one held point before practice, after reach practice and after strike practice."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .tt4_transfer_trial import _hash, _read, _write
from .tt31_replay import replay
from .tt34_results import paired
from .tt35_replies import varied_opponent
from .tt36_course import OUT


def film(directory):
    paths = [
        OUT / f"{name}.json.gz"
        for name in ("local-held-fixed", "local-held-dense", "staged-held-dense")
    ]
    stages = ["Before practice", "After reach practice", "After strike practice"]
    movies = []
    captions = []
    audits = []
    streams = [_read(path) for path in paths]
    for d in streams[1:]:
        paired(streams[0], d)
    for j, (stage, d, path) in enumerate(zip(stages, streams, paths, strict=True)):
        i = next(
            i
            for i, (c, s) in enumerate(zip(d["cards"], d["schedule"], strict=True))
            if c["id"] == 12373 and s["kind"] == "reply"
        )
        card = d["cards"][i]
        row = d["rows"][i]
        seed = d["schedule"][i]["reply_seed"]
        tmp = directory / f"stage-{j}.json.gz"
        movie = directory / f"stage-{j}.mp4"
        receipt = directory / f"stage-{j}.json"
        _write(
            tmp,
            {"cards": [card], "ready": d["ready"], "opening_side": 1, "rows": [row]},
        )
        with varied_opponent(card["index"], seed):
            replay(tmp, card["id"], receipt, movie, cad=True)
        audit = _read(receipt)
        own = row["score"]["learner"]
        label = f"{own} legal arm return" if own == 1 else f"{own} legal arm returns"
        touched = row["placements"]["delivered"][0]["near_contact"] is not None
        captions.append(
            (
                stage,
                label
                if own
                else ("Contact, return missed" if touched else "Ball missed"),
            )
        )
        movies.append(movie)
        audits.append(
            {
                "source": path.name,
                "source_sha256": _hash(path),
                "card": card,
                "reply_seed": seed,
                "events_sha256": audit["events_sha256"],
                "frames": audit["frames"],
                "own": own,
                "first_touch": touched,
                "clearance": audit["clearance"],
            }
        )
    panels = []
    for i, (label, score) in enumerate(captions):
        panels.append(
            f"[{i}:v]crop=868:650:18:84,scale=448:336,pad=448:410:0:74:color=0x16222a,"
            f"drawtext=text='{label}':x=12:y=8:fontcolor=white:fontsize=23,"
            f"drawtext=text='{score}':x=12:y=40:fontcolor=0xBFD9E2:fontsize=18[p{i}]"
        )
    filters = ";".join([*panels, "[p0][p1][p2]hstack=inputs=3[out]"])
    output = OUT / "learning-12373.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *[v for m in movies for v in ("-i", str(m))],
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    _write(
        OUT / "learning-12373.json", {"stages": audits, "movie_sha256": _hash(output)}
    )


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        film(Path(folder))
