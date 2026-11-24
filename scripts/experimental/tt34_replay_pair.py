"""Render matched physical action replays under one frame of camera delay."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .tt4_transfer_trial import _read, _write
from .tt31_replay import replay
from .tt34_latency import OUT


def render(directory):
    movies = []
    for name in ("frozen", "online"):
        source = _read(OUT / f"switch-{name}.json.gz")
        index = next(
            i
            for i, (c, lag) in enumerate(
                zip(source["cards"], source["schedule"], strict=True)
            )
            if c["id"] == 6323 and lag == 1
        )
        sample = source | {k: [source[k][index]] for k in ("cards", "rows", "schedule")}
        tape = directory / f"{name}.json.gz"
        movie = directory / f"{name}.mp4"
        _write(tape, sample)
        replay(tape, 6323, directory / f"{name}.json", movie, cad=False)
        movies.append(movie)
    panels = []
    for i, heading in enumerate(
        (
            "Frozen timing  |  1 legal arm return",
            "Learned timing  |  2 legal arm returns",
        )
    ):
        panels.append(
            f"[{i}:v]crop=868:650:18:84,scale=640:480,pad=640:544:0:64:color=0x16222a,"
            f"drawtext=text='{heading}':x=15:y=18:fontcolor=white:fontsize=23[p{i}]"
        )
    filters = ";".join([*panels, "[p0][p1]hstack=inputs=2[out]"])
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(movies[0]),
            "-i",
            str(movies[1]),
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
            str(OUT / "timing-6323.mp4"),
        ],
        check=True,
    )


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        render(Path(folder))
