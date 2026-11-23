"""Replay the matched regulation-size point with and without the visual continuation."""

import argparse
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

from .tt4_transfer_trial import _hash, _read, _write
from .tt21_recovery_movie import font
from .tt26_grip_course import load_match
from .tt26_transfer_movie import match_frames
from .tt27_flight_feedback import OUT


def run(path):
    shots, _ = load_match(path)
    games = _read(OUT / "full-points.json.gz")["rows"]
    row = max(games, key=lambda r: (r["by"]["repeat"]["near"], r["by"]["repeat"]["far"]))
    game = dict(angle=0.0, shot=shots[row["id"]], results={})
    for name in ("opening", "repeat"):
        record = row["by"][name]
        game["results"][name] = dict(record, rival_actions=record["opponent"])
    left, right = (match_frames(game, mode) for mode in ("opening", "repeat"))
    out = OUT / "regulation-point.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "780x355",
        "-r",
        "30",
        "-i",
        "-",
        "-vf",
        "pad=780:356",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        count = max(len(left), len(right)) + 12
        for tick in range(count):
            screen = Image.new("RGB", (780, 355), "#16212b")
            pen = ImageDraw.Draw(screen)
            pen.text(
                (10, 5),
                "Regulation table  ·  Same recorded launch and opponent",
                font=font(23, True),
                fill="white",
            )
            for x, frames, caption in [
                (10, left, "Opening policy"),
                (398, right, "Opening + visual reply correction"),
            ]:
                frame, score = frames[min(tick, len(frames) - 1)]
                screen.paste(frame, (x, 65))
                pen.text((x, 38), caption, font=font(16), fill="#e3ebec")
                pen.text((x, 313), f"SO-101 legal returns   {score}", font=font(17, True), fill="#a4e1cf")
            proc.stdin.write(screen.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("Video encoding failed")
    _write(
        OUT / "regulation-point.json",
        dict(
            id=row["id"],
            group=row["group"],
            frames=count,
            fps=30,
            hash=_hash(out),
            selection="longest full point under the repeat controller, ties broken by opponent returns",
            referee_replay=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    run(parser.parse_args().feeds)
