"""Show a representative matched solo drop from the dose experiment at real-time speed."""

import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt21_recovery_movie import font
from .tt24_public_body import stock
from .tt26_grip_course import Grip
from .tt28_dose import HELD_CAP, MOUNTS, OUTPUT


def frames(row, mount, dose):
    recorded = row["results"][dose]
    arena = stock()
    arena.max_steps = HELD_CAP
    Grip(arena._model).set(*MOUNTS[mount])
    result = []
    try:
        arena.reset(seed=row["seed"], regime=row["regime"], split="held", render=False)
        for t, cmd in enumerate(recorded["actions"]):
            if t % 2 == 0:
                result.append(
                    (
                        Image.fromarray(arena.render_external()).resize((336, 252)),
                        Image.fromarray(arena.render_wrist()).resize((160, 120)),
                        arena.judge_metrics()["rebounds"],
                    )
                )
            arena.step(cmd, render=False)
        metrics = arena.judge_metrics()
        if digest_json(metrics) != recorded["referee"]:
            raise ValueError("Solo movie failed physical replay")
        result.append((result[-1][0], result[-1][1], metrics["rebounds"]))
        return result
    finally:
        arena.close()


def run():
    scenes = []
    selections = []
    for mount in MOUNTS:
        candidates = _read(OUTPUT / f"dose-solo-{mount}.json.gz")["held"]

        def score(row):
            before = row["results"]["0"]["streak"]
            after = row["results"]["192"]["streak"]
            return np.array([after - before, after])

        center = np.median([score(row) for row in candidates], axis=0)
        selected = min(candidates, key=lambda row: tuple(abs(score(row) - center)))
        scenes.append((mount, [frames(selected, mount, d) for d in ("0", "192")]))
        selections.append(
            dict(
                mount=mount,
                seed=selected["seed"],
                regime=selected["regime"],
                streak={d: selected["results"][d]["streak"] for d in ("0", "192")},
            )
        )
    target = OUTPUT / "wrist-bouncing.mp4"
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-s",
            "720x490",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-threads",
            "1",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        stdin=subprocess.PIPE,
    )
    caption = font(17, True)
    small = font(15)
    with process:
        for mount, pair in scenes:
            for t in range(max(map(len, pair)) + 12):
                canvas = Image.new("RGB", (720, 490), "#142431")
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (14, 7), "Same held drop and physical paddle mount", font=font(24, True), fill="white"
                )
                for i, (tag, data) in enumerate(
                    zip(("Before solo practice", "After 192 practice drops"), pair, strict=True)
                ):
                    x = 14 + 354 * i
                    outside, wrist, score = data[min(t, len(data) - 1)]
                    draw.text((x, 45), tag, font=caption, fill="#dce7e9")
                    canvas.paste(outside, (x, 73))
                    canvas.paste(wrist, (x, 336))
                    draw.text((x + 173, 352), "Wrist camera", font=small, fill="#dce7e9")
                    draw.text((x + 173, 383), f"Rebounds   {score}", font=caption, fill="#a4e1cf")
                    if t >= len(data):
                        draw.text((x + 173, 415), "Drop ended", font=small, fill="#f1cc9e")
                draw.text(
                    (14, 462),
                    f"Simulated SO-101   ·   {mount} mount   ·   real-time replay",
                    font=small,
                    fill="#dce7e9",
                )
                process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Encoding failed")
    _write(
        OUTPUT / "wrist-bouncing.json",
        dict(
            examples=selections,
            selection="within each mount, nearest to median paired gain then median mature streak on the held roster",
            sha256=_hash(target),
            fps=30,
        ),
    )


if __name__ == "__main__":
    run()
