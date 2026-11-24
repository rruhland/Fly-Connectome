"""Render two matched held drops using the verified native command streams."""

import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt21_recovery_movie import font
from .tt24_public_body import stock
from .tt29_bounce import COURSE, HELD_CAP, OUTPUT

WIDTH, HEIGHT = 864, 518
FPS = 30


def replay(row, dose):
    physical = stock(fixture="table-edge")
    physical.max_steps = HELD_CAP
    physical.reset(seed=row["seed"], regime=row["regime"], split="held", render=False)
    results = row["results"][dose]
    frames = []
    try:
        for tick, command in enumerate(results["actions"]):
            if tick % 2 == 0:
                frame = Image.fromarray(physical.render_external()).resize((404, 303))
                wrist = Image.fromarray(physical.render_wrist()).resize((149, 112))
                frames.append((frame, wrist, physical.judge_metrics()["rebounds"]))
            physical.step(command, render=False)
        end = physical.judge_metrics()
        if digest_json(end) != results["referee"]:
            raise ValueError("The visualized tape changed physical outcome")
        return frames + [(frames[-1][0], frames[-1][1], end["rebounds"])]
    finally:
        physical.close()


def frame_of(panels, label, tick, dose):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#19212b")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), label, fill="#f3f5f7", font=font(22, True))
    for index, panel in enumerate(panels):
        x = 20 + 420 * index
        frame, wrist, rebounds = panel[min(len(panel) - 1, tick)]
        title = ("Before warm-up", f"After {dose} practice drops")[index]
        draw.text((x, 47), title, fill="#e0e5e4", font=font(17, True))
        canvas.paste(frame, (x, 75))
        canvas.paste(wrist, (x, 391))
        draw.text((x + 162, 401), "Onboard wrist RGB", fill="#ececea", font=font(15))
        draw.text((x + 162, 432), f"Qualified rebounds  {rebounds}", fill="#aeddd5", font=font(17, True))
        if tick >= len(panel):
            draw.text((x + 162, 465), "Drop ended", fill="#e1c4a8", font=font(14))
    return canvas


def run():
    data = _read(COURSE)
    valid = data["held"]
    final = str(max(map(int, data["snapshots"])))
    values = np.array([row["results"][final]["streak"] - row["results"]["0"]["streak"] for row in valid])
    typical = int(np.argmin(np.abs(values - np.median(values))))
    stronger = sorted(range(len(valid)), key=lambda i: valid[i]["results"][final]["streak"])[-2]
    cases = ((typical, "Held drop at the median paired improvement"),
             (stronger, "Held drop with the second-longest warm streak"))
    target = OUTPUT / "tt29_bounce_held.mp4"
    with subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
         "-threads", "1", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        stdin=subprocess.PIPE,
    ) as process:
        metadata = []
        for index, label in cases:
            row = valid[index]
            panels = [replay(row, mode) for mode in ("0", final)]
            for tick in range(max(map(len, panels)) + 10):
                process.stdin.write(frame_of(panels, label, tick, final).tobytes())
            metadata.append(dict(label=label, seed=row["seed"], regime=row["regime"],
                                 streak={mode: row["results"][mode]["streak"] for mode in ("0", final)}))
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("The held movie could not be encoded")
    _write(OUTPUT / "tt29_bounce_held_movie.json", dict(replays=metadata, fps=FPS,
           source_sha256=_hash(COURSE), movie_sha256=_hash(target)))
    print(target, metadata, flush=True)


if __name__ == "__main__":
    run()
