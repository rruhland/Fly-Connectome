"""Render an actual receiving-table action tape through the learner's cameras."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt4_arena import TT4Arena
from .tt4_transfer_trial import Wiring


def load(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as stream:
        return json.load(stream)


def caption(frame, title, status, count):
    width = 680
    h = round(width * frame.shape[0] / frame.shape[1])
    output = Image.new("RGB", (width, h + 76), "#141d25")
    output.paste(Image.fromarray(frame).resize((width, h), Image.Resampling.NEAREST), (0, 40))
    draw = ImageDraw.Draw(output)
    try:
        title_font, text_font = (ImageFont.truetype("DejaVuSans.ttf", n) for n in (18, 16))
    except OSError:
        title_font = text_font = ImageFont.load_default()
    draw.text((12, 8), title, fill="#f1f4f9", font=title_font)
    draw.text((12, h + 47), status, fill="#e1ecf4", font=text_font)
    draw.text((width - 125, h + 47), f"Returns   {count}", fill="#a9e9ce", font=text_font)
    return output


@contextmanager
def recording(output, first, dt):
    """Stream long MP4 recordings and keep the existing animated-image format."""
    if output.suffix.lower() != ".mp4":
        frames = [first]
        yield frames.append
        durations = [round(1000 * dt)] * (len(frames) - 1) + [1000]
        first.save(output, save_all=True, append_images=frames[1:], method=4, quality=90,
                   duration=durations, loop=0)
        return
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        path = Path(temp) / "rally.mp4"
        command = ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                   "-video_size", f"{first.width}x{first.height}", "-framerate", str(1 / dt),
                   "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                   "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            last = first

            def append(frame):
                nonlocal last
                encoder.stdin.write(frame.tobytes())
                last = frame

            try:
                append(first)
                yield append
                for _ in range(max(0, round(1 / dt) - 1)):
                    append(last)
            finally:
                encoder.stdin.close()
        if encoder.returncode:
            raise RuntimeError(f"ffmpeg exited with code {encoder.returncode}")
        path.replace(output)


def replay(path, output, condition, course_index=0, point_index=0, view="first_person"):
    receipt = load(path)
    course = receipt["courses"][view][course_index]
    row = course["runs"][condition]["points"][point_index]
    actions = np.array(row["actions"], dtype="<f8")
    if hashlib.sha256(actions.tobytes()).hexdigest() != row["action_sha256"]:
        raise ValueError("action tape hash mismatch")
    schedule = course["schedule"]
    kwargs = receipt["geometry"]["arena"].copy()
    arena = TT4Arena(seed=schedule["arena_seed"], view_seed=schedule["view_seed"], view=view, **kwargs)
    if receipt["geometry"].get("fast_hall"):
        from .tt5_hall_trial import simplify_hall_lighting
        simplify_hall_lighting(arena)
    interface = Wiring(arena, schedule["arena_seed"])
    native = kwargs['render_mode'] == 'mujoco'
    camera = "player’s left eye" if view == "first_person" else "overhead camera"
    title = f"MuJoCo table tennis  ·  {camera}  ·  {'shaded RGB' if native else 'projected RGB'}"
    try:
        # Keep the exact resolution, colors and camera projection used during the point.
        images = arena.reset(seed=row["seed"])
        status = "Opponent serves"
        with recording(output, caption(images[0], title, status, 0), arena.control_dt) as append:
            for action in actions:
                images, _, _, _, info = interface.step(action)
                for event in info["events"]:
                    who = "Arm" if event.get("side") == 1 else "Opponent"
                    if event["kind"] == "contact":
                        status = f"{who} strikes the ball"
                    elif event["kind"] == "legal_return":
                        status = f"{who} lands a legal return"
                if arena.terminated:
                    status = ("Arm wins" if arena.reward > 0 else "Opponent wins") + "  ·  " + arena.reason.replace("_", " ")
                elif arena.truncated:
                    status = "Time limit reached · no point scored"
                append(caption(images[0], title, status, arena.legal_returns))
            if arena.point_result() != row["evaluator"]:
                raise ValueError("physical replay diverged from the recorded outcome")
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--condition", default="first_person_mini")
    parser.add_argument("--course", type=int, default=0)
    parser.add_argument("--point", type=int, default=0)
    parser.add_argument("--view", default="first_person", choices=["first_person", "overhead_side"])
    args = parser.parse_args()
    replay(args.receipt, args.output, args.condition, args.course, args.point, args.view)
