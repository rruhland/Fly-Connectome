"""Render an actual receiving-table action tape through the learner's cameras."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

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
    interface = Wiring(arena, schedule["arena_seed"])
    native = kwargs['render_mode'] == 'mujoco'
    camera = "player’s left eye" if view == "first_person" else "overhead camera"
    title = f"MuJoCo table tennis  ·  {camera}  ·  {'shaded RGB' if native else 'projected RGB'}"
    try:
        # Keep the exact resolution, colors and camera projection used during the point.
        images = arena.reset(seed=row["seed"])
        status = "Opponent serves"
        frames = [caption(images[0], title, status, 0)]
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
            frames.append(caption(images[0], title, status, arena.legal_returns))
        if arena.point_result() != row["evaluator"]:
            raise ValueError("physical replay diverged from the recorded outcome")
        frame_ms = round(1000 * arena.control_dt)
        frames[0].save(output, save_all=True, append_images=frames[1:], method=4, quality=90,
                       duration=[*([frame_ms] * (len(frames) - 1)), 1000], loop=0)
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
