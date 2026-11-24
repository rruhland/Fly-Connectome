"""Replay a physical teacher point at its original simulation speed."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .tt4_transfer_trial import _hash, _read, _write
from .tt21_recovery_movie import font
from .tt29_partner import PartnerArena
from .tt29_rally_physics import OUTPUT, RallyArena

FPS = 30
WIDTH, HEIGHT = 720, 540


def _arena(record):
    settings = record["configuration"].copy()
    lanes = settings.pop("partner_lanes", None)
    fast = settings.pop("partner_fast", False)
    settings.update(width=WIDTH, height=HEIGHT)
    if lanes is None:
        return RallyArena(**settings)
    if fast:
        from .tt29_partner_fast import FastPartnerArena

        return FastPartnerArena(targets=lanes, **settings)
    return PartnerArena(targets=lanes, **settings)


def _overlay(rgb, time, landed, total, last_speed):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((15, 12, 438, 71), radius=7, fill="#172332")
    title = "Physics-state coach · alternating legal returns"
    draw.text((27, 19), title, font=font(15, True), fill="#f7f8fc")
    pace = f"   ·   Last departure {last_speed:.1f} m/s" if last_speed is not None else ""
    draw.text((27, 42), f"{landed:3d} / {total}   ·   {time:05.1f} s{pace}", font=font(15), fill="#d9e7df")
    return image.tobytes()


def render(record, output):
    arena = _arena(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-threads", "1", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    state = sha256()
    frame = landed = 0
    speed = None
    events = record["volleys"]
    try:
        arena.launch(record["launch"])
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for near, far in record["actions"]:
                arena.exchange(near, far)
                state.update(arena.data.qpos.astype("<f8").tobytes())
                state.update(arena.data.qvel.astype("<f8").tobytes())
                time = float(arena.data.time)
                while landed < len(events) and events[landed]["landing_time"] <= time + 1e-9:
                    speed = float(np.linalg.norm(events[landed]["out_speed"]))
                    landed += 1
                if time + 1e-9 >= (frame + 1) / FPS:
                    pixels = _overlay(arena.render_broadcast(), time, landed, len(events), speed)
                    encoder.stdin.write(pixels)
                    frame += 1
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed to encode the replay")
        observed = state.hexdigest()
        expected = record.get("replay", {}).get("state_sha256", record.get("live_state_sha256"))
        if expected is not None and observed != expected:
            raise ValueError("Video replay state differs from the verified receipt")
        if len(events) != landed or json.dumps(record["events"]) != json.dumps(arena.events):
            raise ValueError("Video replay ended at a different referee outcome")
        return dict(fps=FPS, frames=frame, physical_seconds=float(arena.data.time),
                    state_sha256=observed, returns=landed, sha256=_hash(output))
    finally:
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT / "rally-physics-teacher.mp4")
    args = parser.parse_args()
    data = _read(args.receipt)
    cases = data if isinstance(data, list) else [data]
    chosen = max(cases, key=lambda row: row["returns"])
    result = render(chosen, args.output)
    _write(args.output.with_suffix(".json"), dict(source_sha256=_hash(args.receipt), **result))
    print(result, flush=True)


if __name__ == "__main__":
    main()
