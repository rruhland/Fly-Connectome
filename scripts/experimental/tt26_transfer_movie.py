"""Replay a paired solo release and full-size receiving point from the transfer tests."""

import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt21_recovery_movie import font
from .tt24_bounce import BounceArena
from .tt24_integrated_stress import InstalledArm
from .tt24_ready import prepare_step
from .tt26_grip_course import Grip, OUT, ReferenceReceiver


def bounce_frames(row, name):
    plan = row["plan"]
    saved = row["results"]["tool_and_adjustment" if name == "bounce" else name]
    install = InstalledArm("stock")
    install.arena.max_steps = 480
    arena = install.arena
    server = BounceArena(view="none")
    frames = []
    try:
        Grip(arena._model).set(row["angle"])
        server.reset(seed=plan["launcher_seed"], split="held", render=False)
        loc, velocity, spin = [
            v.copy() for v in (server._ball_position, server._ball_velocity, server._ball_spin)
        ]
        install.reset(plan, loc, False)
        prep = plan["prep_ticks"]
        changes = dict(plan["changes"])
        for t, action in enumerate(saved["actions"]):
            if t % 2 == 0:
                left = Image.fromarray(arena.render_external()).resize((372, 244))
                corner = Image.fromarray(arena.render_wrist()).resize((128, 96))
                left.paste(corner, (244, 0))
                frames.append((left, len(arena._rebounds)))
            if t - prep in changes:
                install.switch_gravity(t - prep, changes[t - prep])
            if t < prep:
                prepare_step(arena, np.asarray(action), loc)
                if t + 1 == prep:
                    arena._ball_velocity[:] = velocity
                    arena._ball_spin[:] = spin
                    mujoco.mj_forward(arena._model, arena._data)
            else:
                arena.step(action, render=False)
        if digest_json(arena.judge_metrics()) != digest_json(saved["metrics"]):
            raise ValueError("Solo replay did not match scored drop")
        frames.append((frames[-1][0], len(arena._rebounds)))
        return frames
    finally:
        server.close()
        arena.close()


def match_frames(row, name):
    record = row["results"][name]
    arena = ReferenceReceiver()
    arena.max_steps = 450
    frames = []
    try:
        Grip(arena.model).set(row["angle"])
        arena.launch(row["shot"])
        with mujoco.Renderer(arena.model, width=640, height=360) as render:
            for near, far in zip(record["actions"], record["rival_actions"], strict=True):
                frames.append(
                    (
                        Image.fromarray(arena.render_broadcast(render)).resize((372, 244)),
                        arena.legal_returns,
                    )
                )
                arena._aim = arena._command_to = lambda *args, far=far: np.asarray(far)
                _, _, done, cap, _ = arena.step(near, render=False)
        if not (done or cap) or digest_json(arena.events) != digest_json(record["events"]):
            raise ValueError("Match movie failed referee replay")
        frames.append((frames[-1][0], arena.legal_returns))
        return frames
    finally:
        arena.close()


def run():
    drops = _read(OUT / "tool-recall.json.gz")["rows"]
    # First qualified release in the predetermined evaluation roster.
    solo = next(
        r
        for r in drops
        if r["results"]["tool_and_adjustment"]["metrics"]["longest_consecutive_rebounds"] >= 10
    )
    games = _read(OUT / "live-points.json.gz")["cases"]
    game = next(
        r
        for r in games
        if r["angle"] == solo["angle"]
        and r["results"]["bounce"]["far"] > 0
        and r["results"]["cold"]["near"] == 0
    )
    followup = next(
        case for case in _read(OUT / "match-continuation.json.gz")["held"] if case["result"]["near"] >= 2
    )
    opening = next(r for r in games if r["angle"] == followup["angle"] and r["idx"] == followup["idx"])
    played = dict(
        angle=opening["angle"],
        shot=opening["shot"],
        results={"cold": opening["results"]["bounce"], "bounce": followup["result"]},
    )
    segments = [
        ("Wrist-camera paddle bouncing", "Same release, spin and arm", bounce_frames, solo, "Rebounds"),
        (
            "Full-size table tennis",
            "Same incoming shot and live opponent",
            match_frames,
            game,
            "Legal returns",
        ),
    ]
    segments.append(
        (
            "Recovering for the opponent’s reply",
            "Same opening and grip, after separate match practice",
            match_frames,
            played,
            "Legal returns",
        )
    )
    out = OUT / "bounce-to-table.mp4"
    args = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "780x360",
        "-r",
        "30",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    frames = 0
    with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
        for title, note, render, row, counter in segments:
            left, right = render(row, "cold"), render(row, "bounce")
            for t in range(max(len(left), len(right)) + 24):
                canvas = Image.new("RGB", (780, 360), "#15212c")
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 6), title, fill="white", font=font(22, True))
                draw.text((12, 35), note, fill="#c8dce4", font=font(15))
                recovery = row is played
                for x, images, label in [
                    (12, left, "Before reply practice" if recovery else "Before solo practice"),
                    (396, right, "After reply practice" if recovery else "After solo practice"),
                ]:
                    view, score = images[min(t, len(images) - 1)]
                    canvas.paste(view, (x, 84))
                    draw.text((x, 58), label, fill="white", font=font(17, True))
                    draw.text((x, 333), f"{counter}   {score}", fill="#aadfc5", font=font(16))
                proc.stdin.write(canvas.tobytes())
                frames += 1
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("Video encoder failed")
    _write(
        OUT / "bounce-to-table.json",
        dict(
            video=_hash(out),
            frames=frames,
            fps=30,
            grip=game["angle"],
            bounce_seed=solo["plan"]["learner_seed"],
            point=game["idx"],
            recording_id=game["shot"]["id"],
            recovery_variant=[followup["angle"], followup["idx"]],
            recovery_selection="first held continuation point with a second legal return",
            selection="first fresh attitude-aware wrist release reaching ten consecutive and first matching-grip positive paired point",
            verified=True,
        ),
    )


if __name__ == "__main__":
    run()
