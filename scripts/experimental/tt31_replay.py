"""Render the two-sided practice or stroke-search game and audit the real paddle surface."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_conditions import cooperative_targets
from .tt29_endurance import audit
from .tt29_vision_replay import (
    BROADCAST,
    HEIGHT,
    WIDTH,
    clearance_final,
    label,
    start_encoder,
    start_renderer,
)
from .tt31_contacts import BladeContacts, summarize


def make_arena(source, point, **settings):
    if "cards" in source:
        from .tt31_curriculum_opponent import replay_arena

        return replay_arena([source["cards"][point]], net_rescue=True, **settings)
    from .tt31_faces import SETTINGS, FaceArena

    return FaceArena(
        neutral=source["mode"] != "original",
        targets=cooperative_targets("lateral", source["spread"]),
        guard_table=True,
        net_rescue=True,
        **SETTINGS,
        **settings,
    )


def draw_frame(arena, renderer, observer, title, legal_total):
    scene = Image.new("RGB", (WIDTH, HEIGHT), "#17222b")
    label(scene, (18, 13), title, 27, bold=True)
    label(
        scene,
        (18, 48),
        "SO-101 at the near end   ·   Full point at simulation speed",
        17,
        color="#b9c9cf",
    )
    scene.paste(
        Image.fromarray(arena.render_broadcast(renderer)).resize(BROADCAST), (18, 83)
    )
    left = 908
    label(scene, (left, 90), f"{legal_total} legal returns", 30, bold=True)
    label(scene, (left, 139), f"{arena.data.time:.2f} s", 21)
    label(scene, (left, 226), "Near-arm contact positions", 21, bold=True)
    width = 286
    x0 = left + width / 2
    y = 337
    d = ImageDraw.Draw(scene)
    d.line((left, y, left + width, y), fill="#b9c9cf", width=3)
    d.line((x0, y - 11, x0, y + 11), fill="#b9c9cf", width=2)
    label(scene, (left, y + 16), "Left", 16)
    label(scene, (left + 102, y + 16), "Shoulder", 16)
    label(scene, (left + 245, y + 16), "Right", 16)
    colors = {"negative": "#69d8b6", "positive": "#fab160", "rim": "#b9c9cf"}
    names = {"negative": "Face A", "positive": "Face B", "rim": "Rim"}
    for i, c in enumerate(observer.rows):
        xp = x0 + np.clip(c["side_x_m"] / 0.7625, -1, 1) * width / 2
        yp = y - 22 - 12 * (i % 5)
        d.ellipse(
            (xp - 5, yp - 5, xp + 5, yp + 5), fill=colors[c["face"]], outline="#17222b"
        )
    if observer.rows:
        last = observer.rows[-1]
        side = "left" if last["side_x_m"] < 0 else "right"
        label(scene, (left, 416), "Most recent contact", 19)
        label(
            scene,
            (left, 452),
            names[last["face"]],
            30,
            bold=True,
            color=colors[last["face"]],
        )
        label(
            scene,
            (left, 500),
            f"{100 * abs(last['side_x_m']):.1f} cm {side} of shoulder",
            19,
        )
    label(scene, (left, 655), "A and B identify opposite", 17, color="#b9c9cf")
    label(scene, (left, 681), "physical faces of the blade.", 17, color="#b9c9cf")
    return scene


def replay(source_path, shot, output, movie, *, cad=False, seed=None):
    from .tt31_ready import centered

    source = _read(source_path)
    with centered(source["ready"]) if "ready" in source else nullcontext():
        return replay_game(source, source_path, shot, output, movie, cad=cad, seed=seed)


def replay_game(source, source_path, shot, output, movie, *, cad, seed):
    matches = [
        (i, r)
        for i, r in enumerate(source["rows"])
        if r["shot"]["id"] == shot and (seed is None or r["opponent_seed"] == seed)
    ]
    if len(matches) != 1:
        raise ValueError("Select one archived point by shot and opponent seed")
    point, row = matches[0]
    arena = make_arena(source, point)
    full_cad = make_arena(source, point, detail="cad") if cad else None
    clearance = None
    observer = BladeContacts(arena)
    side = source.get("opening_side", -1)
    renderer = encoder = None
    frames = 0
    try:
        arena.launch(row["shot"])
        arena.partner.reset(seed=row["opponent_seed"])
        if full_cad is not None:
            from .tt29_arm_clearance import ArmClearance
            from .tt29_rally_physics import mujoco

            guard = ArmClearance(full_cad.model, tolerance=0.0)
            shadow = mujoco.MjData(full_cad.model)
            arena.sample_clearance(True)
            arena._cad_reader = guard, shadow
        if movie:
            renderer = start_renderer(arena)
            encoder = start_encoder(movie, 30)
        with patch.object(arena, "_racket_contact", observer.observe):
            for tick, action in enumerate(row["actions"]):
                if encoder and tick % 2 == 0:
                    count = max(arena.legal_returns + arena.rival_returns - 1, 0)
                    name = (
                        "Continuously varied placement"
                        if "cards" in source
                        else "Changing paddle faces in a live rally"
                    )
                    if "cards" in source and source["cards"][point]["mode"] == "periodic":
                        name = "Alternating left and right placements"
                    encoder.stdin.write(
                        draw_frame(arena, renderer, observer, name, count).tobytes()
                    )
                    frames += 1
                arena.exchange(action, np.zeros(5))
        if digest_json(row["events"]) != digest_json(arena.events):
            raise ValueError("Replayed physical events diverged")
        actual = audit(arena.events, opening_side=side)
        if actual["returns"] != row["score"]["returns"]:
            raise ValueError("Replayed legal returns diverged")
        contacts = observer.scored(arena.events, opening_side=side)
        if full_cad is not None:
            clearance = clearance_final(arena, guard, shadow)
            if (
                clearance["samples"]
                != round(arena.data.time / arena.model.opt.timestep) + 1
            ):
                raise ValueError("Clearance scan missed physics states")
    finally:
        if encoder:
            encoder.stdin.close()
            code = encoder.wait()
        if renderer:
            renderer.close()
        arena.close()
        if full_cad is not None:
            full_cad.close()
    if encoder and code:
        raise RuntimeError(f"Video encoder exited with {code}")
    _write(
        output,
        {
            "source_sha256": _hash(source_path),
            "id": shot,
            "opponent_seed": row["opponent_seed"],
            "events_sha256": digest_json(row["events"]),
            "contacts": contacts,
            "summary": summarize([{"contacts": contacts}]),
            "returns": actual["returns"],
            "clearance": clearance,
            "frames": frames,
            "fps": 30,
            "video_sha256": _hash(movie) if movie else None,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("shot", type=int)
    p.add_argument("output", type=Path)
    p.add_argument("--movie", type=Path)
    p.add_argument("--cad", action="store_true")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    replay(
        args.source, args.shot, args.output, args.movie, cad=args.cad, seed=args.seed
    )
