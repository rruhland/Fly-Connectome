"""Render the two-sided practice or stroke-search game and audit the real paddle surface."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_conditions import cooperative_targets
from .tt29_endurance import audit
from .tt29_vision_replay import BROADCAST, HEIGHT, WIDTH, label, start_encoder, start_renderer
from .tt31_contacts import BladeContacts, summarize


def make_arena(source, row, point):
    if "cards" in source:
        from .tt31_curriculum_opponent import replay_arena
        return replay_arena([source["cards"][point]], net_rescue=True, robust_contact=True)
    from .tt31_faces import FaceArena, SETTINGS
    return FaceArena(neutral=source["mode"] != "original", targets=cooperative_targets("lateral", source["spread"]),
                     guard_table=True, net_rescue=True, **SETTINGS)


def draw_frame(arena, renderer, observer, title, legal_total):
    scene = Image.new("RGB", (WIDTH, HEIGHT), "#17222b")
    label(scene, (18, 13), title, 27, bold=True)
    label(scene, (18, 48), "SO-101 at the near end   ·   Full point at simulation speed", 17, color="#b9c9cf")
    scene.paste(Image.fromarray(arena.render_broadcast(renderer)).resize(BROADCAST), (18, 83))
    left = 908
    label(scene, (left, 90), f"{legal_total} legal returns", 30, bold=True)
    label(scene, (left, 139), f"{arena.data.time:.2f} s", 21)
    label(scene, (left, 226), "Near-arm contact positions", 21, bold=True)
    width = 286
    x0 = left + width / 2
    y = 337
    d = ImageDraw.Draw(scene)
    d.line((left, y, left + width, y), fill="#b9c9cf", width=3)
    d.line((x0, y-11, x0, y+11), fill="#b9c9cf", width=2)
    label(scene, (left, y+16), "Left", 16)
    label(scene, (left+102, y+16), "Shoulder", 16)
    label(scene, (left+245, y+16), "Right", 16)
    colors = {"negative": "#69d8b6", "positive": "#fab160", "rim": "#b9c9cf"}
    names = {"negative":"Face A", "positive":"Face B", "rim":"Rim"}
    for i, c in enumerate(observer.rows):
        xp = x0 + np.clip(c["side_x_m"] / .7625, -1, 1) * width / 2
        yp = y-22-12*(i % 5)
        d.ellipse((xp-5, yp-5, xp+5, yp+5), fill=colors[c["face"]],outline="#17222b")
    if observer.rows:
        last = observer.rows[-1]
        side = "left" if last["side_x_m"] < 0 else "right"
        label(scene,(left,416),"Most recent contact",19)
        label(scene,(left,452),names[last["face"]],30,bold=True,color=colors[last["face"]])
        label(scene,(left,500),f'{100*abs(last["side_x_m"]):.1f} cm {side} of shoulder',19)
    label(scene, (left, 655), "A and B identify opposite", 17, color="#b9c9cf")
    label(scene, (left, 681), "physical faces of the blade.", 17, color="#b9c9cf")
    return scene


def replay(source_path, shot, output, movie):
    source = _read(source_path)
    point, row = next((i,r) for i,r in enumerate(source["rows"]) if r["shot"]["id"] == shot)
    arena = make_arena(source, row, point)
    observer = BladeContacts(arena)
    side = source.get("opening_side", -1)
    renderer = encoder = None
    frames = 0
    try:
        arena.launch(row["shot"])
        arena.partner.reset(seed=row["opponent_seed"])
        if movie:
            renderer = start_renderer(arena)
            encoder = start_encoder(movie, 30)
        with patch.object(arena, "_racket_contact", observer.observe):
            for tick, action in enumerate(row["actions"]):
                if encoder and tick % 2 == 0:
                    count = max(arena.legal_returns + arena.rival_returns - 1, 0)
                    name = "Continuously varied placement" if "cards" in source else "Changing paddle faces in a live rally"
                    encoder.stdin.write(draw_frame(arena, renderer, observer, name, count).tobytes())
                    frames += 1
                arena.exchange(action, np.zeros(5))
        if digest_json(row["events"]) != digest_json(arena.events):
            raise ValueError("Replayed physical events diverged")
        actual = audit(arena.events, opening_side=side)
        if actual["returns"] != row["score"]["returns"]:
            raise ValueError("Replayed legal returns diverged")
        contacts = observer.scored(arena.events, opening_side=side)
    finally:
        if encoder:
            encoder.stdin.close()
            code = encoder.wait()
        if renderer:
            renderer.close()
        arena.close()
    if encoder and code:
        raise RuntimeError(f"Video encoder exited with {code}")
    _write(output, dict(source_sha256=_hash(source_path), id=shot, events_sha256=digest_json(row["events"]),
                        contacts=contacts, summary=summarize([dict(contacts=contacts)]), returns=actual["returns"],
                        frames=frames, fps=30, video_sha256=_hash(movie) if movie else None))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("shot", type=int)
    p.add_argument("output", type=Path)
    p.add_argument("--movie", type=Path)
    args = p.parse_args()
    replay(args.source, args.shot, args.output, args.movie)
