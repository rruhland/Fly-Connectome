"""Render the first registered drop from the three-way wrist calibration comparison."""
import argparse
import gzip
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt24_bounce import BounceArena
from .tt24_motor_movie import ACCENT, BACKGROUND, BOLD, FONT, INK, MUTED
from .tt24_world_arm import DIR


def frames(arena, row):
    arena.reset(seed=row["seed"], split="held", render=False)
    views = []
    for tick in range(len(row["actions"])+1):
        if tick % 2 == 0 or tick == len(row["actions"]):
            judge = arena.judge_metrics()
            views.append((arena.render_external(), arena.render_wrist(), judge["rebounds"], judge["done"]))
        if tick < len(row["actions"]):
            arena.step(row["actions"][tick], render=False)
    if judge["rebounds"] != row["score"] or judge["longest_consecutive_rebounds"] != row["run"]:
        raise ValueError("movie and held score differ")
    return views


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor",action="store_true")
    cfg=parser.parse_args()
    if cfg.donor:
        with gzip.open(DIR/"tt24_donor-refined.json.gz","rt") as src:
            doc=json.load(src)
        first=doc["held_stock"][0]
        rows={mode:dict(seed=first["seed"],actions=r["actions"],score=r["metrics"]["rebounds"],
                        run=r["metrics"]["longest_consecutive_rebounds"])
              for mode,r in first["conditions"].items()}
        names={"public":"Wrist camera policy","clean":"Perfect-state diagnostic","hold":"Stationary paddle"}
        horizon=300
        heading="The first held drop on a new arm"
        caption="Five-joint motor control learned from rebound rewards on three other arms."
        path=DIR/"donor-first-drop.mp4"
    else:
        with gzip.open(DIR/"public-carry.json.gz","rt") as src:
            doc=json.load(src)
        rows={mode:r[0] for mode,r in doc["modes"].items()}
        names={"carried":"Self-calibrated wrist","direct":"Previous calibration","hold":"Stationary paddle"}
        horizon=doc["horizon"]
        heading="One drop, three wrist conditions"
        caption="First held release. Both active arms use the same five-joint motor weights."
        path=DIR/"public-carry-first-drop.mp4"
    arena = BounceArena(view="none", width=192, height=144, max_steps=horizon)
    try:
        clips = {mode:frames(arena, rows[mode]) for mode in names}
    finally:
        arena.close()
    title = ImageFont.truetype(BOLD, 25)
    label = ImageFont.truetype(BOLD, 17)
    text = ImageFont.truetype(FONT, 15)
    size = (928, 592)
    command = ["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pixel_format","rgb24",
               "-video_size","928x592","-framerate","30","-i","-","-c:v","libx264",
               "-vf","tpad=stop_mode=clone:stop_duration=0.7","-pix_fmt","yuv420p","-preset","fast",
               "-crf","18","-movflags","+faststart",str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for tick in range(max(map(len, clips.values()))):
            canvas = Image.new("RGB",size,BACKGROUND)
            draw = ImageDraw.Draw(canvas)
            draw.text((16,9),heading,font=title,fill=INK)
            draw.text((16,44),caption,font=text,fill=MUTED)
            draw.text((912,17),f"{tick/30:.2f} s",anchor="ra",font=label,fill=MUTED)
            for col, (mode, name) in enumerate(names.items()):
                x = 16+col*304
                external, wrist, score, done = clips[mode][min(tick,len(clips[mode])-1)]
                draw.text((x,75),name,font=label,fill=INK)
                draw.text((x,99),f"{score} verified rebounds"+("   ·   ended" if done else ""),font=text,fill=ACCENT)
                for y, view in ((126,external),(370,wrist)):
                    canvas.paste(Image.fromarray(view).resize((288,216), Image.Resampling.NEAREST),(x,y))
                draw.text((x,346),"Wrist camera",font=text,fill=MUTED)
            encoder.stdin.write(np.asarray(canvas).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg failed")


if __name__ == "__main__":
    main()
