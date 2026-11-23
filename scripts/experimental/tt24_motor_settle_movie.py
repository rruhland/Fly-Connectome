"""Replay the first ten-bounce release with the newly learned interface, beside uncorrected control."""
import gzip
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_donor_trial import ROOT
from .tt24_motor_movie import ACCENT, BACKGROUND, BOLD, FONT, INK, MUTED


def frames(seed,case):
    arena=BounceArena(view="none",width=224,height=168,max_steps=420)
    try:
        arena.reset(seed=seed,split="held",render=False)
        views=[]
        for tick,action in enumerate(case["tape"]):
            arena.step(action,render=False)
            if tick%2==0 or tick+1==len(case["tape"]):
                views.append((arena.render_external(),arena.render_wrist(),len(arena._rebounds)))
        if digest_json(arena.judge_metrics())!=digest_json(case["metrics"]):
            raise ValueError("camera witness failed physical replay")
        return views
    finally:
        arena.close()


def main():
    with gzip.open(ROOT/"motor-settle-0.json.gz","rt") as f:
        session=json.load(f)["sessions"][1]
    game=next(g for g in session["games"] if g["conditions"]["calibrated"]["metrics"]["longest_consecutive_rebounds"]>=10)
    branches=[frames(game["seed"],game["conditions"][k]) for k in ("calibrated","frozen")]
    path=ROOT/"motor-interface-paired.mp4"
    size=(864,640)
    title=ImageFont.truetype(BOLD,23)
    large=ImageFont.truetype(BOLD,17)
    small=ImageFont.truetype(FONT,15)
    args=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","864x640","-r","30",
          "-i","-","-c:v","libx264","-vf","tpad=stop_mode=clone:stop_duration=1","-pix_fmt","yuv420p",
          "-preset","fast","-crf","20","-movflags","+faststart",str(path)]
    with subprocess.Popen(args,stdin=subprocess.PIPE) as proc:
        for t in range(max(map(len,branches))):
            img=Image.new("RGB",size,BACKGROUND)
            draw=ImageDraw.Draw(img)
            draw.text((16,10),"The same fall, with five unfamiliar motor controls",font=title,fill=INK)
            draw.text((16,45),"Both arms use the same wrist policy. Only one measured how its new commands move its joints.",font=small,fill=MUTED)
            for i,branch in enumerate(branches):
                x=16+428*i
                draw.text((x,77),("1.45 s of encoder calibration","Original motor mapping")[i],font=large,fill=INK)
                view,wrist,score=branch[min(t,len(branch)-1)]
                state="release finished" if t>=len(branch)-1 else "playing"
                draw.text((x,106),f"{score} qualified rebounds   ·   {state}",font=small,fill=ACCENT)
                img.paste(Image.fromarray(view).resize((405,272)),(x,136))
                draw.text((x,420),"Policy wrist camera",font=small,fill=MUTED)
                img.paste(Image.fromarray(wrist).resize((248,186)),(x,446))
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("video encoder failed")
    print(game["seed"],[b[-1][2] for b in branches],flush=True)


if __name__=="__main__":
    main()
