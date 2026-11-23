"""Replay the first registered wrist drop through five physical gravity changes."""
import gzip
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt24_bounce import BounceArena
from .tt24_continuous_gravity import CAP, GRAVITIES, SEGMENT
from .tt24_donor_trial import ROOT
from .tt24_motor_movie import ACCENT, BACKGROUND, BOLD, FONT, INK, MUTED


def main():
    with gzip.open(ROOT/"continuous-gravity-optical.json.gz","rt") as f:
        row=json.load(f)["rows"][0]
    # The camera and simulation model dimensions are identical to the experiment.
    arena=BounceArena(view="none",width=192,height=144,max_steps=CAP)
    txt=ImageFont.truetype(FONT,16)
    large=ImageFont.truetype(BOLD,24)
    small=ImageFont.truetype(BOLD,16)
    out=ROOT/"continuous-gravity-first.mp4"
    args=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","840x436",
          "-r","30","-i","-","-vcodec","libx264","-pix_fmt","yuv420p","-preset","fast","-crf","19",
          "-movflags","+faststart",str(out)]
    try:
        nominal=arena._model.opt.gravity.copy()
        arena.reset(seed=row["seed"],split="held",render=False)
        measurements={r[0]:r[2] for r in row["measurements"]}
        estimate=1.
        with subprocess.Popen(args,stdin=subprocess.PIPE) as process:
            for tick,a in enumerate(row["actions"]):
                gravity=GRAVITIES[min(tick//SEGMENT,len(GRAVITIES)-1)]
                arena._model.opt.gravity[:]=nominal*gravity
                estimate=measurements.get(tick,estimate)
                _,_,done=arena.step(a,render=False)
                if tick%2==0 or done:
                    canvas=Image.new("RGB",(840,436),BACKGROUND)
                    brush=ImageDraw.Draw(canvas)
                    brush.text((18,12),"A wrist-controlled arm plays through five gravity changes",font=large,fill=INK)
                    brush.text((18,53),f"{(tick+1)/60:4.1f} s    ·    {len(arena._rebounds)} scored rebounds",font=small,fill=ACCENT)
                    brush.text((428,53),f"Gravity  {gravity:.2f} g     ·     Wrist estimate  {estimate:.2f} g",font=txt,fill=INK)
                    for x,fr,caption in [(18,arena.render_external(),"External view"),(428,arena.render_wrist(),"Only camera used by the arm")]:
                        brush.text((x,85),caption,font=small,fill=MUTED)
                        canvas.paste(Image.fromarray(fr).resize((390,292),Image.Resampling.NEAREST),(x,112))
                    brush.text((18,409),"First of 12 held releases. The arm also receives its five joint encoder readings.",font=txt,fill=MUTED)
                    process.stdin.write(np.asarray(canvas).tobytes())
            process.stdin.close()
            if process.wait():
                raise RuntimeError("video encoding failed")
        metrics=arena.judge_metrics()
        if metrics["rebounds"]!=row["score"] or metrics["longest_consecutive_rebounds"]!=row["run"]:
            raise ValueError("rendered trajectory diverged")
    finally:
        arena.close()


if __name__=="__main__":
    main()
