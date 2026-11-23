"""Show the same reviewed release under native-selected and shuffled coordinate conventions."""
import json
import gzip
import subprocess

import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .tt24_bounce import BounceArena
from .tt24_donor_trial import ROOT
from .tt24_posture import WidePosture
from .tt24_ready import prepare_step
from .tt24_motor_movie import ACCENT, BACKGROUND, BOLD, FONT, INK, MUTED


def frames(row):
    server=BounceArena(view="none")
    arena=WidePosture(view="none",width=192,height=144,max_steps=360)
    try:
        seed=row["seed"]
        server.reset(seed=seed,split="held",render=False)
        location=server._ball_position.copy()
        assert np.array_equal(location,row["release_xyz"])
        arena.reset(seed=seed,split="held",render=False)
        arena._ball_position[:]=location
        arena._ball_velocity[:]=0.
        mujoco.mj_forward(arena._model,arena._data)
        result=[]
        lead=row["preparation_ticks"]
        for tick,a in enumerate(row["actions"]):
            if tick<lead:
                prepare_step(arena,np.array(a),location)
                if tick+1==lead:
                    arena._ball_velocity[:]=server._ball_velocity
                    arena._ball_spin[:]=server._ball_spin
                    mujoco.mj_forward(arena._model,arena._data)
            else:
                arena.step(a,render=False)
            if tick%2==0 or tick+1==len(row["actions"]):
                result.append((arena.render_external(),arena.render_wrist(),len(arena._rebounds),tick<lead))
        if arena.judge_metrics()["longest_consecutive_rebounds"]!=row["run"]:
            raise ValueError("rendered motor trajectory failed physical verification")
        return result
    finally:
        arena.close()
        server.close()


def main():
    with gzip.open(ROOT/"reference-bank-0.json.gz","rt") as f:
        group=json.load(f)
    stock=group["groups"]["stock"]["by_alpha"]
    p=stock[str(group["selections"]["native"])]
    s=stock[str(group["selections"]["shuffled"])]
    index=next(i for i in range(len(p)) if p[i]["run"]-s[i]["run"]>=10)
    curves=[frames(p[index]),frames(s[index])]
    title=ImageFont.truetype(BOLD,24)
    label=ImageFont.truetype(BOLD,17)
    txt=ImageFont.truetype(FONT,15)
    path=ROOT/"reference-transfer-paired.mp4"
    args=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","800x652",
          "-r","30","-i","-","-c:v","libx264","-vf","tpad=stop_mode=clone:stop_duration=0.7",
          "-pix_fmt","yuv420p","-preset","fast","-crf","19","-movflags","+faststart",str(path)]
    with subprocess.Popen(args,stdin=subprocess.PIPE) as encoder:
        for tick in range(max(map(len,curves))):
            canvas=Image.new("RGB",(800,652),BACKGROUND)
            draw=ImageDraw.Draw(canvas)
            draw.text((18,11),"What transfers from Pong to a wrist-controlled arm?",font=title,fill=INK)
            draw.text((18,48),"Identical ball release, arm posture and previously learned five-joint motor policy.",font=txt,fill=MUTED)
            for i,(clip,mode) in enumerate(zip(curves,("Pong reward · familiar reference","Shuffled reward · reset reference"))):
                x=18+391*i
                frame=clip[min(tick,len(clip)-1)]
                draw.text((x,77),mode,font=label,fill=INK)
                status="preparing" if frame[3] else "ended" if tick>=len(clip)-1 else "in flight"
                draw.text((x,103),f"{frame[2]} scored rebounds   ·   {status}",font=txt,fill=ACCENT)
                canvas.paste(Image.fromarray(frame[0]).resize((362,250),Image.Resampling.NEAREST),(x,132))
                draw.text((x,391),"Wrist RGB used for control",font=txt,fill=MUTED)
                canvas.paste(Image.fromarray(frame[1]).resize((316,232),Image.Resampling.NEAREST),(x,415))
            encoder.stdin.write(np.asarray(canvas).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("movie encoding failed")
    print("source replica 0, first ten-plus paired difference",index,p[index]["seed"],p[index]["run"],s[index]["run"])


if __name__=="__main__":
    main()
