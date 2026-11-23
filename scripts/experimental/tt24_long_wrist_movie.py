"""Present one complete 120-second release from verified commands with real-time and accelerated sections."""
import gzip
import json
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm
from .tt24_long_wrist import CAP
from .tt24_motor_movie import ACCENT, BACKGROUND, BOLD, FONT, INK, MUTED
from .tt24_ready import prepare_step


def main():
    with gzip.open(ROOT/"long-wrist-4.json.gz","rt") as f:
        r=json.load(f)
    plan=r["plan"]
    row=r["results"]["live"]
    arm=InstalledArm(plan["body"])
    env=arm.arena
    env.max_steps=CAP
    server=BounceArena(view="none")
    title=ImageFont.truetype(BOLD,24)
    text=ImageFont.truetype(FONT,16)
    label=ImageFont.truetype(BOLD,17)
    target=ROOT/"long-wrist-428.mp4"
    result=[]
    cmd=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","768x430","-r","30",
         "-i","-","-c:v","libx264","-vf","tpad=stop_mode=clone:stop_duration=1","-pix_fmt","yuv420p",
         "-preset","fast","-crf","20","-movflags","+faststart",str(target)]
    try:
        server.reset(seed=plan["launcher_seed"],split="held",render=False)
        location=server._ball_position.copy()
        arm.reset(plan,location,False)
        lead=plan["prep_ticks"]
        changes=dict(plan["changes"])
        gravity=1.
        switch_count=0
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
            for t,action in enumerate(row["actions"]):
                tick=t-lead
                if tick in changes:
                    gravity=changes[tick]
                    result.append(arm.switch_gravity(tick,gravity))
                    switch_count+=1
                if t<lead:
                    prepare_step(env,np.array(action),location)
                    if t+1==lead:
                        env._ball_velocity[:]=server._ball_velocity
                        env._ball_spin[:]=server._ball_spin
                        mujoco.mj_forward(env._model,env._data)
                else:
                    env.step(action,render=False)
                normal=tick<300 or 3450<=tick<3690 or tick>=CAP-240
                interval=2 if normal else 16
                if tick%interval and tick!=CAP-1:
                    continue
                since=max(0,tick+1)/60.
                canvas=Image.new("RGB",(768,430),BACKGROUND)
                pen=ImageDraw.Draw(canvas)
                pen.text((16,12),"Two minutes of wrist-controlled bouncing",font=title,fill=INK)
                pen.text((16,51),f"{len(env._rebounds)} qualified rebounds   ·   {since:05.1f} s   ·   {'real time' if normal else '8× playback'}",font=label,fill=ACCENT)
                pen.text((16,90),"Outside view",font=text,fill=MUTED)
                pen.text((394,90),"Wrist camera used for play",font=text,fill=MUTED)
                for x,img in ((16,env.render_external()),(394,env.render_wrist())):
                    canvas.paste(Image.fromarray(img).resize((355,266)),(x,117))
                pen.text((16,397),f"Gravity {gravity:.2f} × nominal     ·     {switch_count} changes so far",font=text,fill=INK)
                proc.stdin.write(canvas.tobytes())
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("video encoding failed")
        if digest_json(env.judge_metrics())!=digest_json(row["metrics"]) or digest_json(result)!=digest_json(row["events"]):
            raise ValueError("rendered video did not reproduce the full physical record")
    finally:
        env.close()
        server.close()


if __name__=="__main__":
    main()
