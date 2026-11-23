"""Render all three previously unseen measured feeds returned by the camera pupil."""

import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .tt4_transfer_trial import _hash, _read, _write
from .tt21_recovery_movie import font
from .tt25_pose_demonstrations import episode
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def run(source):
    path=OUTPUT/"recorded-nearby-lessons.json.gz"
    archive=_read(path)
    rows=[r for r in archive["cases"] if r["results"]["camera"]["legal"]]
    if not rows:
        raise ValueError("No held receiving point")
    shots={r["id"]:r for r in json.loads(source.read_bytes())}
    with np.load(OUTPUT/"recorded-pose-imitation.npz",allow_pickle=False) as model:
        memory=(model["image_keys"],model["joint_goals"],model["scale"])
    arena=ReferenceReceiver()
    target=OUTPUT/"recorded-nearby-returns.mp4"
    frames_written=0
    try:
        with mujoco.Renderer(arena.model,width=640,height=360) as renderer:
            command=["ffmpeg","-y","-nostdin","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
                     "-s","760x472","-r","30","-i","-","-an","-c:v","libx264","-crf","19","-threads","1",
                     "-pix_fmt","yuv420p","-movflags","+faststart",str(target)]
            with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
                for index,case in enumerate(rows,1):
                    shot=shots[case["id"]]
                    speed=np.linalg.norm([shot["vel_"+a] for a in "xyz"])
                    def frame(tick,game,landed=False):
                        nonlocal frames_written
                        canvas=Image.new("RGB",(760,472),"#13212a")
                        draw=ImageDraw.Draw(canvas)
                        draw.text((16,9),"Receiving a recorded rally shot",font=font(25,True),fill="white")
                        draw.text((17,45),f"Full-size simulated table  |  Incoming {speed:.1f} m/s  |  Shot {case['id']}",
                                  font=font(16),fill="#d1dfe8")
                        view=Image.fromarray(game.render_broadcast(renderer)).resize((736,384),Image.Resampling.BILINEAR)
                        canvas.paste(view,(12,70))
                        draw.rectangle((12,421,748,460),fill="#13212a")
                        status="Legal return landed" if landed else "Camera-controlled SO-101"
                        draw.text((17,431),status,font=font(18,True),fill="#a6efb0" if landed else "white")
                        draw.text((740,432),f"{index} of {len(rows)}   /   {tick/30:.2f} s",font=font(16),anchor="ra",fill="white")
                        encoder.stdin.write(canvas.tobytes())
                        frames_written += 1
                    outcome,_=episode(arena,shot,memory=memory,observer=frame)
                    if outcome!=case["results"]["camera"]:
                        raise ValueError("Spectator output changed a played shot")
                    for _ in range(18):
                        frame(outcome["frames"],arena,True)
                encoder.stdin.close()
                if encoder.wait()!=0:
                    raise RuntimeError("Video encoding failed")
        probe=subprocess.check_output(["ffprobe","-v","error","-show_entries","stream=nb_frames","-of","json",str(target)])
        if int(json.loads(probe)["streams"][0]["nb_frames"])!=frames_written:
            raise ValueError("Missing video frames")
        _write(target.with_suffix(".json"),dict(source=_hash(path),video=_hash(target),frames=frames_written,
               frame_rate=30,terminal_hold_frames=18,ids=[r["id"] for r in rows],verified=True))
    finally:
        arena.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind",type=Path)
    run(p.parse_args().deepmind)
