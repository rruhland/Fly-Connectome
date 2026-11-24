"""Capture causal RGB estimates, encoder history and separately marked simulation diagnostics."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read
from .tt29_fast_camera import native_ball_pixels
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt36_impact import triangulate
from .tt37_opponent import arena


def capture(paths, out, limit):
    groups = []
    sensor = []
    evaluator = []
    keys = set()
    for path in paths:
        d = _read(path)
        with centered(d["ready"]):
            a = arena()
            stereo, rig, *_ = resources(a)
            try:
                for card, row in zip(d["cards"], d["rows"], strict=False):
                    key = (card["id"], card["seed"], card["reply_seed"])
                    if key in keys:
                        continue
                    keys.add(key)
                    a.launch(row["shot"], card=card)
                    a.partner.reset(seed=card["seed"])
                    eye = Vision(rig, a.control_dt, raw=True)
                    for tick, cmd in enumerate(row["actions"]):
                        if tick % 2 == 0:
                            imgs = stereo.capture()
                            eye.update(tick, imgs, a.joints)
                            f = eye.eye.latest
                            rays = [
                                native_ball_pixels(imgs[s]) for s in ("left", "right")
                            ]
                            xyz = (
                                None
                                if any(r is None or r.clipped for r in rays)
                                else triangulate([r.center for r in rays], rig)
                            )
                            now = tick * a.control_dt
                            optical = (
                                np.full(8, np.nan)
                                if f is None
                                else np.r_[*f.advance(now), now - f.time, f.bounced]
                            )
                            sensor.append(
                                np.r_[
                                    now,
                                    np.full(3, np.nan) if xyz is None else xyz,
                                    optical,
                                ]
                            )
                            evaluator.append(
                                np.r_[
                                    a.ball,
                                    a.velocity,
                                    a.spin,
                                    a._last_hitter,
                                    a._received_bounce,
                                ]
                            )
                            groups.append(key)
                        a.exchange(cmd, FAR_COMMAND)
                    print(out, card["id"], len(sensor), flush=True)
                    if len(keys) >= limit:
                        break
            finally:
                stereo.close()
                a.close()
        if len(keys) >= limit:
            break
    np.savez_compressed(out, sensor=sensor, evaluation=evaluator, groups=groups)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("source", type=Path, nargs="+")
    p.add_argument("--limit", type=int, default=12)
    args = p.parse_args()
    capture(args.source, args.output, args.limit)
