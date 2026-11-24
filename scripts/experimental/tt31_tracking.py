"""Replay the learner's stereo tracking and score incoming predictions against a physical crossing."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_conditions import cooperative_targets
from .tt29_fast_camera import CachedStereoRenderer, native_ball_pixels
from .tt29_partner_fast import FastPartnerArena, NativeBallFlight
from .tt29_vision import FlightPractice, OpticalFlight, Rig


def crossing(start, end, y):
    if start[2] >= y > end[2]:
        return start + (end-start) * (y-start[2])/(end[2]-start[2])
    return None


def run(source, output):
    data = _read(source)
    plane = -1.10  # Ahead of the receiving blade and after most table bounces.
    arena = FastPartnerArena(strike_ready="lateral", fast_search=True, clearance_guard=True, velocity_guard=True,
                             rate_scale=.8, motor_envelope=True, guard_table=True, max_steps=9000,
                             targets=cooperative_targets("lateral","practice"), net_rescue=True, robust_contact=True)
    stereo = CachedStereoRenderer(arena)
    rig = Rig.from_arena(arena)
    practice = FlightPractice()
    nominal = NativeBallFlight(arena)
    rows = []
    try:
        for row in data["rows"]:
            arena.launch(row["shot"])
            arena.partner.reset(seed=row["opponent_seed"])
            rig.begin()
            eye = OpticalFlight(rig, practice, learn=True, detector=native_ball_pixels)
            forecasts, actual = [], None
            first_motion = None
            for i, action in enumerate(row["actions"]):
                t = i * arena.control_dt
                before = np.r_[arena.data.time, arena.ball]
                untouched = arena.contacts == 0
                if first_motion is None and np.max(np.abs(action)) > .01:
                    first_motion = t
                if i % 2 == 0:
                    eye.observe(t, stereo.capture(), arena.joints)
                    f = eye.latest
                    if (actual is None and untouched and f is not None and t-f.time < .105 and
                            f.velocity[1] < -.9 and f.rms < 2.4 and f.depth_sigma < .16):
                        p, v = f.advance(t)
                        path = nominal.path(p, v, np.zeros(3), 1, bool(f.bounced and p[1] < .1))
                        if len(path) > 1 and path[-1,2] < plane < path[0,2]:
                            projection = [np.interp(plane, path[::-1,2], col[::-1]) for col in path.T[:4]]
                            forecasts.append(dict(t=t, arrival=t+projection[0], x=projection[1], z=projection[3]))
                arena.exchange(action, np.zeros(5))
                if untouched and arena.contacts == 0 and actual is None:
                    actual = crossing(before, np.r_[arena.data.time, arena.ball], plane)
            if digest_json(arena.events) != digest_json(row["events"]):
                raise ValueError("Tracking replay diverged")
            samples = []
            if actual is not None:
                for advance in (.30,.20,.10):
                    available = [f for f in forecasts if f["t"] <= actual[0] - advance]
                    if available:
                        f = available[-1]
                        samples.append(dict(lead=advance, age_s=actual[0]-f["t"], ex=f["x"]-actual[1],
                                            ez=f["z"]-actual[3], et=f["arrival"]-actual[0]))
            rows.append(dict(id=row["shot"]["id"], landed=row["score"]["learner"], plane_y_m=plane,
                             measured_crossing=None if actual is None else actual.tolist(),
                             first_motion_s=first_motion, forecasts=forecasts, samples=samples))
    finally:
        stereo.close()
        arena.close()
    _write(output,dict(source_sha256=_hash(source),rows=rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.source,args.output)
