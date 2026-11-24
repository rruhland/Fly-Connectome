"""Export only observable sensory/issued-action histories from answered physical rescues."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_fast_camera import native_ball_pixels
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_opponent import arena

OUT = Path('artifacts/tt40')


def playback(a, camera, rig, entry, exposures=True):
    r = entry['result']
    a.launch(r['shot'],card=entry['card'])
    a.partner.reset(seed=entry['card']['seed'])
    eye = Vision(rig,a.control_dt,raw=True)
    steps = []
    fork = r['fork_tick']
    low = max(0,fork-14)
    for tick, cmd in enumerate(r['actions']):
        q = a.joints.copy()
        if tick >= low and exposures:
            frame = camera.capture() if tick%2==0 else None
            eye.update(tick,frame,q)
            f = eye.eye.latest
            now = tick*a.control_dt
            if f is not None and now-f.time < .104 and f.rms < 2.4 and f.depth_sigma < .16:
                lead = np.r_[*f.advance(now)].tolist()
                bounce = bool(f.bounced)
            else:
                lead = bounce = None
            pixels = {}
            if frame is not None:
                for k, image in frame.items():
                    seen = native_ball_pixels(image)
                    pixels[k] = None if seen is None or seen.clipped else seen.center.tolist()
            steps.append(dict(tick=tick, tick_from_fork=tick-fork, own_encoders=q.tolist(),
                              issued=cmd, stereo_centers_px=pixels, rgb_flight=lead,
                              visually_bounced=bounce))
        a.exchange(cmd,FAR_COMMAND)
    if digest_json(a.events)!=digest_json(r['events']):
        raise ValueError('Physical rescue did not reproduce on untouched court')
    if digest_json(audit(a.events,opening_side=1))!=digest_json(r['score']):
        raise ValueError('Score did not independently reproduce')
    return steps


def export(sources,output):
    sheets = [_read(s) for s in sources]
    if any(not d.get('complete') for d in sheets):
        raise ValueError('Do not export a partial physical case sequence')
    d = sheets[0]
    if any(s['ready'] != d['ready'] or s['source_sha256'] != d['source_sha256'] for s in sheets):
        raise ValueError('Observable examples mix different fixture or development tapes')
    items = []
    issued = set()
    with centered(d['ready']):
        a = arena(protocol='random')
        s, rig, *_ = resources(a)
        try:
            for source, sheet in zip(sources,sheets,strict=True):
                for entry in sheet['rows']:
                    r = entry['result']
                    if r is None or not r['answered'] or digest_json(r['actions']) in issued:
                        continue
                    issued.add(digest_json(r['actions']))
                    optical = playback(a,s,rig,entry)
                    items.append(dict(card=entry['card'], source_leg=entry['leg'], origin=source.name,
                                      fork_tick=r['fork_tick'], physical_legal_and_answered=True,
                                      frames=optical))
                    print(entry['id'],entry['leg'],source.name,len(optical),flush=True)
        finally:
            s.close()
            a.close()
    _write(output,dict(schema='tt40-low-observable-examples-1',
                       provenance='coach selected the tape once, not a deployed policy',
                       sources={s.name:_hash(s) for s in sources}, camera_resolution=[384,288], capture_hz=30,
                       groups=items))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,nargs='+',default=[OUT/'low-feasibility.json.gz'])
    p.add_argument('--output',type=Path,default=OUT/'low-feasibility-examples.json.gz')
    args=p.parse_args()
    export(args.source,args.output)
