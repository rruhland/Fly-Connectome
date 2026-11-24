"""Measure when a different broad random request first becomes visible on a common earned prefix."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from . import tt37_opponent as live
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, resources
from .tt37_cue_vision import PaddleEye
from .tt41_anticipate import OUT

TT40 = Path("artifacts/tt40")
STOCK = Path("artifacts/tt37/mix-rollin.json.gz")
WITNESS = TT40/"long-low-diverse-optical.json.gz"
DELTAS = (104729, 786433)


def bytes_ahead(frame, eye):
    result = {}
    for name in ("left", "right"):
        x0, y0, x1, y1 = eye.crop[name]
        result[name] = frame[name][y0:y1, x0:x1].copy()
    return result


def projected(offer):
    if offer is None or offer["receive_line"] is None:
        return None
    cross = offer["receive_line"]
    z = cross["xyz"][2]
    vy, vz = cross["velocity"][1:]
    if vy > -.9:
        return None
    dt = (-1.53-cross["xyz"][1])/vy
    return z+vz*dt-4.905*dt*dt


def trial(a, sensor, rig, c, r, leg, fork, seed):
    painter = PaddleEye({name: rig.fixed[name] for name in ("left","right")})
    stock_draw = live.draw
    captures, measures = {}, {}
    starts = np.zeros_like(r["actions"][0])
    a.launch(r["shot"], card=c)
    a.partner.reset(seed=c["seed"])

    def one_request(card, k, protocol, rng):
        # For a paired response-time assay, all other draws are precisely the original ones.
        if k == leg and seed is not None:
            card = dict(card,reply_seed=seed)
        return stock_draw(card,k,protocol,rng)

    with patch.object(live,"draw",one_request):
        for k in range(fork+round(2.10/a.control_dt)):
            if fork-12 <= k <= fork+22 and k%2==0:
                element = sensor.capture()
                captures[k] = bytes_ahead(element,painter)
                v = painter.update(k*a.control_dt,element)
                if v is not None:
                    measures[k] = v
            action = r["actions"][k] if k<len(r["actions"]) else starts
            _, _, ended, cap, _ = a.exchange(action,FAR_COMMAND)
            if ended or cap or (k>fork+23 and leg+1 in a.crossings):
                break
    placements = a.point_placements()
    reqs = placements["requested"]
    recv = placements["delivered"]
    if len(reqs) <= leg:
        raise ValueError("Expected reply target was never requested")
    near_time = fork*a.control_dt
    prefix = [e for e in a.events if e["time"] < near_time-1e-9]
    stock_prefix = [e for e in r["events"] if e["time"] < near_time-1e-9]
    if digest_json(prefix) != digest_json(stock_prefix):
        raise ValueError("Matched native event prefix before the preparation fork diverged")
    if [o["target"] for o in reqs[:leg]] != [o["target"] for o in r["placements"]["requested"][:leg]]:
        raise ValueError("Redraw changed a previous independent request")
    offer = recv[leg] if len(recv)>leg else None
    return dict(target=reqs[leg]["target"],draw_s=reqs[leg]["time_s"],
                native_height=projected(offer),native_far_hit_s=None if offer is None else offer["far_contact"]["time"],
                prefix_events_sha256=digest_json(prefix), frames=captures, motions=measures)


def paired(base, other, time):
    frames = []
    for k in sorted(base["frames"].keys()&other["frames"].keys()):
        px, contrast = 0,0
        for side in ("left","right"):
            lhs, rhs = base["frames"][k][side],other["frames"][k][side]
            d = np.max(np.abs(lhs.astype(np.int16)-rhs.astype(np.int16)),axis=2)
            px += int(np.sum(d>0))
            contrast += int(np.sum(d>=9))
        delta = None
        if k in base["motions"] and k in other["motions"]:
            x,y = base["motions"][k],other["motions"][k]
            delta = dict(center_m=float(np.linalg.norm(x[:3]-y[:3])),
                         speed_mps=float(np.linalg.norm(x[3:6]-y[3:6])))
        elif (k in base["motions"]) != (k in other["motions"]):
            delta = {"only_one_visible": True}
        frames.append(dict(tick=k,after_preparation_start_s=time(k),
                           changed_far_rgb_pixels=px,changed_high_contrast_pixels=contrast,blade_delta=delta))
    return dict(new_target=other["target"],new_draw_s=other["draw_s"],new_native_height=other["native_height"],
                new_native_far_hit_s=other["native_far_hit_s"],frames=frames,
                first_arbitrary_rgb_change_s=next((f["after_preparation_start_s"] for f in frames
                                                  if f["changed_far_rgb_pixels"]>0),None),
                first_10_high_contrast_pixels_s=next((f["after_preparation_start_s"] for f in frames
                                                    if f["changed_high_contrast_pixels"]>=10),None),
                first_triangulated_center_10mm_change_s=next((
                    f["after_preparation_start_s"] for f in frames
                    if (f["blade_delta"] or {}).get("center_m",0.)>.010),None))


def run(out):
    stereo_doc, stock = _read(WITNESS),_read(STOCK)
    tape = {(c["id"],c["seed"],c["reply_seed"]):(c,r) for c,r in zip(stock["cards"],stock["rows"],strict=True)}
    done, results = set(),[]
    with centered(stock["ready"]):
        a = live.arena()
        s, rig, *_ = resources(a)
        try:
            for w in stereo_doc["groups"]:
                card,leg,fork = w["card"],w["leg"],w["command_prefix_end_tick"]
                key = (card["id"],card["seed"],card["reply_seed"])
                if (key,leg) in done:
                    continue
                done.add((key,leg))
                c,r = tape[key]
                baseline = trial(a,s,rig,c,r,leg,fork,None)
                close = r["placements"]["delivered"][leg]
                if (baseline["native_height"] != projected(close)
                        or baseline["target"] != close["target"]):
                    raise ValueError("Stock continuation differs for role under audit")
                item = dict(id=c["id"],leg=leg,prep_start_s=fork*a.control_dt,
                            prep_duration_s=(w["preparation_end_tick"]-fork)*a.control_dt,
                            baseline_target=baseline["target"],baseline_draw_s=baseline["draw_s"],
                            baseline_native_height=baseline["native_height"],
                            baseline_far_hit_s=baseline["native_far_hit_s"],
                            prefix_events_sha256=baseline["prefix_events_sha256"],
                            contrasts=[])
                for offset in DELTAS:
                    alt = trial(a,s,rig,c,r,leg,fork,c["reply_seed"]+offset)
                    if alt["target"] == baseline["target"]:
                        raise ValueError("New independent draw accidentally repeats the stock request")
                    if baseline["prefix_events_sha256"] != alt["prefix_events_sha256"]:
                        raise ValueError("Draw had already altered the referee events before the fork")
                    difference = paired(baseline,alt,lambda k:(k-fork)*a.control_dt)
                    item["contrasts"].append(difference|{"alt_reply_seed":c["reply_seed"]+offset})
                print(item["id"],[[(v[k]) for k in ("new_native_height","first_arbitrary_rgb_change_s",
                                                     "first_triangulated_center_10mm_change_s")]
                                   for v in item["contrasts"]],flush=True)
                results.append(item)
                _write(out,dict(input_sha256={str(p):_hash(p) for p in (WITNESS,STOCK)},rows=results,
                                role="Counterfactual information audit only. No target or audited timestamp is an input to a policy.",
                                original_near_commands=True,complete=False))
        finally:
            s.close()
            a.close()
    result = _read(out)
    result["complete"] = True
    _write(out,result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT/"draw-visibility.json")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    run(args.out)
