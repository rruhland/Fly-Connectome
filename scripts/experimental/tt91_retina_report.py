"""Report true RGB timings and unchanged causal detector, estimator, and action bytes."""
from collections import Counter

import numpy as np

from . import tt91_retina_benchmark as bench
from . import tt91_retina_action as actor
from .tt4_transfer_trial import _hash,_read,_write

REPORT=bench.ROOT/"report.json"
PACK=bench.ROOT/"packing.json.gz"
PACK_PLAN=bench.ROOT/"packing.protocol.json"


def summarize(archive,protocol):
    r=_read(archive)
    p=_read(protocol)
    if r["plan"]!=_hash(protocol):
        raise ValueError("Changed optical protocol")
    modes=p["modes"]
    out=dict(complete=r["complete"],receipt=_hash(archive),rows=[],summary={})
    count=len(r["rows"])
    seconds=sum(row["native_seconds"] for row in r["rows"])
    active=sum(row["captured_seconds"] for row in r["rows"])
    for mode in modes:
        cols=[row["sensors"][mode] for row in r["rows"]]
        sums=Counter()
        for row in cols:
            for k in ("exposures","wide","valid","sensor_cpu_s","fit_cpu_s","crop_prepare_s","crop_render_s","submitted","culled"):
                if row[k] is not None:
                    sums[k]+=row[k]
        totals=Counter()
        for row in r["rows"]:
            totals.update(row["parity"].get(mode,{}))
        metrics=Counter()
        error=[]
        for row in r["rows"]:
            own=[]
            for ex in row["measures"]:
                ref=ex["estimates"]["original"]
                new=ex["estimates"][mode]
                metrics.update(consumed=1,fit=int(new is not None),
                               fit_availability_same=int((ref is None)==(new is None)),
                               fit_packet_exact=int(ref==new))
                if ref is None or new is None or ex["truth"]["pv"][4]>=-1.:
                    continue
                own.append(abs(np.r_[new["pv"],new["spin"]]-np.r_[ex["truth"]["pv"],ex["truth"]["spin"]]))
                metrics["matched_inbound_fits"]+=1
            if own:
                error.append(np.median(np.asarray(own),axis=0))
        cost=dict(ms_per_exposure=1000*sums["sensor_cpu_s"]/sums["exposures"],
                  cpu_seconds_per_captured_sim_second=sums["sensor_cpu_s"]/active,
                  cpu_seconds_per_actual_sim_second=sums["sensor_cpu_s"]/seconds,
                  sensor_cpu_s_per_offer=sums["sensor_cpu_s"]/count)
        color=Counter()
        for row in r["rows"]:
            c=row["frame_channels"].get(mode,{})
            color.update({k:v for k,v in c.items() if k!="max_abs"})
            color["max_abs"]=max(color["max_abs"],c.get("max_abs",0))
        out["summary"][mode]=dict(totals=dict(sums),parity=dict(totals),fits=dict(metrics),price=cost,
            released_median_abs_truth_error=None if not error else np.mean(np.asarray(error),axis=0).tolist(),
            raster_changes=dict(color))
    for row in r["rows"]:
        out["rows"].append(dict(id=row["case"]["id"],role=row["case"]["role"],native_prefix_exact=row["native_prefix_exact"],
                                exposures=len(row["clock"]),captures=[row["sensors"][mode]["valid"] for mode in modes],
                                measured_ms_per_exposure=[row["sensors"][mode]["sensor_cpu_s"]/len(row["clock"])*1000 for mode in modes],
                                points_exact=[row["parity"].get(mode,{}).get("optical_point_exact",len(row["clock"])) for mode in modes]))
    return out


def run():
    result=dict(paired=summarize(bench.OUT,bench.PLAN))
    if PACK.exists():
        result["packing_development"]=summarize(PACK,PACK_PLAN)
    if actor.OUT.exists():
        d=_read(actor.OUT)
        if d["plan"]!=_hash(actor.PLAN):
            raise ValueError("Changed action test assignment")
        p=_read(actor.PLAN)
        result["action"]=dict(receipt=_hash(actor.OUT),parity=d.get("parity"),id=p["item"]["card"]["id"],rows={})
        for mode,row in d["rows"].items():
            t=row["observer"]
            result["action"]["rows"][mode]=dict(score=row["score"]["learner"],opponent=row["score"]["opponent"],
                native_seconds=row["native_seconds"],ticks=len(row["actions"]),whole_cpu_s=row["cpu_s"],
                sensor_cpu_s=t["camera_cpu_seconds"],fit_cpu_s=t["fit_cpu_seconds"],
                exposures=t["camera_calls"],valid=t["valid_stereo"],selection=t["selection"],complete=row["complete"])
    _write(REPORT,result)
    print(result["paired"]["complete"],result["paired"]["summary"],flush=True)
    if "action" in result:
        print(result["action"],flush=True)


if __name__=="__main__":
    run()
