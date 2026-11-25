"""Pair the fixed TT83 motor populations with TT75's independent causal RGB sensor."""
import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt83_population_eval as tt
from .tt75_dense_fit import DenseFlight
from .tt75_retina import Retina
from .tt83_observer_adapter import PublicTracker
from .tt4_transfer_trial import _hash, _read, _write

ROOT=tt.ROOT
PLAN=ROOT/"dense.protocol.json"
OUT=ROOT/"dense.forks.json.gz"
SETTINGS=Path("artifacts/tt75/protocol.json")
CAL=Path("artifacts/tt75/sensor-calibration.json")
RECOGNITION=Path("artifacts/tt75/held.json.gz")
BASE_OBSERVE=tt.school.observed_prefix


def freeze():
    old=tt.read_plan()
    p=_read(SETTINGS)
    if not _read(RECOGNITION)["complete"]:
        raise ValueError("Seal the stand-alone dense observer assessment first")
    cells=[x for x in old["cases"] if x["roster"]=="prior-veto"][:2]
    cells += [next(x for x in old["cases"] if x["roster"]==g) for g in ("independent-miss","routine")]
    _write(PLAN,dict(old,cases=cells,settings=p["settings"],
                     sources={**old["sources"],**{str(q):_hash(q) for q in (SETTINGS,CAL,RECOGNITION,tt.PLAN)}},
                     role="TT83 physical input paired on the same sealed fixed prefix, replacing only the public flight estimate. First two prior-veto roles, first independent loss, and the named routine control are chosen by roster order, independent of both branch outcomes. Prior motor basis, direct path budget, trust and issuance policy stay fixed. Fallback is the original optics when the dense estimate is unavailable. No truth values from the observer benchmark enter control."))


class Capture:
    """Optical host side. Receives only actual RGB frames and simulator clock."""

    def __init__(self,a,rig,wide,settings,start):
        self.a=a
        self.r=Retina(a,rig,wide)
        self.f=DenseFlight(self.r.period,settings)
        self.next=float(start)
        self.original=a._contacts

    def collect(self, incoming):
        self.original(incoming)
        t=float(self.a.data.time)
        if t>=self.next-1.e-8 and not (self.a.terminated or self.a.truncated):
            p,_,_=self.r.capture(t)
            self.f.ingest(t,p)
            self.next=t+self.r.period

    def close(self):
        self.r.close()


def provider(settings):
    def observed(a,cam,rig,card,row,tick,axes):
        s=Capture(a,rig,cam,settings,max(0,tick-92)*a.control_dt)
        try:
            with patch.object(a,"_contacts",s.collect):
                eye,x,q,ref,old,feature,hist=BASE_OBSERVE(a,cam,rig,card,row,tick,axes)
            t=float(a.data.time)
            bridge=PublicTracker(eye.eye)
            value=bridge.update(t,x,s.f.fit(t))
            if bridge.dense is not None:
                eye.eye=bridge
                previous=hist[-1][2]
                feature=tt.school.context(value,q,ref,(q-previous)/a.control_dt,bridge.latest.bounced)
            print("dense sensor",card["id"],s.r.calls,s.r.valid,bridge.dense is not None,flush=True)
            return eye,value,q,ref,old,feature,hist
        finally:
            s.close()
    return observed


def run(n):
    p=_read(PLAN)
    with patch.multiple(tt,PLAN=PLAN,HELD=OUT), patch.object(tt.school,"observed_prefix",provider(p["settings"])):
        tt.forks(n)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","forks"))
    p.add_argument("--n",type=int,default=4)
    a=p.parse_args()
    freeze() if a.mode=="plan" else run(a.n)
