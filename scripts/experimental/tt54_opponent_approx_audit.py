"""Audit every approximate score on a real referee-verified normal physical action tape."""

import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt29_partner as opponent
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt48_rim_recovery import STOCK
from .tt54_opponent_approx import BASE, approximate, compiled

OUT=Path("artifacts/tt54_exact_speed/opponent-kernel.json")


def run(out,index):
    d=_read(STOCK)
    row,card=d["rows"][index],d["cards"][index]
    if compiled() is None:
        raise ValueError("No compiled kernel")
    results=[]
    def paired(*inputs):
        start=process_time();old=BASE(*inputs);stock_cpu=process_time()-start
        start=process_time();new=approximate(*inputs);fast_cpu=process_time()-start
        if not np.array_equal(old.view(np.uint64),new.view(np.uint64)):
            raise ValueError("Any finite or infinite candidate score changed")
        results.append(dict(rows=len(old),stock_cpu=stock_cpu,packed_cpu=fast_cpu,
                            scores_exact=True,sort_exact=bool(np.array_equal(np.argsort(old),np.argsort(new)))))
        return old
    with mpc.centered(d["ready"]),patch.object(opponent,"_flight_approx",paired):
        a=mpc.arena()
        try:
            a.launch(row["shot"],card=card)
            a.partner.reset(seed=card["seed"])
            for action in row["actions"]:
                a.exchange(action,mpc.FAR_COMMAND)
            actual=digest_json(a.events)
            if actual!=digest_json(row["events"]):
                raise ValueError("Scoring kernel audit disturbed the native executed tape")
        finally:
            a.close()
    q=dict(id=card["id"],input_sha256=_hash(STOCK),events_exact=True,event_sha256=actual,
           candidates=sum(r["rows"] for r in results),stock=sum(r["stock_cpu"] for r in results),
           packed=sum(r["packed_cpu"] for r in results),batches=results)
    _write(out,q)
    print({k:v for k,v in q.items() if k!="batches"},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--index",type=int,default=0)
    a=p.parse_args()
    run(a.out,a.index)
