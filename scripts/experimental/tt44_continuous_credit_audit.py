"""Audit actual new early motor rescues against their exact archived starts."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from .tt4_transfer_trial import _hash, _read, _write
from .tt44_continuous_credit import PAIRED, ROOT
from .tt44_paired_credit_audit import audit


def run(training, paired, output, ids):
    new,base=_read(training),_read(paired)
    if new["pairs_sha256"]!=_hash(paired):
        raise ValueError("Changed matched preparation references")
    rows=[]
    for r in new["rows"]:
        if r["stock_landed"] or not r["modified_landed"] or ids and r["id"] not in ids:
            continue
        paired_r=next(x for x in base["rows"] if x["id"]==r["id"])
        stock=r["trials"][0] if new["outbound"] else paired_r["variants"]["stock"]
        changed=next(x for x in r["trials"] if x["landed"] and any(x["parameters"]))
        rows.append(dict(r,variants=dict(stock=stock,continuous=changed)))
    if not rows:
        raise ValueError("No actual rescues to audit")
    with TemporaryDirectory(prefix="tt44-continuous-audit-") as d:
        candidate=Path(d)/"physical.json.gz"
        _write(candidate,dict(rows=rows))
        audit(candidate,output,len(rows),ids)
    receipt=_read(output)
    receipt.update(corpus_sha256=_hash(training),paired_sha256=_hash(paired))
    _write(output,receipt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training",type=Path,default=ROOT/"training.json.gz")
    p.add_argument("--paired",type=Path,default=PAIRED)
    p.add_argument("--output",type=Path,default=ROOT/"audit.json")
    p.add_argument("--ids",type=int,nargs="+")
    a=p.parse_args()
    run(a.training,a.paired,a.output,a.ids)
