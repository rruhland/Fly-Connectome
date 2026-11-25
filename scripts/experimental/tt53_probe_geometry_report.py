"""Aggregate identical-budget probe geometry comparisons by decision and release."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt53_probe_geometry import OUT, PLAN, TRIAL
from .tt53_signed_practice import ROOT


def summary(p,rows):
    tally=[]
    for r in rows:
        required=r["exact_per_axis_marginals"]
        if not required or len(r["learned_legal"])!=4 or len(r["sham"])!=4:
            raise ValueError("Probes no longer equal budget")
        a=int(sum(r["learned_legal"]))
        b=int(sum(v["scored"] for v in r["sham"]))
        tally.append(dict(id=r["id"],tick=r["tick"],learned=a,sham=b,
                          sham_matched=sum(v["matched"] for v in r["sham"])))
    releases=set(t["id"] for t in tally)
    first=[next(t for t in tally if t["id"]==r) for r in sorted(releases)]
    groups={k:sorted({t["id"] for t in tally if t[k]}) for k in ("learned","sham")}
    return dict(n=len(tally),releases=len(releases),equal_marginals=len(tally),tally=tally,
                probe_legal={k:sum(t[k] for t in tally) for k in groups},
                forks_rescued={k:sum(t[k]>0 for t in tally) for k in groups},
                unmatched_sham=int(sum(4-t["sham_matched"] for t in tally)),
                unique_releases_any_fork=groups,
                unique_releases_first_fork={k:[t["id"] for t in first if t[k]] for k in groups},
                learned_only_release=sorted(set(groups["learned"])-set(groups["sham"])),
                sham_only_release=sorted(set(groups["sham"])-set(groups["learned"])),
                source_sha256=_hash(TRIAL),plan_sha256=_hash(p))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=OUT)
    p.add_argument("--out",type=Path,default=ROOT/"geometry-report.json")
    a=p.parse_args()
    data=_read(a.input)
    if data["protocol_sha256"]!=_hash(PLAN):
        raise ValueError("Wrong geometry schedule")
    scored=summary(PLAN,data["rows"])
    _write(a.out,dict(scored,complete=data["complete"],input_sha256=_hash(a.input)))
    print({key:value for key,value in scored.items() if key!="tally"},flush=True)
