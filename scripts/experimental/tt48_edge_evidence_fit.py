"""Qualify a signed optical-outcome discrepancy gate before issuing an override."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt46_continuous_season import PLAN as SEASON_PLAN
from .tt48_edge_evidence import CAPTURE
from .tt48_contact_patch import ROOT
from .tt48_rim_recovery import glance

COLLECTED = ROOT/"discrepancy-fit.json"
SCALE = np.array([.22, 1.05, .21, 2.4, 3.5, .39, 2.7, 3.5, .35])
V_SCALE = np.array([1., 2.0, 2.4])


def pair_features(state, stock, correction, q4):
    a, b = np.asarray(stock["virtual_feedback"]), np.asarray(correction["virtual_feedback"])
    x = np.asarray(state)
    return np.r_[
        abs(x[0]), x[4], x[2], a[3], a[4],
        correction["virtual_alignment"] or 0.,
        b[3]-a[3], b[4]-a[4], q4,
    ]


def examples(data):
    out = []
    for case in data["rows"]:
        if not case["valid"] or "edge" not in case["arms"]:
            continue
        a,b=case["arms"]["stock"],case["arms"]["edge"]
        steps=min(len(a["issued"]),len(b["issued"]),17)
        wrist=float(np.asarray(b["issued"])[steps-1,3]-np.asarray(a["issued"])[steps-1,3])
        sensor=np.asarray(case["context"])
        f0,f1=(np.asarray(v["virtual_feedback"]) for v in (a,b))
        q=pair_features(sensor,a,b,wrist)
        real_a=a["measured"]["outgoing"] if a["measured"] is not None else None
        real_b=b["measured"]["outgoing"] if b["measured"] is not None else None
        # Retain single-arm RGB measurements even if the other side could
        # not be optically reconstructed. The separate paired contrast is
        # never filled with a native ball velocity.
        out.append(dict(
            id=case["id"], x=q.tolist(), nominal=f1[2:5].tolist(), nominal_p=f1[5:8].tolist(),
            nominal_before=f0[2:5].tolist(), native_legal=b["native_legal"], actual_legal=b["landed"],
            rgb_before=real_a,rgb_after=real_b,
            weak=bool(glance(f0) and not a["native_legal"]
                      and ((a["virtual_alignment"] or 0.)<.88 or f0[3]<.9 or f0[4]<.8)),
            virtual_contact=bool(f1[1]>.5 and .060<f1[8]<.081), radius=float(f1[8]),
            face=b["virtual_alignment"], stock_landed=a["landed"],
        ))
    return out


def expected(path, approach):
    """Conservative incoming-side ballistic envelope on camera-scored departure."""
    p, v = np.asarray(path), np.asarray(approach)
    if p[1]>=-.72 or p[1]<=-1.81 or v[1]<=1.8 or v[2]<=.4:
        return False
    # This is an economical ranker for evidence, not a law of legal contact.
    # Drag shrinks horizontal advance before the first possible table return.
    b = (v[2]+np.sqrt(max(0.,v[2]**2+19.62*(p[2]-.778))))/9.81
    y = p[1]+.93*b*v[1]
    x = p[0]+.93*b*v[0]
    net = -p[1]/(.93*v[1])
    z = p[2]+net*v[2]-4.905*net**2
    return bool(.08 < y < 1.27 and abs(x)<.58 and z>.956)


class Evidence:
    def __init__(self, rows):
        self.rows=list(rows)

    def assess(self,item,mode):
        c=np.asarray(item["x"])
        rows=[r for r in self.rows if r["id"]!=item["id"] and r["virtual_contact"] and r["rgb_after"] is not None]
        if not rows:
            return dict(accepted=False,reason="no independent RGB departure")
        keys=np.asarray([r["x"] for r in rows])
        dist=np.sqrt(np.mean(((keys-c)/SCALE)**2,axis=1))
        weights=np.exp(-.5*(dist/.92)**2)
        v0=np.asarray(item["nominal"])
        miss=np.asarray([np.asarray(r["rgb_after"])-r["nominal"] for r in rows])
        # A bounded signed discrepancy. The reversed control uses the same
        # measurements, supports, outcomes and direction magnitudes.
        residual=(weights @ miss)/(1.5+weights.sum())
        sign=-1. if mode=="reverse" else 1.
        alternate=v0+sign*np.clip(residual,[-.85,-2.2,-3.1],[.85,2.2,3.1])
        earned=np.asarray([r["actual_legal"] for r in rows],dtype=bool)
        p=float((.50+weights@earned)/(1.0+weights.sum()))
        close=weights>.23
        positive=int(np.count_nonzero(close&earned))
        negative=int(np.count_nonzero(close&~earned))
        strong=(positive>=2 and negative>=1 and p>=.55 and float(weights.sum())>=1.45)
        legal_est=expected(item["nominal_p"],alternate)
        eligible=(item["weak"] and item["virtual_contact"] and not item["native_legal"]
                  and item["face"] is not None and item["face"]>.66)
        # Preserve authoritative stock safety, model geometry and actuator
        # screens. The binary miss is overridden only for likely ball flight.
        accepted=bool(strong and eligible and legal_est)
        return dict(accepted=accepted,support_missed=negative,support_earned=positive,
                    p_physically_earned=p,corrected_velocity=alternate.tolist(),
                    geometry_eligible=bool(eligible),optical_outcome_support=float(weights.sum()),
                    corrected_arc_playable=legal_est)


def fit(data):
    rows=examples(data)
    predicted=[]
    for item in rows:
        model=Evidence(rows)
        methods={k:model.assess(item,k) for k in ("learned","reverse")}
        methods.update(
            nominal=dict(accepted=bool(item["weak"] and item["native_legal"]),
                         corrected_velocity=item["nominal"]),
            permissive=dict(accepted=bool(item["weak"] and item["virtual_contact"]),
                            corrected_velocity=item["nominal"]))
        predicted.append(dict(item,methods=methods))
    measures={}
    for variant in ("nominal","permissive","learned","reverse"):
        accepted=[r for r in predicted if r["methods"][variant]["accepted"]]
        rgb=[np.asarray(r["methods"][variant]["corrected_velocity"])-r["rgb_after"]
             for r in predicted if r["rgb_after"] is not None and "corrected_velocity" in r["methods"][variant]]
        measures[variant]=dict(
            accepted=len(accepted), true_rescues=sum(r["actual_legal"] and not r["stock_landed"] for r in accepted),
            failed_edges=sum(not r["actual_legal"] for r in accepted),
            replaced_already_good=sum(r["stock_landed"] for r in accepted),
            optical_labels=len(rgb), camera_departure_rmse_mps=float(np.sqrt(np.mean(np.asarray(rgb)**2))) if rgb else None,
        )
    m=measures["learned"]
    qualifies=(m["true_rescues"]>=2 and not (m["failed_edges"] or m["replaced_already_good"])
               and m["true_rescues"]>measures["reverse"]["true_rescues"]
               and m["camera_departure_rmse_mps"] is not None
               and m["camera_departure_rmse_mps"]<measures["nominal"]["camera_departure_rmse_mps"])
    return dict(
        rows=predicted, measures=measures, qualifies=qualifies,
        training_scope="One exact same-delivery edge contrast per release. Official physical scored and failed outcomes are retained, with RGB-measured velocities where visible. Each assessed release is removed entirely from its signed discrepancy bank.",
        decision_scope="The learned and reversed controls share fixed local support and use opposed vectors drawn from the same real observations. The permissive control shows the effect of opening the nominal gate without empirical support. No policy issue unless this retrospective evidence passes.",
    )


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=CAPTURE)
    p.add_argument("--out",type=Path,default=COLLECTED)
    a=p.parse_args()
    d=_read(a.input)
    if not d["complete"]:
        raise ValueError("Wait for the full fixed RGB practice roster")
    if {r["id"] for r in d["rows"]} & {c["id"] for c in _read(SEASON_PLAN)["assessment"]}:
        raise ValueError("A sealed assessment release cannot teach the discrepancy rule")
    r=fit(d)
    r.update(evidence_sha256=_hash(a.input),code_sha256=_hash(Path(__file__)),held_sha256=_hash(SEASON_PLAN))
    _write(a.out,r)
    print(r["measures"],"qualifies",r["qualifies"],flush=True)
