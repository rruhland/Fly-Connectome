"""Release-held precision, recall and timing for the causal TT41 low-ball observer."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_cue_fit import predict, solve, weights
from .tt41_anticipate import GATES, LOW_THRESHOLDS, OUT, outbound_projection, physical_projection

MODELS = ("mean", "ball", "paddle", "ball_paddle")
PENALTIES = (.015, .06, .24, .96, 3.84)
DEADLINES = (.88, 1.0, 1.20, 1.33)
HELD_SOURCE = "artifacts/tt39/endurance-05.json.gz"


def datum(marker, mode):
    p, v = np.asarray(marker["ball"]).reshape(2, 3)
    x, _, _ = p
    vx, _, _ = v
    base = np.r_[p, v, abs(x), abs(vx), x*vx]
    if marker["gate"].startswith("out_"):
        bounce = outbound_projection(marker["ball"],marker["bounced"])
        if bounce is None:
            b = np.array([0.,.75,0.,0.,0.])
        else:
            b = np.asarray(bounce)
        base = np.r_[base,b,abs(b[0]),np.maximum(0.,np.array([.30,.65,1.05])-b[1]),
                     bounce is not None,marker["bounced"]]
    else:
        projection = physical_projection(marker["ball"], marker["bounced"])
        previous = marker["previous_out"] or [0.]*6
        base = np.r_[base, previous, projection or [0.,1.03,0.], projection is not None,
                     marker["bounced"]]
    if mode == "ball":
        return base
    q, w, contour = np.split(np.asarray(marker["paddle"]), [3, 6])
    xx, yy, xy = contour.reshape(2,3).T
    trace = np.maximum(xx+yy,1e-5)
    paddle = np.r_[q,w,contour,(xx-yy)/trace,2*xy/trace,
                   q[0]*w[0],abs(q[0]),abs(w[0]),np.linalg.norm(w)]
    return paddle if mode == "paddle" else np.r_[base,paddle]


def examples(document, sources, *, min_leg=1):
    all_rows = []
    seen = set()
    for point in document["rows"]:
        if point["source"] not in sources:
            continue
        c = point["card"]
        for e in point["rows"]:
            if e["leg"] < min_leg:
                continue
            key = (c["id"], c["seed"], c["reply_seed"], e["leg"])
            if key in seen:
                raise ValueError("Duplicate optical opportunity")
            seen.add(key)
            all_rows.append(dict(e, uid=key, release=c["id"], source=point["source"]))
    return all_rows


def group_folds(groups):
    releases = np.unique(groups)
    folds = np.array_split(np.random.default_rng(411917).permutation(releases), min(5,len(releases)))
    for fold in folds:
        yield np.isin(groups, fold)


def regression(rows, gate, mode):
    if len(rows) < 24 or len(set(r["release"] for r in rows)) < 7:
        return None, None
    marker = [r["gate"][gate] for r in rows]
    x = np.asarray([datum(m, mode) for m in marker])
    y = np.asarray([r["optical"]["height"] for r in rows])
    g = np.asarray([r["release"] for r in rows])
    scores = np.empty((len(PENALTIES), len(rows)))
    for take in group_folds(g):
        for j, reg in enumerate(PENALTIES):
            beta = solve(x[~take], y[~take, None], g[~take], reg)
            scores[j,take] = predict(beta,x[take])[:,0]
    w = weights(g)
    q = np.sum(w*np.abs(scores-y)*(1.+1.5*(y < .99)), axis=1)
    j = int(np.argmin(q))
    return solve(x,y[:,None],g,PENALTIES[j]), scores[j]


def rank_table(scores, positives):
    """All equal-score examples enter together, including the prevalence baseline."""
    finite = np.isfinite(scores)
    indices = np.flatnonzero(finite)[np.argsort(scores[finite], kind="stable")]
    if not len(indices):
        return []
    boundaries = np.r_[np.flatnonzero(np.diff(scores[indices]) > 1e-10), len(indices)-1]
    truth = positives[indices]
    tp = np.cumsum(truth)[boundaries]
    n = boundaries+1
    return list(zip(scores[indices[boundaries]], n, tp, strict=True))


def lock(scores, positives, target):
    choices = [p for p in rank_table(scores, positives) if p[1] >= 4 and p[2] >= 3 and p[2]/p[1] >= target]
    if not choices:
        return None
    return float(max(choices, key=lambda p: (p[2], p[2]/p[1], -p[0]))[0])


def average_precision(scores, positives):
    n = sum(positives)
    if n == 0:
        return None
    pairs = rank_table(scores, positives)
    if not pairs:
        return None
    prev, ap = 0,0.
    for _,window,tp in pairs:
        ap += (tp-prev)/n * tp/window
        prev = tp
    return ap


def precision_recall_curve(scores, positives, *, maximum=24):
    if not positives.any():
        return []
    ranks = rank_table(scores,positives)
    if not ranks:
        return []
    positions = np.unique(np.linspace(0,len(ranks)-1,min(maximum,len(ranks))).round().astype(int))
    return [dict(threshold=float(ranks[i][0]),calls=int(ranks[i][1]),
                 precision=float(ranks[i][2]/ranks[i][1]),recall=float(ranks[i][2]/sum(positives)))
            for i in positions]


def grouped_gain(left, right, positives, groups):
    """Conditional paired uncertainty, resampling whole releases and keeping model fits frozen."""
    ids = np.unique(groups)
    if len(ids) < 5 or int(sum(positives)) < 2:
        return None
    rng = np.random.default_rng(411191)
    pools = [np.flatnonzero(groups==i) for i in ids]
    gains = []
    for picks in rng.integers(len(ids),size=(600,len(ids))):
        sample = np.concatenate([pools[j] for j in picks])
        y = positives[sample]
        if y.any():
            gains.append(average_precision(left[sample],y)-average_precision(right[sample],y))
    return dict(nonempty_resamples=len(gains), interval_95=np.quantile(gains,[.025,.975]).tolist()) if gains else None


def operation(scores, labels, cutoff, lead, bounce=None):
    direct = np.full(len(scores), False) if cutoff is None else scores <= cutoff
    z = dict(cutoff=cutoff, flagged=int(sum(direct)), true_low=int(sum(direct & labels)),
             conditional_precision=float(np.mean(labels[direct])) if direct.any() else None,
             eligible_recall=float(sum(direct & labels)/sum(labels)) if labels.any() else None)
    for name, time in (("deadline",lead),("deadline_before_bounce",bounce)):
        if time is not None:
            z[name] = {str(t): dict(flagged_in_time=int(sum(direct & (time>=t))),
                                   low_in_time=int(sum(direct & labels & (time>=t))),
                                   recall_in_time=float(sum(direct & labels & (time>=t))/sum(labels)) if labels.any() else None)
                       for t in DEADLINES}
    return z


def split(document, held):
    if held not in document["sources"]:
        raise ValueError("Held optical tape not present")
    train = examples(document, set(document["sources"])-{held})
    validation = examples(document, {held})
    overlap = {r["release"] for r in validation} & {r["release"] for r in train}
    disjoint = [r for r in train if r["release"] not in overlap]
    return disjoint, validation, sorted(overlap)


def eval_gate(train, held, gate):
    train_gate = [r for r in train if r["optical"] is not None and gate in r["gate"]]
    train_pair = [r for r in train_gate if r["gate"][gate]["paddle"] is not None]
    other = [r for r in held if r["optical"] is not None]
    valid = np.array([gate in r["gate"] for r in other], dtype=bool)
    paired = np.array([gate in r["gate"] and r["gate"][gate]["paddle"] is not None for r in other], dtype=bool)
    if not paired.any() or len(train_pair) < 24:
        return dict(train=len(train_gate), train_paired=len(train_pair), held=len(other),
                    held_gate=int(valid.sum()), held_paired=int(paired.sum())), {}
    truth = np.asarray([r["optical"]["height"] for r in other])
    fit_truth = np.asarray([r["optical"]["height"] for r in train_pair])
    t = np.asarray([r["gate"][gate]["decision_s"] if ok else np.nan for r,ok in zip(other,valid,strict=True)])
    facing = np.asarray([r["far_contact_s"] for r in other])-t
    bounce = np.asarray([r["near_bounce_s"] for r in other])-t
    lead = np.asarray([r["optical"]["at_s"] for r in other])-t
    times = lambda a: np.quantile(a[paired],[.10,.5,.90]).tolist()
    values = dict(train=len(train_gate), train_paired=len(train_pair), held=len(other),
                  held_gate=int(valid.sum()), held_paired=int(paired.sum()),
                  far_lead_s=times(facing), bounce_lead_s=times(bounce), receive_lead_s=times(lead),
                  held_below={str(x):int(sum(truth<x)) for x in LOW_THRESHOLDS}, fitted={})
    bank, forecasts = {},{}
    for mode in MODELS + (("outbound_depth",) if gate.startswith("out_") else ("stock_projection",)):
        if mode == "mean":
            beta = None
            p = np.full(len(other), float(np.mean(fit_truth)))
            oof = np.full(len(train_pair), float(np.mean(fit_truth)))
        elif mode in ("stock_projection","outbound_depth"):
            def physical(row):
                m = row["gate"][gate]
                project = physical_projection if mode == "stock_projection" else outbound_projection
                value = project(m["ball"],m["bounced"])
                return np.nan if value is None else value[1]

            beta = None
            p = np.asarray([physical(r) if ok else np.nan for r,ok in zip(other,valid,strict=True)])
            oof = np.asarray([physical(r) for r in train_pair])
        else:
            beta, oof = regression(train_pair, gate, mode)
            if beta is None:
                continue
            valid_mode = valid if mode == "ball" else paired
            p = np.asarray([
                float(predict(beta,datum(r["gate"][gate],mode))[0]) if ok else np.nan
                for r,ok in zip(other,valid_mode,strict=True)
            ])
        if beta is not None:
            bank[mode] = beta
        forecasts[mode] = p
        estimate = np.isfinite(p) & valid
        compare = paired & estimate
        result = dict(mae_m=float(np.mean(np.abs(truth[compare]-p[compare])))
                      if compare.any() and mode != "outbound_depth" else None)
        for limit in LOW_THRESHOLDS:
            labels, tr = truth<limit, fit_truth<limit
            cutoff = {f"{p:.2f}":lock(oof,tr,p) for p in (.60,.80)}
            # AP/MAE is a like-for-like paired comparison, while actual policy recall counts
            # each optically labelled opportunity in the denominator including absent gates.
            result[str(limit)] = dict(ap_paired=average_precision(p[compare],labels[compare]),
                diagnostic_pr_curve_paired=precision_recall_curve(p[compare],labels[compare]),
                paired_low=int(np.sum(labels&compare)), train_paired_low=int(sum(tr)),
                operations={k:operation(np.where(estimate,p,np.nan),labels,c,lead,bounce) for k,c in cutoff.items()})
            if mode == "outbound_depth":
                # Diagnostic transfer of the physically tested coarse shortness flag.
                result[str(limit)]["fixed_short_rule"] = operation(
                    np.where(estimate&(p>.05),p,np.nan),labels,.65,lead,bounce)
        values["fitted"][mode] = result
    if "ball" in forecasts and "ball_paddle" in forecasts:
        gold = np.asarray([r["release"] for r in other])[paired]
        values["paired_ap_gain_bootstrap"] = {
            str(h):grouped_gain(forecasts["ball_paddle"][paired],forecasts["ball"][paired],truth[paired]<h,gold)
            for h in LOW_THRESHOLDS}
    return values, bank


def optical_availability(document, rows):
    tapes = {p:_read(Path(p)) for p in document["sources"]}
    index = {}
    for name, d in tapes.items():
        for c, row in zip(d["cards"],d["rows"]):
            for leg, offer in enumerate(row["placements"]["delivered"]):
                recv = offer["receive_line"]
                if recv is None or recv["velocity"][1] >= -.90:
                    continue
                z=recv["xyz"][2]
                arc=(-1.53-recv["xyz"][1])/recv["velocity"][1]
                height=z+arc*recv["velocity"][2]-.5*9.81*arc*arc
                index[(name,c["id"],c["seed"],c["reply_seed"],leg)] = height
    errors, heights, checked = [],[],[]
    for row in rows:
        key=(row["source"],*row["uid"])
        if key in index:
            actual=index[key]
            checked.append((actual,row["optical"] is not None))
            if row["optical"] is not None:
                errors.append(row["optical"]["height"]-actual)
                heights.append(actual)
    return dict(n_all=len(rows), optical_labels=sum(r["optical"] is not None for r in rows),
                native_crossing_pairs=len(checked), paired_labels=len(errors),
                comparison_native_gravity_only_mae_m=float(np.mean(np.abs(errors))) if errors else None,
                selection_audit={str(t):{
                    "native_low":sum(h<t for h,_ in checked),
                    "native_low_labelled":sum(h<t and ok for h,ok in checked),
                    "same_binary":sum((h<t)==(h+e<t) for h,e in zip(heights,errors,strict=True))
                } for t in LOW_THRESHOLDS})


def run(path, out, models, held_source):
    d = _read(path)
    train, held, overlap = split(d, held_source)
    report = dict(source_sha256=_hash(path), complete=d["completed"], held_source=held_source,
                  held_releases=len(set(r["release"] for r in held)), train_removed_release_overlap=overlap,
                  later_only=True, audit=dict(held=optical_availability(d,held),train=optical_availability(d,train)),
                  gates={})
    bank = {}
    for gate in GATES:
        report["gates"][gate], bank[gate] = eval_gate(train,held,gate)
    _write(out,report)
    if models is not None:
        _write(models,dict(receipts_sha256=_hash(path),held=held_source,model=bank))
    print(report["complete"],report["audit"],flush=True)
    for gate,r in report["gates"].items():
        if "fitted" in r:
            print(gate,r["held_paired"],round(r["receive_lead_s"][1],3),
                  {m:(round(v["mae_m"],3) if v["mae_m"] is not None else None,
                      round(v["0.99"]["ap_paired"],3) if v["0.99"]["ap_paired"] is not None else None,
                      v["0.99"]["operations"]["0.60"]["true_low"],
                      v["0.99"]["operations"]["0.60"]["flagged"])
                   for m,v in r["fitted"].items()},flush=True)


def teacher_timing(out):
    """Compare the earliest trustworthy inward camera fit to actually issued low preparation."""
    sensor = Path("artifacts/tt40/long-low-diverse-optical.json.gz")
    physical = Path("artifacts/tt40/long-low-diverse-audit.json")
    doc, audit = _read(sensor), _read(physical)
    by_sha = {r["events_sha256"]:r for r in audit["rows"]}
    rows = []
    for group in doc["groups"]:
        witness = by_sha[group["events_sha256"]]
        leg = group["leg"]
        contacts = witness["contacts"]
        far = [e for e in contacts if e["side"] == -1][leg]["time_s"]
        near = next(e["time_s"] for e in contacts if e["side"] == 1 and e["time_s"] > far)
        hz = doc["control_hz"]
        prep = group["command_prefix_end_tick"]/hz
        ready = group["preparation_end_tick"]/hz
        eligible = [f for f in group["frames"] if f["rgb_flight"] is not None and
                    f["rgb_flight"][4] < -1.15 and
                    any(v is not None for v in f["stereo_centers_px"].values()) and far <= f["tick"]/hz < near]
        first = min(eligible,key=lambda f:f["tick"],default=None)
        t = None if first is None else first["tick"]/hz
        lookahead = []
        for f in group["frames"]:
            ft = f["tick"]/hz
            if not (prep-.20 <= ft < far and f["rgb_flight"] is not None
                    and any(v is not None for v in f["stereo_centers_px"].values())):
                continue
            outbound = outbound_projection(f["rgb_flight"],f["visually_bounced"])
            if outbound is not None:
                lookahead.append(dict(time_s=ft,after_preparation_start_s=ft-prep,
                                      outward_y=f["rgb_flight"][1],estimated_far_bounce_x=outbound[0],
                                      estimated_far_bounce_y=outbound[1]))
        consistent_low = []
        for f in eligible:
            prediction = physical_projection(f["rgb_flight"],f["visually_bounced"])
            if prediction is not None and prediction[1] < .99:
                consistent_low.append((f["tick"]/hz,prediction[1]))
        rows.append(dict(id=group["card"]["id"],leg=leg,origin=group["origin"],
                         planned_preparation_s=ready-prep,preparation_start_s=prep,
                         preparation_end_s=ready,far_hit_s=far,native_repaired_near_hit_s=near,
                         first_trustworthy_inward_s=t,first_generic_low=consistent_low[0] if consistent_low else None,
                         first_outbound_estimate=lookahead[0] if lookahead else None,
                         first_outbound_under_point65=next((q for q in lookahead if .05 < q["estimated_far_bounce_y"]<.65),None),
                         inward_to_physical_near_s=None if t is None else near-t,
                         vision_lateness_vs_issued_prep_s=None if t is None else t-prep,
                         remaining_prepare_s=None if t is None else ready-t))
    _write(out,dict(sources={str(p):_hash(p) for p in (sensor,physical)},rows=rows,
                    role="Timing only. Neither actual contacts nor teacher command timestamps enter an observer."))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input",type=Path,nargs="?",default=OUT/"receipts.json.gz")
    cli.add_argument("--out",type=Path,default=OUT/"forecast.json")
    cli.add_argument("--models",type=Path)
    cli.add_argument("--held",default=HELD_SOURCE)
    cli.add_argument("--teachers",action="store_true")
    args = cli.parse_args()
    if args.teachers:
        teacher_timing(args.out)
    else:
        run(args.input,args.out,args.models,args.held)
