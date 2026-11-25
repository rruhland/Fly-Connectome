"""Learn a low-rank direct-joint repair from paired earned and missed strokes."""
import argparse
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np

from . import tt57_motor_school as school
from . import tt57_motor_school_mature as maturity
from . import tt57_motor_school_transfer as transfer
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_optical_clinic import OpticalClinic

ROOT = Path("artifacts/tt59_fewshot_joints")
PLAN = ROOT / "protocol.json"
MODEL = ROOT / "readout.npz"
OUT = ROOT / "forks.json.gz"
SOURCES = tuple(
    Path("artifacts") / (n + ".json.gz")
    for n in ("tt42/after48", "tt42/direct-execution-held", "tt42/diverse-evaluation",
              "tt44_combined/combined", "tt48_transfer/continued-64")
)
MODES = ("paired", "shuffled")
CANONICAL = .24
RANK = 5
CANDIDATES = 48
ATTEMPTS = 4
SHUFFLE_SEED = 590731
H = school.H
mpc = school.mpc


def freeze():
    old = _read(school.PLAN)
    forbidden = {c["id"] for deck in ("practice", "held") for c in old[deck]}
    forbidden.update(c["id"] for c in _read(transfer.CONTROL_PLAN)["rows"])
    pool = []
    for p in SOURCES:
        d = _read(p)
        for row in school.course_rows(d, forbidden, "practice"):
            pool.append(dict(row, source=str(p)))
    # Only archived ordinary results enter this roster. Plan receives no new physics.
    eligible = [r for r in pool if r["lane"] != "right"]
    right = [r for r in pool if r["lane"] == "right"]
    rng = np.random.default_rng(590733)
    shuffled = [eligible[i] for i in rng.permutation(len(eligible))]
    chosen = []
    ids = set()
    for lane in ("middle", "left") * 8 + ("middle",) * 4:
        remain = [r for r in shuffled if r["id"] not in ids and r["lane"] == lane]
        if remain:
            v = min(remain, key=lambda x: x["tick"] // 700)
            chosen.append(v)
            ids.add(v["id"])
        if len(chosen) == 12:
            break
    for r in sorted(right, key=lambda x: x["tick"]):
        if r["id"] not in ids:
            chosen.append(r)
            ids.add(r["id"])
        if len(chosen) == 16:
            break
    if len(chosen) < 12 or ids & forbidden:
        raise ValueError("Insufficient independent ordinary misses")
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(PLAN, dict(
        schema="tt59-disjoint-fewshot-joints-1", cases=chosen, excluded=sorted(forbidden),
        sources={str(p): _hash(p) for p in SOURCES}, school_protocol=_hash(school.PLAN),
        school_practice=_hash(school.OUT), earned=_hash(maturity.OUT),
        optics=_hash(school.course.OPTIC), bank=_hash(mpc.BANK),
        budgets=dict(path_frames=H, physical_attempts=ATTEMPTS, candidates_per_nominal_setting=CANDIDATES,
                     coarse_settings=1, refinement_settings=2, refinement_candidates=12, retained_rank=RANK),
        scope="Disjoint recorded releases from TT42/TT44/TT48 ordinary flights, excluding all TT57 releases including its qualification and TT58 anchors. Touches the blade but fails to score, first stock optical search in the prescribed early window. Alternate left and middle when available, then reserve four unseen right-wing cases as extrapolation. Source releases may be familiar to upstream frozen stock components but not to the paired correction fit.",
        controls="Fixed source-context derangement shared across cases. Both arms use identical public baseline anchors, source labels, candidate count, gains, correction magnitudes as a source multiset, search, attempts, binary failure repulsion, and native referee. Stop after a physically scored, CAD-clean stroke, and report whether the ordinary opponent answered. A new native branch is an attempt, unlike a public twin rollout or the audit replay.",
    ))
    print("frozen", len(chosen), dict(Counter(c["lane"] for c in chosen)), flush=True)


def validate_plan():
    p = _read(PLAN)
    checks = dict(p["sources"])
    checks.update({str(path): p[key] for path, key in (
        (school.PLAN, "school_protocol"), (school.OUT, "school_practice"), (maturity.OUT, "earned"),
        (school.course.OPTIC, "optics"), (mpc.BANK, "bank"),
    )})
    for path, signature in checks.items():
        if _hash(Path(path)) != signature:
            raise ValueError(f"Frozen TT59 source changed: {path}")
    if p["budgets"]["physical_attempts"] != ATTEMPTS or p["budgets"]["retained_rank"] != RANK:
        raise ValueError("Changed frozen TT59 method")
    return p


def fit():
    plan = validate_plan()
    doc = _read(maturity.OUT)
    if (not doc["complete"] or doc["original_practice_sha256"] != plan["school_practice"]
            or doc["protocol_sha256"] != plan["school_protocol"]):
        raise ValueError("Unsealed physical practice")
    keys, states, failures, repaired, ids, labels = [], [], [], [], [], []
    with school.enabled(cameras=False):
        a = mpc.arena(width=64, height=48)
        try:
            axes = a._arm_ids[1][2]
            stride = a._joint_increments[axes] * a.control_dt / a.model.opt.timestep
            lo, hi = a._joint_low[axes], a._joint_high[axes]
            for row in doc["rows"]:
                for arm in row.get("arms", []):
                    if not (arm["landed"] and arm.get("issued_relative")):
                        continue
                    stock = row["stock"]
                    if (stock["landed"] or stock.get("unmatched") or not arm["landed"]
                            or not arm["matched_far_hit"] or not arm["matched_near_bounce"]
                            or arm["clearance"]["penetration_samples"] or arm["clearance"]["nearest"]["signed_m"] <= 0.
                            or row["case"]["id"] in {v["id"] for v in plan["cases"]}):
                        raise ValueError("Unpaired, unsafe, or leaked training movement")
                    inp = row["inputs"]
                    ref, x = np.asarray(inp["ref"]), np.asarray(inp["x"])
                    u = np.asarray(stock["controls"][:H])
                    old = mpc.tail(school.atlas.reference_path(ref, u, stride, lo, hi)[1:], H) - ref
                    new = np.asarray(arm["issued_relative"])
                    retraced = school.atlas.reference_path(ref, np.asarray(arm["controls"][:H]), stride, lo, hi)[1:] - ref
                    if not np.allclose(new, retraced, rtol=0., atol=1e-10):
                        raise ValueError("Scored movement differs from the issued joint path")
                    sweep = transfer.eta(x) / CANONICAL
                    failures.append(school.warp(old, sweep, 0.))
                    repaired.append(school.warp(new, sweep, 0.))
                    keys.append(inp["features"])
                    states.append(x)
                    ids.append(row["case"]["id"])
                    labels.append(arm["event_sha256"])
        finally:
            a.close()
    ids = np.asarray(ids)
    if len(np.unique(ids)) < 12:
        raise ValueError("Too few distinct causal lessons")
    failures = np.asarray(failures)
    delta = np.asarray(repaired) - failures
    flat = delta.reshape(len(delta), -1)
    _, s, basis = np.linalg.svd(flat, full_matrices=False)
    basis = basis[:RANK]
    weights = flat @ basis.T
    rng = np.random.default_rng(SHUFFLE_SEED)
    for _ in range(10000):
        permutation = rng.permutation(len(ids))
        if np.all(ids[permutation] != ids):
            break
    else:
        raise ValueError("Could not derange correlated source releases")
    np.savez_compressed(MODEL, features=keys, x=states, failures=failures, basis=basis,
                        weights=weights, ids=ids, permutation=permutation, lesson_events=labels,
                        protocol=_hash(PLAN), frozen_fraction=float(np.sum(s[:RANK]**2)/np.sum(s**2)))
    print("learned", len(ids), len(np.unique(ids)), "rank energy", round(np.sum(s[:RANK]**2)/np.sum(s**2), 4), flush=True)


class Corrections:
    def __init__(self, path=MODEL):
        with np.load(path) as f:
            if str(f["protocol"]) != _hash(PLAN):
                raise ValueError("Correction parameters and frozen roster differ")
            self.features = f["features"] * transfer.SCALE
            self.failures = f["failures"]
            self.basis = f["basis"]
            self.weights = f["weights"]
            self.ids = f["ids"]
            self.permutation = f["permutation"]

    def proposals(self, x, features, baselines, mode):
        distance = np.linalg.norm(self.features - features * transfer.SCALE, axis=1)
        indices, ids = [], set()
        for ii in np.argsort(distance, kind="stable"):
            if self.ids[ii] not in ids:
                indices.append(ii)
                ids.add(self.ids[ii])
            if len(indices) >= 6:
                break
        matching = np.array(indices)
        directions = matching if mode == "paired" else self.permutation[matching]
        weight = np.exp(-.5 * (distance[matching] - distance[matching[0]])**2)
        coeff = self.weights[directions]
        vectors = np.vstack((coeff, np.average(coeff, axis=0, weights=weight)))
        residuals = (vectors @ self.basis).reshape(-1, H, 6)
        scale = CANONICAL / transfer.eta(x)
        output, meta = [], []
        for n in range(CANDIDATES):
            i = n % len(residuals)
            block = n // len(residuals)
            # Common deterministic phase/gain grid. Baselines and geometric
            # neighbor distances are shared. Only the repair labels can change.
            root, gain, phase, prior = (
                (0, 1.0, 0., .0), (1, .82, -1.5, .0), (1, 1.14, 1.3, .0),
                (2, .76, 0., .0), (2, 1.0, 1.4, .42), (1, 1.0, 0., .42),
                (0, .68, -1.3, .50)
            )[block]
            baseline = baselines[root]
            if prior and i < len(matching):
                backing = school.warp(self.failures[matching[i]], scale, 0.)
                baseline = (1.-prior)*baseline + prior*backing
            residual = school.warp(residuals[i], scale, phase)
            proposal = baseline + np.clip(gain * residual, -.52, .52)
            output.append(proposal)
            meta.append(dict(context_release=int(self.ids[matching[i]]) if i < len(matching) else None,
                             repair_release=int(self.ids[directions[i]]) if i < len(matching) else None,
                             distance=float(distance[matching[min(i,len(matching)-1)]]),
                             anchor=root, gain=gain, phase_tick=phase, failure_prior=prior,
                             delta_rms=float(np.sqrt(np.mean((proposal - baseline)**2)))))
        return np.asarray(output), meta


def anchor_paths(engine, raw, ready, x, q, ref, base):
    """Stock-like local forecasts from the public twin, no recorded future branch."""
    trace = np.full((1, H, 17), np.nan)
    static = np.broadcast_to(base, (H, 6)).copy()
    engine.predict.trajectories(static[None], school.AIM, (H+1)*mpc.DT, traces=trace)
    last = base
    follow = []
    for row in trace[0]:
        if np.isfinite(row).all():
            joint, confidence = raw.get(row[:6], row[6:11])
            if joint is not None:
                last = ready + confidence * (joint - ready)
        follow.append(last)
    follow = np.asarray(follow)
    ease = np.minimum(school.PHASE / max(4., .50*transfer.eta(x)/mpc.DT), 1.)[:, None]
    return np.stack((ease * (base-ref), np.clip(follow-ref, -.72, .72),
                     ease * np.clip(.62*(base-ref)+.38*(q-ref), -.72, .72)))


def rank_paths(engine, paths, ref, metadata):
    """Rank only in the separately built RGB/encoder twin; no actual trial here."""
    rates, y, feedback, loss, legal = school.native(engine.predict, paths, ref)
    nominal = engine.twin
    p, v = nominal.ball.copy(), nominal.velocity.copy()
    d = nominal._ball_dof
    w = nominal.data.qvel[d+3:d+6].copy()
    # Two errors around the same public twin. Only the nominal top 12 are
    # perturbed, with identical budgets and proxy for both methods.
    order = np.argsort(loss, kind="stable")[:12]
    votes = np.zeros(len(order), dtype=int)
    risk = np.zeros(len(order))
    try:
        for side in (-1., 1.):
            nominal.ball[:] = p + side*np.array([.0018,-.0023,.0028])
            nominal.velocity[:] = v + side*np.array([.023,.043,.073])
            nominal.data.qvel[d+3:d+6] = w + [16.+17.*side,0.,0.]
            _, _, _, trial_loss, trial_legal = school.native(engine.predict, paths[order], ref)
            votes += trial_legal
            risk += np.minimum(trial_loss, 15.0)
    finally:
        nominal.ball[:] = p
        nominal.velocity[:] = v
        nominal.data.qvel[d+3:d+6] = w
    out = []
    for j, vote, smooth in zip(order, votes, risk, strict=True):
        verdict = dict(metadata[j], candidate=int(j), legal=bool(legal[j]),
                       robust=int(vote), nominal=float(loss[j]), separated=bool(feedback[j,1] > .5),
                       contact_delay=float(y[j,1]) if legal[j] else None,
                       score=float(loss[j] + .36*smooth - .50*vote))
        out.append((rates[j,:H].copy(), verdict))
    return out, dict(candidates=len(paths), refinement=len(order)*2,
                     nominal_legal=int(legal.sum()), robust=int((votes > 0).sum()))


class Attempts:
    """Binary feedback excludes a command and discounts nearly identical retries."""
    def __init__(self, ranked):
        self.ranked = ranked
        self.failed = []

    def choose(self):
        options = []
        for commands, meta in self.ranked:
            dis = min((float(np.sqrt(np.mean((commands-u)**2))) for u in self.failed), default=1.)
            if dis <= .008:
                continue
            options.append((meta["score"] + 2.3*np.exp(-dis/.11), commands, meta, dis))
        if not options:
            return None
        _, command, info, dist = min(options, key=lambda z: z[0])
        return command, dict(info, previous_failure_distance=dist)

    def miss(self, commands):
        self.failed.append(np.array(commands, copy=True))


def visible(x, feature):
    return x is not None and feature is not None and np.isfinite(feature).all() and x[4] < -.9 and -1.03 < x[1] < -.47


def score_forks(count):
    p = validate_plan()
    readout = Corrections()
    receipt = dict(schema="tt59-fewshot-physical-forks-1", protocol=_hash(PLAN), weights=_hash(MODEL),
                   operator=_hash(Path(__file__)), rows=[], complete=False)
    if OUT.exists():
        old = _read(OUT)
        if any(old[k] != receipt[k] for k in ("protocol", "weights", "operator")):
            raise ValueError("Previously issued trial belongs to another experiment")
        receipt = old
    docs = {}
    with np.load(mpc.BANK) as f:
        raw = mpc.DeepField(f, "local")
    with school.enabled():
        cad = mpc.arena(width=64, height=48, detail="cad")
        try:
            guard = school.CourtClearance(cad.model, articulated=True)
            for spec in p["cases"][len(receipt["rows"]):count]:
                start = process_time()
                src = spec["source"]
                if src not in docs:
                    docs[src] = _read(Path(src))
                doc = docs[src]
                ready = np.asarray(doc["ready"])
                i, tick = spec["i"], spec["tick"]
                card, row = doc["cards"][i], doc["rows"][i]
                with mpc.centered(ready):
                    a = mpc.arena()
                    cam, rig, *_ = mpc.resources(a)
                    engine = OpticalClinic(body=True)
                    try:
                        axes = a._arm_ids[1][2]
                        eye, x, q, ref, _, feature, hist = school.observed_prefix(a, cam, rig, card, row, tick, axes)
                        saved = school.State(a)
                        stock = school.physical(a, saved, card, row, spec, row["actions"][tick:])
                        if stock["landed"] or stock.get("unmatched") or stock["event_sha256"] != school.digest_json(row["events"]):
                            raise ValueError("Physical stock failure did not replay exactly")
                        if not visible(x, feature):
                            raise ValueError("Prespecified fork did not reproduce its public visibility")
                        transfer.warm_clinic(engine, rig, a.control_dt, hist, eye, tick, q, ref)
                        if not engine.sync():
                            raise ValueError("Public nominal twin unavailable")
                        j, confidence = raw.get(x, q)
                        base = ready if j is None else ready + confidence * (j - ready)
                        baselines = anchor_paths(engine, raw, ready, x, q, ref, base)
                        modes = {}
                        for mode in MODES:
                            paths, meta = readout.proposals(x, feature, baselines, mode)
                            ranks, diagnostic = rank_paths(engine, paths, ref, meta)
                            session = Attempts(ranks)
                            attempts = []
                            for number in range(1, ATTEMPTS+1):
                                suggestion = session.choose()
                                if suggestion is None:
                                    break
                                command, evidence = suggestion
                                full = np.vstack((command, np.zeros((school.T+94-H, 6))))
                                verify = school.physical(a, saved, card, row, spec, full)
                                reward = False
                                if verify.get("landed"):
                                    audit = school.physical(a, saved, card, row, spec, full, guard)
                                    if audit["event_sha256"] != verify["event_sha256"]:
                                        raise ValueError("CAD changed physical referee sequence")
                                    verify["clearance"] = audit["clearance"]
                                    reward = (not audit["clearance"]["penetration_samples"]
                                              and audit["clearance"]["nearest"]["signed_m"] > 0.)
                                attempts.append(dict(number=number, decision=evidence, reward=reward,
                                                     planned_prefix=command.tolist(), **verify))
                                if reward:
                                    break
                                session.miss(command)
                            modes[mode] = dict(diagnostic, trials=attempts,
                                               first=next((v["number"] for v in attempts if v["reward"]), None))
                        receipt["rows"].append(dict(case=dict(spec, card=card, ready=doc["ready"]),
                            stock=stock, arms=modes, optics=dict(x=x.tolist(), feature=feature.tolist(), ref=ref.tolist(), q=q.tolist()),
                            cpu_s=process_time()-start))
                        receipt["complete"] = len(receipt["rows"]) == len(p["cases"])
                        _write(OUT, receipt)
                        print(len(receipt["rows"]), spec["id"], spec["lane"],
                              [(m, v["first"], v["nominal_legal"], v["robust"],
                                [z["landed"] for z in v["trials"]]) for m, v in modes.items()], flush=True)
                    finally:
                        engine.twin.close()
                        cam.close()
                        a.close()
        finally:
            cad.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "fit", "forks"))
    parser.add_argument("--n", type=int, default=16)
    a = parser.parse_args()
    if a.mode == "plan":
        freeze()
    elif a.mode == "fit":
        fit()
    else:
        score_forks(a.n)


if __name__ == "__main__":
    main()
