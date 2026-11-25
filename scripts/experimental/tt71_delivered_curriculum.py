"""A practice partner learns which continuous request actually produces a given receiving ball."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt56_scored_motor_atlas as archive
from . import tt68_clamp_transfer as body
from . import tt37_opponent as opponent
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_opportunity_map import projected_height

ROOT = Path("artifacts/tt71_delivered_curriculum")
MODEL = ROOT / "partner.npz"
PLAN = ROOT / "course.json"
BOUNDS = np.array(opponent.PROFILES["reach"])
STANDARD = opponent.draw
XEDGES = [-0.10, 0.10]
ZEDGES = [0.985, 1.0, 1.02, 1.045]


def featurize(goal, ball, v, w):
    return np.r_[goal, ball, v, w]


def actual(o):
    line = o.get("receive_line")
    if not line:
        return None
    return np.array([line["xyz"][0], projected_height(line)])


def cell(y):
    return np.searchsorted(XEDGES, y[..., 0]) * 5 + np.searchsorted(ZEDGES, y[..., 1])


def fit_model():
    X = []
    Y = []
    ids = []
    seen = set()
    meta = _read(archive.ROOT / "fit.json")
    held = {c["id"] for c in _read(body.dose.PLAN)["practice"]} | {
        c["id"] for c in _read(body.season.PLAN)["practice"]
    }
    for filename, checksum in meta["sources"].items():
        p = Path(filename)
        if _hash(p) != checksum:
            raise ValueError("Changed archive")
        d = _read(p)
        for c, r in zip(d["cards"], d["rows"]):
            if c["id"] in held or c.get("profile") != "reach":
                continue
            for o in r["placements"]["delivered"]:
                v = r["placements"]["requested"][o["leg"]]
                y = actual(o)
                if y is None or v["draw"] != "random":
                    continue
                x = featurize(
                    v["target"],
                    v["observed_ball"],
                    v["observed_incoming"],
                    v["observed_spin"],
                )
                key = np.round(x, 6).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                X.append(x)
                Y.append(y)
                ids.append(c["id"])
    X, Y, ids = map(np.array, (X, Y, ids))
    ROOT.mkdir(exist_ok=True, parents=True)
    order = np.random.default_rng(71099).permutation(np.unique(ids))
    runs = {}
    for kind, dim in [("requested_only", 4), ("incoming_and_request", 13)]:
        pred = np.zeros_like(Y)
        for cohort in np.array_split(order, 5):
            mask = np.isin(ids, cohort)
            reg = Response(X[~mask, :dim], Y[~mask])
            pred[mask] = reg(X[mask, :dim])
        runs[kind] = dict(
            mean_abs=np.abs(Y - pred).mean(0).tolist(),
            rmse=np.sqrt(np.square(Y - pred).mean(0)).tolist(),
            correlation=[float(np.corrcoef(a, b)[0, 1]) for a, b in zip(Y.T, pred.T)],
        )
    np.savez_compressed(MODEL, X=X, Y=Y, ids=ids)
    _write(
        ROOT / "fit.json",
        dict(
            rows=len(Y),
            releases=len(order),
            held=runs,
            model=_hash(MODEL),
            source=_hash(archive.ROOT / "fit.json"),
        ),
    )
    print(runs, flush=True)


class Response:
    def __init__(self, X, Y):
        self.mu = X.mean(0)
        self.scale = np.maximum(X.std(0), 0.0001)
        self.inputs = (X - self.mu) / self.scale
        self.y = Y
        self.tree = cKDTree(self.inputs)

    def __call__(self, x):
        u = (np.atleast_2d(x) - self.mu) / self.scale
        ds, idx = self.tree.query(u, k=min(50, len(self.y)))
        w = np.exp(-0.65 * np.square(ds / np.maximum(0.08, ds[:, -1:])))
        offset = self.inputs[idx] - u[:, None, :]
        ones = np.ones((*offset.shape[:2], 1))
        design = np.concatenate((ones, offset), axis=2)
        m = design.transpose(0, 2, 1) * w[:, None, :]
        ridge = np.diag(np.r_[0.001, np.ones(u.shape[1]) * 0.85])
        return np.linalg.solve(m @ design + ridge, m @ self.y[idx])[:, 0, :]


def plan():
    meta = _read(ROOT / "fit.json")
    if meta["held"]["incoming_and_request"]["correlation"][1] < 0.45:
        raise ValueError("Insufficient independent control of actual delivery height")
    p = _read(body.dose.PLAN)["practice"]
    pool = {c["id"]: c for c in p}
    rng = np.random.default_rng(716817)
    cards = [
        dict(
            pool[i],
            seed=int(rng.integers(1, 2**30)),
            reply_seed=int(rng.integers(1, 2**30)),
        )
        for _ in range(5)
        for i in rng.permutation(sorted(pool))
    ]
    _write(
        PLAN,
        dict(
            cards=cards,
            budget=256,
            model=_hash(MODEL),
            fixture=_hash(body.PLAN),
            base=_hash(body.season.CARRY),
            scope="Teacher-only private incoming and chosen request train a model of realized opponent deliveries. It may choose where to practice. The SO-101 sees normal RGB, encoders and earned scores. 35% of training uses unaltered broad random goals. Both groups credit the first 256 legal opponent deliveries. All assessment games restore independent continuous broad draws.",
        ),
    )


class Partner:
    def __init__(self, kind):
        with np.load(MODEL) as z:
            self.predict = Response(z["X"], z["Y"])
        self.kind = kind
        self.now = None
        self.cache = {}
        self.counts = np.zeros((15, 2))
        self.offers = 0

    def select(self, card, leg, mode, rng):
        key = (card["seed"], card["reply_seed"], leg)
        if key in self.cache:
            return self.cache[key]
        base, tag = STANDARD(card, leg, mode, rng)
        random = np.random.default_rng([*key, 716241])
        if (
            self.kind == "conditional"
            and random.uniform() >= 0.35
            and self.now is not None
        ):
            choices = random.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (32, 4))
            xs = np.column_stack(
                (choices, np.broadcast_to(self.now, (len(choices), len(self.now))))
            )
            estimated = cell(self.predict(xs))
            obs = self.counts.sum(axis=1)
            prob = (self.counts[:, 1] + 2.8) / (obs + 4)
            target = 0.78 - 0.16 * min(1.0, self.offers / 180)
            priority = np.exp(-0.5 * ((prob - target) / 0.18) ** 2) / (obs + 4) ** 0.55
            # Avoid a large easy cell monopolizing the draw by its area alone.
            multiplicity = np.bincount(estimated, minlength=15)
            weights = priority[estimated] / np.sqrt(multiplicity[estimated])
            base = tuple(
                choices[random.choice(len(choices), p=weights / weights.sum())]
            )
            tag = "delivered_curriculum"
        self.cache[key] = (base, tag)
        return base, tag

    def observe(self, r):
        for o in r["placements"]["delivered"]:
            y = actual(o)
            if y is not None:
                self.counts[cell(y), int(bool(o["near_landed"]))] += 1


def practice(kind, feeds, teacher_factory=Partner):
    p = _read(PLAN)
    d = _read(body.PLAN)
    ready = d["ready"]
    if (
        p["model"] != _hash(MODEL)
        or p["base"] != _hash(body.season.CARRY)
        or p["fixture"] != _hash(body.PLAN)
    ):
        raise ValueError("Training conditions changed")
    teacher = teacher_factory(kind)
    cards = p["cards"]
    ids = {c["id"] for c in cards}
    shots = body.mpc.measured_launches(feeds)
    with np.load(body.mpc.BANK) as f:
        raw = body.fixture.BodyRelativeRaw(body.mpc.DeepField(f, "local"), body.DX)
    original = opponent.LivePartner.advance

    def advance(partner):
        a = partner.arena
        if (
            a is arena
            and a.card is not None
            and a._last_hitter == 1
            and a.velocity[1] > 0.32
            and partner.reception != a.contacts
        ):
            teacher.now = np.r_[a.ball.copy(), a.velocity.copy(), a.spin.copy()]
            teacher.plant = a
        return original(partner)

    exchange = opponent.LiveArena.exchange

    def count(a, *args):
        result = exchange(a, *args)
        if a is not arena:
            return result
        teacher.offers += sum(
            e["kind"] == "legal_return" and e["side"] == -1
            for e in result[-1]["events"]
        )
        return result

    out = ROOT / (kind + ".json.gz")
    bank = out.with_suffix("").with_suffix(".npz")
    doc = dict(
        protocol=_hash(PLAN),
        cards=cards,
        ready=ready,
        rows=[],
        mode="live",
        fixture_dx_m=body.DX,
    )
    rank = body.season.fresh.shot.Coverage(_read(body.season.fresh.RISK))
    with body.enabled(), body.fixture.slid(body.DX), body.mpc.centered(ready):
        arena = body.mpc.arena()
        cam, rig, stride, lo, hi = body.mpc.resources(arena)
        try:
            with patch.object(
                body.mpc,
                "Fragments",
                lambda file, engine: body.course.Carried(
                    file, engine, practice=[body.course.PRACTICE], held=ids
                ),
            ):
                actor = body.Registered(
                    body.mpc.BANK, raw, variant="elastic", budget=28, coverage=rank
                )
            try:
                actor.priors.home = np.asarray(ready)
                base = _read(body.season.CARRY)
                body.course.resume(
                    actor, body.season.CARRY.with_suffix("").with_suffix(".npz"), base
                )
                keep = actor.priors.keep
                actor.priors.keep = lambda *args: (
                    keep(*args) if teacher.offers <= p["budget"] else 0
                )
                with (
                    patch.object(opponent, "draw", teacher.select),
                    patch.object(opponent.LivePartner, "advance", advance),
                    patch.object(opponent.LiveArena, "exchange", count),
                ):
                    for i, c in enumerate(cards):
                        before = teacher.offers
                        r = body.mpc.play(
                            arena,
                            cam,
                            rig,
                            stride,
                            lo,
                            hi,
                            ready,
                            c,
                            shots[c["id"]],
                            raw,
                            actor,
                            body.course.OPTIC,
                            "live",
                            240,
                            20,
                        )
                        if teacher.offers - before != r["score"]["opponent"]:
                            raise ValueError("Practice exposure mismatch")
                        teacher.observe(r)
                        doc["rows"].append(r)
                        doc.update(
                            checkpoint_sha256=body.course.save(actor, bank),
                            received=teacher.offers,
                            practice_diagnostics=getattr(teacher, "diagnostics", None),
                            cells=teacher.counts.tolist(),
                            complete=teacher.offers >= p["budget"],
                        )
                        _write(out, doc)
                        print(
                            i + 1,
                            c["id"],
                            r["score"]["learner"],
                            teacher.offers,
                            flush=True,
                        )
                        if doc["complete"]:
                            break
            finally:
                actor.close()
        finally:
            cam.close()
            arena.close()
            if hasattr(teacher, "close"):
                teacher.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "plan", "uniform", "conditional"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    if a.phase == "fit":
        fit_model()
    elif a.phase == "plan":
        plan()
    else:
        practice(a.phase, a.feeds)
