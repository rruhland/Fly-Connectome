"""Generate joint options around a legal return with a physically learned next-ball objective."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt43_outgoing_coverage as coverage
from .tt4_transfer_trial import _hash, _read, _write

RISK_TRIGGER = .065
RISK_WEIGHT = 7.0
QUALITY_CAP = .34
GAIN = .16
SCALES = np.array([.06, .07, .10, .11, .10, .055, .07, .10, .10, .10])


def retarget(base, noise, eta):
    """Two smooth joint-reference perturbations, expressed around likely impact."""
    t = (np.arange(len(base)) + 1) * mpc.DT
    reach = np.clip(t / max(.05, eta - .016), 0., 1.)
    reach = reach * reach * (3 - 2 * reach)
    drive = np.clip((t - eta + .078) / .083, 0., 1.)
    drive = drive * drive * (3 - 2 * drive)
    displace = np.outer(reach, noise[:5]) + np.outer(drive, noise[5:])
    return base + np.c_[displace, np.zeros(len(base))]


def choose(quality, agreement, risks, supported, stock_quality, stock_risks):
    """Reject losses on the current stroke and unstable improvements across optical hypotheses."""
    delta = np.asarray(stock_risks) - np.asarray(risks)
    objective = RISK_WEIGHT * delta.mean(axis=1) - np.maximum(quality - stock_quality, 0.)
    valid = ((agreement == 2) & supported & (delta.mean(axis=1) > .024)
             & (delta.min(axis=1) >= -.012) & (quality < stock_quality + QUALITY_CAP)
             & (objective > GAIN))
    return None if not valid.any() else int(np.argmax(np.where(valid, objective, -np.inf)))


class Generator(coverage.Actor):
    def begin(self):
        super().begin()
        self.last_generation = -1000
        self.capture = None
        self.values = {}
        self.install = None

    def search(self, tick, x, q, reference, base):
        self.values = {}
        self.install = None
        super().search(tick, x, q, reference, base)
        if self.install is not None:
            tapes, note = self.install
            self.best = tapes
            self.notes[-1].update(
                chosen="coverage_generated", robust=2, score=note["quality"][1],
                target_errors=note["replacement"], contact_delay_s=note["replacement_eta"],
            )

    def trial(self, goals, *, long=False, traces=False):
        r = super().trial(goals, long=long, traces=traces)
        if long and self.capture is not None:
            self.capture.append(r[3][:, 3:10].copy())
        elif long:
            for g, okay, v in zip(goals, r[1], r[3], strict=True):
                if okay:
                    self.values[np.asarray(g).tobytes()] = v.copy()
        return r

    def replay_risk(self, goals, nominal, x):
        """Read all three outgoing flows using the original two uncertainty trials."""
        self.capture = []
        self.in_rank = True
        try:
            agree, quality = self.uncertainty(goals, nominal)
            data = [x, *self.capture]
        finally:
            self.capture = None
            self.in_rank = False
        out = np.full((len(goals), len(data)), .5)
        local = np.ones(len(goals), dtype=bool)
        for j, v in enumerate(data):
            okay = np.isfinite(v).all(axis=1)
            local &= okay
            if okay.any():
                out[okay, j], support = self.coverage.risk(v[okay])
                local[okay] &= support
        return agree, quality, out, local

    def robust(self, goals, rank, nominal):
        values = np.asarray([self.values[np.asarray(g).tobytes()] for g in goals])
        actual = values[:, 3:10]
        agree, quality, risks, support = self.replay_risk(goals, nominal, actual)
        stock = int(np.argmin(quality))
        gap, tick = values[stock, 1], self.decision_tick
        # Leave routine shots, geometry outside physical practice, and unsupported
        # current contacts to the original ranker. Calm windows skip generation.
        if (agree[stock] < 2 or not support[stock] or risks[stock].mean() < RISK_TRIGGER
                or not .064 < gap < .32 or tick - self.last_generation < 10):
            return int(rank[stock]), int(agree[stock]), float(quality[stock])
        self.last_generation = tick
        n = self.make_variations(goals[stock], gap, tick, quality[stock], risks[stock])
        if n is not None:
            tape, metrics, selected_cost, selected_risk = n
            note = dict(tick=tick, generated=True, prior=int(rank[stock]), selected=-1,
                        original=actual[stock].tolist(), replacement=metrics[3:10].tolist(),
                        original_eta=float(gap), replacement_eta=float(metrics[1]),
                        quality=[float(quality[stock]), selected_cost],
                        risks=[float(risks[stock].mean()), float(selected_risk.mean())],
                        hypothesis_risks=[risks[stock].tolist(), selected_risk.tolist()],
                        robustness=[int(agree[stock]), 2])
            self.install = tape, note
            self.reranks.append(note)
        return int(rank[stock]), int(agree[stock]), float(quality[stock])

    def make_variations(self, base, eta, tick, stock_quality, stock_risks):
        rng = np.random.default_rng([430310, tick])
        eps = rng.normal(size=(9, len(SCALES))) * SCALES
        eps[:5] = np.c_[np.diag(SCALES[:5]), .50 * np.diag(SCALES[5:])]
        noise = np.r_[eps, -eps]
        trials = []
        self.diag["coverage_generation_windows"] += 1
        for turn in range(2):
            goals = [retarget(base, n, eta) for n in noise]
            cost, legal, _, values, _ = self.trial(goals, long=True)
            allowed = np.flatnonzero(legal)
            if len(allowed):
                predicted, support = self.coverage.risk(values[allowed, 3:10])
                for j, risk in zip(allowed[support], predicted[support], strict=True):
                    desirability = (cost[j] + RISK_WEIGHT * risk
                                    + .018 * np.linalg.norm(noise[j] / SCALES))
                    trials.append((float(desirability), cost[j], noise[j], goals[j], values[j]))
            if not trials:
                break
            trials.sort(key=lambda a: a[0])
            if turn == 0:
                elite = np.array([r[2] for r in trials[:4]])
                seed = (elite[np.arange(12) % len(elite)] +
                        .47 * rng.normal(size=(12, len(SCALES))) * SCALES)
                noise = np.clip(seed, -2 * SCALES, 2 * SCALES)
        elite = [r for r in trials if float(r[1]) < stock_quality + QUALITY_CAP + .18][:7]
        if not elite:
            return None
        stock_nominal = np.array([u[1] for u in elite])
        goals = [u[3] for u in elite]
        scores = np.array([u[4] for u in elite])
        agree, quality, risks, supported = self.replay_risk(goals, stock_nominal, scores[:, 3:10])
        j = choose(quality, agree, risks, supported, stock_quality, stock_risks)
        if j is None:
            return None
        self.diag["coverage_generation_issued"] += 1
        return goals[j], scores[j], float(quality[j]), risks[j]


def run(args):
    generator_hash = _hash(Path(__file__))
    if args.output.exists() and _read(args.output).get("generation_sha256") != generator_hash:
        raise ValueError("The recording was generated by a different actor")

    def writer(path, doc):
        _write(path, doc | dict(generation_sha256=generator_hash))

    # With the original runner, all exact-prefix, score, placement draw, optical,
    # memory and action invariance checks remain in force.
    with patch.object(coverage, "Actor", Generator), patch.object(coverage, "_write", writer):
        coverage.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=coverage.ROOT / "outgoing-risk.json")
    parser.add_argument("--control", type=Path, default=course.ROOT / "no-more-practice.json.gz")
    parser.add_argument("--output", type=Path, default=coverage.ROOT / "outgoing-generation-live.json.gz")
    parser.add_argument("--limit", default=6, type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)
