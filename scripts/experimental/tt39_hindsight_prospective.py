"""Test a directly replayable causal first-rank repair without knowing the point will fail."""

import argparse
from pathlib import Path
from time import process_time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import card, opening_candidates
from .tt31_ready import centered
from .tt37_joint_deep import DeepField
from .tt37_joint_report import paired, read_complete
from .tt38_joint_mpc_report import metrics
from .tt39_hindsight import DT, OPTICS, OUT, flight, paths, propose


DECK = "prospective"
BUDGET = 34
MAX_ATTEMPTS = 2
RETRY_GAP = 4
LEASE = mpc.HORIZON


def ready(x, fresh):
    """Observed court region and upward motion cue, not a chosen interception location."""
    return (fresh and x is not None and -.99 < x[1] < -.63 and x[4] < -.9 and x[5] > .20)


def opportunity(sensory):
    last = -10
    for i, frame in enumerate(sensory):
        if frame["xyz"] is not None:
            last = i
        if ready(frame["x"], i - last <= 1):
            return i
    return None


def freeze(folder, feeds, optic):
    practice = read_complete(paths(folder, "practice"))
    outcomes = read_complete(paths(folder, "verification"))
    refs = {c["id"]: r for c, r in zip(practice["cards"], practice["rows"], strict=True)}
    checks = []
    for r in outcomes["rows"]:
        a = r["results"][0]
        f = a["causal"]["physical"]
        q = a["causal"]["search"]
        checks.append(dict(id=r["id"], retrospective=a["fit"]["tick"],
                           prospective=opportunity(refs[r["id"]]["sensory"]),
                           first_rank_trials=q["trials"],
                           first_rank_physical_legal=bool(f and f["score"]["learner"])))
    src = [folder / "hindsight-plan.json", mpc.BANK.with_suffix(".json"),
           *(Path(p) for p in ("artifacts/tt37/opponent-plan.json", "artifacts/tt37/school-plan.json",
                               "artifacts/tt39/endurance-plan.json", "artifacts/tt37/clinic-learning-plan.json",
                               "artifacts/tt37/memory-course.json"))]
    used = set()
    def ids(v):
        if isinstance(v, dict):
            if isinstance(v.get("id"), int) and "seed" in v:
                used.add(v["id"])
            for r in v.values():
                ids(r)
        if isinstance(v, list):
            for r in v:
                ids(r)
    for source in src:
        d = _read(source)
        ids(d)
        used.update(i for i in d.get("exclusions", []) if isinstance(i, int))
    shots = measured_launches(feeds)
    eligible = [s["id"] for s in opening_candidates(shots) if s["id"] not in used]
    narrow_count = len(eligible)
    if len(eligible) < 6:
        # The earlier independent courses exhaust the narrow geometry pool. Rank a
        # slightly wider measured envelope by the same pose/velocity similarity.
        names = ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z")
        center = np.array([0., -1.18, .29, 0., 4.1, .65])
        scale = np.array([.12, .35, .25, .65, 1., 1.])
        wide = [s for s in shots.values() if s["id"] not in used
                and abs(s["pos_x"]) < .56 and -2.02 < s["pos_y"] < -.44
                and .10 < s["pos_z"] < .75 and abs(s["vel_x"]) < 2.0
                and 2.40 < s["vel_y"] < 6.90 and -.60 < s["vel_z"] < 3.30]
        eligible = [s["id"] for s in sorted(wide, key=lambda s: np.linalg.norm(
            (np.array([s[n] for n in names]) - center) / scale))[:24]]
    if len(eligible) < 6:
        raise ValueError("Too few unused measured paths in the broadened compatibility envelope")
    rng = np.random.default_rng(39107042)
    chosen = rng.choice(eligible, size=6, replace=False)
    roster = [dict(card(int(i), "full", j, int(rng.integers(1, 2**30))), profile="reach",
                   opening="random", reply_seed=int(rng.integers(1, 2**30)))
              for j, i in enumerate(chosen)]
    _write(folder / "hindsight-prospective-plan.json", dict(
        prospective=roster, historical_cue_check=checks, budget=BUDGET, attempts=MAX_ATTEMPTS,
        retry_gap=RETRY_GAP, lease=LEASE, feed_sha256=_hash(feeds), optic_sha256=_hash(optic),
        bank_sha256=_hash(mpc.BANK), verification_sha256=_hash(paths(folder, "verification")),
        narrow_unused_recordings=narrow_count, candidate_pool=len(eligible),
        roster_rule="Six disjoint measured releases. No outcome or delivery screening. If the original pool "
                    "is exhausted, rank the expanded measured pose/velocity envelope then draw from closest 24.",
    ))


class CausalCommit(mpc.MPC):
    def begin(self):
        super().begin()
        self.tries = 0
        self.released = False

    def step(self, tick, eye, x, q, reference, base):
        latest = eye.latest
        sample = dict(x=x, damp=None if latest is None else float(latest.damping),
                      bounced=bool(latest is not None and latest.bounced))
        # Match the camera-only body/flight initialization used in the repair probe.
        self.engine.observe(tick, SimpleNamespace(latest=flight(sample, tick)), q, reference.copy())
        if x is not None and x[4] > .9:
            self.best = None
            self.tries = 0
            self.released = False
            return base
        if self.best is not None and 0 <= tick - self.tick < LEASE:
            self.diag["screened_ticks"] += 1
            return self.best[tick - self.tick]
        if self.released:
            return base
        track = eye.filter.track
        fresh = bool(track and -.001 < tick * DT - track[-1][0] <= DT + 1e-9)
        if self.tries >= MAX_ATTEMPTS or tick < self.next or not ready(x, fresh):
            return base
        self.tries += 1
        self.next = tick + RETRY_GAP
        self.rng = np.random.default_rng(382785)
        self.best = None
        self.search(tick, x, q, reference, base)
        note = self.notes[-1] if self.notes and self.notes[-1]["tick"] == tick else {}
        if note.get("chosen") is None or note.get("robust", 0) < 1:
            self.best = None
            return base
        self.released = True
        self.lease = LEASE
        self.diag["ranked_launches"] += 1
        self.diag["screened_ticks"] += 1
        return self.best[0]


def evaluate(folder, feeds, optic, mode):
    manifest = folder / "hindsight-prospective-plan.json"
    receipt = _read(manifest)
    if (_hash(feeds), _hash(optic), _hash(mpc.BANK)) != (
            receipt["feed_sha256"], receipt["optic_sha256"], receipt["bank_sha256"]):
        raise ValueError("The planned optics or instruction set changed")
    actor = CausalCommit if mode == "commit" else mpc.MPC
    output = paths(folder, f"prospective-{mode}")
    with patch.object(mpc, "MPC", actor):
        mpc.run(feeds, mpc.BANK, manifest, DECK, "frozen", output, optic=optic,
                cap=24, variant="elastic", budget=BUDGET)
    d = read_complete(output)
    d["causal_commit"] = mode == "commit"
    _write(output, d)


def report(folder):
    arms = {k: read_complete(paths(folder, f"prospective-{k}")) for k in ("stock", "commit")}
    current = arms["stock"]
    if any(d["plan_sha256"] != current["plan_sha256"] or d["budget"] != current["budget"]
           or d["source_sha256"] != current["source_sha256"] for d in arms.values()):
        raise ValueError("Different cases, compute budget or instruction set in prospective comparison")
    d = _read(folder / "hindsight-prospective-plan.json")
    _write(folder / "hindsight-prospective-results.json", dict(
        stats={k: metrics(v) for k, v in arms.items()}, shared_budget=BUDGET,
        comparison=paired(arms["stock"], arms["commit"]), historical_cue_check=d["historical_cue_check"],
    ))


def replay_cost(folder):
    """Time the already pinned practice decisions without trying a second physical action."""
    practice = read_complete(paths(folder, "practice"))
    proof = read_complete(paths(folder, "verification"))
    states = {c["id"]: r["sensory"] for c, r in zip(practice["cards"], practice["rows"], strict=True)}
    with np.load(mpc.BANK) as d:
        raw = DeepField(d, "local")
    out = []
    with centered(practice["ready"]):
        actor = mpc.MPC(mpc.BANK, raw, variant="elastic", budget=BUDGET)
        try:
            twin = actor.engine.twin
            axes = actor.engine.actuators
            stride = twin._joint_increments[axes] * (twin.control_dt / twin.model.opt.timestep)
            lo, hi = twin._joint_low[axes], twin._joint_high[axes]
            for saved in proof["rows"]:
                f = saved["results"][0]
                t = f["fit"]["tick"]
                stream = states[saved["id"]]
                repair = f["causal"]["physical"]
                start = process_time()
                command, note = propose(actor, stream, f["fit"], False)
                clock = process_time() - start
                matches = repair is None and command is None
                difference = None
                if repair is not None and command is not None:
                    ref = np.array(stream[t]["ref"])
                    actual = repair["actions"][t:t + LEASE]
                    generated = []
                    for desired in command[:len(actual)]:
                        u = np.clip((np.clip(desired, lo, hi) - ref) / stride, -.8, .8)
                        generated.append(u)
                        ref = np.clip(ref + u * stride, lo, hi)
                    matches = np.array_equal(generated, actual)
                    difference = float(np.max(np.abs(np.asarray(generated) - actual)))
                out.append(dict(id=saved["id"], first_checkpoint=t, exact_command_match=bool(matches),
                                max_command_difference=difference,
                                physical_legal=bool(repair and repair["score"]["learner"]),
                                search_cpu_s=actor.diag["search_cpu_s"], total_cpu_s=clock,
                                native_trials=note["trials"]))
        finally:
            actor.close()
    _write(folder / "hindsight-causal-timing.json", dict(rows=out, verification_sha256=_hash(paths(folder, "verification"))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "stock", "commit", "report", "timing"))
    p.add_argument("--folder", type=Path, default=OUT)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--optic", type=Path, default=OPTICS)
    cli = p.parse_args()
    if cli.stage == "plan":
        freeze(cli.folder, cli.feeds, cli.optic)
    elif cli.stage == "report":
        report(cli.folder)
    elif cli.stage == "timing":
        replay_cost(cli.folder)
    else:
        evaluate(cli.folder, cli.feeds, cli.optic, cli.stage)
