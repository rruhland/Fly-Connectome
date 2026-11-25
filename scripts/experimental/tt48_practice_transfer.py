"""Deploy the arm's real exploration experience on new independent random-opponent points."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_context_search as exploration
from . import tt46_continuous_season as season
from . import tt47_season_retention as retention
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt48_transfer")
PLAN = ROOT / "streams.json"
SOURCE = exploration.ROOT / "practice-48.json.gz"


def plan():
    earlier = _read(season.PLAN)
    pool = {
        c["id"]: c for c in [*_read(fresh.PLAN)["assessment"], *earlier["assessment"]]
    }
    if len(pool) != 32 or set(pool) & {c["id"] for c in earlier["practice"]}:
        raise ValueError("Changed release roster")
    rng = np.random.default_rng(48186427)
    games = []
    for block in range(2):
        for i in rng.permutation(list(pool)):
            games.append(
                dict(
                    pool[i],
                    seed=int(rng.integers(2**30)),
                    reply_seed=int(rng.integers(2**30)),
                )
            )
    ROOT.mkdir(parents=True, exist_ok=True)
    _write(
        PLAN,
        dict(
            practice=games,
            training=_hash(SOURCE),
            original_plan=_hash(season.PLAN),
            rule="Two shuffled blocks of 32 different opening recordings absent from the perturbation course and the original readouts. Every game has new opening and reply placements. All launches count.",
        ),
    )


class Deployed(exploration.Player):
    def step(self, tick, eye, x, q, reference, base):
        # Frozen gameplay does not call reward callbacks. Drop the previous
        # attempt when its ball leaves so it cannot act on the first few frames
        # of the next ball before the new sensory context arrives.
        if not self.local.learn and x is not None and x[4] > 0.9:
            self.exploration = None
        return super().step(tick, eye, x, q, reference, base)


class Practice(exploration.LocalSearch):
    def act(self, k):
        if not self.learn:
            return super().act(k)
        self.learn = False
        try:
            return super().act(k)
        finally:
            self.learn = True


def run(a):
    if not PLAN.exists():
        plan()
    d = _read(PLAN)
    if d["training"] != _hash(SOURCE):
        raise ValueError("Exploratory practice fingerprint mismatch")
    use_plan = season.PLAN if a.deck == "assessment" else PLAN
    trained = _read(a.policy)
    fit_ids = {c["id"] for c in trained["cards"][: len(trained["rows"])]}
    ids = {c["id"] for c in _read(use_plan)[a.deck][: a.n]}
    if ids & fit_ids or ids & set(
        _read(exploration.MOTORS.with_suffix(".json"))["releases"]
    ):
        raise ValueError("Practice leaked into fresh play")
    a.out.parent.mkdir(exist_ok=True, parents=True)
    with np.load(exploration.MOTORS) as z:
        local = Practice(z["basis"], a.mode == "live")
    for row in trained["rows"]:
        for t in row["early_trials"]:
            local.rows.append(
                (np.asarray(t["context"]), np.asarray(t["coordinates"]), t["reward"])
            )
    if a.actor == "plain":
        cls = season.Player
        extra = {}
        bank = course.Carried
    else:
        cls = Deployed
        extra = dict(local=local)
        bank = exploration.Memory
    rank = fresh.shot.Coverage(_read(fresh.RISK))
    native = course.mpc.play
    source = getattr(a, "resume_from", None)
    if source is not None:
        if a.actor != "plain" or a.mode != "live" or source == a.out:
            raise ValueError("Continuation requires an independent plain-live output")
        prior = _read(source)
        prefix = prior["rows"]
        cards = _read(use_plan)[a.deck][: a.n]
        if (
            not prefix
            or len(prefix) != len(prior["cards"])
            or len(prefix) >= len(cards)
            or prior.get("actor") != a.actor
            or prior["cards"] != cards[: len(prefix)]
            or prior["mode"] != a.mode
            or prior["carry_sha256"] != _hash(a.carry)
            or prior["policy"] != _hash(a.policy)
            or prior["plan_sha256"] != _hash(use_plan)
            or any(
                u["memory"]["after"] != v["memory"]["before"]
                for u, v in zip(prefix, prefix[1:])
            )
        ):
            raise ValueError(
                "Changed practice policy, opening sequence or point-boundary memory"
            )
        original = source.with_suffix("").with_suffix(".npz")
        if _hash(original) != prior["checkpoint_sha256"]:
            raise ValueError("Invalid continuation checkpoint")
        if not a.out.exists():
            a.out.with_suffix("").with_suffix(".npz").write_bytes(original.read_bytes())
            _write(a.out, dict(prior, cards=cards, continuation_source=_hash(source)))
        elif _read(a.out).get("continuation_source") != _hash(source):
            raise ValueError("Another course already owns this output")
        a.resume = True

    def restore(model, path, document):
        if model.serial == 0:
            retention.ancestral_restore(model, path, document, season.CARRY)
        else:
            retention.season_original(model, path, document)

    def play(*args, **kw):
        row = native(*args, **kw)
        row["early_trials"] = getattr(args[10], "attempts", [])
        return row

    with (
        patch.object(course, "PLAN", use_plan),
        patch.object(course, "Carried", bank),
        patch.object(course.mpc, "play", play),
        patch.object(
            course.mpc,
            "MPC",
            lambda *args, **kw: cls(*args, coverage=rank, **extra, **kw),
        ),
        patch.object(
            course,
            "resume",
            restore,
        ),
    ):
        course.run(a)
    receipt = _read(a.out)
    receipt.update(
        actor=a.actor,
        policy=_hash(a.policy),
        practice_plan=_hash(use_plan),
        policy_definition=_hash(Path(__file__)),
    )
    _write(a.out, receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("actor", choices=("plan", "plain", "exploit"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--deck", choices=("assessment", "practice"), default="practice")
    p.add_argument("--mode", choices=("frozen", "live"), default="live")
    p.add_argument("--carry", type=Path, default=SOURCE)
    p.add_argument("--policy", type=Path, default=SOURCE)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(cap=240, resume=False)
    a = p.parse_args()
    plan() if a.actor == "plan" else run(a)
