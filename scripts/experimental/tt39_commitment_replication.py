"""Full-game replication of camera-triggered joint commitment on all 24 assigned cards."""

import argparse
from pathlib import Path
from unittest.mock import patch

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_report import paired, read_complete, replay_case
from .tt38_joint_mpc_report import metrics
from .tt39_carry_joints import Carried
from .tt39_hindsight_prospective import BUDGET, LEASE, RETRY_GAP, CausalCommit


OUT = Path("artifacts/tt39")
COURSE = Path("artifacts/tt37/memory-course.json")
OPTICS = Path("artifacts/tt37/rebound.npz")
PRACTICE = Path("artifacts/tt37/camera-time-school.npz")
REFERENCES = {"planner_free": OUT / "action-baseline.json.gz",
              "integrated": Path("artifacts/tt37/integrated-replication.json.gz")}
CAP = 200
DECK = "replication"
ARMS = ("stock", "commit", "carry")


def destination(folder, arm):
    return folder / f"commitment-replication-{arm}.json.gz"


def freeze(folder, feeds):
    source = _read(COURSE)
    cards = source[DECK]
    hashes = {name: _hash(path) for name, path in REFERENCES.items()}
    archived = {name: read_complete(path) for name, path in REFERENCES.items()}
    if (len(cards) != 24 or len({c["id"] for c in cards}) != 24
            or any(d["cards"] != cards for d in archived.values())
            or source["feed_sha256"] != _hash(feeds)):
        raise ValueError("The 24 preassigned replication starts or measured file changed")
    roles = {c["id"] for c in cards}
    for source_path in (mpc.BANK, PRACTICE):
        training = _read(source_path.with_suffix(".json"))
        if roles & {r["card"]["id"] for r in training["practice"]}:
            raise ValueError("Joint action teaching contains a replication recording")
        if training["bank_sha256"] != _hash(source_path):
            raise ValueError("The declared action memory changed")
    _write(folder / "commitment-replication-plan.json", dict(
        replication=cards, caps={"alternating": CAP, "fresh_attempts": 2}, budget=BUDGET,
        command_lease_ticks=LEASE, retry_gap_ticks=RETRY_GAP,
        feed_sha256=_hash(feeds), optics_sha256=_hash(OPTICS), instructions_sha256=_hash(mpc.BANK),
        practice_sha256=_hash(PRACTICE), practice_receipt_sha256=_hash(PRACTICE.with_suffix(".json")),
        course_sha256=_hash(COURSE), archived=hashes,
        inclusion="All 24 assigned starts. No filtering, retries by release, hidden state or hindsight.",
        carry="Fixed score-credited camera-time-school practice only. No updates within or across tested points.",
        empty_prior="Ordinary standing instructions only, no real-match additions.",
    ))


def assess(folder, feeds, arm, *, resume=False):
    plan = folder / "commitment-replication-plan.json"
    d = _read(plan)
    if (d["instructions_sha256"], d["practice_sha256"], d["optics_sha256"], d["feed_sha256"],
            d["practice_receipt_sha256"]) != (
            _hash(mpc.BANK), _hash(PRACTICE), _hash(OPTICS), _hash(feeds), _hash(PRACTICE.with_suffix(".json"))):
        raise ValueError("A frozen physical training or evaluation input changed")
    if d[DECK] != _read(COURSE)[DECK]:
        raise ValueError("Not the declared full 24-case computation")
    actor = mpc.MPC if arm == "stock" else CausalCommit
    fragments = (lambda bank, engine: Carried(
        bank, engine, practice=[PRACTICE], held={r["id"] for r in d[DECK]})) if arm == "carry" else mpc.Fragments
    path = destination(folder, arm)
    earlier = None
    if resume and path.exists():
        earlier = _read(path)
        if (earlier["mode"] != "frozen" or earlier["plan_sha256"] != _hash(plan)
                or earlier["source_sha256"] != d["instructions_sha256"]
                or earlier["optical_sha256"] != d["optics_sha256"] or earlier["cards"] != d[DECK]
                or earlier["cap"] != CAP or earlier["budget"] != BUDGET):
            raise ValueError("The last saved point belongs to a different benchmark")
    done = 0 if earlier is None else len(earlier["rows"])
    if done < len(d[DECK]):
        # Core writes a durable partial after each real point. A frozen run can
        # restart at the next card because its physical point and priors reset.
        if done:
            target = folder / f".commitment-replication-{arm}-resume.json.gz"
        else:
            target = path
        run = mpc.run
        emit = mpc._write

        def save_progress(file, doc):
            if done and file == target:
                if doc["model_source_sha256"] != earlier["model_source_sha256"]:
                    raise ValueError("Core simulation changed since the archived prefix")
                merged = dict(earlier, rows=earlier["rows"] + doc["rows"], resumed_after=done)
                emit(path, merged)
            return emit(file, doc)

        with (patch.object(mpc, "MPC", actor), patch.object(mpc, "Fragments", fragments),
              patch.object(mpc, "cards_for", return_value=d[DECK][done:]),
              patch.object(mpc, "_write", save_progress)):
            run(feeds, mpc.BANK, plan, DECK, "frozen", target, optic=OPTICS,
                cap=CAP, variant="elastic", budget=BUDGET, progress=0)
        complete = read_complete(target)
        if earlier is not None and done:
            if complete["model_source_sha256"] != earlier["model_source_sha256"]:
                raise ValueError("Live physics code changed after the partial frozen assessment")
            earlier["rows"] += complete["rows"]
            complete = earlier
            complete["resumed_after"] = done
            target.unlink()
    else:
        complete = earlier
    complete["arm"] = arm
    complete["practice_sha256"] = d["practice_sha256"] if arm == "carry" else None
    complete["actor_source_sha256"] = _hash(Path(__file__).with_name("tt39_hindsight_prospective.py"))
    _write(path, complete)


def archived_counts(d):
    """Do not invent CPU counters absent from the separately recorded actor."""
    own = [r["score"]["learner"] for r in d["rows"]]
    return dict(
        points=len(own), own=sum(own), first=sum(v > 0 for v in own),
        later=sum(max(v - 1, 0) for v in own), longest_own=max(own, default=0),
        longest_alternating=max((r["score"]["returns"] for r in d["rows"]), default=0),
        no_offer=sum(not r["placements"]["delivered"] for r in d["rows"]),
    )


def check(folder, *, audit=False):
    p = folder / "commitment-replication-plan.json"
    manifest = _read(p)
    all_results = {}
    for arm in ARMS:
        path = destination(folder, arm)
        d = read_complete(path)
        if (d["cards"] != manifest[DECK] or d["cap"] != CAP or d["arm"] != arm or d["mode"] != "frozen"
                or len(d["rows"]) != 24 or d["source_sha256"] != manifest["instructions_sha256"]
                or d["optical_sha256"] != manifest["optics_sha256"] or d["plan_sha256"] != _hash(p)):
            raise ValueError("A declared start was omitted or the matched physical run changed")
        memories = [r["memory"] for r in d["rows"]]
        if any(r["before"] != r["after"] or r["before"] != memories[0]["before"] for r in memories):
            raise ValueError("The frozen controller learned from held play")
        if bool(memories[0]["before"]["rows"]) != (arm == "carry"):
            raise ValueError("Unexpected carried-state boundary")
        all_results[arm] = d
    if any(x["model_source_sha256"] != all_results["stock"]["model_source_sha256"]
           or x["actor_source_sha256"] != all_results["stock"]["actor_source_sha256"]
           for x in all_results.values()):
        raise ValueError("Arms ran on different core simulation or cue implementations")
    archived = {}
    for role, path in REFERENCES.items():
        if _hash(path) != manifest["archived"][role]:
            raise ValueError("Archived reference moved")
        old = read_complete(path)
        archived[role] = dict(metrics=archived_counts(old), versus_commit=paired(old, all_results["commit"]))
    stats = {k: metrics(d) for k, d in all_results.items()}
    out = {
        "stats": stats,
        "stock_to_commit": paired(all_results["stock"], all_results["commit"]),
        "commit_to_fixed_carry": paired(all_results["commit"], all_results["carry"]),
        "stock_to_fixed_carry": paired(all_results["stock"], all_results["carry"]),
        "no_offer_assignments": {k: sum(not r["placements"]["delivered"] for r in d["rows"])
                                 for k, d in all_results.items()},
        "archived_references": archived,
        "source_hashes": {k: _hash(destination(folder, k)) for k in ARMS},
    }
    _write(folder / "commitment-replication-results.json", out)
    if audit:
        verified = []
        for arm, d in all_results.items():
            i = max(range(len(d["rows"])), key=lambda i: d["rows"][i]["score"]["learner"])
            q = replay_case(d["ready"], "random", d["cards"][i], d["rows"][i])
            verified.append(dict(arm=arm, source_sha256=out["source_hashes"][arm], **q))
        _write(folder / "commitment-replication-audit.json", dict(rows=verified))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan", *ARMS, "report", "audit"))
    p.add_argument("--folder", type=Path, default=OUT)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--resume", action="store_true")
    cli = p.parse_args()
    cli.folder.mkdir(parents=True, exist_ok=True)
    if cli.phase == "plan":
        freeze(cli.folder, cli.feeds)
    elif cli.phase in ARMS:
        assess(cli.folder, cli.feeds, cli.phase, resume=cli.resume)
    else:
        check(cli.folder, audit=cli.phase == "audit")
