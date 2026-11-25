"""Check whether the two copies changed action before or after earning new experience."""

from . import tt107_resume_interception as course
from .tt4_transfer_trial import _read, _write, _hash
from .tt29_endurance import audit
from .tt68_clamp_transfer import describe
from .tt80_course_report import matched_subset

ROOT = course.ROOT


def run():
    live = _read(ROOT / "after-first.json.gz")
    cold = _read(ROOT / "frozen.json.gz")
    c = live["cards"][0]
    a, b = live["rows"][0], cold["rows"][0]
    if c != cold["cards"][0] or c != _read(course.PLAN)["practice"][0]:
        raise ValueError("Different initial release or draws")
    if not (
        a["memory"]["before"] == b["memory"]["before"] == b["memory"]["after"]
        and a["memory"]["after"] != a["memory"]["before"]
    ):
        raise ValueError("Memory mismatch")
    for r in (a, b):
        if audit(r["events"], opening_side=1) != r["score"]:
            raise ValueError("Tape and referee differ")
    first = next(
        i for i, (x, y) in enumerate(zip(a["actions"], b["actions"])) if x != y
    )
    before = audit([e for e in a["events"] if e["step"] <= first], opening_side=1)[
        "learner"
    ]
    if before < 1:
        raise ValueError("Commands diverged before new experience")
    witness = ROOT / "live-record-1-67.json"
    proof = _read(witness)
    if (
        proof["own"] != a["score"]["learner"]
        or proof["launches"] != 1
        or proof["clearance"]["penetration_samples"]
    ):
        raise ValueError("Replay does not certify point")
    half = ROOT / "first-frozen-tape.json.gz"
    _write(half, dict(ready=cold["ready"], cards=[c], rows=[b], opening_side=1))
    out = dict(
        ready=live["ready"],
        card=c,
        weight_checksum=_hash(course.CARRY.with_suffix("").with_suffix(".npz")),
        after_first_sha256=_hash(ROOT / "after-first.npz"),
        replay_sha256=_hash(witness),
        frozen_tape=_hash(half),
        live=describe([a]),
        frozen=describe([b]),
        first_different_command_zero_based=first,
        own_returns_before_divergence=before,
        identical_incoming=matched_subset([b], [a]),
        updates=a["planner"]["updates"],
        rewarded_fragments=a["planner"]["credited_fragments"],
        live_cpu_s=a["runtime_cpu_s"],
        frozen_cpu_s=b["runtime_cpu_s"],
    )
    _write(ROOT / "first-point-learning.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    run()
