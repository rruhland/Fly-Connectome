"""Separate planner-assisted ceiling for session-to-session rewarded movement retention."""

import argparse
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from . import tt29_action_memory as motor
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_partner_fast import FastPartnerArena
from .tt29_rally_physics import LATERAL_STRIKE_READY
from .tt31_curriculum import ERRORS
from .tt31_curriculum_opponent import CourseArena
from .tt31_ready import centered
from .tt31_recovery import Recovery
from .tt35_replies import varied_opponent
from .tt37_opponent import LiveArena

ROOT = Path(__file__).resolve().parents[2] / "artifacts"
OUT = ROOT / "tt37"
WARM = ROOT / "tt31" / "ready-staged-checkpoint.json.gz"
COOPERATIVE = ROOT / "tt29" / "coupled-173-movements.json.gz"
READY = ROOT / "tt31" / "left-ready.json"
PILOT = (3180, 12644, 9258, 4250, 4805, 13509)


def roster(phase, path=None):
    if path:
        return _read(path)[phase]
    if phase == "lane":
        return [dict(id=sid, index=i, seed=370941+3*i, key=f"lane-{i}-{sid}",
                     reply_seed=370975+13*i, profile="lane", mode="full")
                for i, sid in enumerate((7271, 4187, 7345, 11292, 7637))]
    if phase != "pilot":
        raise ValueError("Provide a shared course for held assessment or practice")
    candidates = {item["id"]: item for item in _read(ROOT / "tt36" / "course-new.json.gz")["cards"]}
    return [dict(candidates[i], reply_seed=370819 + index) for index, i in enumerate(PILOT)]


class Lane(LiveArena):
    """Accept an already recorded serve toward the learner for the cooperative comparison."""

    def launch(self, shot, *, card=None):
        self.card = next(self.cards) if card is None else card
        if self.card["id"] != shot["id"] or shot["pos_y"] <= 0 or shot["vel_y"] >= 0:
            raise ValueError("A lane card needs the corresponding incoming launch")
        self.crossings, self.before_step = {}, None
        FastPartnerArena.launch(self, shot)
        self._last_hitter = -1
        self.launches += 1

    def point_placements(self):
        # The measured opening is already incoming here, unlike the shared
        # course. It has no requested far-goal entry and is excluded from this
        # placement calculation while remaining in the independent referee log.
        first_far_contact = next((i for i, e in enumerate(self.events)
                                  if e["kind"] == "contact" and e["side"] == -1), len(self.events))
        view = copy(self)
        view.events = [e for i, e in enumerate(self.events)
                       if not (i < first_far_contact and e["kind"] == "legal_return" and e["side"] == -1)]
        view.crossings = {i-1: x for i, x in self.crossings.items() if i > 1}
        return super(Lane, view).point_placements()


def play(feeds, temporary, card, checkpoint, *, cap, update, protocol):
    live = []

    def factory(**settings):
        if protocol == "pilot":
            arena = CourseArena(cards=[card], **settings)
        elif protocol == "lane":
            arena = Lane(protocol="random", cards=[card], **settings)
        else:
            from .tt37_opponent import arena as make_arena
            arena = make_arena(cards=[card], protocol=protocol, **settings)
        live.append(arena)
        return arena

    cooperative = protocol == "lane"
    optical = None if cooperative else checkpoint if checkpoint else ERRORS
    if protocol == "pilot":
        branch = varied_opponent(card["index"], card["reply_seed"])
    else:
        branch = nullcontext()
    posture = nullcontext() if cooperative else centered(_read(READY)["ready"])
    recovery = nullcontext() if cooperative else patch.object(motor, "RallyObjective", Recovery)
    with posture, branch, recovery:
        motor.run(feeds, temporary, [card["id"]], "left+right", "lateral",
                  checkpoint=checkpoint, optical_checkpoint=optical, arena_factory=factory,
                  opening_side=-1 if cooperative else 1,
                  opponent_seeds=[card["seed"]], return_target=cap, rated=True, motor_envelope=True,
                  robust_contact=True, rally_objective=True, recent=True, blend=True, fast_partner=True,
                  cached_stereo=True, net_rescue=True, camera_period=2,
                  optical_outcomes=None if cooperative else "learn", censored_optical_outcomes=not cooperative,
                  no_memory=not update, freeze_optical_outcomes=not update and not cooperative)
    receipt = _read(temporary)
    row = receipt["rows"][0]
    row["placements"] = live[0].point_placements()
    kept = {key: receipt[key] for key in ("checkpoint", "optical_outcome_checkpoint") if key in receipt}
    return row, kept


def run(args):
    cards = roster(args.phase, args.course)[:args.limit]
    protocol = args.phase if args.phase in ("pilot", "lane") else args.protocol
    cooperative = protocol == "lane"
    checkpoint = args.checkpoint or ((COOPERATIVE if cooperative else WARM) if args.mode == "warm" else None)
    update = args.mode == "carry"
    ready = [float(x) for x in LATERAL_STRIKE_READY] if cooperative else _read(READY)["ready"]
    output = args.output or OUT / f"online-planner-{args.phase}-{args.mode}.json.gz"
    result = dict(planner_assisted=True, mode=args.mode, phase=args.phase, protocol=protocol, cards=cards,
                  ready=ready, opening_side=-1 if cooperative else 1, cap=args.cap, feed_sha256=_hash(args.feeds),
                  start_sha256=_hash(checkpoint) if checkpoint else None, rows=[])
    with TemporaryDirectory(prefix="tt37-planner-") as directory:
        temporary = Path(directory) / "trial.json.gz"
        for card in cards:
            row, kept = play(args.feeds, temporary, card, checkpoint, cap=args.cap, update=update, protocol=protocol)
            result["rows"].append(row)
            if update:
                checkpoint = output.with_name(output.name.removesuffix(".json.gz") + "-checkpoint.json.gz")
                _write(checkpoint, kept)
                result["end_sha256"] = _hash(checkpoint)
            _write(output, result)
            print(args.mode, card["id"], row["score"]["learner"], row["score"]["returns"],
                  row["initial_examples"], row["final_examples"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase")
    parser.add_argument("mode", choices=("cold", "carry", "warm"))
    parser.add_argument("--protocol", default="random")
    parser.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    parser.add_argument("--course", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cap", type=int, default=24)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
