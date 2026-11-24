"""Screen calibrated body replacements on archived failed low-central exchanges."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_joint_capture import observed
from .tt37_joint_deep import DeepField
from .tt37_opponent import arena
from .tt37_visual_inertia import LearnedOptics
from .tt39_terminal_servo import OUT, TerminalServo

# Diagnostic failure roster. Deployment neither knows these identifiers nor a future miss.
SOURCES = (
    ("joint-mpc-extended", 9412), ("joint-mpc-held_openings-live", 13628),
    ("joint-mpc-held_openings-live", 12387), ("joint-mpc-guarded-challenge-elastic", 10842),
    ("joint-mpc-extended", 5142),
)


def eligible(row):
    offer = row["placements"]["delivered"][-1]
    if offer["near_landed"] or offer["receive_line"] is None:
        raise ValueError("The diagnostic fixture no longer has a failed offer")
    if max(abs(offer["receive_line"]["xyz"][0]), 0.) >= .10:
        raise ValueError("The diagnostic fixture is not a central delivery")
    return [n["tick"] for n in row["searches"] if n["tick"] >= offer["far_contact"]["step"]
            and n.get("chosen") is None and -1.50 < n["p"][1] < -.53]


def course(bank, optics, output, limit):
    selected = SOURCES[:limit]
    manifests = {}
    rows = []
    ready = _read(bank.with_suffix(".json"))["ready"]
    with np.load(bank) as states:
        raw = DeepField(states, "local")
    with centered(ready):
        a = arena(protocol="random")
        renderer, rig, stride, low, high = resources(a)
        model = TerminalServo(bank, raw, variant="elastic")
        try:
            for name, ident in selected:
                path = mpc.OUT / f"{name}.json.gz"
                d = _read(path)
                if d["ready"] != ready or d["source_sha256"] != _hash(bank) or d["optical_sha256"] != _hash(optics):
                    raise ValueError("The replay does not use the same body start, cameras and instruction bank")
                key = d["cards"].index(next(c for c in d["cards"] if c["id"] == ident))
                source, card = d["rows"][key], d["cards"][key]
                ticks = eligible(source)
                manifests[path.name] = _hash(path)
                played = play(a, renderer, rig, stride, low, high, raw, model, source, card, optics, ticks)
                rows.append(played)
                _write(output, dict(ready=ready, cards=[r["card"] for r in rows], rows=rows, sources=manifests,
                                    bank_sha256=_hash(bank), optics_sha256=_hash(optics), cap=200, mode="terminal-fork",
                                    note="Archived failed offers chosen for diagnostic prefixes only. No hidden state feeds the new actor. No point reset inside the trial."))
                print(ident, "stock", source["score"]["learner"], "candidate", played["score"]["learner"],
                      "switch", played["fork_tick"], played["planner"], flush=True)
        finally:
            model.close()
            renderer.close()
            a.close()


def play(a, renderer, rig, stride, low, high, raw, model, source, card, optics, ticks):
    a.launch(source["shot"], card=card)
    a.partner.reset(seed=card["seed"])
    eye = Vision(rig, a.control_dt, raw=True)
    eye.eye = LearnedOptics(eye.eye, optics)
    model.begin()
    home = model.priors.home.copy()
    ref = home.copy()
    commands = []
    switched = None
    inbound = False

    def skip_old_search(self, tick, x, q, reference, base):
        # The archived stock planner already had no legal candidate at the bit-matched prefix.
        self.notes.append(dict(tick=tick, p=x[:3].tolist(), chosen=None, archived_probe=True))

    for tick in range(a.max_steps):
        q = a.joints
        x = observed(eye, tick, renderer.capture() if tick % 2 == 0 else None, q)
        desired = home
        if x is not None and x[4] < -.9 and x[1] < -.90:
            inbound = True
        if x is not None and (x[4] < -.9 or (inbound and x[1] < .50)):
            goal, conf = raw.get(x, q, reference=ref)
            if goal is not None:
                desired = home + conf * (goal - home)
        if switched is None:
            model.engine.observe(tick, eye.eye, q, ref.copy())
            if tick in ticks and x is not None and x[4] < -.9 and model.engine.sync():
                # Only the diagnostic activation time comes from tape metadata.
                # Candidate trajectory inputs are the current stereo and self-motion.
                with patch.object(mpc.MPC, "search", skip_old_search):
                    model.search(tick, x, q, ref, desired)
                if model.tick == tick and model.notes[-1].get("chosen") == "terminal_body_shooting":
                    prefix = [e for e in source["events"] if e["step"] <= tick]
                    if digest_json(a.events) != digest_json(prefix):
                        raise ValueError("Matched physical prefix diverged before issuing a new command")
                    switched = tick
                    model.next = tick + mpc.LEASE
                    desired = model.best[0]
            if switched is None:
                u = np.asarray(source["actions"][tick])
            else:
                u = np.clip((np.clip(desired, low, high) - ref) / stride, -.8, .8)
        else:
            desired = model.step(tick, eye.eye, x, q, ref, desired)
            u = np.clip((np.clip(desired, low, high) - ref) / stride, -.8, .8)
        ref = np.clip(ref + u * stride, low, high)
        commands.append(u.tolist())
        _, _, done, capped, _ = a.exchange(u, FAR_COMMAND)
        if done or capped or a.legal_returns + a.rival_returns - 1 >= 200:
            break
    if switched is None and digest_json(a.events) != digest_json(source["events"]):
        raise ValueError("Frozen episode failed exact replay")
    return dict(card=card, shot=source["shot"], opponent_seed=card["seed"], actions=commands,
                events=a.events, score=audit(a.events, opening_side=1), placements=a.point_placements(),
                control_own=source["score"]["learner"], fork_tick=switched, eligible_ticks=ticks,
                planner=dict(model.diag), searches=model.notes, motor_audit=a.motor_audit())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path, default=OUT/"forks.json.gz", nargs="?")
    p.add_argument("--limit", type=int, default=len(SOURCES))
    p.add_argument("--bank", type=Path, default=mpc.BANK)
    p.add_argument("--optics", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    args = p.parse_args()
    course(args.bank, args.optics, args.output, args.limit)
