"""Diagnostic-only physical ceiling: substitute measured truth in the private planner."""

import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_optical_clinic import OpticalClinic
from .tt39_carry_joints import Carried
from .tt41_bounce_joint import CHECKPOINT, OPTIC, PRACTICE, SRC, initial_bank

OUT = Path("artifacts/tt42_private_init")
FROZEN = Path("artifacts/tt40/matched-within.json.gz")
GROUPS = {
    "stock": (),
    "phase": ("phase",),
    "pv": ("pv",),
    "spin": ("spin",),
    "body": ("body",),
    "ball": ("pv", "spin", "phase"),
    "all": ("pv", "spin", "phase", "body"),
}


def mark(a, name):
    joint = a.model.joint(name)
    return int(joint.qposadr[0]), int(joint.dofadr[0])


class PrivateInit(OpticalClinic):
    """The evaluator truth never touches the stock real camera path, gates or scorer."""

    def __init__(self, plant, mode, *, body=True):
        super().__init__(body=body)
        self.plant = plant
        self.mode = mode
        self.fields = GROUPS[mode]
        self.live_joints, _, self.live_actuators, _, _ = plant._arm_ids[1]
        self.live_dofs = plant.model.jnt_dofadr[
            plant.model.actuator_trnid[self.live_actuators, 0]
        ]
        self.live_paddle_q, self.live_paddle_v = mark(plant, "learner_paddle_free")
        self.records = []

    def sync(self):
        if not super().sync():
            return False
        src, dst = self.plant, self.twin
        p, v = dst.ball.copy(), dst.velocity.copy()
        w = dst.data.qvel[dst._ball_dof + 3 : dst._ball_dof + 6]
        d, d_native = dst.data, src.data
        q = self.paddle_q
        body_mm = 1000.0 * np.linalg.norm(
            d.qpos[q : q + 3]
            - d_native.qpos[self.live_paddle_q : self.live_paddle_q + 3]
        )
        # Each quantity is measured and logged before changing any counterpart.
        # The late correction and its truth stay in this evaluator module.
        self.records.append(
            {
                "tick": self.tick,
                "optical_phase": bool(dst._received_bounce),
                "native_phase": bool(src._received_bounce),
                "dp_mm": (1000.0 * (src.ball - p)).tolist(),
                "dv_m_s": (src.velocity - v).tolist(),
                "dw_rad_s": (src.spin - w).tolist(),
                "paddle_origin_error_mm": float(body_mm),
                "dqdot_rad_s": (
                    d_native.qvel[self.live_dofs] - d.qvel[self.dofs]
                ).tolist(),
            }
        )
        if "pv" in self.fields:
            dst.ball[:] = src.ball
            dst.velocity[:] = src.velocity
        if "spin" in self.fields:
            # _refresh gives the private ball an identity quaternion. The true
            # spin is converted out of the live free joint's body frame first.
            w[:] = src.spin
        if "phase" in self.fields:
            dst._received_bounce = bool(src._received_bounce)
        if "body" in self.fields:
            d.qpos[self.positions] = d_native.qpos[self.live_joints]
            d.qvel[self.dofs] = d_native.qvel[self.live_dofs]
            # Keep own issued reference in its ordinary slot and copy only the
            # rotational/translational DOFs of the held near blade, not the ball.
            f, g = self.live_paddle_q, self.live_paddle_v
            i, j = self.paddle_q, self.paddle_v
            d.qpos[i : i + 7] = d_native.qpos[f : f + 7]
            d.qvel[j : j + 6] = d_native.qvel[g : g + 6]
        if self.fields:
            if not np.isclose(
                d_native.time, self.tick * dst.control_dt, atol=1e-8, rtol=0
            ):
                raise RuntimeError(
                    "Private initialization disagrees with the physical command clock"
                )
            mujoco.mj_forward(dst.model, d)
        return True


def runner(a, res, source, card, index, mode, raw, shot, capacity):
    kwargs = {"practice": [PRACTICE], "held": {c["id"] for c in source["cards"]}}
    with (
        patch.object(
            mpc, "OpticalClinic", lambda *, body: PrivateInit(a, mode, body=body)
        ),
        patch.object(
            mpc, "Fragments", lambda bank, engine: Carried(bank, engine, **kwargs)
        ),
    ):
        model = mpc.MPC(
            mpc.BANK, raw, variant=source["variant"], budget=source["budget"]
        )
    old = source["rows"][index]
    if model.priors.online.maxlen != capacity:
        raise ValueError("Carried bank lost its reference capacity")
    initial_bank(model, index + 1, old["memory"]["before"])
    model.serial = index
    try:
        row = mpc.play(
            a,
            *res,
            source["ready"],
            card,
            shot,
            raw,
            model,
            OPTIC,
            "frozen",
            source["cap"],
        )
        row["private_state_discrepancies"] = model.engine.records
    finally:
        model.close()
    if row["memory"]["before"] != row["memory"]["after"] or row["planner"].get(
        "credited_fragments", 0
    ):
        raise ValueError("A truth-conditioned command must never earn motor memory")
    return row


def run(args):
    source = _read(SRC)
    control = _read(FROZEN)
    if (
        _hash(SRC) != control["source_sha256"]
        or _hash(CHECKPOINT) != source["checkpoint_sha256"]
        or _hash(OPTIC) != source["optical_sha256"]
        or _hash(mpc.BANK) != source["source_sha256"]
        or source["previous_practice"] != {PRACTICE.name: _hash(PRACTICE)}
    ):
        raise ValueError(
            "The stock and prior archive differ from the earned-memory control"
        )
    ids = args.ids
    modes = args.modes
    if len(set(ids)) != len(ids) or len(set(modes)) != len(modes):
        raise ValueError(
            "Duplicate oracle arms or releases are not independent comparisons"
        )
    cases = {c["id"]: (i, c) for i, c in enumerate(source["cards"])}
    launches = mpc.measured_launches(args.feeds)
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    out = {
        "schema": "tt42-private-init-v1",
        "source_sha256": _hash(SRC),
        "control_sha256": _hash(FROZEN),
        "carry_checkpoint_sha256": _hash(CHECKPOINT),
        "cap": source["cap"],
        "code_sha256": {
            name: _hash(Path(mpc.__file__).with_name(name))
            for name in (
                "tt42_private_init.py",
                "tt38_joint_mpc.py",
                "tt37_optical_clinic.py",
                "tt29_rally_rollout.cpp",
            )
        },
        "modes": {m: list(GROUPS[m]) for m in modes},
        "ids": ids,
        "rows": [],
        "isolated": "Frozen earned memory. Private truth affects only candidate-hypothesis initialization. Actor, optical gates, retrieval, target distribution and actual native court unchanged.",
    }
    _write(args.out, out)
    with mpc.centered(source["ready"]):
        a = mpc.arena()
        res = mpc.resources(a)
        try:
            for cid in ids:
                index, c = cases[cid]
                archived = control["rows"][index]
                r = {
                    "card": c,
                    "archived_frozen_score": archived["score"]["learner"],
                    "archived_credited_score": source["rows"][index]["score"][
                        "learner"
                    ],
                    "archived_frozen_events_sha256": digest_json(archived["events"]),
                    "modes": {},
                }
                out["rows"].append(r)
                for mode in modes:
                    if mode == "stock" and args.archived_stock:
                        # Existing complete control was freshly camera-run with the
                        # same priors, protocol and no within-point learning.
                        q = archived | {"baseline_from_existing_frozen_receipt": True}
                    else:
                        q = runner(
                            a, res, source, c, index, mode, raw, launches[cid], 768
                        )
                    r["modes"][mode] = q
                    q["matching_initial_memory"] = (
                        q["memory"]["before"] == archived["memory"]["before"]
                    )
                    if not q["matching_initial_memory"]:
                        raise ValueError(
                            "Mode imported different prior motor experience"
                        )
                    if mode == "stock":
                        q["archived_action_exact"] = q["actions"] == archived["actions"]
                        q["archived_event_exact"] = digest_json(
                            q["events"]
                        ) == digest_json(archived["events"])
                    _write(args.out, out)
                    print(
                        cid,
                        mode,
                        q["score"]["learner"],
                        q["score"]["opponent"],
                        round(q["runtime_cpu_s"], 2),
                        q.get("archived_action_exact"),
                        flush=True,
                    )
        finally:
            res[0].close()
            a.close()
    out["complete"] = True
    _write(args.out, out)


def visual_residuals(records):
    if not records:
        return None
    keys = ("dp_mm", "dv_m_s", "dw_rad_s", "paddle_origin_error_mm", "dqdot_rad_s")
    summary = {}
    for key in keys:
        samples = np.asarray([r[key] for r in records])
        error = (
            np.linalg.norm(samples, axis=1) if samples.ndim == 2 else np.abs(samples)
        )
        summary[key] = {
            "median": float(np.median(error)),
            "p90": float(np.quantile(error, 0.9)),
            "maximum": float(np.max(error)),
        }
    return {
        "decisions": len(records),
        "phase_disagreement": sum(
            r["optical_phase"] != r["native_phase"] for r in records
        ),
        "error_norms": summary,
    }


def compare(args):
    docs = [_read(p) for p in args.sources]
    source = _read(SRC)
    merged = {}
    for d in docs:
        if not d.get("complete") or d["source_sha256"] != _hash(SRC):
            raise ValueError("Require complete matching comparison segments")
        for r in d["rows"]:
            into = merged.setdefault(r["card"]["id"], {"card": r["card"], "modes": {}})
            if set(into["modes"]) & set(r["modes"]):
                raise ValueError("Duplicate mode for one release")
            into["modes"].update(r["modes"])
    result = {"sources": {str(p): _hash(p) for p in args.sources}, "rows": []}
    start = process_time()
    with mpc.centered(source["ready"]):
        a = mpc.arena()
        try:
            for c in merged.values():
                stock = c["modes"]["stock"]
                card = c["card"]
                group = {"card": card, "results": {}}
                for mode, r in c["modes"].items():
                    premiere = lambda t: t["placements"]["delivered"][0]
                    if digest_json(premiere(stock)["far_contact"]) != digest_json(
                        premiere(r)["far_contact"]
                    ) or digest_json(premiere(stock)["near_bounce"]) != digest_json(
                        premiere(r)["near_bounce"]
                    ):
                        raise ValueError("Benchmark first physical delivery changed")
                    reqs = lambda t: [
                        (v["draw"], v["target"]) for v in t["placements"]["requested"]
                    ]
                    u, v = reqs(stock), reqs(r)
                    n = min(len(u), len(v))
                    if u[:n] != v[:n] or any(draw != "random" for draw, _ in v):
                        raise ValueError(
                            "Opponent aim stream or physical profile changed"
                        )
                    tick = next(
                        (
                            i
                            for i, (g, j) in enumerate(
                                zip(stock["actions"], r["actions"])
                            )
                            if g != j
                        ),
                        None,
                    )
                    identical = (
                        tick
                        if tick is not None
                        else min(len(stock["actions"]), len(r["actions"]))
                    )
                    events_before = lambda t, end=identical: [
                        e for e in t["events"] if e["step"] <= end
                    ]
                    if digest_json(events_before(stock)) != digest_json(
                        events_before(r)
                    ):
                        raise ValueError(
                            "Physical prefix differs before first control change"
                        )
                    last = r["events"][-1]
                    summary = {
                        "legal": r["score"]["learner"],
                        "opponent": r["score"]["opponent"],
                        "cpu_s": r["runtime_cpu_s"],
                        "ticks": len(r["actions"]),
                        "terminal": last.get("reason", last.get("kind")),
                        "first_action_change": tick,
                        "first_receive_same": True,
                        "first_touched": premiere(r)["near_contact"] is not None,
                        "touches": sum(
                            o["near_contact"] is not None
                            for o in r["placements"]["delivered"]
                        ),
                        "terminal_touch": r["placements"]["delivered"][-1][
                            "near_contact"
                        ]
                        is not None,
                        "residuals": visual_residuals(
                            r.get("private_state_discrepancies", [])
                        ),
                        "target_prefix_draws": n,
                        "random_targets": len(v),
                        "event_sha256": digest_json(r["events"]),
                        "no_credit": r["memory"]["before"] == r["memory"]["after"],
                    }
                    if (
                        args.replay
                        and (not args.replay_ids or card["id"] in args.replay_ids)
                        and (not args.replay_modes or mode in args.replay_modes)
                    ):
                        a.sample_clearance(live=args.cad)
                        a.launch(r["shot"], card=card)
                        a.partner.reset(seed=card["seed"])
                        for cmd in r["actions"]:
                            a.exchange(cmd, mpc.FAR_COMMAND)
                        if digest_json(a.events) != digest_json(r["events"]):
                            raise ValueError(
                                f"{card['id']} {mode}: independently replayed events differ"
                            )
                        if digest_json(audit(a.events, opening_side=1)) != digest_json(
                            r["score"]
                        ):
                            raise ValueError(
                                f"{card['id']} {mode}: independent point score differs"
                            )
                        summary.update(
                            exact_replay=True, clearance=a.clearance_result()
                        )
                    group["results"][mode] = summary
                result["rows"].append(group)
        finally:
            a.close()
    result["audit_cpu_s"] = process_time() - start
    _write(args.out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("run", "verify"))
    p.add_argument("out", type=Path)
    p.add_argument("--ids", nargs="+", type=int, default=[8510, 12109, 14390])
    p.add_argument(
        "--modes",
        nargs="+",
        choices=GROUPS,
        default=["stock", "pv", "spin", "body", "all"],
    )
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--archived-stock", action="store_true")
    p.add_argument("--sources", nargs="+", type=Path)
    p.add_argument("--replay", action="store_true")
    p.add_argument("--replay-ids", type=int, nargs="+")
    p.add_argument("--replay-modes", choices=GROUPS, nargs="+")
    p.add_argument("--cad", action="store_true")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (run if args.task == "run" else compare)(args)
