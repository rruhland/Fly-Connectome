"""Practice small continuous joint changes on altered central feeds, using actual returns as feedback."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

from . import tt54_central_lesson as lesson
from .tt44_paired_credit import State
from .tt4_transfer_trial import _hash, _read, _write

mpc = lesson.mpc
OUT = lesson.ROOT / "central-scored-practice.json.gz"
# First three independent source releases. The last two shifts never update
# motor coefficients; their outcomes measure transfer from the earlier feeds.
SHIFTS = (
    (-0.005, 0),
    (-0.012, 0),
    (-0.020, 0),
    (-0.032, 0),
    (-0.015, 0.022),
    (-0.022, 0.043),
)


def transform(controls, stride, q0, z):
    """Shift a rewarded sequence in phase and add a smooth two-knot joint residual."""
    g = np.vstack([q0, q0 + np.cumsum(controls * stride, axis=0)])
    t = np.arange(len(controls))
    # The smooth corrections are expressed relative to the contact phase
    # estimated from the source's RGB and arm encoders, not the new ball's
    # actual impact. Arm rates are bounded a second time when sent.
    gauss1 = np.exp(-0.5 * ((t - 9) / 5.0) ** 2)
    gauss2 = np.exp(-0.5 * ((t - 18) / 5.0) ** 2)
    xi = np.clip(
        1 + t + z[-1] * 2.5 * np.exp(-0.5 * ((t - 16) / 8) ** 2), 0, len(g) - 1
    )
    positions = np.column_stack([np.interp(xi, np.arange(len(g)), a) for a in g.T])
    scale = np.array([0.16, 0.14, 0.20, 0.23, 0.28])
    positions[:, :5] += (gauss1[:, None] * z[:5] + gauss2[:, None] * z[5:10]) * scale
    return positions


def run():
    received = _read(lesson.OUT)
    examples = _read(lesson.INPUT)["training"]
    table = {(r["release"], r["dz"], r["dy"]): r for r in received["rows"]}
    if any(
        (e["case"]["release"], dz, dy) not in table
        for e in examples
        for dz, dy in SHIFTS
    ):
        raise ValueError("Full matched camera practice must finish first")
    doc = dict(
        plan_sha256=_hash(lesson.INPUT),
        baseline_sha256=_hash(lesson.OUT),
        rows=[],
        control_steps=0,
        complete=False,
        training="Actual native contact practice. Rewind to a predeclared source-camera-phase boundary on training feeds, and vary bounded direct joint rates. Real scored-and-answered only authorizes retention. Final two draws per release test previous coefficients alone.",
    )
    base_draw = lesson.opponent.draw
    for j, e in enumerate(examples):
        spec = e["case"]
        old = _read(Path(spec["source"]))
        c, r = old["cards"][spec["index"]], old["rows"][spec["index"]]
        phase = e["template"]["source_phase_tick"]
        cut = phase - 17
        earned = []
        with mpc.centered(old["ready"]):
            arena = mpc.arena(width=64, height=48)
            axes = arena._arm_ids[1][2]
            stride = arena._joint_increments[axes] * round(
                arena.control_dt / arena.model.opt.timestep
            )
            lo, hi = arena._joint_low[axes], arena._joint_high[axes]
            try:
                anchor = table[c["id"], *SHIFTS[0]]["anchor"]
                arena.launch(r["shot"], card=c)
                arena.partner.reset(seed=c["seed"])
                for u in r["actions"][:anchor]:
                    arena.exchange(u, mpc.FAR_COMMAND)
                doc["control_steps"] += anchor
                shared = State(arena)
                for i, (dz, dy) in enumerate(SHIFTS):
                    row = table[c["id"], dz, dy]
                    measured = row["variants"]["phase"]
                    tape = np.array(measured["controls"])
                    if cut <= anchor or cut >= anchor + len(tape):
                        raise ValueError(
                            "Practice boundary is outside shared receiving window"
                        )

                    def draw(card, k, protocol, rng):
                        target, name = base_draw(card, k, protocol, rng)
                        if k != spec["leg"]:
                            return target, name
                        return tuple(
                            np.clip(
                                np.array(target) + [0, dy, dz, 0],
                                lesson.B[:, 0],
                                lesson.B[:, 1],
                            )
                        ), "central_practice"

                    with patch.object(lesson.opponent, "draw", draw):
                        shared.restore(arena)
                        for u in tape[: cut - anchor]:
                            arena.exchange(u, mpc.FAR_COMMAND)
                        doc["control_steps"] += cut - anchor
                        state = State(arena)
                        q0 = arena.data.ctrl[axes].copy()
                        reference_actions = tape[cut - anchor :]
                        padded = np.vstack([reference_actions, np.zeros((180, 6))])
                        choices = [("same", np.zeros(11))]
                        choices.extend(("earned", z) for z in earned)
                        if i < 4:
                            sobol = qmc.Sobol(
                                11, scramble=True, seed=540110 + j * 13 + i
                            )
                            choices.extend(
                                ("explore", z) for z in 2 * sobol.random_base2(6) - 1
                            )
                        attempts = []
                        win = None
                        for kind, z in choices:
                            state.restore(arena)
                            goals = transform(padded, stride, q0, z)
                            ref = q0.copy()
                            sent = []
                            for k, target in enumerate(goals[:160]):
                                if kind == "same":
                                    u = padded[k]
                                else:
                                    u = np.clip(
                                        (np.clip(target, lo, hi) - ref) / stride,
                                        -0.8,
                                        0.8,
                                    )
                                ref = np.clip(ref + u * stride, lo, hi)
                                _, _, done, cap, info = arena.exchange(
                                    u, mpc.FAR_COMMAND
                                )
                                sent.append(u.tolist())
                                if (
                                    done
                                    or cap
                                    or any(
                                        ev["kind"] == "legal_return"
                                        and ev["side"] == -1
                                        for ev in info["events"]
                                    )
                                    and len(arena.point_placements()["delivered"])
                                    > spec["leg"] + 1
                                ):
                                    break
                            doc["control_steps"] += len(sent)
                            score = lesson.native_result(arena, spec["leg"])
                            if score.get("incoming") != measured.get("incoming"):
                                raise ValueError(
                                    "Motor practice saw a different incoming ball"
                                )
                            if kind == "same" and (
                                score["legal"],
                                score["answered"],
                            ) != (measured["legal"], measured["answered"]):
                                raise ValueError(
                                    "Baseline motor commands changed referee outcome"
                                )
                            attempts.append(
                                dict(kind=kind, z=z.tolist(), steps=len(sent), **score)
                            )
                            if score["answered"]:
                                win = dict(
                                    kind=kind,
                                    cut=cut,
                                    controls=sent,
                                    events=arena.events.copy(),
                                )
                                if kind == "explore" and i < 4:
                                    earned.insert(0, z.copy())
                                break
                    doc["rows"].append(
                        dict(
                            release=c["id"],
                            index=i,
                            leg=spec["leg"],
                            dz=dz,
                            dy=dy,
                            training=i < 4,
                            attempts=attempts,
                            success=win,
                        )
                    )
                    _write(OUT, doc)
                    print(
                        c["id"],
                        i,
                        len(attempts),
                        attempts[0]["physical_z"],
                        bool(win),
                        win and win["kind"],
                        flush=True,
                    )
            finally:
                arena.close()
    doc["complete"] = True
    _write(OUT, doc)


if __name__ == "__main__":
    run()
