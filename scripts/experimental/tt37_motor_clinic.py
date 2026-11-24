"""Practice joint corrections using isolated physical counterfactuals, then replay with a frozen memory."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from . import tt33_joint as joint
from .tt4_transfer_trial import _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, Rollout, measured_launches
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt37_belief import ArrivalHistory
from .tt37_opponent import arena
from .tt37_posture import Surface, sample

GRIP = -0.1
FAR = np.zeros(5)
AIM = np.array([0.0, 1.29, 1.05, 3.6], dtype=float)
SCALE = np.array([0.15, 0.1, 0.8, 1.2, 1.7, 0.9, 0.9, 0.9, 0.9, 0.9])


def geometry(surf, opt, *, depth=None):
    x, z = opt[1:3]
    vx, _, vz = opt[3:6]
    normal = np.array([-0.18 * x - 0.085 * vx, 1.0, 0.29 - 0.095 * vz])
    normal /= np.linalg.norm(normal)
    qs, v = surf.solution(x, z, normal, depth=depth)
    return qs, v


def generate(n):
    # Position and joint velocity at the action center, discovered by bounded search.
    raw = (
        2
        * qmc.Sobol(10, scramble=True, seed=37091).random_base2(
            int(np.ceil(np.log2(n)))
        )[:n]
        - 1
    )
    return np.r_[
        np.zeros((1, 10)),
        raw * np.r_[[0.23, 0.33, 0.38, 0.36, 0.50], [2.2, 3.0, 2.8, 3.0, 3.6]],
    ]


def schedules(center, motion, eta, params, dt, home, frames):
    t = (1 + np.arange(frames)) * dt
    span = np.clip(t - eta + 0.030, -0.13, 0.065)
    poses = center + params[:, :5]
    velocities = 0.86 * motion + params[:, 5:]
    result = poses[:, None, :] + span[None, :, None] * velocities[:, None, :]
    return np.concatenate((result, np.full((*result.shape[:2], 1), GRIP)), axis=-1)


def recall(bank, x):
    if not bank:
        return np.zeros(10)
    d = np.linalg.norm((np.array([e["x"] for e in bank]) - x) / SCALE, axis=1)
    return np.array(bank[int(np.argmin(d))]["change"])


def consequence_search(history, elite, rng, sizes, count):
    """Use local observed sensitivities of outgoing velocity to propose motor changes."""
    actions = np.asarray([h[0] for h in history]) / sizes
    observations = np.asarray([h[1] for h in history])
    hit = observations[:, 1] > 0
    if hit.sum() < 8:
        return np.empty((0, 10))
    actions = actions[hit]
    observed = observations[hit]
    proposals = []
    for seed in elite[:5]:
        center = seed / sizes
        d = np.linalg.norm(actions - center, axis=1)
        idx = np.argsort(d)[: min(48, len(d))]
        w = np.exp(-0.5 * (d[idx] / max(0.5, np.median(d[idx]))) ** 2)
        w /= w.sum()
        a = actions[idx]
        v = observed[idx, 2:5]
        da = a - w @ a
        dv = v - w @ v
        jac = np.linalg.solve(
            da.T @ (w[:, None] * da) + 0.03 * np.eye(10), da.T @ (w[:, None] * dv)
        ).T
        base = w @ v + jac @ (center - w @ a)
        position = w @ observed[idx, 5:8]
        for vy in (2.9, 3.5, 4.2, 4.9):
            flight = max(0.30, (0.65 - position[1]) / (vy * 0.9))
            target = np.array(
                [
                    -position[0] / flight,
                    vy,
                    (0.78 - position[2]) / flight + 5.0 * flight,
                ]
            )
            change = jac.T @ np.linalg.solve(
                jac @ jac.T + 0.15 * np.eye(3), target - base
            )
            for gain in (0.65, 1.15):
                value = (
                    center + gain * np.clip(change, -1.5, 1.5) + rng.normal(0, 0.04, 10)
                )
                proposals.append(np.clip(value, -2.7, 2.7) * sizes)
    return np.asarray(proposals[:count])


def landing_error(feedback):
    p, v = feedback[:, 5:8], feedback[:, 2:5]
    gy = 9.81
    airtime = np.maximum(
        0.02,
        (v[:, 2] + np.sqrt(np.maximum(0.0, v[:, 2] ** 2 + 2 * gy * (p[:, 2] - 0.78))))
        / gy,
    )
    travel = 0.94 * airtime
    xy = p[:, :2] + v[:, :2] * travel[:, None]
    net_t = -p[:, 1] / np.maximum(0.6, v[:, 1] * 0.94)
    net_h = p[:, 2] + v[:, 2] * net_t - gy * 0.5 * net_t**2
    return (
        np.maximum(0.0, np.abs(xy[:, 0]) - 0.48) * 2
        + np.abs(xy[:, 1] - 0.68) * 0.55
        + np.maximum(0.0, 0.98 - net_h) * 2
        + np.maximum(0.0, 2.0 - v[:, 1]) * 0.6
        + 0.7 * feedback[:, 8]
    )


def search(
    engine,
    fit,
    eta,
    choices,
    dt,
    home,
    rounds,
    rng,
    guided=False,
    experience=None,
    frame_log=None,
):
    beliefs = getattr(engine, "belief_search", None)
    if beliefs is not None:
        beliefs.start()
    evaluations = []
    history = []
    sizes = np.array([0.23, 0.33, 0.38, 0.36, 0.50, 2.2, 3.0, 2.8, 3.0, 3.6])
    count = len(choices)
    distance = []
    for round_index in range(max(1, rounds)):
        goals = schedules(*fit, eta, choices, dt, home, round((eta + 1.05) / dt))
        feedback = np.zeros((len(choices), 9))
        traces = (
            np.full((*goals.shape[:2], 17), np.nan) if frame_log is not None else None
        )
        _, outcomes = engine.trajectories(
            goals, AIM, eta + 0.12, feedback=feedback, traces=traces
        )
        if beliefs is not None:
            beliefs.observe(choices, goals, feedback)
        if frame_log is not None:
            frame_log.add(traces, outcomes, feedback, eta, dt, round_index)
        ok = np.flatnonzero(np.isfinite(outcomes[:, 0]))
        if experience is not None:
            value = np.where(np.isfinite(outcomes[:, 0]), outcomes[:, 0], -1.0)
            experience.append(np.c_[choices, feedback, value].astype(np.float32))
        # The labels below stay inside simulator practice. Only the winning
        # joint corrections and their pre-action sensory contexts are exported.
        b, v = feedback[:, 5:8], feedback[:, 2:5]
        # Nominal outgoing velocity to an interior landing, used on faults only.
        flight = np.maximum(0.32, (0.85 - b[:, 1]) / 4.0)
        wanted = np.c_[
            -b[:, 0] / flight,
            np.full(len(v), 4.4),
            (0.78 - b[:, 2]) / flight + 4.95 * flight,
        ]
        error = (
            np.linalg.norm((v - wanted) * [0.8, 0.8, 1.3], axis=1)
            + 2.0 * feedback[:, 8]
        )
        loss = np.where(
            feedback[:, 1] > 0,
            landing_error(feedback) if guided else error,
            15.0 + 12.0 * feedback[:, 0],
        )
        if guided:
            history.extend(zip(choices, feedback, strict=True))
        distance.append(
            {
                "trials": len(choices),
                "contacts": int(feedback[:, 1].sum()),
                "legal": len(ok),
                "closest": float(feedback[:, 0].min()),
                "best": float(loss.min()),
            }
        )
        if len(ok):
            if getattr(engine, "robust", False):
                selected, nominal, audit_row = engine.select(
                    choices[ok], outcomes[ok], fit, eta, dt, home, rng
                )
                distance.append(audit_row)
                return selected, nominal, distance
            value = getattr(engine, "motor_value", None)
            cost = outcomes[ok, 0] if value is None else value.cost(outcomes[ok])
            idx = ok[np.argmin(cost)]
            return choices[idx], outcomes[idx].tolist(), distance
        if rounds == 0:
            return choices[0], None, distance
        evaluations.extend(zip(loss, choices, strict=True))
        elite = np.asarray([a for _, a in sorted(evaluations, key=lambda x: x[0])[:10]])
        parent = elite[rng.integers(len(elite), size=count)]
        noise = (
            rng.normal(0.0, 1.0, (count, 10))
            * sizes
            * (0.58 if round_index < 2 else 0.35)
        )
        choices = np.clip(parent + noise, -2.2 * sizes, 2.2 * sizes)
        if guided:
            inverse = consequence_search(history, elite, rng, sizes, count // 2)
            choices[: len(inverse)] = inverse

    if beliefs is not None:
        answer = beliefs.select(engine, eta)
        if answer is not None:
            candidate, result, audit_row = answer
            distance.append(audit_row)
            if candidate is not None:
                return candidate, result, distance
    return min(evaluations, key=lambda x: x[0])[1], None, distance


def alternate_reach(surface, obs, shift):
    time = shift / obs[4]
    forecast = obs.copy()
    forecast[0] += time
    forecast[1] += obs[3] * time
    forecast[2] += obs[5] * time - 4.905 * time * time
    forecast[5] -= 9.81 * time
    return forecast, geometry(surface, forecast, depth=surface.depth + shift)


def point(
    a,
    stereo,
    rig,
    model,
    surf,
    stride,
    lo,
    hi,
    card,
    shot,
    bank,
    params,
    mode,
    cap,
    cut,
    rounds,
    guided,
    archive,
    response,
    refine,
    variable,
    frame_log,
    optical_engine=None,
    visual=None,
    refresh=False,
    pupil=None,
    repair=False,
    camera_stride=2,
    economy=False,
    servo=None,
    settled=False,
    rescue=None,
    joint_experience=None,
    fallback=None,
    impact=None,
    preparation=None,
):
    actor = joint.JointPolicy(model | {"ready": surf.ready.tolist()}, stride, lo, hi)
    eye = joint.Vision(rig, a.control_dt)
    if visual:
        from .tt37_visual_inertia import LearnedOptics

        eye.eye = LearnedOptics(eye.eye, visual)
    a.launch(shot, card=card)
    a.partner.reset(seed=card["seed"])
    engine = Rollout(a, 1) if mode == "coach" else None
    if optical_engine is not None:
        engine = optical_engine
        engine.previous = None
        engine.bounced = False
    outcome_pupil = pupil if hasattr(pupil, "record_command") else None
    if outcome_pupil is not None:
        outcome_pupil.begin_point()
    if preparation is not None:
        preparation.begin()
    if impact is not None and hasattr(impact, "begin"):
        impact.begin()
    if joint_experience is not None:
        joint_experience.begin_point()
    start_time = process_time()
    perception_cpu = arena_cpu = 0.0
    online_value = getattr(engine, "motor_value", None)
    if not hasattr(online_value, "feedback"):
        online_value = None
    if online_value is not None:
        online_value.begin_point()
    pending = None
    sensed_inbound = False
    records = []
    actions = []
    tape = None
    deadline = -100
    tape_start = retry_tick = -1
    fallback_until = -1
    fallback_ticks = 0
    lastprep = -100
    prep = surf.ready
    history = ArrivalHistory()
    if servo is not None:
        servo.begin()
    last_servo = -100
    if response is not None:
        response.last = None
    for tick in range(a.max_steps):
        q = a.joints.copy()
        started = process_time()
        obs = eye.update(
            tick, stereo.capture() if tick % camera_stride == 0 else None, q
        )
        perception_cpu += process_time() - started
        if optical_engine is not None:
            optical_engine.observe(tick, eye.eye, q, actor.reference.copy())
        now = tick * a.control_dt
        if refine:
            obs = history.update(now, obs)
        if (
            servo is not None
            and tape is not None
            and obs is not None
            and tick % camera_stride == 0
            and tick - last_servo >= 3
            and tick - tape_start >= 2
            and 0.022 < obs[0] < 0.158
            and optical_engine.sync()
        ):
            adjusted = servo.inspect(
                optical_engine, tape[tick - tape_start :], obs[0], a.control_dt, tick, q
            )
            last_servo = tick
            if adjusted is not None:
                if impact is not None and hasattr(impact, "cancel"):
                    impact.cancel()
                if online_value is not None:
                    online_value.issued(None)
                tape = adjusted
                tape_start = tick
                pending = None
                retry_tick = -1
        if (
            refine
            and engine is not None
            and tape is not None
            and retry_tick == tick
            and obs is not None
            and 0.035 < obs[0] < 0.22
            and 0.76 < obs[2] < 1.38
        ):
            keep = False
            if repair and (optical_engine is None or optical_engine.sync()):
                from .tt37_motion_repair import revise

                residual, note = revise(
                    engine, tape[tick - tape_start :], obs[0], a.control_dt, tick
                )
                if note is not None:
                    records[-1]["late_repair"] = note
                if residual is not None:
                    tape = residual
                    tape_start = tick
                    keep = True
                    if note["status"] == "corrected":
                        pending = None
                        if online_value is not None:
                            online_value.issued(None)
            elif refresh and optical_engine is not None and optical_engine.sync():
                # Check the previously issued motion against the newer images first.
                remainder = tape[tick - tape_start :]
                extra = round(1.08 / a.control_dt)
                follow = np.concatenate([remainder, np.tile(remainder[-1], (extra, 1))])
                _, repeat = engine.trajectories(follow[None], AIM, obs[0] + 0.12)
                keep = np.isfinite(repeat[0, 0])
            if not keep:
                tape = None
                deadline = -100
        if tape is not None and tick - tape_start < len(tape):
            desired = tape[tick - tape_start]
        else:
            tape = None
            desired = surf.ready
            if obs is not None and 0.03 < obs[0] < 0.80 and now > deadline:
                ready_to_commit = (
                    not settled or obs[0] < 0.12 or eye.eye.filter.settled(now)
                )
                if (
                    obs[0] <= cut
                    and ready_to_commit
                    and (not guided or 0.76 < obs[2] < 1.38)
                ):
                    eta = obs[0]
                    fit = geometry(surf, obs)
                    x = np.r_[obs[1:6], q[:5]]
                    if hasattr(online_value, "context"):
                        online_value.context(x)
                    if response is not None and response.last is not None:
                        response.observe(False)
                    remembered = (
                        recall(bank, x) if response is None else response(bank, x)
                    )
                    choices = (
                        np.array([remembered])
                        if engine is None
                        else np.vstack((remembered, params))
                    )
                    shift = 0.0
                    selected = remembered
                    rescued = None
                    impact_review = None
                    trace = []
                    score = None
                    if engine is not None and (
                        optical_engine is None or optical_engine.sync()
                    ):
                        experience = []
                        if frame_log is not None:
                            frame_log.group = (
                                card["id"],
                                card["seed"],
                                card["reply_seed"],
                                tick,
                            )
                        cheap = []
                        if economy:
                            from .tt37_reuse import propose

                            if hasattr(engine, "skill_memory"):
                                propose = engine.skill_memory.propose
                            selected, score, check = propose(
                                engine, bank, x, fit, eta, a.control_dt, surf.ready
                            )
                            cheap.append(check)
                        if score is None:
                            selected, score, trace = search(
                                engine,
                                fit,
                                eta,
                                choices,
                                a.control_dt,
                                surf.ready,
                                rounds,
                                np.random.default_rng(
                                    [37173, 0 if economy else len(bank), tick]
                                ),
                                guided,
                                experience,
                                frame_log,
                            )
                        trace = cheap + trace
                        if archive is not None and experience:
                            table = np.concatenate(experience)
                            archive.append(
                                np.c_[
                                    np.repeat(x[None], len(table), axis=0), table
                                ].astype(np.float32)
                            )
                        if variable and score is None:
                            for other_shift in (0.17, -0.105, 0.08, 0.25):
                                alternative, other_fit = alternate_reach(
                                    surf, obs, other_shift
                                )
                                if not (
                                    0.035 < alternative[0] < 0.45
                                    and 0.78 < alternative[2] < 1.38
                                ):
                                    continue
                                proposal, other_score, notes = search(
                                    engine,
                                    other_fit,
                                    alternative[0],
                                    choices,
                                    a.control_dt,
                                    surf.ready,
                                    min(rounds, 3),
                                    np.random.default_rng(
                                        [37619, tick, round((other_shift + 1) * 1000)]
                                    ),
                                    guided,
                                )
                                trace.extend(notes)
                                if other_score is not None:
                                    shift = other_shift
                                    selected, score, fit, eta = (
                                        proposal,
                                        other_score,
                                        other_fit,
                                        alternative[0],
                                    )
                                    break
                        if rescue is not None and score is None:
                            rescued = rescue.plan(optical_engine, surf, obs, selected)
                            trace.extend(rescued["trace"])
                            if rescued["tape"] is not None:
                                score = rescued["score"]
                                eta = rescued["eta"]
                        if (
                            impact is not None
                            and optical_engine is not None
                            and (rescued is None or rescued["tape"] is None)
                        ):
                            impact_review = impact.plan(
                                optical_engine, surf, obs, selected, score, fit, eta
                            )
                            trace.extend(impact_review["trace"])
                            if impact_review["tape"] is not None:
                                score = impact_review["score"]
                                eta = impact_review["eta"]
                        if optical_engine is not None:
                            pending = (
                                {"x": x.tolist(), "change": selected.tolist()}
                                if shift == 0
                                and (rescued is None or rescued["tape"] is None)
                                and (
                                    impact_review is None
                                    or impact_review["tape"] is None
                                )
                                else None
                            )
                        elif score is not None and shift == 0:
                            bank.append({"x": x.tolist(), "change": selected.tolist()})
                    if online_value is not None:
                        online_value.issued(score)
                    if impact_review is not None and impact_review["tape"] is not None:
                        tape = impact_review["tape"]
                    elif rescued is not None and rescued["tape"] is not None:
                        tape = rescued["tape"]
                    else:
                        tape = schedules(
                            *fit,
                            eta,
                            np.array([selected]),
                            a.control_dt,
                            surf.ready,
                            max(1, round((eta + 0.075) / a.control_dt)),
                        )[0]
                    desired = tape[0]
                    tape_start = tick
                    fallback_until = (
                        tick + len(tape)
                        if score is None and fallback is not None
                        else -1
                    )
                    deadline = now + eta + 0.26
                    retry_tick = (
                        tick + max(3, round((eta - 0.092) / a.control_dt))
                        if (score is None or refresh) and eta > 0.13 and not (impact_review and impact_review.get("commit"))
                        else -1
                    )
                    records.append(
                        {
                            "tick": tick,
                            "surface_shift": shift,
                            "camera_rescue": None
                            if rescued is None
                            else rescued["note"],
                            "context": x.tolist(),
                            "choices": sum(t["trials"] for t in trace),
                            "good": sum(t["legal"] for t in trace),
                            "score": score,
                            "search": trace,
                            "action": selected.tolist(),
                        }
                    )
                    if impact_review is not None:
                        records[-1]["impact_margin"] = impact_review["note"]
                elif not guided or 0.76 < obs[2] < 1.38:
                    if tick - lastprep >= 4:
                        p, v = geometry(surf, obs)
                        prep = np.r_[p - 0.075 * v, GRIP]
                        lastprep = tick
                    desired = prep
                    if (
                        impact is not None
                        and hasattr(impact, "prepare")
                        and optical_engine is not None
                    ):
                        desired = impact.prepare(
                            optical_engine, surf, obs, bank, desired
                        )
        if preparation is not None:
            from .tt38_scored_joints import optical_state

            prep_mix = preparation.observe(optical_state(eye.eye, now), q)
            chart_goal = desired
        if pupil is not None and tape is None:
            # Visit the posture the pupil would actually reach. The physics coach
            # intervenes only inside the receiving window during training.
            desired = surf.ready
            f = eye.eye.latest
            if (
                f is not None
                and now - f.time < 0.105
                and f.rms < 2.4
                and f.depth_sigma < 0.16
            ):
                optical = np.r_[*f.advance(now)]
                sensed_inbound |= optical[4] < -0.9 and optical[1] < -0.9
                if optical[4] < -0.9 or (
                    sensed_inbound and optical[4] > 0.9 and optical[1] < 0.5
                ):
                    proposed, weight = pupil.get(optical, q)
                    if proposed is not None:
                        desired = surf.ready + weight * (proposed - surf.ready)
        if preparation is not None and tape is None:
            desired = chart_goal + prep_mix * (desired - chart_goal)
        if fallback is not None and tick < fallback_until:
            from .tt38_scored_joints import optical_state

            view = optical_state(eye.eye, now)
            if view is not None and abs(view[4]) > 0.9:
                proposal, confidence = fallback.get(view, q)
                if proposal is not None and confidence > 0.35:
                    desired = surf.ready + confidence * (proposal - surf.ready)
                    fallback_ticks += 1
                    pending = None
                    if online_value is not None:
                        online_value.issued(None)
        action = actor.drive(desired, q, 0.8)
        if outcome_pupil is not None:
            from .tt38_scored_joints import optical_state

            outcome_pupil.record_command(
                tick, optical_state(eye.eye, now), q, actor.reference
            )
        if joint_experience is not None:
            from .tt38_scored_joints import optical_state

            joint_experience.step(tick, optical_state(eye.eye, now), q, actor.reference)
        started = process_time()
        _, _, done, capped, info = a.exchange(action, FAR)
        arena_cpu += process_time() - started
        if impact is not None and hasattr(impact, "feedback"):
            hit_reward = any(e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"])
            impact.feedback(hit_reward, done)
        if preparation is not None:
            near = any(
                e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]
            )
            preparation.feedback(near, done)
        if joint_experience is not None and any(
            e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]
        ):
            joint_experience.credit()
        if outcome_pupil is not None and any(
            e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]
        ):
            outcome_pupil.score_feedback(max(0, a.legal_returns - 1))
        if online_value is not None:
            online_value.feedback(
                [e["side"] for e in info["events"] if e["kind"] == "legal_return"], done
            )
        if servo is not None and any(
            e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]
        ):
            servo.repay()
        if pending is not None and any(
            e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]
        ):
            if mode == "optical-learn":
                bank.append(pending)
            pending = None
        if response is not None and response.last is not None:
            success = any(
                e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]
            )
            if success or done or capped:
                response.observe(success)
        actions.append(action.tolist())
        if done or capped or a.legal_returns + a.rival_returns - 1 >= cap:
            break
    if outcome_pupil is not None:
        outcome_pupil.end_point(terminal=bool(done and not capped))
    return {
        "cpu_s": process_time() - start_time,
        "arena_cpu_s": arena_cpu,
        "perception_cpu_s": perception_cpu,
        "simulation_s": float(a.data.time),
        "actions": actions,
        "servo": list(servo.notes) if servo is not None else [],
        "direct_fallback_ticks": fallback_ticks,
        "records": records,
        "events": a.events,
        "placements": a.point_placements(),
        "score": audit(a.events, opening_side=1),
        "shot": shot,
        "opponent_seed": card["seed"],
    }


def run(args):
    if args.learn_value and not (args.motor_value or args.empty_value):
        raise ValueError("Supply a starting motor-value model")
    plan = _read(args.plan)
    cards = plan[args.group][args.start : args.start + args.n]
    shots = measured_launches(args.feeds)
    model = _read(PUPIL)
    a = arena()
    ik = BladeIK(a)
    surface = Surface(ik, sample(ik), args.depth, args.surface)
    a.close()
    joint.RECEIVE_Y = args.depth
    bank = _read(args.memory)["bank"] if args.memory else []
    initial_bank = list(bank)
    out = {
        "cards": cards,
        "forget_between_points": args.forget_between_points,
        "ready": surface.ready.tolist(),
        "depth": args.depth,
        "cut": args.cut,
        "mode": args.mode,
        "rows": [],
        "bank": bank,
    }
    out.update(
        economy=args.economy,
        memory_start=len(bank),
        camera_stride=args.camera_stride,
        repair=args.repair,
        robust=args.robust,
        body=args.body,
        visual=str(args.visual) if args.visual else None,
        refresh=args.refresh,
        variable=args.variable,
        cap=args.cap,
        depth=args.depth,
        refine=args.refine,
        rounds=args.rounds,
        samples=args.samples,
        guided=args.guided,
        group=args.group,
    )
    response = None
    if args.response:
        from .tt37_response import Response

        response = Response(args.response, adaptive=args.online)
        out["pretraining"] = [str(p) for p in args.response]
    if args.forest:
        from .tt37_value import Forest

        response = Forest(args.forest, surface, adaptive=args.online)
        out["forest"] = str(args.forest)
    from .tt37_native_frames import NativeFrames

    frame_log = NativeFrames(args.frames) if args.frames else None
    archive = [] if args.archive else None
    pupil = None
    if args.rollin:
        from .tt37_joint_deep import DeepField

        with np.load(args.rollin) as z:
            pupil = DeepField(z, "local")
        out["pupil_rollin"] = str(args.rollin)
    if args.prep_outcomes:
        from .tt39_action_value import ActionValue, load_data

        if pupil is None:
            raise ValueError("A base direct motor field is required")
        dataset, used = load_data(args.prep_outcomes, future_mix=args.prep_lead)
        if used & {c["id"] for c in cards}:
            raise ValueError("Preparation outcomes came from this assessment course")
        pupil = ActionValue(pupil, dataset, future_mix=args.prep_lead)
        out["preparation_outcome_sources"] = [str(p) for p in args.prep_outcomes]
        out["preparation_lead_fraction"] = args.prep_lead
    fallback = None
    if args.fallback_bank:
        from .tt37_joint_deep import DeepField

        with np.load(args.fallback_bank) as z:
            fallback = DeepField(z, "local")
        out["direct_fallback"] = str(args.fallback_bank)
    vision_source = args.visual
    if args.blank_vision:
        if args.visual or args.learn_vision:
            raise ValueError("Frozen blank vision must run without a visual checkpoint")
        from .tt37_online_vision import BlankInertia

        vision_source = BlankInertia()
        out["blank_vision"] = True
    if args.learn_vision:
        from .tt37_online_vision import StreamingInertia

        vision_source = StreamingInertia(args.visual, period=args.camera_stride / 60)
        out["online_visual_checkpoint"] = str(args.learn_vision)
    servo = None
    if args.servo:
        from .tt38_joint_servo import JointServo

        if args.mode != "optical-learn":
            raise ValueError("Servo learning requires an optical learning run")
        servo = JointServo()
    if args.settled:
        if not args.visual:
            raise ValueError(
                "A calibrated rebound camera is required to time the command"
            )
        with np.load(args.visual) as z:
            if "rebound_states" not in z:
                raise ValueError("The camera must be trained on visible reversals")
        out["settled"] = True
    rescue = None
    if args.camera_rescue:
        from .tt38_camera_rescue import CameraRescue

        if not args.mode.startswith("optical-"):
            raise ValueError(
                "Private rescue requires a sensor-initialized nominal model"
            )
        rescue = CameraRescue(
            search_seed=args.rescue_seed,
            batch=args.rescue_batch,
            rounds=args.rescue_rounds,
        )
        out["camera_rescue"] = {
            "seed": args.rescue_seed,
            "batch": args.rescue_batch,
            "rounds": args.rescue_rounds,
        }
    preparation = None
    if args.prep_memory or args.prep_fixed is not None:
        from .tt39_preparation import Preparation

        if pupil is None:
            raise ValueError("Supply direct joint preparation with --rollin")
        history = (
            _read(args.prep_memory)
            if args.prep_memory
            else {"examples": [], "practice_ids": []}
        )
        if set(history["practice_ids"]) & {c["id"] for c in cards}:
            raise ValueError(
                "The preparations were learned on these assessment releases"
            )
        preparation = Preparation(history["examples"], fixed=args.prep_fixed)
        out["preparation_source"] = str(args.prep_memory) if args.prep_memory else None
        out["fixed_preparation"] = args.prep_fixed
    paired_model = None
    if getattr(args, "paired_practice", None):
        from .tt40_paired_practice import Paired
        paired_model = Paired(args.paired_practice)
        if (not args.mode.startswith("optical-") or args.variable or args.repair or args.camera_rescue
            or set(paired_model.ids) & {c["id"] for c in cards}):
            raise ValueError("Use an optical, nominal-depth policy on disjoint releases")
        out["paired_practice"] = str(args.paired_practice)
    joint_experience = None
    if args.record_joints:
        from .tt38_scored_joints import ScoredJoints

        joint_experience = ScoredJoints()
    changes = generate(args.samples)
    with centered(surface.ready):
        a = arena()
        from .tt37_optical_clinic import OpticalClinic

        optical_engine = (
            OpticalClinic(body=args.body, robust=args.robust)
            if args.mode.startswith("optical-")
            else None
        )
        if args.skill_memory:
            from .tt39_skilled_memory import SkilledMemory

            if optical_engine is None or not args.economy:
                raise ValueError(
                    "Scored skill proposals require the optical economy controller"
                )
            skill_card_ids = _read(args.skill_memory)["practice"]
            if set(skill_card_ids) & {r["id"] for r in cards}:
                raise ValueError("The skill was trained on this test release")
            optical_engine.skill_memory = SkilledMemory(args.skill_memory)
            out["skill_memory"] = str(args.skill_memory)
        if args.belief_search:
            from .tt39_belief_search import BeliefSearch

            if optical_engine is None:
                raise ValueError(
                    "Visual uncertainty requires the independent camera model"
                )
            optical_engine.belief_search = BeliefSearch(args.belief_search)
            out["belief_search"] = str(args.belief_search)
        if args.motor_value or args.empty_value:
            from .tt38_motor_value import MotorValue, OnlineMotorValue

            if optical_engine is None:
                raise ValueError(
                    "Physical value learning requires an independent sensor-driven model"
                )
            cls = OnlineMotorValue if args.learn_value else MotorValue
            start_value = (
                _read(args.motor_value)
                if args.motor_value
                else {"x": [np.zeros(11)], "y": [[0.5, 0.5]]}
            )
            if args.rally_value:
                from .tt38_rally_value import OnlineRallyValue, RallyValue

                cls = OnlineRallyValue if args.learn_value else RallyValue
                if args.empty_value:
                    start_value["y"] = [[0.5, 0.5, 0.5, 0.5]]
                if len(start_value["y"][0]) != 4:
                    raise ValueError("Rally value requires a four-outcome checkpoint")
                out["rally_value"] = True
            optical_engine.motor_value = cls(start_value)
            out["motor_value"] = str(args.motor_value)
        if getattr(args, "local_critic", None):
            from .tt40_local_critic import Conditional
            if not args.motor_value or args.rally_value or args.learn_value or optical_engine is None:
                raise ValueError("Local updates require the existing two-outcome camera prior")
            local_prior = _read(args.local_critic)
            if set(local_prior["ids"]) & {c["id"] for c in cards}:
                raise ValueError("Local critic has already practiced an assessment release")
            optical_engine.motor_value = Conditional(_read(args.motor_value), local_prior, update=args.local_updates)
            out["local_critic"] = str(args.local_critic)
        stereo, rig, stride, lo, hi = joint.resources(a)
        try:
            for c in cards:
                if args.forget_between_points:
                    bank[:] = initial_bank
                start_size = len(bank)
                r = point(
                    a,
                    stereo,
                    rig,
                    model,
                    surface,
                    stride,
                    lo,
                    hi,
                    c,
                    shots[c["id"]],
                    bank,
                    changes,
                    args.mode,
                    args.cap,
                    args.cut,
                    args.rounds,
                    args.guided,
                    archive,
                    response,
                    args.refine,
                    args.variable,
                    frame_log,
                    optical_engine,
                    vision_source,
                    args.refresh,
                    pupil,
                    args.repair,
                    args.camera_stride,
                    args.economy,
                    servo,
                    args.settled,
                    rescue,
                    joint_experience,
                    fallback,
                    preparation=preparation,
                    impact=paired_model,
                )
                if servo is not None:
                    out["late_motor_memory"] = [
                        {"state": y.tolist(), "delta": v.tolist()}
                        for y, v in servo.memory
                    ]
                r["memory_size"] = [start_size, len(bank)]
                if response is not None:
                    out["online_outcomes"] = [
                        {"context": x.tolist(), "action": u.tolist(), "reward": v}
                        for x, u, v in response.experience
                    ]
                if getattr(args, "local_critic", None):
                    out["local_critic_updates"] = optical_engine.motor_value.online
                if paired_model is not None:
                    out["paired_online"] = [[x.tolist(), j, v] for x, j, v in paired_model.history]
                if args.learn_vision:
                    r["optical_labels_so_far"] = vision_source.added
                    vision_source.save(args.learn_vision)
                if preparation is not None:
                    out["preparation_experience"] = preparation.new
                if args.learn_value:
                    out["value_experience"] = optical_engine.motor_value.examples
                    r["value_examples_so_far"] = len(out["value_experience"])
                out["rows"].append(r)
                _write(args.out, out)
                if joint_experience is not None:
                    joint_experience.complete(c)
                    joint_experience.save(
                        args.record_joints, surface.ready, args.depth, args.out
                    )
                if frame_log is not None:
                    frame_log.save()
                if archive:
                    tmp = args.archive.with_suffix(".partial.npz")
                    np.savez_compressed(tmp, rows=np.concatenate(archive))
                    tmp.replace(args.archive)
                print(
                    c["id"],
                    r["score"]["learner"],
                    [(x["good"], x["choices"]) for x in r["records"]],
                    flush=True,
                )
        finally:
            if optical_engine is not None:
                optical_engine.twin.close()
            stereo.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode", choices=("coach", "recall", "optical-learn", "optical-fixed")
    )
    p.add_argument("out", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument(
        "--plan", type=Path, default=Path("artifacts/tt37/opponent-plan.json")
    )
    p.add_argument("--group", default="practice")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--samples", type=int, default=80)
    p.add_argument("--memory", type=Path)
    p.add_argument("--motor-value", type=Path)
    p.add_argument("--learn-value", action="store_true")
    p.add_argument("--empty-value", action="store_true")
    p.add_argument("--belief-search", type=Path)
    p.add_argument("--skill-memory", type=Path)
    p.add_argument("--paired-practice", type=Path)
    p.add_argument("--local-critic", type=Path)
    p.add_argument("--local-updates", action="store_true")
    p.add_argument("--rally-value", action="store_true")
    p.add_argument("--cap", type=int, default=24)
    p.add_argument("--depth", type=float, default=-1.13)
    p.add_argument("--surface", type=float, default=0.35)
    p.add_argument("--cut", type=float, default=0.185)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--guided", action="store_true")
    p.add_argument("--archive", type=Path)
    p.add_argument("--response", type=Path, nargs="+")
    p.add_argument("--online", action="store_true")
    p.add_argument("--refine", action="store_true")
    p.add_argument("--forest", type=Path)
    p.add_argument("--variable", action="store_true")
    p.add_argument("--frames", type=Path)
    p.add_argument("--record-joints", type=Path)
    p.add_argument("--visual", type=Path)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--body", action="store_true")
    p.add_argument("--robust", action="store_true")
    p.add_argument("--rollin", type=Path)
    p.add_argument("--prep-outcomes", type=Path, nargs="+")
    p.add_argument("--prep-lead", type=float, default=0.0)
    p.add_argument("--prep-memory", type=Path)
    p.add_argument("--prep-fixed", type=float, choices=(0.0, 0.5, 1.0))
    p.add_argument("--fallback-bank", type=Path)
    p.add_argument("--blank-vision", action="store_true")
    p.add_argument("--learn-vision", type=Path)
    p.add_argument("--repair", action="store_true")
    p.add_argument("--camera-stride", type=int, choices=(1, 2), default=2)
    p.add_argument("--economy", action="store_true")
    p.add_argument("--servo", action="store_true")
    p.add_argument("--settled", action="store_true")
    p.add_argument("--camera-rescue", action="store_true")
    p.add_argument("--rescue-seed", type=int, default=3814647)
    p.add_argument("--rescue-batch", type=int, default=56)
    p.add_argument("--rescue-rounds", type=int, default=3)
    p.add_argument("--forget-between-points", action="store_true")
    run(p.parse_args())
