"""Score complete regulation points from the first image with a reusable reply controller."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import COACHED, PublicBody
from .tt26_grip_course import Grip, PRIOR, ReferenceReceiver, load_match
from .tt26_match_continuation import archived_step
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT, SOURCE, command
from .tt27_reply import DEV_DONORS, Flight, PoseMap, Recall, StereoReply, examples
from .tt27_reply_recorded import transported


def one(arena, tape, shot, initial, bank, poses, flight, ready, mode, opener=None):
    arena.launch(shot)
    tape.steps.clear()
    body = PublicBody(arena)
    visual = ViewHistory()
    eye = StereoReply(arena.width, arena.height)
    pilot = Recall(eye, body, poses, flight, bank, "track")
    keys, goals, scales = initial
    posture = None
    triggers = []
    completed = 0
    actions = []
    measured = []
    for tick in range(arena.max_steps):
        frames = arena.render_cameras()
        eye.read(frames, tick)
        if tick <= 3:
            k = visual.observe(frames, arena.joints, body.target)[:14]
            if tick == 3:
                posture = (
                    opener(k) if opener else goals[np.argmin(np.linalg.norm((keys - k) / scales, axis=1))]
                )
        if eye.outgoing:
            # Visible travel away from the near end starts the next cycle.
            completed += 1
            eye = StereoReply(arena.width, arena.height)
            eye.reached_near = True
            pilot = Recall(eye, body, poses, flight, bank, "track")
        if eye.incoming is not None and (not triggers or triggers[-1] != eye.incoming[0]):
            triggers.append(eye.incoming[0])
        a = pilot.act(tick) if mode != "opening" else None
        if a is None:
            target = (
                ready if completed and mode == "recover" else (body.target[:5] if completed else posture)
            )
            a = command(target, body)
        measured.append(arena.joints.tolist())
        actions.append(a.tolist())
        _, _, done, cap, _ = arena.step(a, render=False)
        body.issued(a)
        if done or cap:
            break
    poses = np.array(measured)[:, :5]
    return dict(
        near=arena.legal_returns,
        far=max(0, arena.rival_returns - 1),
        contacts=arena.contacts,
        triggers=triggers,
        cycles=completed,
        events=arena.events.copy(),
        outcome=arena.point_result(),
        path=float(np.linalg.norm(np.diff(poses, axis=0), axis=1).sum()),
        frames=len(actions),
        actions=actions,
        opponent=tape.tape(len(actions)),
    )


def run(path, align_to_end=False, nearby=False):
    lookup, initial = load_match(path)
    arena = ReferenceReceiver(stationary_opponent=False)
    check = ReferenceReceiver()
    arena.max_steps = check.max_steps = 450
    install_rally_opponent(
        arena,
        target_profile=_read(COACHED)["metadata"]["opening"]["far_profile"],
        align_to_end=align_to_end,
    )
    tape = FarTape(arena)
    grip, poses, flight = Grip(arena.model), PoseMap(arena), Flight()
    traces = _read(OUT / "reply-traces.json.gz")["traces"]
    samples = [e for e in examples(traces) if e["id"] in DEV_DONORS]
    bank = transported(samples, grip, poses, 0.0)
    ready = np.median(
        [r["trace"][0]["joints"][:5] for r in traces if r["selected"] and r["id"] in DEV_DONORS], axis=0
    )
    cooked = _read(PRIOR / "recorded-live-opponent.json.gz")
    raw = _read(SOURCE / "fresh-recordings.json.gz")["records"][0]["rows"]
    rng = np.random.default_rng(430991)
    course = [("prior_first_return", k) for k in cooked["conditional_roster"]]
    others = [
        r["id"] for r in raw if r["id"] not in cooked["conditional_roster"] and r["id"] not in DEV_DONORS
    ]
    course.extend(("unselected", int(k)) for k in rng.choice(others, 32, replace=False))
    if nearby:
        varied = np.random.default_rng(435761)
        donors = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
        feeds = [("fresh_nearby", draw(lookup, donors, varied, i)) for i in range(12)]
    else:
        feeds = [(group, lookup[k]) for group, k in course]
    scope = ("-end" if align_to_end else "") + ("-nearby" if nearby else "")
    rows = []
    try:
        for cohort, shot in feeds:
            k = shot["id"]
            data = {}
            for mode in ("opening", "repeat", "recover"):
                # Until an opponent actually sends the ball back all three are identical.
                if mode == "repeat" and not data["opening"]["triggers"]:
                    data[mode] = data["opening"]
                    continue
                if mode == "recover" and not data["repeat"]["cycles"]:
                    data[mode] = data["repeat"]
                    continue
                r = one(arena, tape, shot, initial, bank, poses, flight, ready, mode)
                check.launch(shot)
                for tick, a in enumerate(r["actions"]):
                    _, _, done, cap, _ = archived_step(check, r["opponent"], tick, a)
                if not (done or cap) or digest_json(check.events) != digest_json(r["events"]):
                    raise ValueError("Complete game failed replay")
                data[mode] = r
            rows.append(dict(group=cohort, id=k, shot=shot, by=data))
            _write(
                OUT / ("full-points" + scope + ".json.gz"),
                dict(ready=ready.tolist(), aligned=align_to_end, rows=rows),
            )
            print(cohort, k, {m: (r["near"], r["far"], r["triggers"]) for m, r in data.items()}, flush=True)
    finally:
        arena.close()
        check.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("--align-to-end", action="store_true")
    p.add_argument("--nearby", action="store_true")
    args = p.parse_args()
    run(args.feeds, args.align_to_end, args.nearby)
