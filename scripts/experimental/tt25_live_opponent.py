"""Continue actual measured-launch receiving points with the physical far opponent."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import COACHED, PublicBody, STEP_SIZE
from .tt25_pose_demonstrations import OBSERVE
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def run(path, development=False):
    shots = {r["id"]: r for r in json.loads(path.read_bytes())}
    previous = _read(OUTPUT / "recorded-block-confirmation.json.gz")
    # Conditional continuation. Every previously returned point enters, even if
    # the moving opponent changes the pupil's initial choice.
    roster = [s["id"] for group in previous["cohorts"] for s in group if s["legal"][s["choice"]]]
    if development:
        roster = _read(OUTPUT / "recorded-nearby-lessons.json.gz")["donors"]
    with np.load(OUTPUT / "recorded-pose-imitation.npz", allow_pickle=False) as f:
        keys, goals, scale = [f[name] for name in ("image_keys", "joint_goals", "scale")]
    arena = ReferenceReceiver(stationary_opponent=False)
    replay = ReferenceReceiver(stationary_opponent=False)
    arena.max_steps = replay.max_steps = 450
    # Use the existing full physical opponent. Its internal planner is a
    # privileged teacher/opponent, never part of the receiving camera policy.
    profile = _read(COACHED)["metadata"]["opening"]["far_profile"]
    install_rally_opponent(arena, target_profile=profile)
    tape = FarTape(arena)
    cases = []
    try:
        for sid in roster:
            shot = shots[sid]
            arena.launch(shot)
            tape.steps.clear()
            body, vision = PublicBody(arena), ViewHistory()
            goal = None
            actions = []
            for tick in range(arena.max_steps):
                if tick <= OBSERVE:
                    state = vision.observe(arena.render_cameras(), arena.joints, body.target)
                    if tick == OBSERVE:
                        goal = goals[np.argmin(np.linalg.norm((keys - state[:14]) / scale, axis=1))]
                command = np.zeros(arena.action_size)
                if goal is not None:
                    command[:5] = np.clip((goal - body.target[:5]) / STEP_SIZE[:5], -1, 1)
                _, _, done, cap, _ = arena.step(command, render=False)
                body.issued(command)
                vision.issued(command)
                actions.append(command.tolist())
                if done or cap:
                    break
            far = tape.tape(len(actions))
            replay.launch(shot)
            for near, rival in zip(actions, far, strict=True):
                replay._aim = replay._command_to = lambda *args, far=rival: np.asarray(far)
                _, _, done, cap, _ = replay.step(near, render=False)
            if not (done or cap) or digest_json(arena.events) != digest_json(replay.events):
                raise ValueError("Full two-arm referee tape changed")
            # Exclude the original incoming hit from replies by the physical far arm.
            far_contacts = sum(e["kind"] == "contact" and e["side"] == -1 for e in arena.events)
            replies = max(0, arena.rival_returns - 1)
            cases.append(
                dict(
                    id=sid,
                    near=arena.legal_returns,
                    far_contacts=far_contacts,
                    replies=replies,
                    actions=actions,
                    rival_actions=far,
                    events=arena.events,
                    outcome=arena.point_result(),
                )
            )
            print(sid, arena.legal_returns, far_contacts, replies, flush=True)
        name = "recorded-live-development.json.gz" if development else "recorded-live-opponent.json.gz"
        _write(OUTPUT / name, dict(conditional_roster=roster, cases=cases))
    finally:
        arena.close()
        replay.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    p.add_argument("--development", action="store_true")
    args = p.parse_args()
    run(args.deepmind, args.development)
