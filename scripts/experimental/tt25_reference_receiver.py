"""Play measured DeepMind rally launches at native scale with the compact-court pupil."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np

from .tt4_arena import AIR_DRAG, BALL_MASS, MAGNUS_COEFFICIENT, TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _write
from .tt18_plant import PlantSettings
from .tt19_view_full_game import digest_json
from .tt20_fixture import ARCHIVED_FIXTURE, TABLE_EDGE, default_fixture
from .tt23_sequence import CONFIG
from .tt24_rally_bank import CACHE, DIMS
from .tt24_rally_improve_trial import policy_for
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import PublicBody
from .tt25_arm_replay import SolidArm
from .tt25_recorded_flight import OUTPUT


class ReferenceReceiver(SolidArm):
    """Real post-strike launch, regular physical referee, far body at rest."""

    def __init__(self, *, stationary_opponent=True, fixture=None, view_mode="fixed",
                 width=DIMS[0], height=DIMS[1], **settings):
        report = json.loads((OUTPUT / "recorded-flight.json").read_text())
        drag, lift = report["held_scores"]["match_fit"]["coefficients"]
        self.delta_drag, self.delta_lift = drag - AIR_DRAG, lift - MAGNUS_COEFFICIENT
        fixture = default_fixture() if fixture is None else fixture
        optics = "board" if fixture == TABLE_EDGE else CONFIG["wrist_optics"]
        configuration = dict(CONFIG, plant=PlantSettings(), max_steps=110, wrist_optics=optics,
                             fixture=fixture, view_mode=view_mode, width=width, height=height)
        super().__init__(**(configuration | settings))
        stationary = np.zeros(len(self._arm_ids[-1][2]))
        if stationary_opponent:
            self._aim = self._command_to = lambda *args: stationary

    def _aerodynamics(self):
        super()._aerodynamics()
        v, w = self.velocity, self.spin
        self.data.xfrc_applied[self._ball_body, :3] += BALL_MASS * (
            -self.delta_drag * np.linalg.norm(v) * v + self.delta_lift * np.cross(w, v)
        )

    def launch(self, shot):
        self.reset(seed=shot["id"], server=-1, render=False)
        self._serving = False
        self.ball[:] = [shot["pos_x"], shot["pos_y"], TABLE_HEIGHT + shot["pos_z"]]
        self.velocity[:] = [shot["vel_" + a] for a in "xyz"]
        self.data.qpos[self._ball_joint + 3 : self._ball_joint + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._ball_dof + 3 : self._ball_dof + 6] = [shot["w_vel_" + a] for a in "xyz"]
        mujoco.mj_forward(self.model, self.data)


def play(arena, shot, actor=None, tape=None):
    arena.launch(shot)
    body = PublicBody(arena)
    history = ViewHistory()
    actions = []
    first_image = None
    for tick in range(arena.max_steps):
        if tape is not None:
            command = tape[tick]
        else:
            if actor is not None or tick == 0:
                frames = arena.render_cameras()
                if tick == 0:
                    first_image = sha256(
                        b"".join(frames[s].tobytes() for s in ("left", "right"))
                    ).hexdigest()
            if actor is None:
                command = np.zeros(arena.action_size)
            else:
                observed = history.observe(frames, arena.joints, body.target)
                command = actor.command(observed, body.target)
        _, _, done, truncated, _ = arena.step(command, render=False)
        actions.append(np.asarray(command).tolist())
        body.issued(command)
        history.issued(command)
        if done or truncated or arena.legal_returns:
            break
    return dict(
        id=shot["id"],
        actions=actions,
        contact=arena.contacts,
        legal=arena.legal_returns,
        incoming_legal=arena.rival_returns,
        first_rgb=first_image,
        reason="legal_landing" if arena.legal_returns else arena.reason,
        events=digest_json(arena.events),
    )


def run(path):
    raw = path.read_bytes()
    source = json.loads((OUTPUT / "recorded-secondary-reference.json").read_text())
    if sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Unexpected DeepMind file")
    lookup = {r["id"]: r for r in json.loads(raw)}
    roster = json.loads((OUTPUT / "recorded-reach.json").read_text())["cases"][:32]
    arena = ReferenceReceiver(fixture=ARCHIVED_FIXTURE)
    with np.load(CACHE, allow_pickle=False) as bank:
        actor = policy_for("retina_rate", bank)
    cases = []
    try:
        for start in roster:
            shot = lookup[start["id"]]
            pair = {}
            for mode, pupil in (("trained", actor), ("stationary", None)):
                row = play(arena, shot, pupil)
                repeated = play(arena, shot, tape=row["actions"])
                if repeated["events"] != row["events"] or len(repeated["actions"]) != len(row["actions"]):
                    raise ValueError("Changed referee on replay")
                pair[mode] = row
            if pair["trained"]["first_rgb"] != pair["stationary"]["first_rgb"]:
                raise ValueError("Different starting image")
            cases.append(dict(id=shot["id"], near_sampled_blade=start["near_blade"], results=pair))
            print(
                len(cases),
                pair["trained"]["incoming_legal"],
                pair["trained"]["contact"],
                pair["trained"]["legal"],
                flush=True,
            )
    finally:
        arena.close()
    _write(
        OUTPUT / "recorded-moderate-receive.json.gz",
        dict(
            cases=cases,
            verified=2 * len(cases),
            source_sha256=source["sha256"],
            policy_bank_sha256=_hash(CACHE),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    run(p.parse_args().deepmind)
