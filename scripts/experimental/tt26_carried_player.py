"""Carry a rebound-earned tool adjustment and a reward-trained visual receiving memory."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_integrated_stress import InstalledArm, episode as release, schedule
from .tt25_pose_demonstrations import episode
from .tt26_free_joints import OPTIONS
from .tt26_grip_course import Grip, OUT, PRIOR, ReferenceReceiver, load_match
from .tt26_receiving_speed import draw
from .tt26_tool_recall import RecalledTool
from .tt26_visual_acquisition import VisualRewardMemory


class CarriedPlayer:
    def __init__(self, adjustment, goals):
        self.adjustment = np.asarray(adjustment, float).copy()
        self.goals = np.asarray(goals, float).copy()
        self.receiving = VisualRewardMemory(len(goals))

    def wrist_player(self):
        return RecalledTool(self.adjustment)

    def select(self, key, order, practice):
        action = self.receiving.choose(key, order, practice)
        return action, self.goals[action] + self.adjustment

    def save(self, path):
        np.savez_compressed(
            path,
            schema="tt26-carried-player-1",
            wrist_hash=_hash(BUNDLE),
            adjustment=self.adjustment,
            goals=self.goals,
            keys=np.asarray(self.receiving.keys),
            actions=self.receiving.selected,
            legal=self.receiving.legal,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            if saved["schema"].item() != "tt26-carried-player-1" or saved["wrist_hash"].item() != _hash(
                BUNDLE
            ):
                raise ValueError("Incompatible player checkpoint")
            player = cls(saved["adjustment"], saved["goals"])
            for key, action, legal in zip(saved["keys"], saved["actions"], saved["legal"], strict=True):
                player.receiving.update(key, int(action), int(legal))
        return player


def run(path):
    shots, (_, goals, _) = load_match(path)
    donors = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    starts = _read(OUT / "five-joint-practice.json.gz")["records"]
    arena = ReferenceReceiver()
    grip = Grip(arena.model)
    arm = InstalledArm("stock")
    arm.arena.max_steps = 480
    wrist_grip = Grip(arm.arena._model)
    launcher = BounceArena(view="none")
    plans = [p for p in schedule(62798431, 8) if p["body"] == "stock"]
    results = []
    try:
        for index, src in enumerate(starts[:2]):
            grip.set(src["mount"])
            wrist_grip.set(src["mount"])
            rng = np.random.default_rng(1952761)
            vocabulary = goals[rng.permutation(len(goals))]
            players = {
                "bounce": CarriedPlayer(OPTIONS[src["selected"]], vocabulary),
                "cold": CarriedPlayer(OPTIONS[0], vocabulary),
            }
            before = players["bounce"].wrist_player()
            scores = []
            for tick in range(144):
                shot = draw(shots, donors, rng, tick)
                _, key = episode(arena, shot, collect=True)
                order = rng.permutation(len(goals))
                point = {}
                for name, player in players.items():
                    selected, target = player.select(key, order, tick < 96)
                    perf = episode(arena, shot, target)[0]
                    point[name] = dict(option=selected, score=perf["legal"], events=perf["events"])
                    if tick < 96:
                        player.receiving.update(key, selected, perf["legal"])
                if tick == 95:
                    for name, player in list(players.items()):
                        destination = OUT / f"carried-player-{index}-{name}.npz"
                        player.save(destination)
                        players[name] = CarriedPlayer.load(destination)
                scores.append(point)
            after = players["bounce"].wrist_player()
            revisits = []
            for plan in plans:
                launcher.reset(seed=plan["launcher_seed"], split="held", render=False)
                launch = [
                    a.copy()
                    for a in (launcher._ball_position, launcher._ball_velocity, launcher._ball_spin)
                ]
                old = release(arm, before, plan, launch, "bundle")
                now = release(arm, after, plan, launch, "bundle")
                if now["actions"] != old["actions"] or now["metrics"] != old["metrics"]:
                    raise ValueError("Receiving practice or checkpoint reload changed wrist play")
                revisits.append(
                    dict(
                        seed=plan["learner_seed"],
                        steps=len(now["actions"]),
                        consecutive=now["metrics"]["longest_consecutive_rebounds"],
                    )
                )
            results.append(dict(mount=src["mount"], scores=scores, revisited=revisits))
            _write(OUT / "carried-player.json.gz", dict(results=results))
            print(
                src["mount"],
                [sum(s[k]["score"] for s in scores[96:]) for k in players],
                [r["consecutive"] for r in revisits],
                flush=True,
            )
    finally:
        arena.close()
        arm.arena.close()
        launcher.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
