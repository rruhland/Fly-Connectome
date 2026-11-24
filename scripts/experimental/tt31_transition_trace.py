"""Identify which observed decision changed the paired 12451 live game."""

import argparse
from bisect import bisect_right
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_curriculum import OUTPUT, play_card
from .tt31_ready import centered
from .tt31_transition import RETAINED, START


SHOT = 12451


def explain(folder, feeds):
    traces = {}
    originals = {}
    sources = {}
    for name, tape, checkpoint in (
        ("drills", "ready-periodic-staged.json.gz", START),
        ("carried", "transition-periodic.json.gz", RETAINED),
    ):
        source = folder / tape
        movement = folder / checkpoint
        game = _read(source)
        row = next(r for r in game["rows"] if r["shot"]["id"] == SHOT)
        originals[name] = row
        if _hash(movement) != row["initial_checkpoint_sha256"]:
            raise ValueError("Decision replay loaded the wrong checkpoint")
        decision = []

        class TracePlanner(tt29_action_memory.RecallPlanner):
            def plan(self):
                recalled = self.retrieved
                searched = self.fallback
                commands = super().plan()
                method = "no-stroke"
                if commands is not None:
                    method = "recall" if self.retrieved > recalled else "search"
                decision.append(dict(tick=self.arena.steps, method=method,
                                     counted_fallback=self.fallback > searched,
                                     command_ticks=len(commands) if commands is not None else 0))
                return commands

        with centered(game["ready"]), patch.object(tt29_action_memory, "RecallPlanner", TracePlanner), TemporaryDirectory() as work:
            repeated, _ = play_card(feeds, row["placements"]["card"], movement, movement,
                                    Path(work) / "point.json.gz", practice=False, cap=24)
        for field in ("actions", "events", "placements"):
            if digest_json(repeated[field]) != digest_json(row[field]):
                raise ValueError(f"Trace changed the original {name} {field}")
        far = [v["step"] for v in row["events"] if v["kind"] == "contact" and v["side"] == -1]
        for plan in decision:
            plan["incoming_ball"] = bisect_right(far, plan["tick"])
        traces[name] = dict(sources=decision, retrieved=row["retrieved"], planned=row["planned"])
        sources[name] = dict(tape_sha256=_hash(source), checkpoint_sha256=_hash(movement))
        print(name, decision, flush=True)

    before, after = originals["drills"], originals["carried"]
    base, carried = [r["placements"]["delivered"] for r in (before, after)]
    paired = []
    for a, b in zip(base, carried):
        paired.append({key: digest_json(a[key]) == digest_json(b[key])
                       for key in ("target", "far_contact", "near_bounce", "receive_line", "near_contact")})
    x, y = [np.asarray(r["actions"]) for r in (before, after)]
    latest = min(len(x), len(y))
    changed = np.flatnonzero(np.any(x[:latest] != y[:latest], axis=1))
    first_different = next((i, a, b) for i, (a, b) in enumerate(zip(before["events"], after["events"]))
                           if digest_json(a) != digest_json(b))
    _write(folder / "transition-12451-trace.json", dict(
        id=SHOT, sources=sources, decision_traces=traces, identical_placement_events=paired,
        first_different_event=first_different, first_changed_action_tick=int(changed[0]),
        first_changed_actions=[x[changed[0]].tolist(), y[changed[0]].tolist()],
        changed_command_ticks_among_shared_duration=len(changed),
        first_near_separations=[next(e for e in r["events"] if e["kind"] == "separation"
                                    and e.get("collider") == "learner_racket") for r in (before, after)],
        second_incoming=dict(drills=base[1], carried=carried[1]),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    explain(args.output, args.feeds)
