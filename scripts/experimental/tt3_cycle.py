"""Carry one numerical visual preference through table play and back to paired Pong."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import torch

from tt3_arena import TT3Arena, heldout_cases
from tt3_learning import play_point
from tt3_numeric_transfer import PortableMemory, start_memory
from tt3_pong import cases_from_seeds, evaluate_pong, assess_retention, state_sha256
from tt3_pong_probe import ShorterDwell
from tt3_recovery_learning import TimingPlayer, DelayMemory
from tt3_visual import RoleModel


def rank_of(matrix):
    strengths = np.linalg.svd(matrix, compute_uv=False)
    if not len(strengths) or strengths[0] < 1e-6:
        return None
    return min(2, int(np.sum(strengths >= 0.22 * strengths[0])))


class ConditionedNumeric:
    """Each observed motor span keeps outcome evidence; updates also reach the common table."""

    def __init__(self, initial):
        self.shared = PortableMemory.restore(initial)
        self.profiles = {1: PortableMemory.restore(initial)}
        self.active = None
        self.exposures = {}

    def identify(self, matrix):
        rank = rank_of(matrix)
        if rank is None:
            return
        self.exposures[rank] = self.exposures.get(rank, 0) + 1
        if self.active is None or rank > self.active:
            self.active = rank
            if rank not in self.profiles:
                self.profiles[rank] = PortableMemory.restore(self.shared.state())

    def choose(self, rng, explore=0.0):
        return self.profiles.get(self.active, self.shared).choose(rng, explore)

    def choose_for_response(self, matrix, rng):
        return self.profiles.get(rank_of(matrix), self.shared).choose(rng)

    def observe(self, option, reward):
        self.shared.observe(option, reward)
        if self.active is not None:
            self.profiles[self.active].observe(option, reward)

    def goal(self, *args, **kwargs):
        return self.shared.goal(*args, **kwargs)

    def state(self):
        return dict(
            shared=self.shared.state(),
            profiles={k: v.state() for k, v in self.profiles.items()},
            active=self.active,
            exposures=self.exposures.copy(),
        )

    @classmethod
    def restore(cls, state):
        m = cls(state["shared"])
        m.profiles = {
            int(k): PortableMemory.restore(s) for k, s in state["profiles"].items()
        }
        m.active = state["active"]
        m.exposures = {int(k): n for k, n in state["exposures"].items()}
        return m


class CyclePlayer(TimingPlayer):
    def _update_motor(self, controlled):
        super()._update_motor(controlled)
        if self.frame >= self.probe_frames - 1 and self.motor.samples >= 12:
            self.memory.identify(self.motor.matrix(np.asarray(controlled.position)))


def course(job):
    index, seed, count, roles, numeric, conditions, all_times, clock = job
    out = {}
    local = seed + index
    schedule = heldout_cases(local, count)
    for condition in conditions:
        cpu_start = process_time()
        full = condition not in ("scratch", "cold_clock")
        role = RoleModel(**roles) if full else RoleModel.untrained()
        model = start_memory(
            "scratch" if condition == "cold_clock" else condition,
            numeric,
            np.random.default_rng(local),
        )
        memory = ConditionedNumeric(model.state())
        timing = DelayMemory()
        player = CyclePlayer(
            role,
            memory,
            seed=local,
            forecast=True,
            recall=True,
            learn_roles=True,
            timing_memory=timing,
            timing_seed=local + 72401,
            require_rebound=not all_times,
        )
        if clock is not None:
            from scripts.experimental.tt3_role_discovery import DiscoveryBinder

            # In the fresh control the source clock is supplied to all arms. This
            # makes the attribution mechanism identical while resetting both sets
            # of learned role/geometry values. It is a conservative comparison.
            if condition == "cold_clock":
                from tt3_cold_clock import OnlineDiscoveryBinder

                player.binder.binder = OnlineDiscoveryBinder(role, local + 823)
            else:
                player.binder.binder = DiscoveryBinder(role, clock, learn=True)
        pts = []
        best_tape = None
        for j, c in enumerate(schedule):
            arena = TT3Arena(c.seed, view_seed=schedule[0].view_seed)
            commands = []

            def capture(frames, act, reward):
                commands.append(act.tolist())

            r = play_point(
                arena, player, capture=capture if condition == "trained" else None
            )
            pts.append(
                {
                    k: r[k]
                    for k in (
                        "legal_returns",
                        "rival_returns",
                        "points_for",
                        "contacts",
                        "frames",
                        "probes",
                        "action_sha256",
                        "public_reward",
                        "delay_choice",
                        "delay_exposed",
                        "visual_continuations",
                    )
                }
            )
            if commands and r["legal_returns"] >= 2:
                priority = (bool(r["points_for"]), r["legal_returns"])
                if best_tape is None or priority > tuple(best_tape["priority"]):
                    best_tape = dict(
                        seed=c.seed,
                        view_seed=schedule[0].view_seed,
                        expected=arena.point_result(),
                        point=j + 1,
                        match=index,
                        priority=priority,
                        actions=commands,
                    )
        out[condition] = dict(
            points=pts,
            memory=memory.state(),
            roles=asdict(player.binder.model),
            timing=timing.state(),
            replay=best_tape,
            cpu_seconds=process_time() - cpu_start,
        )
    return out


def metrics(rows):
    return dict(
        points=len(rows),
        landed=sum(r["legal_returns"] > 0 for r in rows),
        later_returns=sum(max(0, r["legal_returns"] - 1) for r in rows),
        opponent=sum(r["rival_returns"] for r in rows),
        wins=sum(r["points_for"] for r in rows),
        sustained=sum(r["legal_returns"] >= 2 for r in rows),
        frames=sum(r["frames"] for r in rows),
        probes=sum(r["probes"] for r in rows),
    )


def write(path, data):
    with gzip.open(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roles", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=62057193)
    p.add_argument("--matches", type=int, default=6)
    p.add_argument("--points", type=int, default=20)
    p.add_argument(
        "--conditions", nargs="+", default=["trained", "fresh_goal", "scratch"]
    )
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--reward-clock",
        type=Path,
        help="fitted event-time classifier shared by the three receiving arms",
    )
    p.add_argument(
        "--all-times",
        action="store_true",
        help="offer every reachable visual meeting time, including before reversals",
    )
    p.add_argument("--native", action="store_true")
    p.add_argument("--native-seed", type=int, default=77019031)
    p.add_argument("--existing", action="store_true")
    args = p.parse_args()
    source_roles = json.loads(args.roles.read_text())
    roles = source_roles.get("role_model", source_roles)
    source = json.loads(args.source.read_text())
    clock = json.loads(args.reward_clock.read_text()) if args.reward_clock else None
    if args.existing:
        with gzip.open(args.out, "rt") as f:
            result = json.load(f)
        if result["source"] != source or result["roles"] != roles:
            raise ValueError("source changed")
    else:
        jobs = [
            (
                i,
                args.seed,
                args.points,
                roles,
                source,
                args.conditions,
                args.all_times,
                clock,
            )
            for i in range(args.matches)
        ]
        started = perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            matches = list(pool.map(course, jobs))
        seconds = perf_counter() - started
        stats = {
            label: {
                key: metrics([r for m in matches for r in m[label]["points"][lo:hi]])
                for key, lo, hi in [
                    ("first_four", 0, 4),
                    ("first_eight", 0, 8),
                    ("after_eight", 8, None),
                    ("all", 0, None),
                ]
            }
            for label in args.conditions
        }
        result = dict(
            seed=args.seed,
            all_times=args.all_times,
            reward_clock=clock,
            wall_seconds=seconds,
            source=source,
            roles=roles,
            stats=stats,
            matches=matches,
        )
        write(args.out, result)
        print(json.dumps(stats), flush=True)
    if args.native:
        torch.set_num_threads(1)
        rng = np.random.default_rng(args.native_seed)
        cases = cases_from_seeds(
            rng.choice(100_000_000, 48, replace=False) + 1740_000_000
        )
        actual = result["matches"][0]["trained"]
        common = dict(max_ticks=1440, points_per_match=5, motor_factory=ShorterDwell)
        before = evaluate_pong(
            cases, RoleModel(**roles), ConditionedNumeric(source), **common
        )
        live = ConditionedNumeric.restore(actual["memory"])
        checksum = state_sha256(live.state())
        after = evaluate_pong(cases, RoleModel(**actual["roles"]), live, **common)
        if state_sha256(live.state()) != checksum:
            raise AssertionError("evaluation changed learned state")
        if sum(live.shared.counts) - sum(source["counts"]) != len(actual["points"]):
            raise AssertionError("unaccounted outcomes")
        report = assess_retention(before, after)
        result["native"] = dict(
            seed=args.native_seed,
            match=0,
            actual_model_sha256=checksum,
            before_points=before,
            after_points=after,
            comparison=report,
        )
        write(args.out, result)
        print(
            "native",
            report["before"],
            report["after"],
            report["established"],
            flush=True,
        )


if __name__ == "__main__":
    main()
