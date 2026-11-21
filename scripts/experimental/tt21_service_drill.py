"""Learn a camera-conditioned service motion online with a short, reward-only drill."""

import argparse
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt20_camera_carry import SETTINGS, summarize, verify
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt21_online_strokes import FACTORY, OUTPUT, STANDARD_PICK

PRACTICE = 213817226
HELD = 637115986
# Relative excursions in public rate-command units, and control frames for onset.
NOISE = np.array([0.26, 0.22, 0.25, 0.32, 0.18, 0.28, 0.50, 0.55, 0.52, 0.26, 1.20, 0.58, 1.15])
OPTICAL_TICKS = (0, 2, 4)
END = 27


def tape(program, horizontal):
    base, swing = program[:5].copy(), program[5:10]
    base[0] += program[12] * np.clip(horizontal, -0.18, 0.18)
    when = int(np.ceil(program[10]))
    width = max(2, int(np.rint(program[11])))
    age = np.arange(END) - when
    mix = np.clip((age + 1) / width, 0.0, 1.0) * np.clip(1.0 - (age - width - 3) / width, 0.0, 1.0)
    goals = base + mix[:, None] * swing
    goals[:4] = 0.0
    movement = np.zeros(5)
    commands = np.zeros((END + 8, 6))
    for tick in range(len(commands)):
        desired = goals[tick] if tick < END else np.zeros(5)
        cmd = np.clip(desired - movement, -0.85, 0.85)
        commands[tick, :5] = cmd
        movement += cmd
    return commands


class ServiceDrill:
    def __init__(self, *, strong=False):
        self.arena = TT20Arena(**SETTINGS, view_mode="left")
        if strong:
            from .tt20_opponent import install_opponent

            install_opponent(self.arena)
        self.eyes = NaturalViewTransport(
            "left", lens_fov={"left": self.arena.lens_intrinsics()["left"]}, flight=False
        )

    def point(self, seed, program):
        a = self.arena
        a.reset(seed=seed, server=1, render=False)
        tracker = self.eyes.tracker()
        xs = []
        commands = []
        for t in range(a.max_steps):
            if t in OPTICAL_TICKS:
                tracker.update(a.render_cameras(), tick=t)
                if tracker.ball is not None and tracker.ball.observed:
                    xs.append(float(tracker.ball.position[0]))
            if t == 4:
                x = float(np.median(xs)) if xs else 0.0
                stroke = tape(np.asarray(program), x)
            cmd = stroke[t] if 4 <= t < END + 8 else np.zeros(6)
            commands.append(cmd.tolist())
            _, reward, done, cap, _ = a.step(cmd, render=False)
            if done or cap:
                break
        return {
            "seed": seed,
            "role": 1,
            "actions": commands,
            "context": x,
            "reward": reward,
            "outcome": a.point_result(),
            "rgb_sha256": tracker.digest.hexdigest(),
            "events": a.events.copy(),
        }


class Population:
    """Interleave exploration and independent-score qualification of an incumbent."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.base = np.array(_read(FACTORY)["programs"])
        self.best = self.base[STANDARD_PICK].copy()
        self.generation = 0
        self.history = []

    def epoch(self, bench):
        n = 10
        programs = [self.best] + [p.copy() for p in self.base]
        while len(programs) < n:
            scale = 0.9 if self.generation < 2 else 0.55
            origin = self.best if self.rng.random() < 0.8 else self.base[self.rng.integers(len(self.base))]
            p = origin + scale * self.rng.normal(size=13) * NOISE
            p[10:12] = np.clip(p[10:12], (5.0, 2.0), (22.0, 7.0))
            programs.append(p)
        reward = np.zeros(n)
        counts = np.zeros(n)
        rows = []
        # Each attempt has its own new toss. Only assigned candidates receive credit.
        for _ in range(2):
            for i in self.rng.permutation(n):
                r = bench.point(int(self.rng.integers(2**31)), programs[i])
                rows.append(dict(r, candidate=int(i)))
                reward[i] += r["reward"]
                counts[i] += 1
        score = reward / counts
        leaders = np.argsort(-score, kind="stable")[:3].tolist()
        if 0 not in leaders:
            leaders[-1] = 0
        # Independent additional practice from the same stream reduces promotion by luck.
        support = np.zeros(n)
        for _ in range(5):
            for i in self.rng.permutation(leaders):
                r = bench.point(int(self.rng.integers(2**31)), programs[i])
                rows.append(dict(r, candidate=int(i)))
                support[i] += r["reward"]
        winner = max(leaders, key=lambda i: support[i] + 0.4 * reward[i] + 0.05 * (i == 0))
        self.best = programs[winner].copy()
        self.history.append(
            {
                "programs": np.asarray(programs).tolist(),
                "winner": winner,
                "selected": self.best.tolist(),
                "exploratory_score": reward.tolist(),
                "confirmation_score": support.tolist(),
                "points": rows,
            }
        )
        self.generation += 1


def train(generations, strong=False):
    bench = ServiceDrill(strong=strong)
    learner = Population(PRACTICE)
    start = perf_counter()
    path = OUTPUT / f"service-drill-{'strong' if strong else 'standard'}-practice.json.gz"
    try:
        for _ in range(generations):
            learner.epoch(bench)
            info = learner.history[-1]
            rows = info["points"]
            _write(
                path,
                {
                    "factory_sha256": _hash(FACTORY),
                    "practice_seed": PRACTICE,
                    "strong": strong,
                    "arena": SETTINGS,
                    "eye": "left",
                    "generations": learner.history,
                    "selected": learner.best.tolist(),
                    "points_seen": sum(len(e["points"]) for e in learner.history),
                    "wall_s": perf_counter() - start,
                },
            )
            print("generation", learner.generation, info["winner"], summarize(rows), flush=True)
    finally:
        bench.arena.close()


def held(count, strong):
    source = OUTPUT / "service-drill-standard-practice.json.gz"
    learned = _read(source)
    settings = {"fixed": _read(FACTORY)["programs"][STANDARD_PICK], "learned": learned["selected"]}
    points = {}
    for condition, param in settings.items():
        bench = ServiceDrill(strong=strong)
        try:
            rows = [
                bench.point(s, param)
                for s in np.random.default_rng(HELD).integers(2**31, size=count).tolist()
            ]
        finally:
            bench.arena.close()
        verify({"arena": SETTINGS, "eye": "left", "strong": strong, "points": rows})
        points[condition] = rows
        print(condition, summarize(rows), flush=True)
    _write(
        OUTPUT / f"service-drill-{'strong' if strong else 'standard'}-held.json.gz",
        {
            "source": source.name,
            "source_sha256": _hash(source),
            "seed": HELD,
            "strong": strong,
            "arena": SETTINGS,
            "points": points,
            "summary": {k: summarize(rs) for k, rs in points.items()},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int)
    parser.add_argument("--held", type=int, default=32)
    parser.add_argument("--strong", action="store_true")
    a = parser.parse_args()
    if a.train:
        train(a.train, a.strong)
    else:
        held(a.held, a.strong)
