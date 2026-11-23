"""Apply retained visual material estimates to continuous full-size arm control."""

from pathlib import Path
import argparse

import numpy as np

from .tt4_transfer_trial import _write
from .tt24_rally_trial import PublicBody
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT, Eyes, Player, command, load_bank
from .tt27_opening_course import Physics
from .tt27_optical_dynamics import MaterialMemory, observed_bounce, predict


class FlightPlayer(Player):
    def __init__(self):
        super().__init__()
        traces = [r["trace"] for r in load_bank()["rows"]]
        vals = [observed_bounce(r) for r in traces]
        self.source = np.median([v for v in vals if v is not None], axis=0)
        self.trace = [np.asarray(t) for t in traces]
        self.end = np.array([t[-1, [1, 3]] for t in self.trace])
        self.y = np.array([t[-1, 2] for t in self.trace])


def point(arena, shot, player, material, *, gain=0.7, choose_geometry=False, output=False):
    arena.launch(shot)
    eye = Eyes(arena)
    body = PublicBody(arena)
    goal = None
    use = None
    track = []
    poses = []
    source_key = None
    updated = []
    for tick in range(arena.max_steps):
        k, pos = eye.observe(arena, body)
        if pos is not None and (not track or pos[1] < track[-1][2] + 0.025):
            track.append([tick, *pos])
        if tick == 3:
            source_key = k
            use = player.choose(k)
            goal = player.templates[use].goal
        if tick in (5, 8, 11, 14, 17, 20, 23, 26) and len(track) >= 4 and use is not None and gain:
            if choose_geometry and tick in (5, 8):
                forecasts = [predict(track, y, material) for y in player.y]
                predicted = np.array([p if p is not None else np.full(5, 20.0) for p in forecasts])
                normal_dist = np.linalg.norm((player.keys - source_key) / player.scale, axis=1)
                costs = np.linalg.norm(
                    (predicted[:, :2] - player.end) / [0.13, 0.09], axis=1
                ) + 0.06 * np.clip(normal_dist, 0, 12)
                use = int(np.argmin(costs))
            template = player.templates[use]
            dst = predict(track, template.final_y, material)
            # Correct model bias at a similar optical depth by subtracting
            # the same forecast from the original successful demonstration.
            ctr = player.trace[use]
            idx = int(np.argmin(abs(ctr[:, 2] - track[-1][2])))
            src = predict(ctr[: max(3, idx + 1)], template.final_y, player.source)
            if src is not None and dst is not None:
                dx, dz = np.clip(dst[:2] - src[:2], -0.16, 0.16) * gain
                delta = template.matrix @ np.r_[dx, 0.0, dz, 0.0, 0.0, 0.0]
                goal = template.goal + np.clip(delta, -0.45, 0.45)
                updated.append([tick, use, dx, dz])
        u = command(goal, body)
        poses.append(arena.joints.copy())
        _, _, done, cap, _ = arena.step(u, render=False)
        body.issued(u)
        eye.view.issued(u)
        if done or cap or arena.legal_returns:
            break
    vals = observed_bounce(track)
    q = np.asarray(poses)
    return dict(
        legal=arena.legal_returns,
        contact=arena.contacts,
        reason=arena.reason,
        frames=len(q),
        path=float(np.linalg.norm(np.diff(q[:, :5], axis=0), axis=1).sum()),
        choice=use,
        measured=vals.tolist() if vals is not None else None,
        trajectory=track if output else [],
        updates=updated if output else [],
    )


def run(path):
    lookup, _ = load_match(path)
    player = FlightPlayer()
    ids = [r["id"] for r in load_bank()["rows"]]
    arena = ReferenceReceiver()
    phy = Physics(arena)
    records = []
    try:
        for name, surface, grav in (
            ("base", 1.0, 1.0),
            ("soft", 2.5, 1.0),
            ("firm", 0.65, 1.0),
            ("heavy", 1.3, 1.14),
        ):
            phy.set(surface, grav)
            prior = MaterialMemory(player.source)
            rng = np.random.default_rng(86192)
            jobs = []
            for t in range(36):
                shot = draw(lookup, ids, rng, t)
                pair = {}
                for label, gain, geom, mat in [
                    ("old", 0.0, False, player.source),
                    ("fixed", 0.65, False, player.source),
                    ("adaptive", 0.65, False, prior.current),
                    ("geometric", 0.75, True, prior.current),
                ]:
                    row = point(arena, shot, player, mat, gain=gain, choose_geometry=geom)
                    pair[label] = row
                # The geometric diagnostic shares this learner's previous material estimate.
                # Only the adaptive branch owns and updates the recurring state.
                if pair["adaptive"]["measured"] is not None:
                    prior.history.append(np.array(pair["adaptive"]["measured"]))
                jobs.append(dict(t=t, donor=shot["id"], material=prior.current.tolist(), results=pair))
            records.append(dict(world=name, rows=jobs))
            _write(OUT / "physics-course.json.gz", dict(source=player.source.tolist(), records=records))
            print(
                name,
                player.source,
                prior.current,
                {k: sum(r["results"][k]["legal"] for r in jobs) for k in pair},
                flush=True,
            )
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
