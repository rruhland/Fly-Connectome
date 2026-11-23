"""Play the compact benchmark live with moving-link/table contact enabled."""

import argparse
import gzip
import json

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt23_coach_trial import PRACTICE, ReplyRanking
from .tt23_sequence import CONFIG
from .tt24_rally_bank import CACHE, DIMS
from .tt24_rally_improve_trial import policy_for
from .tt24_rally_remap import driver
from .tt24_rally_trial import COACHED, Station
from .tt25_arm_replay import SolidArm
from .tt25_recorded_flight import OUTPUT, ROOT


class SolidStation(Station):
    def __init__(self, profile):
        self.arena = SolidArm(**dict(CONFIG, width=DIMS[0], height=DIMS[1]))
        self.rival = install_rally_opponent(self.arena, target_profile=profile)
        self.rival.confirm = ReplyRanking(self.rival, _read(PRACTICE)).confirm
        self.tape = FarTape(self.arena)


def replay(arena, row):
    arena.lane = row["lane"]
    arena.reset(seed=row["seed"], server=-1, render=False)
    for i, (near, far) in enumerate(zip(row["actions"], row["rival_actions"], strict=True)):
        rates = np.asarray(far)
        arena._aim = arena._command_to = lambda *args, rates=rates: rates
        _, reward, done, cap, _ = arena.step(near, render=False)
        if bool(done or cap) != (i+1 == len(row["actions"])):
            raise ValueError("Solid court duration changed on replay")
    if reward != row["reward"] or digest_json(arena.events) != digest_json(row["events"]):
        raise ValueError("Solid court referee changed on replay")


def run():
    profile = _read(COACHED)["metadata"]["opening"]["far_profile"]
    station = SolidStation(profile)
    check = SolidArm(**CONFIG)
    with np.load(CACHE, allow_pickle=False) as bank:
        actor = policy_for("retina_rate", bank)
    results = []
    try:
        for i in range(2):
            path = ROOT / f"artifacts/tt24/rolling-match-inspected-{i}.json.gz"
            previous = json.loads(gzip.decompress(path.read_bytes()))
            settings = previous["settings"]
            truth, inverse = np.array(settings["truth"]), np.linalg.inv(settings["fit"])
            for case in previous["cases"]:
                row = station.play(case, "calibrated", actor, live_opening=True,
                                   execute=lambda q: driver(truth, inverse, q))
                replay(check, row)
                results.append(row)
                print(len(results), row["outcome"]["legal_returns"], row["longest"], flush=True)
    finally:
        check.close()
        station.arena.close()
    _write(OUTPUT / "recorded-arm-live.json.gz", dict(policy_bank_sha256=_hash(CACHE),
           opponent=profile, cases=results, verified=len(results)))


def stock():
    source = ROOT / "artifacts/tt24/rally-improve-held.json.gz"
    previous = _read(source)
    station = SolidStation(previous["metadata"]["profile"])
    check = SolidArm(**CONFIG)
    with np.load(CACHE, allow_pickle=False) as bank:
        actor = policy_for("retina_rate", bank)
    cases = []
    try:
        for case in previous["cases"]:
            results = {}
            for name in ("retina_rate", "hold"):
                row = station.play(case, name, actor, live_opening=True)
                replay(check, row)
                results[name] = row
            if results["retina_rate"]["prefix_rgb"] != results["hold"]["prefix_rgb"]:
                raise ValueError("Paired policies observed different initial images")
            cases.append(dict(seed=case["seed"], results=results))
            print(len(cases), results["retina_rate"]["longest"], flush=True)
    finally:
        station.arena.close()
        check.close()
    _write(OUTPUT / "recorded-arm-stock.json.gz", dict(source_sha256=_hash(source),
           policy_bank_sha256=_hash(CACHE), cases=cases, verified=2*len(cases)))


def movie(stock_course=False):
    from .tt23_sequence_coach_movie import render_tape

    source = OUTPUT / ("recorded-arm-stock.json.gz" if stock_course else "recorded-arm-live.json.gz")
    results = _read(source)
    if len(results["cases"]) * (2 if stock_course else 1) != results["verified"]:
        raise ValueError("Incomplete solid-arm evaluation")
    rows = [c["results"]["retina_rate"] for c in results["cases"]] if stock_course else results["cases"]
    longest = max(rows, key=lambda r: r["longest"])
    render_tape(longest, source, OUTPUT / ("recorded-arm-stock.mp4" if stock_course else "recorded-arm-live.mp4"), branch="solid_arm",
                subtitle="End-clamped arm on the compact court",
                terminal_caption=f"Live camera policy, selected longest point from {len(rows)}",
                arena_factory=lambda: SolidArm(**dict(CONFIG, width=DIMS[0], height=DIMS[1])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--stock", action="store_true")
    args = parser.parse_args()
    if args.movie:
        movie(args.stock)
    elif args.stock:
        stock()
    else:
        run()
