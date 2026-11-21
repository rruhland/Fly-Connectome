"""Calibrate an SO-101 strike simulator by comparing predicted and observed outgoing arcs."""

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_flight import FlightLaw, optical_start
from .tt19_model_rollout import observed_ball
from .tt19_shoot import HORIZON
from .tt22_rally import RallyPlayer, visual_return
from .tt22_rally_trial import Station


def designs():
    # Symmetric spin and velocity adjustments. The zero correction is an explicit candidate.
    return [(0.0, 0.0, 0.0, 0.0)] + [
        p
        for p in product((0.0, 0.65, 1.0), (-60.0, 0.0, 60.0), (-0.12, 0.0, 0.12), (-0.15, 0.0, 0.15))
        if p != (0.0, 0.0, 0.0, 0.0)
    ]


def adjusted(obs, params):
    shrink, rotation, lateral_speed, vertical_speed = params
    return replace(
        obs,
        spin=(1 - shrink) * obs.spin + np.array((rotation, 0.0, 0.0)),
        velocity=obs.velocity + np.array((0.0, lateral_speed, vertical_speed)),
    )


def discrepancy(expected, reading):
    if not expected["hit"] or expected["landing"] is None:
        return 1.0
    error = min(0.70, np.linalg.norm(np.subtract(expected["landing"], reading["landing"])))
    if reading["net"] is not None:
        error += 0.30 if expected["net"] is None else min(0.30, 1.5 * abs(expected["net"] - reading["net"]))
    return error


def measurements(row, player, options):
    points = dict(row["optical"])
    hist = [(i, points.get(i)) for i in range(len(row["actions"]))]
    acts = np.asarray(row["actions"])
    notes = {note["tick"]: note for note in row["shots"]}
    samples = []
    player.reset()
    for i, act in enumerate(acts):
        if i in notes:
            shot = notes[i]
            reading = visual_return(hist, shot)
            if reading is not None:
                estimate = optical_start(hist[: i + 1], player.law.gravity)
                obs = observed_ball(hist[: i + 1], estimate, player.law)
                offsets = []
                for opt in options:
                    changed = adjusted(obs, opt)
                    offsets.append(
                        np.r_[np.zeros(3), changed.velocity - obs.velocity, changed.spin - obs.spin]
                    )
                duration = int(np.clip(np.ceil(shot["eta"]) + 3, 3, HORIZON))
                tape = np.zeros((HORIZON, 5))
                actual = acts[i : i + duration, :5]
                tape[: len(actual)] = actual
                player.sim.current = player.sim.twin.state()
                predicted = player.sim.predict(
                    np.broadcast_to(tape, (len(options), HORIZON, 5)),
                    np.full(len(options), duration),
                    obs,
                    np.array(offsets),
                )
                samples.append(
                    dict(opening=i == min(notes), errors=[discrepancy(p, reading) for p in predicted])
                )
        player.sim.twin.advance(act)
    return samples


def fit(samples, options):
    if not samples:
        return list(options[0])
    errors = np.array([s["errors"] for s in samples])
    # Weak absolute preference for the unchanged simulator with little data.
    prior = np.linalg.norm(np.array(options) / [1.0, 60.0, 0.12, 0.15], axis=1) * 0.06
    return list(options[np.argmin(errors.sum(axis=0) + prior)])


def calibrate(source, output):
    doc = _read(source)
    if doc["privileged"]:
        raise ValueError("requires optical practice")
    player = RallyPlayer(FlightLaw())
    options = designs()
    rows = []
    try:
        for row in doc["rows"]:
            rows.append(dict(seed=row["seed"], samples=measurements(row, player, options)))
    finally:
        player.close()
    settings = {}
    for state in (True, False):
        samples = [s for row in rows for s in row["samples"] if s["opening"] == state]
        settings["opening" if state else "rally"] = fit(samples, options)
    _write(output, dict(options=options, rows=rows, settings=settings))
    print(settings, flush=True)


class Calibrated(RallyPlayer):
    def __init__(self, law, settings):
        super().__init__(law)
        self.settings = settings

    def search(self, estimate, observation, target):
        opt = self.settings["rally" if self.shots else "opening"]
        return super().search(estimate, adjusted(observation, opt), target)


def play(source, output, seed, count):
    stationary = Station(0.30, 0.82, "anchor")
    settings = _read(source)["settings"]
    law = stationary.player.law
    stationary.player.close()
    stationary.player = Calibrated(law, settings)
    result = dict(
        rows=[], settings=settings, spread=0.30, pace=0.82, rival="anchor", privileged=False, seed=seed
    )
    try:
        for s in np.random.default_rng(seed).integers(2**31, size=count):
            r = stationary.point(int(s))
            result["rows"].append(r)
            _write(output, result)
            print(int(s), r["longest"], flush=True)
    finally:
        stationary.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "play"))
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, default=8351422)
    p.add_argument("--count", type=int, default=5)
    args = p.parse_args()
    if args.mode == "fit":
        calibrate(args.source, args.output)
    else:
        play(args.source, args.output, args.seed, args.count)
