"""Estimate Pong flight and opposition from timed public entity positions."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from pong_reachable_placement import ShotGeometry


def fit_geometry(recordings):
    samples, flights, wall, centers, caps, pursuits = [], [], [], [], [], []
    for recording in recordings:
        array = np.array([[time, *(own or [np.nan] * 2), *(ball or [np.nan] * 2),
                           *(opponent or [np.nan] * 2)]
                          for time, own, ball, opponent in recording])
        t, own_y, _, ball_y, ball_x, opponent_y, opponent_x = array.T
        opponent_y = np.where(abs(opponent_x) > .38, opponent_y, np.nan)
        centers.extend(opponent_x[opponent_x > .38])
        wall.extend(abs(ball_y[np.isfinite(ball_y)]))
        indices = np.flatnonzero(np.isfinite(ball_x))
        for before, after in zip(indices, indices[8:]):
            span = t[after] - t[before]
            if after - before != 8 or span < .1:
                continue
            if abs(ball_x[before]) < .34 and abs(ball_x[after]) < .34 and max(abs(ball_y[before]), abs(ball_y[after])) < .41:
                speed = np.linalg.norm(array[after, 3:5] - array[before, 3:5]) / span
                if .42 < speed < .66:
                    flights.append(speed)
        last = -1.
        for index in indices:
            now = t[index]
            if ball_x[index] > -.36 or now - last < .35:
                continue
            left = indices[(t[indices] >= now - .38) & (t[indices] <= now - .09)]
            right = indices[(t[indices] >= now + .09) & (t[indices] <= now + .38)]
            if len(left) < 8 or len(right) < 8:
                continue
            pre = np.polyfit(t[left] - now, array[left, 3:5], 1)
            post = np.polyfit(t[right] - now, array[right, 3:5], 1)
            if pre[0, 1] >= -.28 or post[0, 1] <= .28:
                continue
            offset_time = (post[1, 1] - pre[1, 1]) / (pre[0, 1] - post[0, 1])
            if abs(offset_time) > .055:
                continue
            paddle = np.flatnonzero(np.isfinite(own_y))
            near = paddle[np.argmin(abs(t[paddle] - now - offset_time))]
            if abs(t[near] - now - offset_time) > .06:
                continue
            contact_y, contact_x = post[1] + offset_time * post[0]
            _, incoming_x = pre[1] + offset_time * pre[0]
            if abs(contact_y) > .44 or abs(contact_x - incoming_x) > .01:
                continue
            residuals = np.polyval(post[:, 0], t[right] - now) - ball_y[right]
            if np.max(abs(residuals)) > .016:
                continue
            offset = contact_y - own_y[near]
            angle = np.arctan2(*post[0])
            if abs(offset) <= .12 and abs(angle) <= 1.1:
                samples.append((contact_x, offset, angle))
                last = now
        pair = np.isfinite(opponent_y[:-8]) & np.isfinite(opponent_y[8:])
        dt = t[8:] - t[:-8]
        velocities = (opponent_y[8:] - opponent_y[:-8]) / dt
        middle = slice(4, -4)
        errors = ball_y[middle] - opponent_y[middle]
        valid = pair & np.isfinite(errors) & (abs(opponent_y[middle]) < .365) & (abs(velocities) < .85)
        if valid.sum() > 30:
            cap = np.quantile(abs(velocities[valid]), .975)
            caps.append(cap)
            pursuits.extend(zip(errors[valid], velocities[valid], np.repeat(cap, valid.sum()), strict=True))
    if len(samples) < 8 or not flights or len(caps) < 4:
        raise ValueError('insufficient observed flights, own bounces or opponent motion')
    planes, offsets, angles = np.array(samples).T
    informative = abs(offsets) >= .018
    if informative.sum() < 4:
        raise ValueError('insufficient offset variation')
    ratio = np.median(angles[informative] / offsets[informative])
    gaps, opponent_velocities, limits = np.array(pursuits).T
    gains = np.linspace(1., 8., 57)
    residual = np.clip(gains[:, None] * gaps, -limits, limits) - opponent_velocities
    gain = gains[np.argmin(np.median(abs(residual), axis=1))]
    geometry = ShotGeometry(float(np.median(planes)), float(-np.median(planes)),
        float(np.quantile(wall, .997)), float(np.median(flights)), float(ratio), float(gain),
        tuple(float(x) for x in np.quantile(caps, [.23, 1.])))
    return geometry, dict(contacts=samples, flight_windows=len(flights), observed_opponent_speed_limits=caps,
                          pursuit_windows=len(pursuits), typical_opposing_entity_station=float(np.median(centers)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    files = sorted(args.recordings.glob('capture-*.json'))
    geometry, diagnostics = fit_geometry([json.loads(path.read_bytes()) for path in files])
    with args.output.open('x') as stream:
        json.dump(dict(geometry=asdict(geometry), diagnostics=diagnostics,
            public_recordings={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}),
            stream, indent=2)
