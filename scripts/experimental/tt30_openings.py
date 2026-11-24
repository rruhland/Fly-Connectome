"""Select new measured openings from launch similarity before evaluating policies."""

import argparse
from pathlib import Path

import numpy as np

from .tt29_rally_physics import measured_launches


def nearest_openings(feeds):
    shots = measured_launches(feeds)
    anchors = (7271, 7345, 11292, 4187)
    exclude = set(anchors) | {10542, 4230, 4527, 15489}
    ids = sorted(shots)
    columns = ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z")
    features = np.array([[shots[i][k] for k in columns] for i in ids])
    scale = features.std(axis=0)
    selected = []
    for anchor in anchors:
        distance = np.linalg.norm((features - features[ids.index(anchor)]) / scale, axis=1)
        order = (ids[i] for i in np.argsort(distance, kind="stable"))
        pair = [i for i in order if i not in exclude and abs(i - anchor) > 20][:2]
        selected.extend(pair)
        exclude.update(pair)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    print(*nearest_openings(parser.parse_args().feeds))
