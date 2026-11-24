"""Check live camera estimates against independently propagated physical arrivals at the decision tick."""

import argparse
from pathlib import Path

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _read, _write
from .tt31_ready import centered
from .tt37_opponent import arena


def evaluate(sources, output):
    data = []
    for source in sources:
        doc = _read(source)
        depth = doc["depth"]
        joint.RECEIVE_Y = depth
        with centered(doc["ready"]):
            a = arena()
            stereo, rig, *_ = joint.resources(a)
            try:
                for card, row in zip(doc["cards"], doc["rows"]):
                    a.launch(row["shot"], card=card)
                    a.partner.reset(seed=card["seed"])
                    eye = joint.Vision(rig, a.control_dt)
                    chosen = {e["tick"]: e for e in row["records"]}
                    for t, action in enumerate(row["actions"]):
                        raw = eye.update(
                            t, stereo.capture() if t % 2 == 0 else None, a.joints
                        )
                        if (
                            raw is not None
                            and t % 2 == 0
                            and 0.03 < raw[0] < 0.30
                            and a._last_hitter == -1
                        ):
                            path = a.partner.flight.path(
                                a.ball,
                                a.velocity,
                                a.spin,
                                1,
                                bounced=a._received_bounce,
                                horizon=0.7,
                            )
                            after = (
                                np.flatnonzero(path[:, 2] <= depth) if len(path) else []
                            )
                            if len(after) and after[0] > 0:
                                j = after[0]
                                p = path[j - 1]
                                v = path[j] - p
                                s = p + v * ((depth - p[2]) / v[2])
                                data.append(
                                    {
                                        "id": card["id"],
                                        "tick": t,
                                        "source": source.name,
                                        "bounced": a._received_bounce,
                                        "optical": raw[:6].tolist(),
                                        "physical": s[[0, 1, 3, 4, 5, 6]].tolist(),
                                        "commit": t in chosen,
                                    }
                                )
                        a.exchange(action, np.zeros(5))
            finally:
                stereo.close()
                a.close()
    _write(output, {"frames": data})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    args = p.parse_args()
    evaluate(args.sources, args.output)
