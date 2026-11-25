"""Replay full games and collect ordinary pre-contact video to future-camera supervision."""

import argparse
from pathlib import Path

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _read, _write
from .tt50_far_readout import GATES, Predictor, ROOT, TARGET, descriptor


def collect(path, name, start, limit):
    source = _read(path)
    target = ROOT / name
    target.mkdir(parents=True, exist_ok=True)
    observer = Predictor()
    with mpc.centered(source["ready"]):
        a = mpc.arena()
        renderer, rig, *_ = mpc.resources(a)
        try:
            for j in range(start, min(start + limit, len(source["rows"]))):
                file = target / f"game-{j:03d}.json.gz"
                if file.exists():
                    continue
                card, r = source["cards"][j], source["rows"][j]
                a.launch(r["shot"], card=card)
                a.partner.reset(seed=card["seed"])
                eye = mpc.Vision(rig, a.control_dt, raw=True)
                eye.eye = mpc.LearnedOptics(eye.eye, course.OPTIC)
                observer.reset()
                records = []
                pending = {}
                old = None
                for t, u in enumerate(r["actions"]):
                    pixels = renderer.capture() if t % 2 == 0 else None
                    x = mpc.observed(eye, t, pixels, a.joints)
                    pred = observer.update(pixels, x, t)
                    if pred is not None:
                        gate, y = pred
                        feature = descriptor(observer.history, x)
                        if gate == GATES[0] or old is not None:
                            pending.clear()
                        pending[gate] = dict(
                            gate=gate, tick=t, x=feature.tolist(), forecast=y.tolist()
                        )
                        old = None
                    if x is not None and x[4] < -0.9:
                        if old is not None and old[1] > -1.1 >= x[1] and pending:
                            future = (
                                old + (x - old) * ((-1.1 - old[1]) / (x[1] - old[1]))
                            )[TARGET]
                            records.append(
                                dict(
                                    tick=t,
                                    target=future.tolist(),
                                    decisions=list(pending.values()),
                                )
                            )
                            pending = {}
                        old = x
                    a.exchange(u, mpc.FAR_COMMAND)
                if digest_json(a.events) != digest_json(r["events"]):
                    raise RuntimeError(
                        "Issued commands did not reproduce the recorded physical game"
                    )
                _write(
                    file,
                    dict(
                        source=path.name,
                        id=card["id"],
                        index=j,
                        point_events=digest_json(a.events),
                        own_returns=r["score"]["learner"],
                        observed=records,
                    ),
                )
                print(j, len(records), r["score"]["learner"], flush=True)
        finally:
            renderer.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("name")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=48)
    args = p.parse_args()
    collect(args.source, args.name, args.start, args.limit)
