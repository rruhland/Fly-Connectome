"""Compare incoming flight at visible solo impacts with the demonstrated regulation approaches."""

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt27_flight_feedback import load_bank
from .tt27_optical_dynamics import fit_arc
from .tt28_toss_prior import OUT


def summary(values):
    return dict(n=len(values), quantiles=np.percentile(values, [0, 25, 50, 75, 100]).tolist())


def run():
    speeds = {}
    for kind in ("ordinary", "oblique", "staged"):
        rows = _read(OUT / f"toss-{kind}.json.gz")["rows"]
        speeds[kind] = [r["event"]["horizontal_in"] for r in rows if r["event"]]
    speeds["regulation"] = []
    for r in load_bank()["rows"]:
        # These are camera observations strictly before the training demonstration's first actual contact.
        t = np.array(r["trace"])[-5:]
        if np.max(np.diff(t[:, 0])) > 2 or np.any(np.diff(t[:, 2]) > 0):
            continue
        _, v = fit_arc(t)
        speeds["regulation"].append(float(np.linalg.norm(v[:2])))
    receipt = dict(speed_mps=speeds, summary={k: summary(v) for k, v in speeds.items()})
    _write(OUT / "motion-range.json", receipt)
    print(receipt["summary"], flush=True)


if __name__ == "__main__":
    run()
