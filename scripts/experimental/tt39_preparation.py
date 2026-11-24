"""Learn how much to trust two preparatory joint movements from physical exchange rewards."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_joint_capture import observed
from .tt37_opponent import arena
from .tt37_visual_inertia import LearnedOptics

# Scales are in metres, metres/sec and radians. The observed flight and the current
# encoder configuration are the only descriptors used to choose preparation.
SCALE = np.array([0.14, 0.22, 0.10, 0.65, 0.8, 1.0, 0.60, 0.50, 0.65, 0.5, 0.7])
MIX = np.array([0.0, 0.5, 1.0])


class Preparation:
    def __init__(self, examples=(), *, fixed=None, learn=True):
        self.examples = list(examples)
        self.fixed = fixed
        self.learn = learn
        self.new = []
        self.begin()

    def begin(self):
        self.active = None
        self.mix = 1.0

    def choose(self, x):
        if self.fixed is not None:
            return self.fixed
        if not self.examples:
            return 1.0
        data = self.examples[-1200:]
        state = np.asarray([r[0] for r in data])
        weight = np.exp(-0.5 * np.sum(((state - x) / SCALE) ** 2, axis=1))
        votes = np.asarray([r[2] for r in data])
        ratios = np.asarray([r[1] for r in data])
        # Interpolate motor mixtures across nearby real observations. Mild shrinkage
        # to fleet-wide success prevents an unsupported local option from dominating.
        similarity = np.exp(-0.5 * ((ratios[:, None] - MIX) / 0.30) ** 2)
        marginal = (np.array([0.50, 0.55, 0.60]) * 6 + votes @ similarity) / (
            6 + similarity.sum(axis=0)
        )
        local = weight[:, None] * similarity
        estimate = (3 * marginal + votes @ local) / (3 + local.sum(axis=0))
        return float(MIX[np.argmax(estimate)])

    def observe(self, optical, q):
        if (
            self.active is None
            and optical is not None
            and optical[4] < -0.9
            and -0.25 < optical[1] < 0.85
        ):
            x = np.r_[optical, q[:5]]
            self.mix = self.choose(x)
            self.active = (x.tolist(), self.mix)
        return self.mix

    def feedback(self, near_score, finished):
        if self.active is not None and (near_score or finished):
            entry = (*self.active, float(near_score))
            self.new.append(entry)
            if self.learn:
                self.examples.append(entry)
            self.active = None


def gather(sources, optical, output):
    data, ids, summaries = [], set(), []
    starting = _read(sources[0])["ready"]
    with centered(starting):
        a = arena()
        renderer, rig, *_ = resources(a)
        try:
            for source in sources:
                d = _read(source)
                if len(d["rows"]) != len(d["cards"]) or d["ready"] != starting:
                    raise ValueError(f"Insufficient or incompatible course: {source}")
                ratio = float(bool(d.get("pupil_rollin")))
                prep = Preparation(fixed=ratio, learn=False)
                for c, r in zip(d["cards"], d["rows"], strict=True):
                    ids.add(c["id"])
                    a.launch(r["shot"], card=c)
                    a.partner.reset(seed=c["seed"])
                    eye = Vision(rig, a.control_dt, raw=True)
                    eye.eye = LearnedOptics(eye.eye, optical)
                    prep.begin()
                    for tick, u in enumerate(r["actions"]):
                        q = a.joints
                        x = observed(
                            eye, tick, renderer.capture() if tick % 2 == 0 else None, q
                        )
                        prep.observe(x, q)
                        _, _, done, _, info = a.exchange(u, FAR_COMMAND)
                        near = any(
                            e["kind"] == "legal_return" and e["side"] == 1
                            for e in info["events"]
                        )
                        prep.feedback(near, done)
                    if digest_json(a.events) != digest_json(r["events"]):
                        raise ValueError("Practice tape failed deterministic replay")
                print(source.name, len(prep.new), flush=True)
                data.extend(prep.new)
                summaries.append(
                    {
                        "source": source.name,
                        "sha256": _hash(source),
                        "mix": ratio,
                        "outcomes": dict(Counter(int(y) for _, _, y in prep.new)),
                    }
                )
        finally:
            renderer.close()
            a.close()
    _write(
        output,
        {
            "examples": data,
            "practice_ids": sorted(ids),
            "sources": summaries,
            "optics": _hash(optical),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    p.add_argument("--optics", type=Path, default=Path("artifacts/tt37/rebound.npz"))
    a = p.parse_args()
    gather(a.sources, a.optics, a.output)
