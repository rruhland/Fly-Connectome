"""Select robust markerless-motion features on practice releases, then measure new streams."""


import numpy as np

from .tt50_far_readout import ROOT, GATES, examples, fit
from .tt4_transfer_trial import _write


def columns(mode):
    if mode == "ball":
        return np.arange(6)
    # Both cameras contribute current silhouette and its short motion history.
    which = {
        "position": (0, 1),
        "face": (2, 3, 4, 5, 8, 9),
        "position_face": (0, 1, 2, 3, 4, 5, 8, 9),
        "full": tuple(range(14)),
    }[mode]
    return np.array(
        [
            *range(6),
            *[
                6 + block * 28 + cam * 14 + k
                for block in (0, 1, 2)
                for cam in (0, 1)
                for k in which
            ],
        ]
    )


def candidates():
    return [
        (name, a)
        for name in ("ball", "position", "face", "position_face", "full")
        for a in (4.0, 32.0, 160.0, 640.0)
    ]


def solve_gate(X, Y, group):
    ops = candidates()
    score = np.zeros((len(ops), Y.shape[1]))
    folds = np.array_split(np.sort(np.unique(group)), 4)
    for i, (name, alpha) in enumerate(ops):
        idx = columns(name)
        for ids in folds:
            held = np.isin(group, ids)
            m, s, c, b = fit(X[~held][:, idx], Y[~held], group[~held], alpha)
            err = np.abs((X[held][:, idx] - m) / s @ c + b - Y[held])
            score[i] += np.sum(err, axis=0)
    # Each outgoing quantity can be more or less observable from the face.
    # Collect all five linear maps into a common design with frozen dimensions.
    win = np.argmin(score, axis=0)
    W = np.zeros((X.shape[1], Y.shape[1]))
    b = np.empty(Y.shape[1])
    for j, i in enumerate(win):
        name, a = ops[i]
        idx = columns(name)
        m, s, c, v = fit(X[:, idx], Y, group, a)
        W[idx, j] = c[:, j] / s
        b[j] = v[j] - m @ W[idx, j]
    return W, b, [ops[i] for i in win]


def main():
    paths = sorted((ROOT / "observations").glob("case-*.npz"))
    with np.load(ROOT / "model.npz") as z:
        bg = z["background"]
    cache = ROOT / "features.npz"
    if cache.exists():
        with np.load(cache) as z:
            gate, x, y, group, hold = (
                z[k] for k in ("gate", "x", "y", "group", "hold")
            )
    else:
        if len(paths) != 24:
            raise ValueError("Incomplete registered sensor test")
        rows = [(r, i >= 16) for i, p in enumerate(paths) for r in examples(p, bg)]
        gate, x, y, group, hold = [
            np.array([r[k] for r, h in rows]) for k in (0, 3, 4, 2)
        ] + [np.array([h for r, h in rows])]
        np.savez_compressed(cache, gate=gate, x=x, y=y, group=group, hold=hold)
    model = dict(background=bg)
    report = {}
    for j, g in enumerate(GATES):
        op = (gate == g) & ~hold
        te = (gate == g) & hold
        W, b, winners = solve_gate(x[op], y[op], group[op])
        pred = x[te] @ W + b
        shuf = x[te].copy()
        for k in np.unique(group[te]):
            ii = group[te] == k
            shuf[ii, 6:] = np.roll(shuf[ii, 6:], 1, axis=0)
        report[str(g)] = dict(
            winners=winners,
            n_train=int(op.sum()),
            n_held=int(te.sum()),
            mae=np.mean(abs(pred - y[te]), axis=0).tolist(),
            scrambled=np.mean(abs(shuf @ W + b - y[te]), axis=0).tolist(),
        )
        for k, v in zip(
            ("mean", "scale", "coef", "bias"), (np.zeros(len(W)), np.ones(len(W)), W, b)
        ):
            model[f"{j}_{k}"] = v
    np.savez_compressed(ROOT / "group-model.npz", **model)
    _write(ROOT / "group-evaluation.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
