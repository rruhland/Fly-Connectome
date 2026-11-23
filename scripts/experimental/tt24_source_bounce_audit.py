"""Measure whether wrist RGB falls in the frozen native duration model's support."""

from collections import deque
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt24_bounce import BounceArena
from .tt24_motor import FEATURES as MOTOR_FEATURES, MotorEffects, MotorPolicy, optical_sightings
from .tt24_source import CHECKPOINT, FEATURES, MODES, VisualConsequence


RECEIPT = CHECKPOINT.parent / 'source-wrist-audit.json.gz'
SCHEDULE = 467742116
SEEDS = 6
MOTORS = 5
HORIZONS = (8, 14)
WINDOWS = (4, 5)


def sight_batch(records, dt, fits):
    windows = {width: [] for width in WINDOWS}
    history = deque(maxlen=max(WINDOWS))
    for tick, ball, paddle in records:
        if ball is None or paddle is None:
            history.clear()
            continue
        history.append((paddle.center, ball.center, paddle.half_span))
        for width in WINDOWS:
            if len(history) < width:
                continue
            a, b, scales = tuple(np.asarray(samples) for samples in zip(*list(history)[-width:], strict=True))
            mode_scores = {name: model.score_history(a, b, dt=dt, scale=np.median(scales), details=True)
                           for name, model in fits.items()}
            support = float(mode_scores['pong']['support'])
            scores = {name: float(item['score'] - fits[name].prior) for name, item in mode_scores.items()}
            windows[width].append(dict(tick=tick, support=support, scores=scores,
                                       features=mode_scores['pong']['features'].tolist()))
    return windows


def compare(launch, candidates, arena, models):
    output = []
    for weights in candidates:
        obs = arena.reset(seed=launch, split='held')
        controller = MotorPolicy(weights, MotorEffects(arena.action_size))
        records = []
        tick = 0
        done = False
        while not done:
            if tick <= HORIZONS[-1]:
                ball, paddle = optical_sightings(obs['rgb'])
                records.append((tick, ball, paddle))
            action = controller.command(obs, sampled='rgb' in obs)
            tick += 1
            obs, _, done = arena.step(action, render=True)
        shots = sight_batch(records, arena.dt, models)
        output.append(dict(weights=weights.tolist(), result=arena.judge_metrics(), windows=shots))
    return output


def summaries(all_rows):
    result = {}
    for width in WINDOWS:
        by_horizon = {}
        for horizon in HORIZONS:
            cue = [[([sample for sample in row['windows'][width] if sample['tick'] <= horizon], row['result']['rebounds'])
                    for row in group] for group in all_rows]
            coverage = np.asarray([s['support'] for group in cue for windows, _ in group for s in windows])
            payload = dict(episodes_with_windows=sum(bool(w) for group in cue for w, _ in group),
                           windows=len(coverage), support_mean=float(np.mean(coverage)),
                           support_fraction=float(np.mean(coverage > 0.)))
            results = {}
            for mode in MODES:
                for reduction in ('mean', 'top_half'):
                    pairs, top, ties = [], [], []
                    for group in cue:
                        valid = np.array([bool(w) for w, _ in group])
                        score = []
                        for windows, _ in group:
                            values = sorted(s['scores'][mode] for s in windows)
                            values = values if reduction == 'mean' else values[len(values) // 2:]
                            score.append(float(np.mean(values)) if values else 0.)
                        y = np.asarray([outcomes for _, outcomes in group])
                        s = np.asarray(score)
                        informative = valid[:, None] & valid[None, :] & (y[:, None] > y[None, :])
                        order = np.sign(s[:, None] - s[None, :])
                        pairs.extend(((order[informative] + 1.) / 2.).tolist())
                        if len(y[valid]):
                            selected = valid & (s >= s[valid].max() - 1e-9)
                            top.append(float(np.mean(y[selected])))
                            ties.append(int(selected.sum()))
                    key = f'{mode}_{reduction}'
                    results[key] = dict(comparisons=len(pairs), concordance=float(np.mean(pairs)) if pairs else None,
                                        selected_rebounds=float(np.mean(top)) if top else None,
                                        mean_tie_count=float(np.mean(ties)) if top else None)
            by_horizon[horizon] = dict(payload, modes=results)
        result[width] = by_horizon
    return result


def main():
    began = perf_counter()
    rng = np.random.default_rng(SCHEDULE)
    launches = rng.integers(1, 2 ** 31 - 1, SEEDS).tolist()
    widths = len(MOTOR_FEATURES)
    weights = rng.normal(0., .45, size=(MOTORS - 1, 5, widths))
    weights = np.concatenate((np.zeros((1, 5, widths)), weights), axis=0)
    modes = {mode: VisualConsequence.load(mode) for mode in MODES}
    arena = BounceArena()
    try:
        rows = []
        for launch in launches:
            played = compare(launch, weights, arena, modes)
            rows.append(played)
            print('wrist', launch, [e['result']['rebounds'] for e in played],
                  [len(e['windows'][4]) for e in played], round(perf_counter() - began, 1), flush=True)
    finally:
        arena.close()
    findings = summaries(rows)
    print(findings, flush=True)
    _write(RECEIPT, dict(schema='tt24-source-wrist-audit-1', source_sha256=_hash(CHECKPOINT), schedule=SCHEDULE,
                         starts=launches, features=FEATURES, summary=findings, rows=rows,
                         seconds=perf_counter() - began))


if __name__ == '__main__':
    main()
