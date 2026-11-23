"""Commission an anonymous RGB silhouette and carry the frozen metric wrist pupil across repainting."""

import argparse
import gzip
from hashlib import sha256
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from scipy.ndimage import label

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_anonvision import AnonymousVision, _angular_distance
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _ellipse


ROOT = Path('artifacts/tt24')
CHECKPOINT = ROOT / 'bounce-ident-pupil.npz'
OUTPUT = ROOT / 'colour-carry.json.gz'
COMMISSION_SEEDS = (33126567, 862730875)
COMMISSION_FRAMES = 26
ASSESSMENT_STREAM = 1196328798
ASSESSMENT_DROPS = 16
REPAINT = ((.58, .07, .93, 1.), (.055, .82, .23, 1.))
CONDITIONS = ('hue_native', 'anonymous_native', 'hue_repainted', 'anonymous_repainted', 'passive')


def _chroma(rgb):
    red, green, blue = np.moveaxis(np.asarray(rgb, dtype=np.float32), -1, 0)
    x = 2. * red - green - blue
    y = np.sqrt(3.) * (green - blue)
    return np.arctan2(y, x), np.hypot(x, y) > 14. + .07 * (red + green + blue)


def _commission_drop(arena, seed):
    tracker = AnonymousVision()
    chosen = {}
    movement = np.zeros(arena.action_size)
    obs = arena.reset(seed=seed, split='practice')
    seen = sha256()
    for tick in range(COMMISSION_FRAMES):
        rgb = obs['rgb']
        seen.update(rgb)
        mover, _ = tracker.observe(rgb, movement)
        track = tracker.mover
        if mover is not None and track.hits >= 3:
            h, w = rgb.shape[:2]
            coords = np.rint(track.region.outline * h + ((w - 1.) / 2, (h - 1.) / 2)).astype(int)
            hue, valid = _chroma(rgb[coords[:, 1], coords[:, 0]])
            history = chosen.setdefault(track, dict(hues=[], motion=[]))
            history['hues'].extend(hue[valid].tolist())
            history['motion'].append(track.mover_score())
        obs, _, done = arena.step(movement)
        if done:
            break
    if not chosen:
        raise ValueError(f'no independently moving region was acquired on unlabeled drop {seed}')
    winner = max(chosen.values(), key=lambda row: sum(row['motion']))
    return np.asarray(winner['hues']), dict(seed=seed, frames=tick + 1, supporting_frames=len(winner['motion']),
                                             contour_pixels=len(winner['hues']), rgb_sha256=seen.hexdigest())


def _commission(arena):
    samples, receipts = zip(*[_commission_drop(arena, seed) for seed in COMMISSION_SEEDS], strict=True)
    centers = np.array([np.angle(np.mean(np.exp(1j * angles))) for angles in samples])
    if _angular_distance(*centers) > .28:
        raise ValueError('the two motions did not identify a consistent appearance')
    hue = float(np.angle(np.mean(np.exp(1j * centers))))
    support = np.concatenate(samples)
    half_width = float(np.clip(np.quantile(_angular_distance(support, hue), .98) + .10, .14, .29))
    return dict(opponent_hue_radians=hue, angular_half_width=half_width, drops=receipts)


def _ellipse_from_mask(mask):
    """Extract the same subpixel silhouette moments as the metric pupil's detector."""
    labels, count = label(mask)
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = sizes.argmax()
    if sizes[component] < 15:
        return None
    yy, xx = np.nonzero(labels == component)
    height, width = mask.shape
    x0, y0 = xx.mean(), yy.mean()
    radius = np.sqrt(len(xx) / np.pi)
    left = np.full(height, width)
    right = np.full(height, -1)
    top = np.full(width, height)
    bottom = np.full(width, -1)
    np.minimum.at(left, yy, xx)
    np.maximum.at(right, yy, xx)
    np.minimum.at(top, xx, yy)
    np.maximum.at(bottom, xx, yy)
    # Exclude clipped image edges when fitting the visible contour.
    a = np.flatnonzero((left > 0) & (left < width))
    b = np.flatnonzero((right >= 0) & (right < width - 1))
    c = np.flatnonzero((top > 0) & (top < height))
    d = np.flatnonzero((bottom >= 0) & (bottom < height - 1))
    edge = np.concatenate((np.column_stack((left[a] - .5, a)), np.column_stack((right[b] + .5, b)),
                           np.column_stack((c, top[c] - .5)), np.column_stack((d, bottom[d] + .5))))
    if len(edge) >= 12:
        x, y = ((edge - (x0, y0)) / radius).T
        design = np.column_stack((x * x, x * y, y * y, x, y))
        a, b, c, d, e = np.linalg.lstsq(design, np.ones(len(x)), rcond=None)[0]
        quad = np.array(((a, b / 2.), (b / 2., c)))
        strength = np.linalg.eigvalsh(quad)
        if min(strength) > .04:
            offset = np.linalg.solve(quad, -.5 * np.array((d, e)))
            if np.linalg.norm(offset) < 2.5:
                x0, y0 = np.array((x0, y0)) + radius * offset
                radius *= np.sqrt((1. + offset @ quad @ offset) / max(strength))
    return np.array((x0 - (width - 1) / 2., (height - 1) / 2. - y0, radius))


def _template_fit(template, rgb):
    hue, present = _chroma(rgb)
    matched = _angular_distance(hue, template['opponent_hue_radians']) <= template['angular_half_width']
    return _ellipse_from_mask(present & matched)


def _set_appearance(arena, materials):
    arena._model.geom_rgba[[arena._ball_geom, arena._blade]] = materials


def _remove_marks(arena):
    model = arena._model
    decorative = (model.geom_contype == 0) & (model.geom_conaffinity == 0)
    thin_table = (model.geom_bodyid == 0) & (abs(model.geom_pos[:, 2] - .7605) < .0001) & (model.geom_size[:, 2] < .001)
    tags = np.array([model.geom(i).name.startswith(('tt18_landmark_', 'tt19_rim_', 'tt19_tip_mark'))
                     for i in range(model.ngeom)])
    removed = np.flatnonzero(decorative & (thin_table | tags))
    if (decorative & thin_table).sum() != 5:
        raise ValueError('expected five purely visual table stripes in the stock fixture')
    model.geom_group[removed] = 3
    if arena._scene_option.geomgroup[3]:
        raise ValueError('the removed markings are still enabled')
    return removed.tolist()


def _initial_state(arena):
    data = arena._data
    return sha256(np.r_[data.qpos, data.qvel, data.ctrl].astype('<f8').tobytes()).hexdigest()


def _play(arena, seed, mode, template, *, tape=None):
    live = mode != 'passive' and tape is None
    obs = arena.reset(seed=seed, split='held', render=live)
    initial = _initial_state(arena)
    policy = WristPupil.load(CHECKPOINT) if live else None
    valid = 0
    detector_s = 0.
    if live:
        def measure(observation):
            nonlocal valid, detector_s
            start = perf_counter()
            rgb = observation['rgb']
            ellipse = _ellipse(rgb) if template is None else _template_fit(template, rgb)
            detector_s += perf_counter() - start
            valid += int(ellipse is not None)
            return policy.sensor.read(ellipse, observation['joints'])
        policy.sensor.observe = measure
    actions = []
    rgb_hash = sha256() if live else None
    for tick in range(arena.max_steps):
        if live:
            rgb_hash.update(obs['rgb'])
            action = policy.command(obs)
        else:
            action = np.zeros(arena.action_size) if tape is None else np.asarray(tape[tick])
        actions.append(action.tolist())
        obs, _, done = arena.step(action, render=live)
        if tape is not None and done != (tick + 1 == len(tape)):
            raise ValueError(f'the physical tape changed termination on seed {seed}')
        if done:
            break
    return dict(metrics=arena.judge_metrics(), actions=actions, initial_state=initial, retinal_detections=valid,
                detector_seconds=detector_s, wrist_rgb_sha256=None if rgb_hash is None else rgb_hash.hexdigest())


def _fresh(seeds):
    probe = re.compile(rb'"seed"\s*:\s*(?:' + b'|'.join(str(n).encode() for n in seeds) + rb')\b')
    for path in ROOT.iterdir():
        if path.name.startswith('colour-carry'):
            continue
        if path.name.endswith('.json.gz'):
            body = gzip.decompress(path.read_bytes())
        elif path.suffix == '.json':
            body = path.read_bytes()
        else:
            continue
        if probe.search(body):
            raise ValueError(f'an assessment release was already used in {path}')


def _summary(cases):
    table = {}
    for mode in CONDITIONS:
        trials = [case['conditions'][mode] for case in cases]
        counts = [row['metrics']['longest_consecutive_rebounds'] for row in trials]
        frames = sum(len(row['actions']) for row in trials)
        table[mode] = dict(mean=float(np.mean(counts)), counts=counts, maximum=max(counts),
                           at_least_ten=sum(value >= 10 for value in counts),
                           retinal_detections=sum(row['retinal_detections'] for row in trials),
                           frames=frames, milliseconds_per_frame=1000. * sum(row['detector_seconds'] for row in trials) / frames)
    pairs = {}
    for this, reference in (('anonymous_native', 'hue_native'), ('anonymous_repainted', 'hue_repainted'),
                            ('anonymous_repainted', 'hue_native'), ('anonymous_repainted', 'anonymous_native')):
        diff = np.array(table[this]['counts']) - table[reference]['counts']
        pairs[f'{this}_vs_{reference}'] = dict(mean_gain=float(diff.mean()), wins=int((diff > 0).sum()),
                                               ties=int((diff == 0).sum()), losses=int((diff < 0).sum()))
    return dict(conditions=table, paired=pairs, verified_tapes=len(cases) * len(CONDITIONS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    seeds = np.random.default_rng(ASSESSMENT_STREAM).integers(1, 2**31 - 1, ASSESSMENT_DROPS).tolist()
    if set(seeds).intersection(COMMISSION_SEEDS) or len(set(seeds)) != len(seeds):
        raise ValueError('the paired assessment needs fresh releases')
    arena = BounceArena(view='wrist', width=192, height=144)
    started = perf_counter()
    try:
        materials = {'native': arena._model.geom_rgba[[arena._ball_geom, arena._blade]].copy(),
                     'repainted': REPAINT}
        removed = _remove_marks(arena)
        meta = dict(schema='tt24-colour-carry-1', checkpoint_sha256=_hash(CHECKPOINT), assessment_seeds=seeds,
                    commission_seeds=COMMISSION_SEEDS, commission_frames_per_drop=COMMISSION_FRAMES,
                    repaint_rgba=REPAINT, removed_noncolliding_mark_geoms=removed, conditions=CONDITIONS,
                    dimensions=[arena.width, arena.height], motor_training_runs=0)
        if args.output.exists():
            result = _read(args.output)
            if digest_json(result['metadata']) != digest_json(meta):
                raise ValueError('the frozen comparison design changed')
        else:
            _fresh(seeds)
            result = dict(metadata=meta, commissioning={}, cases=[])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write(args.output, result)
        for appearance, rgba in materials.items():
            if appearance not in result['commissioning']:
                _set_appearance(arena, rgba)
                result['commissioning'][appearance] = _commission(arena)
                _write(args.output, result)
                print(appearance, result['commissioning'][appearance], flush=True)
        for seed in seeds[len(result['cases']):]:
            case = dict(seed=seed, conditions={})
            for mode in CONDITIONS:
                appearance = 'repainted' if mode.endswith('repainted') else 'native'
                _set_appearance(arena, materials[appearance])
                template = result['commissioning'][appearance] if mode.startswith('anonymous') else None
                played = _play(arena, seed, mode, template)
                tape = _play(arena, seed, mode, None, tape=played['actions'])
                if digest_json(played['metrics']) != digest_json(tape['metrics']):
                    raise ValueError(f'physical replay changed the score or events on seed {seed}')
                case['conditions'][mode] = played
            if len({row['initial_state'] for row in case['conditions'].values()}) != 1:
                raise ValueError('the two palettes started from different physical states')
            result['cases'].append(case)
            _write(args.output, result)
            print(seed, [row['metrics']['longest_consecutive_rebounds'] for row in case['conditions'].values()], flush=True)
    finally:
        arena.close()
    result['summary'] = _summary(result['cases'])
    result['wall_seconds_last_invocation'] = perf_counter() - started
    _write(args.output, result)
    print(result['summary'], flush=True)


if __name__ == '__main__':
    main()
