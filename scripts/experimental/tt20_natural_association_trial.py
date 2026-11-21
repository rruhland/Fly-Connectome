"""Check opt-in RGB association against the frozen TT20 tagged factorial."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt20_arm_tags_trial import DATA, EYES, SOURCE_TAPE, TagReplayArena, _summary
from .tt20_natural import NaturalViewTransport


FACTORIAL = DATA / 'arm-tag-factorial.json.gz'
RECEIPT = DATA / 'natural-association.json.gz'


def _position(track):
    return None if track is None else list(map(float, track.position))


def _pointset(label, settings, tapes, matched):
    _, posts, _, tags = label.split('_')
    arena = TagReplayArena(posts=posts == '1', tags=tags == '1', **settings)
    estimates = []
    try:
        for baseline in matched:
            reference = tapes[baseline['seed']]
            seed, server = reference['seed'], reference['role']
            arena.reset(seed=seed, server=server, render=False)
            eyes = {eye: NaturalViewTransport(eye, lens_fov=arena.lens_intrinsics(), flight=False,
                                             continuous_wrist=True, associate=True).tracker() for eye in EYES}
            pixels, physics = {eye: sha256() for eye in EYES}, sha256()
            frames = []
            for tick, action in enumerate(reference['actions']):
                images = arena.render_cameras()
                frame = {}
                for eye, stream in EYES.items():
                    tracker, image = eyes[eye], images[stream]
                    pixels[eye].update(image)
                    tracker.update({stream: image}, tick=tick)
                    frame[eye] = dict(ball=_position(tracker.ball), blade=_position(tracker.actor),
                                      plane=tracker.actor_normal is not None)
                frames.append(frame)
                arena.step(action, render=False)
                physics.update(np.r_[arena.data.qpos, arena.data.qvel].astype('<f8').tobytes())
            if (arena.point_result() != reference['outcome']
                    or json.dumps(arena.events) != json.dumps(reference['events'])
                    or {eye: h.hexdigest() for eye, h in pixels.items()} != baseline['rgb_sha256']
                    or physics.hexdigest() != baseline['physics_sha256']):
                raise ValueError(f'association replay differs from archived RGB or physics for {label}, seed {seed}')
            estimates.append(dict(seed=seed, server=server, frames=frames,
                                  physics_sha256=baseline['physics_sha256'], rgb_sha256=baseline['rgb_sha256']))
            print(label, seed, len(frames), flush=True)
    finally:
        arena.close()
    return estimates


def _measure(base, after, eye):
    results = {}
    matching = [(original, newer) for source, result in zip(base, after, strict=True)
                for original, newer in zip(source['frames'], result['frames'], strict=True)]
    for role, key in (('ball', 'ball'), ('blade', 'silhouette')):
        pairs = [(a['views'][eye][key], b[eye][role], a['truth'][role]) for a, b in matching]
        distances = [[None if measured is None else 1000. * np.linalg.norm(np.subtract(measured, truth))
                      for measured in (baseline, corrected)] for baseline, corrected, truth in pairs]
        errors = {mode: [pair[i] for pair in distances if pair[i] is not None]
                  for i, mode in enumerate(('baseline', 'associated'))}
        paired = [pair for pair in distances if None not in pair]
        results[role] = dict(frames=len(pairs), **{k: _summary(v) for k, v in errors.items()},
                             within_50mm={k: sum(int(value < 50.) for value in v) for k, v in errors.items()},
                             paired=len(paired), paired_median_mm=[float(np.median(v)) for v in zip(*paired)],
                             changed=sum(baseline != corrected for baseline, corrected, _ in pairs))
    results['planes'] = sum(b[eye]['plane'] for _, b in matching)
    return results


def run(output, workers):
    source = _read(SOURCE_TAPE)
    expected = _read(FACTORIAL)
    tapes = {row['seed']: row for row in source['points']}
    reference = {key: [row for row in rows if row['split'] == 'held']
                 for key, rows in expected['scenarios'].items()}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        jobs = {key: pool.submit(_pointset, key, expected['settings'], tapes, rows) for key, rows in reference.items()}
        cases = {key: value.result() for key, value in jobs.items()}
    metrics = {key: {eye: _measure(reference[key], rows, eye) for eye in EYES} for key, rows in cases.items()}
    names = ('tt20_natural.py', 'tt20_natural_geometry.py', 'tt20_natural_association.py',
             'tt20_natural_association_trial.py', 'tt20_fixture.py')
    _write(output, dict(schema='tt20-natural-association-1', baseline_sha256=_hash(FACTORIAL),
                        source_sha256=_hash(SOURCE_TAPE), scenes=cases, metrics=metrics,
                        implementation={name: _hash(Path(__file__).with_name(name)) for name in names},
                        replays=sum(len(rows) for rows in cases.values())))
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.workers)
