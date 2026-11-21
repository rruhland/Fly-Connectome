"""Carry anonymous trajectory-interaction weights to target tool identification."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .tt3_pong import source_episodes
from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt20_natural_trial import ORIGIN, _camera_truth
from .tt22_anonymous import CandidateFrames, pixel
from .tt22_anonymous_babble import BabblingEye
from .tt22_anonymous_babble_trial import OUTPUT as BABBLE, STREAMS, TARGET, probe
from .tt22_anonymous_interaction import choose, evidence, examples, fit_source, score_source
from .tt22_anonymous_trial import TARGET as PRACTICE
from .tt22_opponent_trial import arena_for
from .tt22_relation_trial import library


OUTPUT = TARGET.with_name('anonymous-interaction.json.gz')
SOURCE_SEED = 701890511
SHUFFLE_SEED = 930585141
TRAIN, HELD = 48, 16
MODES = ('pong_interaction', 'timing_shuffled', 'pure_active', 'pong_control')


def source(shared):
    seeds = np.random.default_rng(SOURCE_SEED).integers(2**31, size=TRAIN+HELD).tolist()
    torch.set_num_threads(1)
    scenes = []
    for samples in source_episodes(seeds, max_ticks=1080, batch_size=16):
        tracker = CandidateFrames(shared)
        tape = [tracker.observe(sample.frames) for sample in samples]
        scenes.append(examples(tape))
    models = fit_source(scenes[:TRAIN])
    scores = score_source(models, scenes[TRAIN:])
    print('Pong held', scores, flush=True)
    return models, dict(seeds=seeds, source_train=TRAIN, source_held=HELD,
                        weights=[model.tolist() for model in models], prediction=scores)


def history(arena, row, patches, *, observation=None):
    arena.reset(seed=row['seed'], server=row['role'], render=False)
    eyes = {key: BabblingEye(patches[key], RoleModel.untrained()) for key in STREAMS}
    tracks = {key: [] for key in STREAMS}
    digest = sha256()
    previous = None
    for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
        for key, frame in arena.render_cameras().items():
            digest.update(key.encode())
            digest.update(frame)
            tape, direct, _ = eyes[key].observe(frame, previous, bind=False)
            if observation is None:
                tracks[key].append(tape)
            else:
                observation(tick, key, frame, direct, eyes[key].votes)
        arena._rival_attempt = True
        arena._aim = lambda *args: np.asarray(far)
        arena.step(near, render=False)
        previous = near
    if digest.hexdigest() != row.get('rgb', row.get('rgb_sha256')):
        raise ValueError('interaction assay RGB differs from the native replay')
    return tracks


def practice(arena, rows, patches, models):
    memories = {mode: {key: defaultdict(float) for key in STREAMS}
                for mode in ('pong_interaction', 'timing_shuffled')}
    rng = np.random.default_rng(SHUFFLE_SEED)
    receipts = []
    for row in rows:
        tape = history(arena, row, patches)
        count = {}
        for key in STREAMS:
            design = examples(tape[key], actor_ids=set(range(len(patches[key]))))
            count[key] = dict(pairs=len(design.rows), changes=int(design.targets.sum()))
            for mode, shuffle in [('pong_interaction', None), ('timing_shuffled', rng)]:
                for patch, value in evidence(models, design, shuffled=shuffle).items():
                    memories[mode][key][patch] += value
        receipts.append(dict(seed=row['seed'], rgb_sha256=row['rgb'], interactions=count))
        print('practice', row['seed'], count, flush=True)
    return memories, receipts


def assess(arena, rows, replayed, patches, memories):
    results = []
    for row, measured in zip(rows, replayed, strict=True):
        measurements = {key: {} for key in STREAMS}

        def observe(tick, key, frame, direct, votes):
            audit = measured['streams'][key][tick]
            if not (audit['before_contact'] and audit['actor']['visible']):
                return
            actors = {mode: choose(votes, memory[key]) for mode, memory in memories.items()}
            actors['pure_active'] = direct
            reference = _camera_truth(arena, key).project(arena.paddle - ORIGIN)[:2]
            errors = {}
            for name, track in actors.items():
                errors[name] = None if track is None else float(np.linalg.norm(pixel(track, frame.shape) - reference))
            errors['pong_control'] = audit['actor']['selected']['pong']
            measurements[key][tick] = errors

        history(arena, row, patches, observation=observe)
        results.append(dict(seed=row['seed'], rgb_sha256=row['rgb_sha256'], measurements=measurements))
        print('held', row['seed'], summarize(results), flush=True)
    return results


def summarize(results):
    summary = {}
    for key in (*STREAMS, 'joint'):
        records = []
        for point in results:
            frames = point['measurements']
            if key == 'joint':
                for tick in frames['left'].keys() & frames['right'].keys():
                    records.append({mode: (frames['left'][tick][mode], frames['right'][tick][mode])
                                    for mode in MODES})
            else:
                records.extend({mode: (error,) for mode, error in record.items()} for record in frames[key].values())
        summary[key] = dict(visible=len(records))
        for mode in MODES:
            estimates = [max(record[mode]) for record in records if None not in record[mode]]
            summary[key][mode] = dict(proposed=len(estimates), correct=sum(error <= 8. for error in estimates),
                                      median_error_px=float(np.median(estimates)) if estimates else None)
    return summary


def run(output):
    data = _read(BABBLE)
    practice_rows = _read(PRACTICE)['variants']['anchor']['rows']
    held = [row['options']['0'] for row in _read(TARGET)['rows']]
    if not data['complete'] or [row['seed'] for row in data['points']] != [row['seed'] for row in held]:
        raise ValueError('complete the same held active-role audit before interaction evaluation')
    seed_sets = [[row['seed'] for row in values] for values in (practice_rows, held)]
    if not set(seed_sets[0]).isdisjoint(seed_sets[1]):
        raise ValueError('practice and held replay seeds overlap')
    with TemporaryDirectory(prefix='tt22-interactions-') as folder:
        models, pong = source(library(folder))
    arena = arena_for('anchor', live=True)
    try:
        patches, hashes = probe(arena)
        if hashes != data['probe_rgb_sha256']:
            raise ValueError('the pre-contact grayscale probe changed')
        memories, receipts = practice(arena, practice_rows, patches, models)
        document = dict(schema='tt22-anonymous-pong-interaction-1', source=pong, practice=receipts,
                        practice_sha256=_hash(PRACTICE), target_sha256=_hash(TARGET),
                        babble_sha256=_hash(BABBLE), memory=memories, complete=False)
        _write(output, document)
        verdicts = assess(arena, held, data['points'], patches, memories)
        document.update(held=verdicts, summary=summarize(verdicts), complete=True)
        _write(output, document)
    finally:
        arena.close()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    run(parser.parse_args().output)
